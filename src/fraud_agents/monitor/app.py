"""The Fraud System Monitor (Part 1 · Step 8; Analyst page in Part 3 · Step 1; the rest in Step 2).

Pages, picked in the sidebar:
- Monitor: every service and agent (running or stopped, how busy), decision and
  final-outcome totals, the latest events, and a control panel to start a replay. Read-only.
- Payment trace: one payment's whole journey, event by event, with the AI calls it used.
- Caught payments: what was flagged or held, checked against the dataset's answer key
  (shown here for testing only: the agents never see it).
- SMS inbox: every text the bank sent and every customer reply, per customer.
- Analyst: the cases the system sent to a human. The analyst reads the evidence and
  clicks Release or Block. That publishes ONE event (case.reviewed); the Decision agent
  turns it into the final outcome and the Notification agent texts the customer.
  The page never changes an outcome itself, and it never shows the answer key.

Run from the project folder:  streamlit run src/fraud_agents/monitor/app.py
Then open http://localhost:8501
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import psycopg
import redis
import streamlit as st

from fraud_agents.shared import bus, db
from fraud_agents.shared.events import EVENT_TYPES as ALL_TYPES
from fraud_agents.shared.events import CaseReviewed, CaseReviewedPayload, derived_event_id, now_utc

GATEWAY_HEALTH = "http://localhost:8000/health"
SMS_HEALTH = "http://localhost:8100/health"
REPLAY_LOG = Path("data/processed/replay_from_monitor.log")
ANSWER_KEY_FILE = Path("data/raw/fraudTest.csv")
SMS_LOG = "sms:log"   # the mock SMS provider's record of every text in and out
RUN_KEY = "monitor:run"   # the current run: when it started, and each agent's count at that moment

AGENTS = [  # (name on the board, consumer group, the streams it reads)
    ("Risk Scoring", "risk-scoring", ["transaction.received"]),
    ("Decision", "decision", ["risk.scored", "verification.completed", "investigation.completed",
                              "case.escalated", "case.reviewed"]),
    ("Investigation (AI)", "investigation", ["transaction.decided"]),
    ("Verification (AI)", "verification", ["investigation.completed", "customer.replied"]),
    ("Notification (AI)", "notification", ["transaction.resolved"]),
]

st.set_page_config(page_title="Fraud System Monitor", layout="wide")


@st.cache_resource
def redis_client():
    return bus.connect()


@st.cache_resource
def pg_connection():
    return db.connect()


@st.cache_resource
def replay_holder():
    return {"process": None}  # shared by every browser tab


def group_info(r, event_type: str, group: str) -> dict:
    """How far a consumer group has got in a stream, and whether anyone is reading it."""
    stream = bus.stream_for(event_type)
    running = bool(r.exists(bus.alive_key(group, stream)))  # the note each agent renews
    try:
        groups = {g["name"]: g for g in r.xinfo_groups(stream)}
    except redis.ResponseError:  # the stream doesn't exist yet
        return {"handled": 0, "waiting": 0, "running": running}
    g = groups.get(group)
    if g is None:
        return {"handled": 0, "waiting": r.xlen(stream), "running": running}
    lag = g.get("lag")
    return {
        "handled": (g.get("entries-read") or 0) - g["pending"],
        "waiting": (lag if lag is not None else 0) + g["pending"],
        "running": running,
    }


def service_up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


def agent_info(r, group: str, streams: list[str], run: dict | None = None) -> dict:
    """An agent's totals over every stream it reads; running only if it reads all of them.

    With a run, 'handled' counts from the run's start; 'waiting' is always the live queue."""
    parts = [group_info(r, t, group) for t in streams]
    start = sum(int((run or {}).get(f"base:{group}:{t}", 0)) for t in streams)
    return {"handled": max(0, sum(x["handled"] for x in parts) - start),
            "waiting": sum(x["waiting"] for x in parts),
            "running": all(x["running"] for x in parts)}


# ---------- Runs: counters from zero, and a clock (Part 3 · Step 3) ----------

def start_run(r, label: str, planned: int = 0) -> None:
    """Mark the start of a new run: remember the time and every agent's count right now."""
    base = {f"base:{group}:{t}": group_info(r, t, group)["handled"]
            for _, group, streams in AGENTS for t in streams}
    r.delete(RUN_KEY)
    r.hset(RUN_KEY, mapping={"started_at": now_utc().isoformat(), "label": label,
                             "planned": planned, **base})


def current_run(r) -> dict | None:
    run = r.hgetall(RUN_KEY)
    if not run:
        return None
    run["started_at"] = datetime.fromisoformat(run["started_at"])
    run["planned"] = int(run.get("planned") or 0)
    return run


def in_run(run: dict | None) -> tuple[str, tuple]:
    """SQL that keeps only payments that arrived during the run (nothing extra without a run)."""
    if run is None:
        return "", ()
    return (" AND published_at >= %s AND txn_id IN (SELECT txn_id FROM events WHERE "
            "event_type = 'transaction.received' AND published_at >= %s)",
            (run["started_at"], run["started_at"]))


def minutes(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s"


def run_banner(conn, r, run: dict | None, replay_running: bool) -> None:
    if run is None:
        st.caption("Counting everything so far. Press ▶ Start replay (or ↺ Reset counters) to "
                   "start counting a new run from zero.")
        return
    sql, args = in_run(run)
    received, decided, stopped, finished, analyst, last = conn.execute(
        "SELECT count(*) FILTER (WHERE event_type = 'transaction.received'), "
        "       count(*) FILTER (WHERE event_type = 'transaction.decided'), "
        "       count(*) FILTER (WHERE event_type = 'transaction.decided' AND decision IN ('flag', 'hold')), "
        "       count(DISTINCT txn_id) FILTER (WHERE event_type = 'transaction.resolved'), "
        "       count(DISTINCT txn_id) FILTER (WHERE event_type = 'transaction.resolved' "
        "                                      AND decision = 'sent_to_analyst'), "
        "       max(published_at) "
        "FROM events WHERE true" + sql, args).fetchone()
    start = run["started_at"]
    planned = f" of {run['planned']}" if run["planned"] else ""
    in_progress = max(0, stopped - finished)
    if run["planned"] and received < run["planned"] and not replay_running:
        planned += " (the rest were repeats of earlier runs, which the gateway drops, or the replay was stopped)"
    lines = [f"**Run started {start:%H:%M:%S} UTC** ({run['label']})",
             f"Payments: **{received}{planned}** arrived, **{decided}** decided by the fast lane",
             f"AI checks (slow lane): **{stopped}** flagged or held → **{finished}** finished"
             + (f" ({analyst} sent to an analyst)" if analyst else "") + f", **{in_progress}** in progress"]
    done = run.get("completed_at")
    if done is None:
        audit_waiting = agent_info(r, "audit", ALL_TYPES)["waiting"]   # the audit log has caught up
        if (not replay_running and received > 0 and decided == received and in_progress == 0
                and audit_waiting == 0 and last is not None):
            r.hsetnx(RUN_KEY, "completed_at", last.isoformat())       # freeze it: later events don't count
            done = last.isoformat()
    if done is not None:
        done = datetime.fromisoformat(done)
        st.success("  \n".join(lines + [f"✅ **Run complete in {minutes((done - start).total_seconds())}** "
                                        f"(last event at {done:%H:%M:%S} UTC)"]))
    else:
        st.info("  \n".join(lines + [f"⏱ **{minutes((now_utc() - start).total_seconds())}** since the start"]))


def status(ok: bool) -> str:
    return "🟢 running" if ok else "⚪ stopped"


# ---------- Control panel (sidebar, Monitor page only) ----------

def control_panel() -> None:
    with st.sidebar:
        st.header("Control panel")
        st.caption("Sends real payments from fraudTest.csv through the gateway.")
        real_mix = st.toggle("Real fraud rate (random payments, ~0.4% fraud)", value=False,
                             help="Off: choose how many genuine and fraud payments (a test mix with more "
                                  "fraud, to see the slow lane work). On: random payments, so fraud comes "
                                  "as rarely as in real card traffic.")
        if real_mix:
            natural = st.number_input("Payments", 50, 20000, 1000, step=50)
            genuine, fraud = natural, 0
        else:
            natural = 0
            genuine = st.number_input("Genuine payments", 0, 5000, 45, step=5)
            fraud = st.number_input("Fraud payments", 0, 500, 5)
        rate = st.slider("Payments per second", 1, 50, 5)
        holder = replay_holder()
        busy = holder["process"] is not None and holder["process"].poll() is None
        st.session_state["sidebar_busy"] = busy   # what the sidebar buttons were drawn for
        if st.button("▶ Start replay", disabled=busy, type="primary"):
            try:
                label = (f"replay: {natural} random payments (real fraud rate) at {rate}/s" if natural
                         else f"replay: {genuine} genuine + {fraud} fraud at {rate}/s")
                start_run(redis_client(), label, planned=genuine + fraud)
            except (redis.ConnectionError, redis.TimeoutError) as error:
                st.error(f"Can't reach Redis (is Docker running?): {error}")
                return
            REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
            holder["process"] = subprocess.Popen(
                [sys.executable, "-m", "fraud_agents.services.replayer.replay", "--rate", str(rate)]
                + (["--natural", str(natural)] if natural else ["--genuine", str(genuine), "--fraud", str(fraud)]),
                stdout=REPLAY_LOG.open("w"), stderr=subprocess.STDOUT,
            )
            st.rerun()
        if busy and st.button("■ Stop replay"):
            holder["process"].terminate()
            st.rerun()
        if st.button("↺ Reset counters", disabled=busy,
                     help="Start counting from zero without sending payments. The audit log keeps everything."):
            try:
                start_run(redis_client(), "counters reset by hand")
            except (redis.ConnectionError, redis.TimeoutError) as error:
                st.error(f"Can't reach Redis (is Docker running?): {error}")
                return
            st.rerun()


# ---------- Live views (refresh every 2 seconds) ----------

@st.fragment(run_every=2)
def live() -> None:
    run = None
    try:
        r = redis_client()
        run = current_run(r)
        process = replay_holder()["process"]
        busy = process is not None and process.poll() is None
        if busy != st.session_state.get("sidebar_busy", busy):
            st.rerun()   # the replay started or finished: redraw the whole page, sidebar buttons too
        banner = st.container()          # drawn above the board, filled after it
        board(run)
        with banner:
            try:
                run_banner(pg_connection(), r, run, busy)
            except psycopg.Error as error:
                pg_connection.clear()
                st.error(f"Can't reach PostgreSQL (is Docker running?): {error}")
    except (redis.ConnectionError, redis.TimeoutError) as error:
        st.error(f"Can't reach Redis (is Docker running?): {error}")
    try:
        records(run)
    except psycopg.Error as error:
        pg_connection.clear()  # reconnect on the next refresh
        st.error(f"Can't reach PostgreSQL (is Docker running?): {error}")


def board(run: dict | None) -> None:
    r = redis_client()

    st.subheader("Agent board")
    cols = st.columns(3)
    for col, (name, url) in zip(cols, [("Gateway", GATEWAY_HEALTH), ("SMS provider", SMS_HEALTH)]):
        up = service_up(url)
        col.metric(name, "up" if up else "down")
        col.caption(status(up))
    audit = agent_info(r, "audit", ALL_TYPES)
    cols[2].metric("Audit log", f"{audit['handled']} recorded", f"{audit['waiting']} waiting",
                   delta_color="off")
    cols[2].caption(status(audit["running"]))
    for col, (name, group, streams) in zip(st.columns(len(AGENTS)), AGENTS):
        info = agent_info(r, group, streams, run)
        col.metric(name, f"{info['handled']} read", f"{info['waiting']} waiting", delta_color="off")
        col.caption(status(info["running"]))
    st.caption(("Agents' 'read' counts start from zero at the start of the run; " if run else "")
               + "'read' counts every event an agent received, including ones it had nothing to do "
               "with (e.g. Investigation reads every decision but only works on flags and holds). "
               "'waiting' is the live queue. The audit log counts everything, always.")

    holder = replay_holder()
    process = holder["process"]
    if process is not None:
        state = "running" if process.poll() is None else "finished"
        last = REPLAY_LOG.read_text().strip().splitlines()[-1:] if REPLAY_LOG.exists() else []
        st.info(f"Replay {state}. {last[0] if last else ''}")
    dead = r.xlen(bus.DEAD_LETTER_STREAM)
    if dead:
        st.warning(f"{dead} event(s) in the dead-letter stream (could not be handled).")



def records(run: dict | None = None) -> None:
    conn = pg_connection()
    sql, args = in_run(run)
    decisions = pd.DataFrame(
        conn.execute(
            "SELECT decision, count(*) FROM events "
            "WHERE event_type = 'transaction.decided'" + sql + " GROUP BY decision", args
        ).fetchall(),
        columns=["decision", "count"],
    )
    feed = pd.DataFrame(
        conn.execute(
            "SELECT to_char(published_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'), "
            "event_type, left(txn_id, 8), producer, score, decision "
            "FROM events ORDER BY published_at DESC, event_type LIMIT 30"
        ).fetchall(),
        columns=["sent (UTC)", "event", "txn", "from", "score", "decision"],
    )

    final = dict(conn.execute(
        "SELECT decision, count(*) FROM (SELECT DISTINCT ON (txn_id) decision FROM events "
        "  WHERE event_type = 'transaction.resolved'" + sql + " ORDER BY txn_id, published_at DESC) latest "
        "GROUP BY decision", args).fetchall())

    st.subheader("Decisions in this run" if run else "Decisions so far")
    totals = dict(zip(decisions["decision"], decisions["count"]))
    a, f, h = st.columns(3)
    a.metric("Approved", totals.get("approve", 0))
    f.metric("Flagged", totals.get("flag", 0))
    h.metric("Held", totals.get("hold", 0))
    st.caption("Final outcomes of the flagged and held payments (after the AI agents, the customer "
               "and the analyst):")
    rel, blk, ana = st.columns(3)
    rel.metric("Released", final.get("released", 0))
    blk.metric("Blocked", final.get("blocked", 0))
    ana.metric("Waiting for an analyst", final.get("sent_to_analyst", 0))

    st.subheader("Live event feed (latest 30)")
    st.dataframe(feed, hide_index=True, width="stretch")


def monitor_page() -> None:
    control_panel()
    st.title("Fraud System Monitor")
    live()


# ---------- Analyst page (Part 3 · Step 1) ----------

REASONS = {  # why a case reached a human, in plain words (rule_id -> text)
    "customer_reply_unclear": "The customer replied, but the reply was unclear.",
    "no_customer_reply": "The customer did not reply in time.",
    "investigator_escalated": "The AI investigator asked for a human to decide.",
    "held_payment_needs_human_release": "The payment was held. The AI investigator found nothing "
                                        "wrong, but only a human (or the customer) may release a held payment.",
    "case_escalated": "An agent could not finish its work, so it handed the case to a human.",
    "decision_could_not_finish": "The Decision agent could not settle the case after several tries, "
                                 "so it handed it to a human.",
}
REVIEW_LOCK = "analyst:reviewed:{case_id}"   # stops two clicks (or two analysts) deciding one case
LOCK_SECONDS = 120   # only needs to cover the audit log catching up; the Decision agent is the real guard


def plain(text: str) -> str:
    """Show $ as a dollar sign: Streamlit reads text between two $ signs as a maths formula."""
    return str(text).replace("$", "\\$")


def body(value) -> dict:
    return value if isinstance(value, dict) else json.loads(value)


def open_cases(conn, r) -> list[dict]:
    """sent_to_analyst outcomes whose analyst decision has not been APPLIED yet, oldest first.

    A case leaves the queue only when the Decision agent has published the final outcome
    for its review, not merely when a review was sent (Part 3 · Step 3)."""
    rows = conn.execute(
        "SELECT o.event_id, o.txn_id, o.published_at, o.body->'payload'->>'rule_id', "
        "       t.body->'payload'->>'amount', t.body->'payload'->>'merchant', "
        "       v.decision, v.body->'payload'->>'analyst', v.published_at "
        "FROM events o "
        "LEFT JOIN events t ON t.txn_id = o.txn_id AND t.event_type = 'transaction.received' "
        "LEFT JOIN events v ON v.event_type = 'case.reviewed' "
        "      AND v.body->'payload'->>'based_on_event_id' = o.event_id "
        "WHERE o.event_type = 'transaction.resolved' AND o.decision = 'sent_to_analyst' "
        "  AND NOT EXISTS (SELECT 1 FROM events f WHERE f.event_type = 'transaction.resolved' "
        "                  AND f.body->'payload'->>'based_on_event_id' = v.event_id) "
        "ORDER BY o.published_at"
    ).fetchall()
    cases = []
    for c, t, at, rule, amount, merchant, v_dec, v_who, v_at in rows:
        pending = None if v_dec is None else {"decision": v_dec, "analyst": v_who, "at": v_at}
        just_sent = r.get(REVIEW_LOCK.format(case_id=c))   # a click the audit log may not show yet
        if pending is None and just_sent:
            try:
                seen = json.loads(just_sent)
                pending = seen | {"at": datetime.fromisoformat(seen["at"])}
            except (ValueError, KeyError, TypeError):   # a lock written by an older version: just the decision
                pending = {"decision": just_sent, "analyst": "an analyst", "at": now_utc()}
        # The case stays in the queue (Part 3 · Step 6: it is no longer hidden after a click),
        # marked as pending until the Decision agent has applied the review.
        cases.append({"case_id": c, "txn_id": t, "sent_at": at, "rule_id": rule, "amount": amount,
                      "merchant": (merchant or "?").removeprefix("fraud_"), "pending": pending})
    return cases


def case_events(conn, txn_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT published_at, event_type, producer, decision, body FROM events "
        "WHERE txn_id = %s ORDER BY published_at, event_type", (txn_id,)).fetchall()
    return [{"at": at, "type": t, "from": p, "result": d, "body": body(b)} for at, t, p, d, b in rows]


def show_case(case: dict, events: list[dict]) -> None:
    first = {}
    for e in events:
        first.setdefault(e["type"], e)          # the first event of each type
    paid = first.get("transaction.received", {}).get("body", {}).get("payload", {})
    scored = first.get("risk.scored", {}).get("body", {}).get("payload", {})
    decided = first.get("transaction.decided", {})
    report = first.get("investigation.completed", {}).get("body", {}).get("payload")

    st.markdown(f"**Why it is here:** {REASONS.get(case['rule_id'], case['rule_id'])}")
    a, b, c, d = st.columns(4)
    a.metric("Amount", f"${float(paid.get('amount', 0)):,.2f}")
    b.metric("Shop", case["merchant"])
    c.metric("Risk score", f"{scored['score']:.2f}" if "score" in scored else "?")
    d.metric("First decision", (decided.get("result") or "?").upper())
    when = first.get("transaction.received", {}).get("body", {}).get("occurred_at")
    when = f"{datetime.fromisoformat(when):%Y-%m-%d %H:%M}" if when else "?"
    st.caption(f"Shop type: {paid.get('category', '?')} · payment time (UTC): {when} · "
               f"card {paid.get('card_id', '?')} · txn {case['txn_id']}")
    if scored.get("top_factors"):
        st.caption("What pushed the score up: " + ", ".join(map(str, scored["top_factors"])))

    st.markdown("##### The AI investigator's report")
    if report:
        st.markdown(f"**Verdict:** {report['verdict'].replace('_', ' ')} · **suggested:** "
                    f"{report['recommended_action'].replace('_', ' ')} · model {report['model']}")
        st.markdown(plain(report["summary"]))
        for item in report.get("evidence", []):
            st.markdown(f"- {plain(item)}")
        st.caption("The report is advice from an AI. Check it against the facts above.")
    else:
        st.caption("No report (the investigation did not finish).")

    texts = []
    for e in events:
        p = e["body"]["payload"]
        if e["type"] == "verification.requested":
            texts.append(("🏦 Bank", p["message"], e["at"]))
        elif e["type"] == "customer.replied":
            texts.append(("👤 Customer", p["reply_text"], e["at"]))
        elif e["type"] == "notification.sent":
            texts.append(("🏦 Bank", p["message"], e["at"]))
    st.markdown("##### Texts with the customer")
    if texts:
        for who, text, at in texts:
            st.markdown(f"**{who}** ({at:%H:%M:%S} UTC): {plain(text)}")
    else:
        st.caption("No texts were exchanged.")

    with st.expander("Full timeline (every event for this payment)"):
        st.dataframe(pd.DataFrame(
            [(f"{e['at']:%Y-%m-%d %H:%M:%S}", e["type"], e["from"], e["result"]) for e in events],
            columns=["sent (UTC)", "event", "from", "result"]), hide_index=True, width="stretch")


def submit_review(case: dict, decision: str, analyst: str, note: str, occurred_at: str) -> str:
    """Publish ONE case.reviewed event. Returns what happened, in words."""
    r = redis_client()
    lock = REVIEW_LOCK.format(case_id=case["case_id"])
    review = CaseReviewed(
        event_id=derived_event_id(case["case_id"], "analyst", "case.reviewed"),
        published_at=now_utc(), txn_id=case["txn_id"],
        occurred_at=datetime.fromisoformat(occurred_at),
        payload=CaseReviewedPayload(decision=decision, analyst=analyst.strip() or "analyst",
                                    note=note.strip() or None, based_on_event_id=case["case_id"]))
    seen = json.dumps({"decision": decision, "analyst": review.payload.analyst, "at": now_utc().isoformat()})
    if not r.set(lock, seen, nx=True, ex=LOCK_SECONDS):
        return "already"
    try:
        bus.publish(r, review)
    except redis.TimeoutError:
        return "unsure"                      # it may have reached Redis: keep the lock
    except Exception:
        r.delete(lock)                       # nothing was sent: allow another try
        raise
    return "sent"


def reviewed_recently(conn) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT to_char(v.published_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'), left(v.txn_id, 8), "
        "       v.body->'payload'->>'analyst', upper(v.decision), coalesce(v.body->'payload'->>'note', ''), "
        "       f.decision, n.body->'payload'->>'message' "
        "FROM events v "
        "LEFT JOIN events f ON f.event_type = 'transaction.resolved' "
        "      AND f.body->'payload'->>'based_on_event_id' = v.event_id "
        "LEFT JOIN events n ON n.event_type = 'notification.sent' "
        "      AND n.body->'payload'->>'based_on_event_id' = f.event_id "
        "WHERE v.event_type = 'case.reviewed' ORDER BY v.published_at DESC LIMIT 20").fetchall()
    return pd.DataFrame(rows, columns=["reviewed (UTC)", "txn", "analyst", "clicked", "note",
                                       "final outcome", "text sent to the customer"])


def analyst_page() -> None:
    st.title("Analyst review")
    st.caption("Cases the system could not settle on its own. Read the evidence, then Release or "
               "Block. The Decision agent records your choice as the final outcome, and the "
               "Notification agent texts the customer.")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    if st.button("🔄 Refresh"):
        st.rerun()
    try:
        conn, r = pg_connection(), redis_client()
        cases = open_cases(conn, r)
        if not cases:
            st.info("No cases are waiting for an analyst.")
        else:
            by_id = {c["case_id"]: c for c in cases}
            st.caption(f"{len(cases)} case(s) waiting, oldest first")
            chosen = st.selectbox(
                "Case", list(by_id), key="case",   # the key keeps your choice when the list changes
                format_func=lambda c: (f"{by_id[c]['sent_at']:%H:%M:%S} · txn {by_id[c]['txn_id'][:8]} · "
                                       f"${float(by_id[c]['amount'] or 0):,.2f} at {by_id[c]['merchant']}"))
            case = by_id[chosen]
            events = case_events(conn, case["txn_id"])
            if case["pending"]:
                p = case["pending"]
                if (now_utc() - p["at"]).total_seconds() < LOCK_SECONDS:
                    st.info(f"⏳ A **{p['decision'].upper()}** by {p['analyst']} was sent at {p['at']:%H:%M:%S} UTC "
                            "and is being applied. Click 🔄 Refresh in a few seconds.")
                else:
                    st.warning(f"A **{p['decision'].upper()}** by {p['analyst']} was sent at {p['at']:%H:%M:%S} UTC, "
                               "but the Decision agent hasn't applied it yet. Is the Decision agent running? "
                               "(See data/logs/decision.log.) You can send the same decision again; "
                               "a different one would conflict with the audit log.")
            show_case(case, events)

            st.markdown("##### Your decision")
            name = st.text_input("Your name", value="analyst", max_chars=60, key="analyst_name")
            note = st.text_input("Note (optional, kept in the audit log)", max_chars=300,
                                 key=f"note-{chosen}")
            occurred = events[0]["body"]["occurred_at"]   # the payment's own time, as on every event
            left, right, _ = st.columns([1, 1, 3])
            choice = None
            only = case["pending"]["decision"] if case["pending"] else None   # a resend must match
            if left.button("✅ Release", key=f"rel-{chosen}", width="stretch",   # same style as Block:
                           help="Approve the payment", disabled=only == "block"):  # no default button
                choice = "release"
            if right.button("⛔ Block", key=f"blk-{chosen}", width="stretch", help="Stop the payment",
                            disabled=only == "release"):
                choice = "block"
            if choice:
                result = submit_review(case, choice, name, note, occurred)
                st.session_state["flash"] = {
                    "sent": f"{choice.title()} sent for txn {case['txn_id'][:8]}. Watch 'Reviewed "
                            "recently' below for the final outcome and the text to the customer (about 10 s).",
                    "already": "Someone already reviewed this case; nothing was sent.",
                    "unsure": "Redis was slow to answer, so your choice may or may not have been sent. "
                              "Check 'Reviewed recently' in a moment before trying again.",
                }[result]
                st.rerun()

        st.subheader("Reviewed recently")
        st.dataframe(reviewed_recently(conn), hide_index=True, width="stretch")
    except (redis.ConnectionError, redis.TimeoutError) as error:
        st.error(f"Can't reach Redis (is Docker running?): {error}")
    except psycopg.Error as error:
        pg_connection.clear()
        st.error(f"Can't reach PostgreSQL (is Docker running?): {error}")


# ---------- Shared helpers for the Step 2 pages ----------

@st.cache_resource(show_spinner="Loading the answer key…")
def _load_answer_key() -> dict[str, int]:
    df = pd.read_csv(ANSWER_KEY_FILE, usecols=["trans_num", "is_fraud"])
    return dict(zip(df["trans_num"], df["is_fraud"]))


def answer_key() -> dict[str, int] | None:
    """trans_num -> is_fraud from the dataset. For testing only: the agents never see it."""
    return _load_answer_key() if ANSWER_KEY_FILE.exists() else None


def truth(key: dict | None, txn_id: str) -> str:
    if key is None or txn_id not in key:
        return "unknown"
    return "FRAUD" if key[txn_id] else "genuine"


def what_happened(e: dict) -> str:
    """One line, in plain words, for one event."""
    p = e["body"]["payload"]
    t = e["type"]
    if t == "transaction.received":
        return f"Payment arrived: ${float(p['amount']):,.2f} at {p['merchant'].removeprefix('fraud_')} ({p['category']})"
    if t == "risk.scored":
        return f"Scored {p['score']:.2f} by model {p['model_version']}" + (
            f" (pushed up by: {', '.join(map(str, p['top_factors']))})" if p.get("top_factors") else "")
    if t == "transaction.decided":
        return f"First decision: {p['decision'].upper()} ({p['rule_id']})"
    if t == "investigation.completed":
        return (f"AI report: {p['verdict'].replace('_', ' ')}; suggests "
                f"{p['recommended_action'].replace('_', ' ')} (tools: {', '.join(p['tools_used']) or 'none'})")
    if t == "verification.requested":
        return f"Texted the customer: {p['message']}"
    if t == "customer.replied":
        return f"Customer replied: {p['reply_text']}"
    if t == "verification.completed":
        return f"Customer check finished: {p['outcome'].replace('_', ' ')}"
    if t == "transaction.resolved":
        return f"FINAL OUTCOME: {p['resolution'].replace('_', ' ').upper()} ({p['rule_id']})"
    if t == "notification.sent":
        return f"Told the customer ({'template' if p['used_template'] else 'AI draft'}): {p['message']}"
    if t == "case.escalated":
        return f"Handed to a human: {p['reason']}"
    if t == "case.reviewed":
        return f"Analyst {p['analyst']} clicked {p['decision'].upper()}" + (f": {p['note']}" if p.get("note") else "")
    return t


# ---------- Payment trace page (Part 3 · Step 2) ----------

def trace_page() -> None:
    st.title("Payment trace")
    st.caption("One payment's whole journey through the system, event by event.")
    try:
        conn = pg_connection()
        recent = conn.execute(
            "SELECT d.txn_id, d.decision, d.published_at, t.body->'payload'->>'amount', "
            "       t.body->'payload'->>'merchant' "
            "FROM events d JOIN events t ON t.txn_id = d.txn_id AND t.event_type = 'transaction.received' "
            "WHERE d.event_type = 'transaction.decided' AND d.decision IN ('flag', 'hold') "
            "ORDER BY d.published_at DESC LIMIT 200").fetchall()
        typed = st.text_input("Payment ID (the first 8 characters are enough)", key="trace_id").strip()
        if typed:
            prefix = typed.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            found = conn.execute("SELECT DISTINCT txn_id FROM events WHERE txn_id LIKE %s LIMIT 5",
                                 (prefix + "%",)).fetchall()
            if not found:
                st.warning("No payment with that ID in the audit log.")
                return
            if len(found) > 1:
                st.warning("More than one payment starts with that; type more characters.")
                return
            txn_id = found[0][0]
        elif recent:
            labels = {t: f"{at:%H:%M:%S} · txn {t[:8]} · {d.upper()} · ${float(a):,.2f} at "
                         f"{m.removeprefix('fraud_')}" for t, d, at, a, m in recent}
            txn_id = st.selectbox("…or pick a recent flagged or held payment", list(labels),
                                  format_func=labels.get, key="trace_pick")
        else:
            st.info("No flagged or held payments yet. Start a replay on the Monitor page.")
            return
        show_trace(conn, txn_id)
    except psycopg.Error as error:
        pg_connection.clear()
        st.error(f"Can't reach PostgreSQL (is Docker running?): {error}")


def show_trace(conn, txn_id: str) -> None:
    events = case_events(conn, txn_id)
    first = {}
    for e in events:
        first.setdefault(e["type"], e)
    resolved = [e for e in events if e["type"] == "transaction.resolved"]
    paid = first.get("transaction.received", {}).get("body", {}).get("payload", {})
    a, b, c, d = st.columns(4)
    a.metric("Amount", f"${float(paid['amount']):,.2f}" if paid else "?")
    b.metric("First decision", (first.get("transaction.decided", {}).get("result") or "—").upper())
    c.metric("Final outcome", resolved[-1]["result"].replace("_", " ") if resolved else "—")
    c.caption("The latest one counts (an analyst can replace 'sent to analyst').")
    waiting = bool(resolved) and resolved[-1]["result"] == "sent_to_analyst"
    d.metric("Answer key", "hidden" if waiting else truth(answer_key(), txn_id))
    d.caption("Hidden until the analyst decides, so it can't sway them." if waiting
              else "For testing only: the agents never see it.")
    st.caption(f"txn {txn_id} · shop {paid.get('merchant', '?').removeprefix('fraud_')} · "
               f"card {paid.get('card_id', '?')}")

    start = events[0]["at"]
    st.markdown("##### The journey (time since the payment arrived · agent · what happened)")
    for e in events:
        st.markdown(f"`+{(e['at'] - start).total_seconds():.1f} s` **{e['from']}** · {plain(what_happened(e))}")

    report = first.get("investigation.completed", {}).get("body", {}).get("payload")
    if report:
        with st.expander("The AI investigator's full report"):
            st.markdown(plain(report["summary"]))
            for item in report.get("evidence", []):
                st.markdown(f"- {plain(item)}")
    calls = conn.execute(
        "SELECT to_char(called_at AT TIME ZONE 'UTC', 'HH24:MI:SS'), agent, model, latency_ms, "
        "input_tokens, output_tokens, ok, coalesce(error, '') FROM llm_calls WHERE txn_id = %s "
        "ORDER BY called_at", (txn_id,)).fetchall()
    st.markdown(f"##### AI calls for this payment: {len(calls)}")
    if calls:
        st.dataframe(pd.DataFrame(calls, columns=["at (UTC)", "agent", "model", "ms", "tokens in",
                                                  "tokens out", "ok", "error"]),
                     hide_index=True, width="stretch")
    with st.expander("Raw events (exactly as stored in the audit log)"):
        for e in events:
            st.markdown(f"**{e['type']}** from {e['from']}")
            st.json(e["body"], expanded=False)


# ---------- Caught payments page (Part 3 · Step 2) ----------

def caught_page() -> None:
    st.title("Caught payments")
    st.caption("Every payment in the audit log, checked against the dataset's answer key (is_fraud). "
               "The answer key is shown here for testing only: the agents never see it.")
    st.warning("**A live testing view, not the overall performance.** To keep the analyst unbiased, "
               "payments waiting for an analyst are left out of the numbers below, so they cover a "
               "changing subset. The official numbers for a run come from the run report "
               "(`python -m fraud_agents.services.metrics.report`), which counts every payment.")
    key = answer_key()
    if key is None:
        st.error(f"Answer key not found at {ANSWER_KEY_FILE}.")
        return
    try:
        rows = pg_connection().execute(
            "SELECT t.txn_id, t.published_at, (t.body->'payload'->>'amount')::numeric, "
            "       t.body->'payload'->>'merchant', t.body->'payload'->>'category', d.score, d.decision, "
            "       (SELECT r.decision FROM events r WHERE r.txn_id = t.txn_id "
            "        AND r.event_type = 'transaction.resolved' ORDER BY r.published_at DESC LIMIT 1) "
            "FROM events t "
            "LEFT JOIN events d ON d.txn_id = t.txn_id AND d.event_type = 'transaction.decided' "
            "WHERE t.event_type = 'transaction.received'").fetchall()
    except psycopg.Error as error:
        pg_connection.clear()
        st.error(f"Can't reach PostgreSQL (is Docker running?): {error}")
        return
    df = pd.DataFrame(rows, columns=["txn_id", "arrived", "amount", "shop", "shop type", "score",
                                     "first decision", "final outcome"])
    if df.empty:
        st.info("No payments yet. Start a replay on the Monitor page.")
        return
    try:
        run = current_run(redis_client())
    except (redis.ConnectionError, redis.TimeoutError):
        run = None                       # Redis down: show everything instead
    if run and st.toggle(f"This run only (since {run['started_at']:%H:%M:%S} UTC)", value=True,
                         key="caught_run_only"):
        df = df[df["arrived"] >= run["started_at"]]
        if df.empty:
            st.info("No payments in this run yet.")
            return
    df["shop"] = df["shop"].str.removeprefix("fraud_")
    df["amount"] = df["amount"].astype(float)
    df["answer key"] = [truth(key, t) for t in df["txn_id"]]
    df["txn"] = df["txn_id"].str[:8]
    df.loc[df["final outcome"] == "sent_to_analyst", "answer key"] = "hidden"   # don't sway the analyst
    hidden = int((df["answer key"] == "hidden").sum())
    if hidden:
        st.info(f"{hidden} payment(s) waiting for an analyst are left out of the numbers below.")
    decided = df[df["first decision"].notna() & df["answer key"].isin(["FRAUD", "genuine"])]
    stopped = decided["first decision"].isin(["flag", "hold"])
    fraud = decided["answer key"] == "FRAUD"
    genuine = decided["answer key"] == "genuine"

    blocked = decided["final outcome"] == "blocked"
    a, b, c, d, e, f = st.columns(6)
    a.metric("Payments decided", len(decided))
    b.metric("Frauds among them", int(fraud.sum()))
    c.metric("Frauds caught", int((fraud & stopped).sum()))
    d.metric("…blocked in the end", int((fraud & blocked).sum()))
    e.metric("Frauds missed", int((fraud & ~stopped).sum()))
    f.metric("False alarms", int((genuine & stopped).sum()))
    if fraud.sum() and genuine.sum():
        st.markdown(f"**Caught {(fraud & stopped).sum() / fraud.sum():.0%} of the frauds** at the first "
                    f"decision, and stopped **{(genuine & stopped).sum() / genuine.sum():.1%} of the genuine "
                    "payments** by mistake.")
    st.caption("Caught = flagged or held by the first decision; 'blocked in the end' = its final outcome "
               "was blocked. A false alarm is a genuine payment that was flagged or held. Payments "
               "waiting for an analyst are left out (their answer is hidden), and so are payments "
               "not in the dataset.")

    cols = ["txn", "amount", "shop", "shop type", "score", "first decision", "final outcome", "answer key"]
    flagged = df[df["first decision"].isin(["flag", "hold"])]   # all of them, including hidden ones
    st.subheader(f"Flagged or held: {len(flagged)}")
    st.dataframe(flagged.sort_values("arrived", ascending=False)[cols], hide_index=True,
                 width="stretch", column_config={"amount": st.column_config.NumberColumn(format="$%.2f"),
                                                 "score": st.column_config.NumberColumn(format="%.2f")})
    st.subheader(f"Frauds the system missed (approved): {int((fraud & ~stopped).sum())}")
    st.dataframe(decided[fraud & ~stopped].sort_values("arrived", ascending=False)[cols[:6]],
                 hide_index=True, width="stretch",
                 column_config={"amount": st.column_config.NumberColumn(format="$%.2f"),
                                "score": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Copy a txn into the Payment trace page to see its whole journey.")


# ---------- SMS inbox page (Part 3 · Step 2) ----------

def sms_page() -> None:
    st.title("SMS inbox")
    st.caption("Every text between the bank and its customers, as recorded by the mock SMS provider.")
    try:
        entries = redis_client().xrevrange(SMS_LOG, count=2000)   # newest first
    except (redis.ConnectionError, redis.TimeoutError) as error:
        st.error(f"Can't reach Redis (is Docker running?): {error}")
        return
    if not entries:
        st.info("No texts yet. They appear when the system checks a payment with a customer.")
        return
    texts = [f for _, f in entries]
    a, b, c = st.columns(3)
    st.caption(f"Showing the latest {len(texts)} texts (the provider keeps about 5,000).")
    a.metric("Texts sent by the bank", sum(t["direction"] == "out" for t in texts))
    b.metric("Replies from customers", sum(t["direction"] == "in" for t in texts))
    c.metric("Customers", len({t["who"] for t in texts}))

    chats: dict[str, list[dict]] = {}
    for t in texts:                                   # newest first, so the first is the latest
        chats.setdefault(t["who"], []).append(t)
    who = st.selectbox(
        "Conversation (latest first)", list(chats), key="sms_who",
        format_func=lambda w: f"{w} · {len(chats[w])} texts · last {chats[w][0]['at']} UTC")
    for t in reversed(chats[who]):                    # oldest first, like a phone
        mine = t["direction"] == "out"
        with st.chat_message("assistant" if mine else "user", avatar="🏦" if mine else "👤"):
            st.markdown(plain(t["text"]))
            st.caption(f"{'Bank' if mine else 'Customer'} · {t['at']} UTC · txn {t['txn_id'][:8] or '—'}")


st.navigation([
    st.Page(monitor_page, title="Monitor", icon="📊", default=True),
    st.Page(trace_page, title="Payment trace", icon="🔎", url_path="trace"),
    st.Page(caught_page, title="Caught payments", icon="🎯", url_path="caught"),
    st.Page(sms_page, title="SMS inbox", icon="💬", url_path="sms"),
    st.Page(analyst_page, title="Analyst", icon="🧑‍💼", url_path="analyst"),
]).run()
