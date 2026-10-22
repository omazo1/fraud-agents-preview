"""The Fraud System Monitor (Part 1 · Step 8; Analyst page added in Part 3 · Step 1).

Two pages, picked in the sidebar:
- Monitor: which services and agents are running and how busy they are, the latest
  events, the decision totals, and a control panel to start a replay. Read-only.
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
REPLAY_LOG = Path("data/processed/replay_from_monitor.log")

AGENTS = [  # (name on the board, consumer group, the stream it reads)
    ("Risk Scoring agent", "risk-scoring", "transaction.received"),
    ("Decision agent", "decision", "risk.scored"),
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


def gateway_up() -> bool:
    try:
        return httpx.get(GATEWAY_HEALTH, timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


def status(ok: bool) -> str:
    return "🟢 running" if ok else "⚪ stopped"


# ---------- Control panel (sidebar, Monitor page only) ----------

def control_panel() -> None:
    with st.sidebar:
        st.header("Control panel")
        st.caption("Sends real payments from fraudTest.csv through the gateway.")
        genuine = st.number_input("Genuine payments", 0, 5000, 45, step=5)
        fraud = st.number_input("Fraud payments", 0, 500, 5)
        rate = st.slider("Payments per second", 1, 50, 5)
        holder = replay_holder()
        busy = holder["process"] is not None and holder["process"].poll() is None
        if st.button("▶ Start replay", disabled=busy, type="primary"):
            REPLAY_LOG.parent.mkdir(parents=True, exist_ok=True)
            holder["process"] = subprocess.Popen(
                [sys.executable, "-m", "fraud_agents.services.replayer.replay",
                 "--genuine", str(genuine), "--fraud", str(fraud), "--rate", str(rate)],
                stdout=REPLAY_LOG.open("w"), stderr=subprocess.STDOUT,
            )
            st.rerun()
        if busy and st.button("■ Stop replay"):
            holder["process"].terminate()
            st.rerun()


# ---------- Live views (refresh every 2 seconds) ----------

@st.fragment(run_every=2)
def live() -> None:
    try:
        board()
    except (redis.ConnectionError, redis.TimeoutError) as error:
        st.error(f"Can't reach Redis (is Docker running?): {error}")
    try:
        records()
    except psycopg.Error as error:
        pg_connection.clear()  # reconnect on the next refresh
        st.error(f"Can't reach PostgreSQL (is Docker running?): {error}")


def board() -> None:
    r = redis_client()

    st.subheader("Agent board")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Gateway", "up" if gateway_up() else "down")
        st.caption(status(gateway_up()))
    for col, (name, group, stream) in zip(cols[1:3], AGENTS):
        info = group_info(r, stream, group)
        with col:
            st.metric(name, f"{info['handled']} handled", f"{info['waiting']} waiting",
                      delta_color="off")
            st.caption(status(info["running"]))
    audit = [group_info(r, t, "audit") for t in ALL_TYPES]
    with cols[3]:
        st.metric("Audit log", f"{sum(a['handled'] for a in audit)} recorded",
                  f"{sum(a['waiting'] for a in audit)} waiting", delta_color="off")
        st.caption(status(all(a["running"] for a in audit)))

    holder = replay_holder()
    process = holder["process"]
    if process is not None:
        state = "running" if process.poll() is None else "finished"
        last = REPLAY_LOG.read_text().strip().splitlines()[-1:] if REPLAY_LOG.exists() else []
        st.info(f"Replay {state}. {last[0] if last else ''}")
    dead = r.xlen(bus.DEAD_LETTER_STREAM)
    if dead:
        st.warning(f"{dead} event(s) in the dead-letter stream (could not be handled).")



def records() -> None:
    conn = pg_connection()
    decisions = pd.DataFrame(
        conn.execute(
            "SELECT decision, count(*) FROM events "
            "WHERE event_type = 'transaction.decided' GROUP BY decision"
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

    st.subheader("Decisions so far")
    totals = dict(zip(decisions["decision"], decisions["count"]))
    a, f, h = st.columns(3)
    a.metric("Approved", totals.get("approve", 0))
    f.metric("Flagged", totals.get("flag", 0))
    h.metric("Held", totals.get("hold", 0))

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
}
REVIEW_LOCK = "analyst:reviewed:{case_id}"   # stops two clicks (or two analysts) deciding one case
LOCK_SECONDS = 600   # only needs to cover the audit log catching up; the Decision agent is the real guard


def plain(text: str) -> str:
    """Show $ as a dollar sign: Streamlit reads text between two $ signs as a maths formula."""
    return str(text).replace("$", "\\$")


def body(value) -> dict:
    return value if isinstance(value, dict) else json.loads(value)


def open_cases(conn, r) -> list[dict]:
    """sent_to_analyst outcomes that nobody has reviewed yet, oldest first."""
    rows = conn.execute(
        "SELECT o.event_id, o.txn_id, o.published_at, o.body->'payload'->>'rule_id', "
        "       t.body->'payload'->>'amount', t.body->'payload'->>'merchant' "
        "FROM events o "
        "LEFT JOIN events t ON t.txn_id = o.txn_id AND t.event_type = 'transaction.received' "
        "WHERE o.event_type = 'transaction.resolved' AND o.decision = 'sent_to_analyst' "
        "  AND NOT EXISTS (SELECT 1 FROM events v WHERE v.event_type = 'case.reviewed' "
        "                  AND v.body->'payload'->>'based_on_event_id' = o.event_id) "
        "ORDER BY o.published_at"
    ).fetchall()
    return [
        {"case_id": c, "txn_id": t, "sent_at": at, "rule_id": rule,
         "amount": amount, "merchant": (merchant or "?").removeprefix("fraud_")}
        for c, t, at, rule, amount, merchant in rows
        if not r.exists(REVIEW_LOCK.format(case_id=c))   # just reviewed; audit log catching up
    ]


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
    if not r.set(lock, decision, nx=True, ex=LOCK_SECONDS):
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
            show_case(case, events)

            st.markdown("##### Your decision")
            name = st.text_input("Your name", value="analyst", max_chars=60, key="analyst_name")
            note = st.text_input("Note (optional, kept in the audit log)", max_chars=300,
                                 key=f"note-{chosen}")
            occurred = events[0]["body"]["occurred_at"]   # the payment's own time, as on every event
            left, right, _ = st.columns([1, 1, 3])
            choice = None
            if left.button("✅ Release", key=f"rel-{chosen}", width="stretch",   # same style as Block:
                           help="Approve the payment"):                       # no button is "the default"
                choice = "release"
            if right.button("⛔ Block", key=f"blk-{chosen}", width="stretch", help="Stop the payment"):
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


st.navigation([
    st.Page(monitor_page, title="Monitor", icon="📊", default=True),
    st.Page(analyst_page, title="Analyst", icon="🧑‍💼", url_path="analyst"),
]).run()
