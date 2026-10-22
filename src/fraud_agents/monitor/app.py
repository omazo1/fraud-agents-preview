"""The basic System Monitor (Part 1 · Step 8).

Shows, live: which services and agents are running and how busy they are, the
latest events, the decision totals, and a control panel to start a replay.
It only READS what the agents publish, so it can never change a decision.

Run from the project folder:  streamlit run src/fraud_agents/monitor/app.py
Then open http://localhost:8501
"""

import subprocess
import sys
from pathlib import Path

import httpx
import pandas as pd
import psycopg
import redis
import streamlit as st

from fraud_agents.shared import bus, db

GATEWAY_HEALTH = "http://localhost:8000/health"
REPLAY_LOG = Path("data/processed/replay_from_monitor.log")

AGENTS = [  # (name on the board, consumer group, the stream it reads)
    ("Risk Scoring agent", "risk-scoring", "transaction.received"),
    ("Decision agent", "decision", "risk.scored"),
]
ALL_TYPES = ["transaction.received", "risk.scored", "transaction.decided"]

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


# ---------- Control panel (sidebar) ----------

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

st.title("Fraud System Monitor")


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


live()
