"""The Verification agent (Part 2 · Step 6): asks the customer, PAUSES, then resumes.

It reads "I investigated it" forms. When the investigation advises asking the customer,
it runs a LangGraph graph that is saved in PostgreSQL (a "checkpointer"), so it can
wait for the reply for as long as needed, even if the agent is stopped and restarted:

    START -> ask_customer -> wait_for_reply ..(PAUSED: interrupt)..
                                   ^              | resumed by the customer's reply
                                   |              v                 (or by the timeout)
                          ask_again (once)  <-  read_reply  (the AI labels the reply)
                                                  |
                                                  v
                                               finish -> verification.completed

- ask_customer: texts "did you just pay $X at Y? Reply YES or NO" through the SMS
  provider, and publishes verification.requested (its event_id is the reference the
  reply comes back with).
- wait_for_reply: interrupt() pauses the graph; the state is saved in PostgreSQL.
- read_reply: plain keyword rules read clear answers (YES / NO); only replies the rules
  can't read go to the AI (Llama), which labels them confirmed / denied / unclear.
  (Mac runs: Llama labelled clear "NO" replies as unclear, so rules now go first; it also
  labelled "who is this?" as denied, so a question back is now "unclear" by rule, and the
  AI's prompt has examples and "when in doubt, choose unclear".)
- ask_again: if the reply was unclear, one follow-up text; then wait again.
- finish: publishes verification.completed (confirmed / denied / unclear / no reply).

The AI only reads the reply. What happens to the payment is decided by rules
(the Decision agent's final-outcome rules).

Run from the project folder:  python -m fraud_agents.agents.verification.agent
"""

import json
import logging
import os
import re
import socket
import threading
import time
from typing import Literal

import httpx
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from fraud_agents.shared import bus, config, db
from fraud_agents.shared.events import (
    CustomerReplied,
    InvestigationCompleted,
    VerificationCompleted,
    VerificationCompletedPayload,
    VerificationRequested,
    VerificationRequestedPayload,
    derived_event_id,
    now_utc,
)
from fraud_agents.shared.llm import LLMCallLogger, chat_model, run_config

PRODUCER = "verification"
WAITING = "verification:waiting"         # Redis hash: reference -> the paused case
REPLY_TIMEOUT_SECONDS = int(os.environ.get("VERIFICATION_TIMEOUT_SECONDS", "60"))
NO_REPLY = "__no_reply__"
SMS_URL = os.environ.get("SMS_PROVIDER_URL", "http://localhost:8100") + "/messages"

log = logging.getLogger(PRODUCER)


class ReplyLabel(BaseModel):
    label: Literal["confirmed", "denied", "unclear"] = Field(
        description="confirmed = the customer says they made the payment; denied = they say "
                    "they did not; unclear = anything else")
    reason: str = Field(description="One short sentence")


class State(TypedDict):
    txn_id: str
    card_id: str
    occurred_at: str
    question: str
    reference: str              # the verification.requested event_id the reply must match
    based_on: str               # the investigation.completed event_id
    reply: str | None
    label: str | None
    labelled_by: str | None     # "rules" or "AI": who read the reply
    follow_ups: int
    outcome: str | None


def keyword_label(text: str) -> str:
    """Plain rules, read before the AI. A question back to the bank is never an answer."""
    t = text.strip().lower()
    if t.endswith("?"):
        return "unclear"
    if re.search(r"\b(no|n|not me|wasn'?t me|did not|didn'?t)\b", t):
        return "denied"
    if re.search(r"\b(yes|y|it was me|i made)\b", t):
        return "confirmed"
    return "unclear"


class Verifier:
    def __init__(self, r, model=None, sms_post=None):
        self.r = r
        self.model = model or chat_model()
        self.labeller = self.model.with_structured_output(ReplyLabel)
        self.logger = LLMCallLogger()
        self.sms_post = sms_post or self._sms_post
        self.sms_key = config.require("SMS_API_KEY")
        self.pool = ConnectionPool(
            conninfo=self._conninfo(), max_size=4,
            kwargs={"autocommit": True, "row_factory": dict_row, "options": "-c TimeZone=UTC"},
        )
        self.checkpointer = PostgresSaver(self.pool)
        self.checkpointer.setup()               # creates the checkpoint tables the first time
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    @staticmethod
    def _conninfo() -> str:
        return (f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
                f"port={os.environ.get('POSTGRES_PORT', '5432')} "
                f"user={config.require('POSTGRES_USER')} password={config.require('POSTGRES_PASSWORD')} "
                f"dbname={config.require('POSTGRES_DB')}")

    def _sms_post(self, payload: dict) -> None:
        response = httpx.post(SMS_URL, json=payload, headers={"X-Api-Key": self.sms_key}, timeout=10)
        response.raise_for_status()

    def _text(self, state: State, text: str, reference: str) -> None:
        """Send one question by SMS and remember that we're waiting for its reply."""
        self.sms_post({"to": state["card_id"], "text": text, "reference": reference,
                       "txn_id": state["txn_id"], "expects_reply": True})
        self.r.hset(WAITING, reference, json.dumps({"thread": thread_id(state["based_on"]),
                                                    "since": time.time()}))
        event = VerificationRequested(
            event_id=reference, published_at=now_utc(), txn_id=state["txn_id"],
            occurred_at=state["occurred_at"],
            payload=VerificationRequestedPayload(card_id=state["card_id"], message=text,
                                                 based_on_event_id=state["based_on"]))
        bus.publish(self.r, event)

    # ----- the graph -----

    def _build(self) -> StateGraph:
        def ask_customer(state: State) -> dict:
            self._text(state, state["question"], state["reference"])
            return {}

        def wait_for_reply(state: State) -> dict:
            # PAUSE here. The graph's state is saved in PostgreSQL; resume() continues it.
            reply = interrupt({"waiting_for": state["reference"]})
            return {"reply": reply}

        def read_reply(state: State, config: RunnableConfig) -> dict:
            if state["reply"] == NO_REPLY:
                return {"label": "no_reply"}
            # 1. Plain rules first: a clear YES or NO, or a question back to the bank,
            #    needs no AI (and the AI can't overrule it).
            label = keyword_label(state["reply"])
            if label != "unclear" or state["reply"].strip().endswith("?"):
                return {"label": label, "labelled_by": "rules"}
            # 2. Only replies the rules can't read go to the AI.
            prompt = (f"A bank texted a customer: \"{state['question']}\"\n"
                      f"The customer replied: \"{state['reply']}\"\n"
                      "Label the reply:\n"
                      "- confirmed: the customer clearly says they made this payment.\n"
                      "- denied: the customer clearly says they did not make this payment.\n"
                      "- unclear: anything else. When in doubt, choose unclear.\n"
                      "Examples: 'that was me at the shop' -> confirmed; "
                      "'I never bought that' -> denied; 'call me later' -> unclear; "
                      "'hmm not sure' -> unclear; 'what is this about' -> unclear.")
            try:
                label = self.labeller.invoke([HumanMessage(prompt)], config=config).label
            except Exception as error:   # the AI failed: keep the rules' answer
                log.warning("AI could not label %r (%s); keeping 'unclear'", state["reply"], error)
                return {"label": "unclear", "labelled_by": "rules"}
            return {"label": label, "labelled_by": "AI"}

        def after_read(state: State) -> Literal["ask_again", "finish"]:
            return "ask_again" if state["label"] == "unclear" and state["follow_ups"] < 1 else "finish"

        def ask_again(state: State) -> dict:
            reference = derived_event_id(state["reference"], PRODUCER, "follow-up")
            self._text(state, "Sorry, we didn't understand. Did you make this payment? "
                              "Please reply YES or NO.", reference)
            return {"reference": reference, "follow_ups": state["follow_ups"] + 1}

        def finish(state: State) -> dict:
            outcome = {"confirmed": "confirmed_by_customer", "denied": "denied_by_customer",
                       "unclear": "unclear", "no_reply": "no_reply"}[state["label"]]
            return {"outcome": outcome}

        g = StateGraph(State)
        for name, fn in [("ask_customer", ask_customer), ("wait_for_reply", wait_for_reply),
                         ("read_reply", read_reply), ("ask_again", ask_again), ("finish", finish)]:
            g.add_node(name, fn)
        g.add_edge(START, "ask_customer")
        g.add_edge("ask_customer", "wait_for_reply")
        g.add_edge("wait_for_reply", "read_reply")
        g.add_conditional_edges("read_reply", after_read, ["ask_again", "finish"])
        g.add_edge("ask_again", "wait_for_reply")
        g.add_edge("finish", END)
        return g

    # ----- starting, resuming, finishing -----

    def _config(self, thread: str, txn_id: str) -> dict:
        return run_config(PRODUCER, txn_id, self.logger) | {"configurable": {"thread_id": thread}}

    def start(self, report: InvestigationCompleted) -> None:
        thread = thread_id(report.event_id)
        cfg = self._config(thread, report.txn_id)
        snapshot = self.graph.get_state(cfg)
        if snapshot.values:
            if snapshot.next and not snapshot.interrupts:
                # Started before but stopped half-way (e.g. the SMS provider was down): finish it.
                self.graph.invoke(None, cfg)
                log.info("txn %s  continued an unfinished start, now WAITING", report.txn_id[:8])
            else:
                log.info("txn %s  already being verified", report.txn_id[:8])
            return
        payment = payment_details(report.txn_id)
        question = (f"Bank alert: did you just pay ${payment['amount']:.2f} at {payment['merchant']}? "
                    "Reply YES or NO.")
        self.graph.invoke({
            "txn_id": report.txn_id, "card_id": payment["card_id"],
            "occurred_at": report.occurred_at.isoformat(), "question": question,
            "reference": derived_event_id(report.event_id, PRODUCER, "verification.requested"),
            "based_on": report.event_id, "reply": None, "label": None, "labelled_by": None,
            "follow_ups": 0,
            "outcome": None,
        }, cfg)
        log.info("txn %s  texted the customer, now WAITING (paused)", report.txn_id[:8])

    def resume(self, reference: str, reply: str) -> None:
        info = self.r.hget(WAITING, reference)
        if info is None:
            log.info("No case waiting for reference %s (late or repeated reply): ignored", reference[:8])
            return
        # Only one resumer at a time per reference (a repeated reply or the timeout may race).
        lock = f"verification:resuming:{reference}"
        if not self.r.set(lock, 1, nx=True, ex=300):
            log.info("Reference %s is already being resumed: ignored", reference[:8])
            return
        try:
            thread = json.loads(info)["thread"]
            snapshot = self.graph.get_state({"configurable": {"thread_id": thread}})
            cfg = self._config(thread, snapshot.values["txn_id"])
            if snapshot.interrupts:                       # paused, waiting: the normal case
                result = self.graph.invoke(Command(resume=reply), cfg)
            elif snapshot.next:                           # an earlier run stopped half-way
                result = self.graph.invoke(None, cfg)
                if "__interrupt__" in result and snapshot.values.get("reply") is None:
                    # It stopped before the first wait, so this reply was never read: use it.
                    # (If it stopped while asking again, it now waits for the NEW question.)
                    result = self.graph.invoke(Command(resume=reply), cfg)
            else:                                         # finished, but publishing had failed
                result = snapshot.values
            if "__interrupt__" in result:
                log.info("txn %s  reply %r was unclear: asked again, WAITING",
                         snapshot.values["txn_id"][:8], reply)
            else:
                self._publish_result(result)
            # Only now, after success, is this reference done. If anything above failed,
            # the entry stays, and the retried reply (or the timeout) tries again.
            self.r.hdel(WAITING, reference)
        finally:
            self.r.delete(lock)

    def _publish_result(self, state: dict) -> None:
        event = VerificationCompleted(
            event_id=derived_event_id(state["based_on"], PRODUCER, "verification.completed"),
            published_at=now_utc(), txn_id=state["txn_id"], occurred_at=state["occurred_at"],
            payload=VerificationCompletedPayload(
                outcome=state["outcome"],
                reply_text=None if state["reply"] == NO_REPLY else state["reply"],
                based_on_event_id=state["reference"]))
        bus.publish(self.r, event)
        log.info("txn %s  RESUMED and finished: %s (reply %r, read by %s)", state["txn_id"][:8],
                 state["outcome"], state["reply"], state.get("labelled_by") or "-")

    def check_timeouts(self) -> None:
        """Resume any case that has waited too long, as 'no reply'. Never crashes the agent."""
        try:
            waiting = self.r.hgetall(WAITING)
        except Exception as error:
            log.warning("Could not check timeouts: %s", error)
            return
        for reference, info in waiting.items():
            try:
                if time.time() - json.loads(info)["since"] > REPLY_TIMEOUT_SECONDS:
                    self.resume(reference, NO_REPLY)
            except Exception:
                log.exception("Timeout handling failed for %s; will try again", reference[:8])


def thread_id(investigation_event_id: str) -> str:
    return f"verification-{investigation_event_id}"


def payment_details(txn_id: str) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT body FROM events WHERE txn_id = %s AND event_type = "
                           "'transaction.received'", (txn_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"payment {txn_id} is not in the audit log yet")
    p = (row[0] if isinstance(row[0], dict) else json.loads(row[0]))["payload"]
    return {"card_id": p["card_id"], "amount": float(p["amount"]),
            "merchant": p["merchant"].removeprefix("fraud_")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "psycopg.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    r = bus.connect()
    r.ping()
    verifier = Verifier(r)

    def on_report(event):
        if isinstance(event, InvestigationCompleted) and \
                event.payload.recommended_action == "verify_with_customer":
            verifier.start(event)

    def on_reply(event):
        if isinstance(event, CustomerReplied):
            verifier.resume(event.payload.in_reply_to, event.payload.reply_text)

    host = f"{socket.gethostname()}-{os.getpid()}"
    for stream, handler in [("investigation.completed", on_report), ("customer.replied", on_reply)]:
        threading.Thread(target=bus.consume, daemon=True, name=stream, args=(
            bus.connect(), bus.stream_for(stream), PRODUCER, handler),
            kwargs={"consumer": f"{host}-{stream}", "batch_size": 1}).start()
    log.info("Verifying with %s; a customer who doesn't reply within %d s counts as 'no reply'. "
             "Press Ctrl + C to stop.", getattr(verifier.model, "model", "the model"), REPLY_TIMEOUT_SECONDS)
    try:
        while True:
            time.sleep(5)
            verifier.check_timeouts()
    except KeyboardInterrupt:
        log.info("Stopped. Waiting cases stay saved in PostgreSQL and continue on the next start.")


if __name__ == "__main__":
    main()
