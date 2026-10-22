"""The Notification agent (Part 2 · Step 7): tells the customer the final outcome.

It reads "final outcome" forms (transaction.resolved) and runs a small LangGraph graph:

    START -> draft (the AI writes the text) -> check (plain rules)
                                                  | passes        | fails
                                                  v               v
                                                send <------ use_template (a safe, fixed text)

- draft: the AI (Llama) writes a short, friendly SMS about the outcome.
- check: plain rules: short enough; mentions the right amount and shop; uses the
  right outcome word; no links; never asks for a PIN, password or card number.
- use_template: if the AI's draft fails ANY rule (or the AI fails), a fixed template
  is sent instead. The customer always gets a correct message.
- send: the text goes out through the SMS provider; notification.sent is published
  (used_template says whether the AI's text or the template was sent).

Run from the project folder:  python -m fraud_agents.agents.notification.agent
"""

import json
import logging
import os
import re
import time

import httpx
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from fraud_agents.shared import bus, config, db
from fraud_agents.shared.events import (
    NotificationSent,
    NotificationSentPayload,
    TransactionResolved,
    derived_event_id,
    now_utc,
)
from fraud_agents.shared.llm import LLMCallLogger, chat_model, run_config

PRODUCER = "notification"
GROUP = "notification"
SMS_URL = os.environ.get("SMS_PROVIDER_URL", "http://localhost:8100") + "/messages"
MAX_LENGTH = 300

log = logging.getLogger(PRODUCER)

# What each outcome means for the customer, and the word the text must contain.
# The key is the resolution, or "resolution/rule_id" where the reason changes the wording.
OUTCOMES = {
    "blocked": ("we blocked the payment and protected your card, because you told us it was not you",
                "blocked"),
    "released/customer_confirmed": ("the payment is approved, because you confirmed you made it",
                                    "approved"),
    "released": ("we checked the payment, it looks normal, and it is approved", "approved"),
    "sent_to_analyst": ("a member of our fraud team is reviewing the payment and will contact you",
                        "review"),
    "blocked/analyst_blocked": ("our fraud team reviewed the payment and blocked it to protect your card",
                                "blocked"),
    "released/analyst_released": ("our fraud team reviewed the payment and it is approved", "approved"),
}

TEMPLATES = {
    "blocked": "Bank: we blocked the ${amount} payment at {merchant} because you told us it "
               "wasn't you. Your card is now protected.",
    "released/customer_confirmed": "Bank: thanks for confirming. The ${amount} payment at "
                                   "{merchant} is approved.",
    "released": "Bank: we checked the ${amount} payment at {merchant}. It is approved, and you "
                "don't need to do anything.",
    "sent_to_analyst": "Bank: our fraud team is reviewing the ${amount} payment at {merchant} "
                       "and will contact you soon.",
    "blocked/analyst_blocked": "Bank: our fraud team reviewed the ${amount} payment at {merchant} "
                               "and blocked it. Your card is now protected.",
    "released/analyst_released": "Bank: our fraud team reviewed the ${amount} payment at {merchant}. "
                                 "It is approved, and you don't need to do anything.",
}

FORBIDDEN = re.compile(r"(https?://|www\.|\b[a-z0-9-]+\.(com|ca|net|org|io|co|info)\b|\bp\.?\s?i\.?\s?n\b|"
                       r"password|card number|\bcvv2?\b|\bcvc\b|click)", re.IGNORECASE)

# Never ask for, or mention, anything a scammer would want (Part 3 · Step 3: more added).
SENSITIVE = re.compile(
    r"(account (number|no\b|details)|routing|\bsin\b|\bssn\b|social (security|insurance)|"
    r"date of birth|birth ?date|\bdob\b|security (code|question)|verification code|one[- ]time|"
    r"\botp\b|passcode|\bcode\b|maiden name|log ?in|sign ?in|username|user name|\bid number|"
    r"passport|driver'?s licen[cs]e|full name|home address|your address|expir\w*|"
    r"last (4|four) digits|security answer|(bank|banking) (details|info\w*)|(3|three)[- ]digit|"
    r"digits on the back|e-?mail|post ?code|postal code|zip code|bit\.ly|tinyurl|\bdot com\b|"
    r"\+?\d[\d\s().-]{7,}\d|\bcall (us|me|this|now)\b|\bverify your\b)",
    re.IGNORECASE)

# Words that would contradict the outcome (e.g. "we blocked it ... it is approved").
CONTRADICTS = {
    "blocked": re.compile(r"(?<!not )\b(approv\w*|releas\w*|authori[sz]ed|processed|completed|went through|"
                          r"gone through|was successful)\b", re.IGNORECASE),
    "approved": re.compile(r"\b(block\w*|declin\w*|stopp\w*|frozen|freez\w*|reject\w*|cancel\w*|"
                           r"on hold|held|denied|refused|reversed|suspend\w*)\b", re.IGNORECASE),
    "review": re.compile(r"\b(approv\w*|releas\w*|block\w*|declin\w*|stopp\w*|reject\w*|cancel\w*)\b",
                         re.IGNORECASE),
}


def template_text(key: str, amount: str, merchant: str) -> str:
    """The safe fixed text. A very long shop name is shortened so the text fits one SMS."""
    text = TEMPLATES[key].format(amount=amount, merchant=merchant)
    if len(text) > MAX_LENGTH:
        room = max(10, len(merchant) - (len(text) - MAX_LENGTH) - 1)
        text = TEMPLATES[key].format(amount=amount, merchant=merchant[:room].rstrip() + "…")
    return text


def outcome_key(resolution: str, rule_id: str) -> str:
    specific = f"{resolution}/{rule_id}"
    return specific if specific in OUTCOMES else resolution


class State(TypedDict):
    resolution: str             # an OUTCOMES key, e.g. "blocked" or "released/customer_confirmed"
    amount: str
    merchant: str
    draft: str | None
    problems: list[str]
    message: str | None
    used_template: bool


def check(text: str, resolution: str, amount: str, merchant: str) -> list[str]:
    """Plain rules the AI's text must pass. Returns the problems (empty = OK)."""
    problems = []
    if not text or len(text) > MAX_LENGTH:
        problems.append(f"length must be 1 to {MAX_LENGTH} characters")
    if not text.startswith("Bank:"):
        problems.append("must start with 'Bank:'")
    plain = text.replace(",", "")                      # $1,042.58 counts as 1042.58
    if not re.search(rf"(?<![\d.]){re.escape(amount)}(?![\d])", plain):
        problems.append(f"must state the exact amount {amount}")
    if merchant.split(",")[0].split(" ")[0].lower() not in text.lower():
        problems.append("must name the shop")
    word = OUTCOMES[resolution][1]
    # The rules below read the text WITHOUT the shop name and the amount, so a shop called
    # e.g. "Yost, Block and Koepp" can't trip the "blocked" rules (Part 3 · Step 3).
    rest = re.sub(re.escape(merchant), " ", text, flags=re.IGNORECASE).replace(amount, " ")
    if not re.search(rf"\b{word}\w*", rest, re.IGNORECASE):          # e.g. review, reviewing
        problems.append(f"must use the word '{word}'")
    if re.search(rf"\b(not|never|no|\w+n['’]t)\W+(\w+\W+)?{word}|\bun{word}", rest, re.IGNORECASE):
        problems.append(f"must not negate '{word}'")
    if FORBIDDEN.search(rest):
        problems.append("no links, and never ask for a PIN, password or card details")
    if SENSITIVE.search(rest):
        problems.append("never ask for or mention personal or security details, codes or phone numbers")
    if CONTRADICTS[word].search(rest):
        problems.append(f"must not contradict '{word}' (e.g. say both blocked and approved)")
    return problems


def build_graph(model=None):
    model = model or chat_model()

    def draft(state: State, config: RunnableConfig) -> dict:
        meaning, word = OUTCOMES[state["resolution"]]
        prompt = (
            "Write one short SMS (under 200 characters) from a bank to its customer.\n"
            f"Facts: a ${state['amount']} card payment at {state['merchant']}; {meaning}.\n"
            f"Rules: start with 'Bank:'; state the amount exactly as ${state['amount']}; name the shop; "
            f"use the word '{word}'; no links; never ask for a PIN, password or card details.\n"
            "Reply with the SMS text only.")
        try:
            text = str(model.invoke([HumanMessage(prompt)], config=config).content).strip().strip('"')
            if "Bank:" in text:                  # drop any chatter before the SMS itself
                text = text[text.index("Bank:"):].strip().strip('"')
        except Exception as error:
            log.warning("AI draft failed (%s); the template will be used", error)
            return {"draft": None, "problems": ["the AI could not write a draft"]}
        return {"draft": text, "problems": check(text, state["resolution"], state["amount"], state["merchant"])}

    def use_template(state: State) -> dict:
        text = template_text(state["resolution"], state["amount"], state["merchant"])
        if problems := check(text, state["resolution"], state["amount"], state["merchant"]):
            log.warning("the safe template itself fails a rule (%s); sent anyway", "; ".join(problems))
        return {"message": text, "used_template": True}

    def accept_draft(state: State) -> dict:
        return {"message": state["draft"], "used_template": False}

    g = StateGraph(State)
    g.add_node("draft", draft)
    g.add_node("accept_draft", accept_draft)
    g.add_node("use_template", use_template)
    g.add_edge(START, "draft")
    g.add_conditional_edges("draft", lambda s: "use_template" if s["problems"] else "accept_draft",
                            ["accept_draft", "use_template"])
    g.add_edge("accept_draft", END)
    g.add_edge("use_template", END)
    return g.compile(), model


def payment_details(txn_id: str) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT body FROM events WHERE txn_id = %s AND event_type = "
                           "'transaction.received'", (txn_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"payment {txn_id} is not in the audit log yet")
    p = (row[0] if isinstance(row[0], dict) else json.loads(row[0]))["payload"]
    return {"card_id": p["card_id"], "amount": f"{float(p['amount']):.2f}",
            "merchant": p["merchant"].removeprefix("fraud_")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    graph, model = build_graph()
    logger = LLMCallLogger()
    sms_key = config.require("SMS_API_KEY")
    r = bus.connect()
    r.ping()

    def handle(event):
        if not isinstance(event, TransactionResolved):
            raise ValueError(f"Expected transaction.resolved, got {event.event_type}")
        current = r.get(f"resolved:{event.txn_id}")   # the Decision agent's current final outcome
        if current and current != event.event_id:
            log.info("txn %s  %s skipped: a later outcome replaced it (e.g. an analyst's decision)",
                     event.txn_id[:8], event.payload.resolution)
            return
        p = payment_details(event.txn_id)
        start = time.time()
        # The text is drafted ONCE and saved before sending. A retry (e.g. after a crash) reuses
        # the saved text, and the Idempotency-Key stops the SMS provider sending it twice, so
        # the customer gets one text and the record shows exactly the text they got.
        text_key = f"notification:text:{event.event_id}"
        saved = r.get(text_key)
        if saved:
            result = json.loads(saved)
        else:
            result = graph.invoke(
                {"resolution": outcome_key(event.payload.resolution, event.payload.rule_id),
                 "amount": p["amount"], "merchant": p["merchant"],
                 "draft": None, "problems": [], "message": None, "used_template": False},
                config=run_config(PRODUCER, event.txn_id, logger))
            result = {k: result[k] for k in ("message", "used_template", "problems")}
            if not r.set(text_key, json.dumps(result), nx=True, ex=7 * 24 * 3600):
                result = json.loads(r.get(text_key))   # another copy saved one first: use it
        httpx.post(SMS_URL, timeout=10, headers={
            "X-Api-Key": sms_key, "Idempotency-Key": f"notification:{event.event_id}"}, json={
            "to": p["card_id"], "text": result["message"], "txn_id": event.txn_id,
        }).raise_for_status()
        bus.publish(r, NotificationSent(
            event_id=derived_event_id(event.event_id, PRODUCER, "notification.sent"),
            published_at=now_utc(), txn_id=event.txn_id, occurred_at=event.occurred_at,
            payload=NotificationSentPayload(card_id=p["card_id"], message=result["message"],
                                            used_template=result["used_template"],
                                            based_on_event_id=event.event_id)))
        source = "TEMPLATE" if result["used_template"] else "AI draft"
        why = f" (AI draft rejected: {'; '.join(result['problems'])})" if result["used_template"] else ""
        log.info("txn %s  %s -> sent %s: %r%s  (%.1f s)", event.txn_id[:8], event.payload.resolution,
                 source, result["message"], why, time.time() - start)

    source = bus.stream_for("transaction.resolved")
    log.info("Listening on %s (group %s) with %s. Press Ctrl + C to stop.",
             source, GROUP, getattr(model, "model", "the model"))
    try:
        bus.consume(r, source, GROUP, handle, batch_size=1, reclaim_after_ms=5 * 60 * 1000)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
