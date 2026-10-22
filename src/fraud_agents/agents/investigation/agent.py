"""The Investigation agent (Part 2 · Step 4): the first fully agentic AI agent.

It reads "I decided" forms. For every flagged or held payment it runs a LangGraph
graph:

    START -> gather_facts -> investigate <---> use_tools     (the AI picks tools; at most 4)
                                  |  <---> nudge          (at most twice: stopped too early, or
                                  |                        wrote a tool call as text)
                                  v
                             write_report  --(report fails the checks once)--> write_report again
                                  |                    (fails twice) --> escalate --> END
                                  v
                                 done --> END             (the report is published)

- gather_facts: always fetches the payment's facts first (no AI needed for that).
- investigate: the AI reads the facts and decides which tool to use next, or stops.
- use_tools: runs the tools the AI asked for (never more than 4 in total).
- nudge: if the AI stopped without checking anything, or wrote a tool request as text
  instead of really calling it, remind it (at most twice).
- write_report: the AI writes a fixed-shape report (verdict, summary, evidence, advice),
  then plain rules check it (check()); one retry, then a human takes over.

The AI only advises: its report goes to the next agents; it never blocks or releases
a payment itself.

Run as a service:   python -m fraud_agents.agents.investigation.agent
Try one payment:    python -m fraud_agents.agents.investigation.agent --once [txn_id]
"""

import argparse
import json
import logging
import operator
import re
import time
from typing import Annotated, Literal

from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from fraud_agents.agents.investigation.tools import (
    _conn,
    card_recent_activity,
    card_spending_profile,
    get_payment_details,
    shop_type_risk,
)
from fraud_agents.shared import bus
from fraud_agents.shared.events import (
    CaseEscalated,
    CaseEscalatedPayload,
    InvestigationCompleted,
    InvestigationCompletedPayload,
    TransactionDecided,
    derived_event_id,
    now_utc,
)
from fraud_agents.shared.llm import LLMCallLogger, chat_model, run_config

PRODUCER = "investigation"
GROUP = "investigation"
MAX_TOOL_CALLS = 4        # the AI may use at most 4 tools per investigation
MAX_REPORT_TRIES = 2      # first try + one retry, then a human takes over
# The payment's facts are always fetched first (gather_facts), so the AI chooses among the others.
AI_TOOLS = [card_recent_activity, card_spending_profile, shop_type_risk]
TOOLS_BY_NAME = {t.name: t for t in AI_TOOLS}
SLOW_RECLAIM_MS = 10 * 60 * 1000   # an investigation is never picked up again while still running
MAX_NUDGES = 2

log = logging.getLogger(PRODUCER)

SYSTEM_PROMPT = """You are a fraud investigator at a bank. A payment was flagged by the
fraud model. Find out whether it looks like fraud by using the tools.
The payment's facts are already below. Use the other tools to check the card's recent
activity, its normal spending habits and the shop type's risk. Call one tool at a time.
When you have enough evidence, reply with a short plain-text summary of what you found,
without calling a tool. Be factual: only use numbers that the tools gave you."""

REPORT_PROMPT = """Write your investigation report about payment {txn_id} as structured data.
- verdict: likely_fraud, likely_genuine or unsure.
- summary: 2 to 4 full sentences explaining the verdict, using the numbers from the tools.
- evidence: 1 to 5 short facts taken from the tool results (each one sentence).
- recommended_action: verify_with_customer (ask the customer if they made it),
  escalate_to_analyst (a human should look), or no_further_action (clearly genuine).
{feedback}"""


class InvestigationReport(BaseModel):
    """The fixed shape the AI must fill in."""

    verdict: Literal["likely_fraud", "likely_genuine", "unsure"] = Field(
        description="Your conclusion about the payment")
    summary: str = Field(description="2 to 4 full sentences explaining the verdict, with numbers")
    evidence: list[str] = Field(description="1 to 5 short facts from the tool results")
    recommended_action: Literal["verify_with_customer", "escalate_to_analyst", "no_further_action"] = Field(
        description="What should happen next")


class State(TypedDict):
    txn_id: str
    nudges: int               # how many times the AI was reminded (at most MAX_NUDGES)
    messages: Annotated[list[AnyMessage], operator.add]   # the conversation, added to at each step
    tool_calls_made: int
    tools_used: Annotated[list[str], operator.add]
    report: dict | None
    report_tries: int
    problems: list[str]


def check(report: InvestigationReport, tools_used: list[str]) -> list[str]:
    """Plain rules the AI's report must pass. Returns the problems found (empty = OK)."""
    problems = []
    if len(report.summary.strip()) < 40 or report.summary.count(" ") < 6:
        problems.append("the summary is too short: write 2 to 4 full sentences")
    if not 1 <= len(report.evidence) <= 5:
        problems.append("give between 1 and 5 pieces of evidence")
    if any(len(e) > 250 for e in report.evidence):
        problems.append("each piece of evidence must be one short sentence")
    if report.verdict != "likely_genuine" and report.recommended_action == "no_further_action":
        problems.append(f"the verdict {report.verdict} cannot have no_further_action "
                        "(only likely_genuine can)")
    if len(set(tools_used)) < 2:
        problems.append("the investigation used fewer than 2 different tools")
    return problems


def build_graph(model=None):
    """Build the investigation graph. `model` can be swapped (e.g. Claude, or a test model)."""
    model = model or chat_model()
    model_with_tools = model.bind_tools(AI_TOOLS)
    report_writer = model.with_structured_output(InvestigationReport)

    def gather_facts(state: State) -> dict:
        facts = get_payment_details.invoke({"txn_id": state["txn_id"]})
        return {
            "messages": [SystemMessage(SYSTEM_PROMPT),
                         HumanMessage(f"Investigate payment {state['txn_id']}.\nFacts: {facts}")],
            "tools_used": ["get_payment_details"],
        }

    def investigate(state: State, config: RunnableConfig) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"], config=config)]}

    def use_tools(state: State) -> dict:
        results, used = [], []
        budget = MAX_TOOL_CALLS - state.get("tool_calls_made", 0)
        for call in state["messages"][-1].tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if len(used) >= budget:
                output = "Tool limit reached: no more tools. Write your summary now."
            elif tool is None:
                output = f"There is no tool called {call['name']}."
            else:
                args = dict(call["args"]) | {"txn_id": state["txn_id"]}  # always this payment
                output = tool.invoke(args)
                used.append(call["name"])
            # Every tool request gets an answer, so the conversation stays valid for the model.
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results, "tools_used": used,
                "tool_calls_made": state.get("tool_calls_made", 0) + len(used)}

    def looks_like_text_tool_call(message) -> bool:
        """Small models sometimes WRITE a tool request as text instead of really calling it."""
        text = str(getattr(message, "content", "")).strip()
        return text.startswith("{") and '"name"' in text

    def nudge(state: State) -> dict:
        names = ", ".join(TOOLS_BY_NAME)
        if looks_like_text_tool_call(state["messages"][-1]):
            text = ("You wrote a tool request as text, so it was not run. To use a tool, call it "
                    f"with the tool-calling function. The only tools are: {names}.")
        else:
            text = f"Before concluding, call at least one tool: {names}."
        return {"nudges": state.get("nudges", 0) + 1, "messages": [HumanMessage(text)]}

    def after_investigate(state: State) -> Literal["use_tools", "nudge", "write_report"]:
        last = state["messages"][-1]
        made = state.get("tool_calls_made", 0)
        if getattr(last, "tool_calls", None) and made < MAX_TOOL_CALLS:
            return "use_tools"
        can_nudge = state.get("nudges", 0) < MAX_NUDGES and made < MAX_TOOL_CALLS
        if can_nudge and (made == 0 or looks_like_text_tool_call(last)):
            return "nudge"  # it stopped too early, or wrote a tool call as text: remind it
        return "write_report"

    def write_report(state: State, config: RunnableConfig) -> dict:
        problems = state.get("problems") or []
        feedback = ("Your previous report was rejected because: " + "; ".join(problems) + ". Fix it."
                    if problems else "")
        # The AI's last tool-call request may be unanswered (limit reached): keep only the
        # conversation up to the last tool result or plain answer, then ask for the report.
        history = list(state["messages"])
        if getattr(history[-1], "tool_calls", None):
            history = history[:-1]
        prompt = HumanMessage(REPORT_PROMPT.format(txn_id=state["txn_id"], feedback=feedback))
        tries = state.get("report_tries", 0) + 1
        try:
            report = report_writer.invoke(history + [prompt], config=config)
            if not isinstance(report, InvestigationReport):
                report = InvestigationReport.model_validate(report)
            problems = check(report, state.get("tools_used", []))
        except (ValidationError, ValueError, TypeError) as error:
            report, problems = None, [f"the report did not have the required shape ({str(error)[:150]})"]
        return {"report": report.model_dump() if report else None,
                "problems": problems, "report_tries": tries}

    def after_check(state: State) -> Literal["done", "write_report", "escalate"]:
        if not state["problems"]:
            return "done"
        return "write_report" if state["report_tries"] < MAX_REPORT_TRIES else "escalate"

    graph = StateGraph(State)
    graph.add_node("gather_facts", gather_facts)
    graph.add_node("investigate", investigate)
    graph.add_node("use_tools", use_tools)
    graph.add_node("nudge", nudge)
    graph.add_node("write_report", write_report)
    graph.add_node("done", lambda state: {})
    graph.add_node("escalate", lambda state: {})
    graph.add_edge(START, "gather_facts")
    graph.add_edge("gather_facts", "investigate")
    graph.add_conditional_edges("investigate", after_investigate, ["use_tools", "nudge", "write_report"])
    graph.add_edge("use_tools", "investigate")
    graph.add_edge("nudge", "investigate")
    graph.add_conditional_edges("write_report", after_check, ["done", "write_report", "escalate"])
    graph.add_edge("done", END)
    graph.add_edge("escalate", END)
    return graph.compile(), model


def model_name(model) -> str:
    return str(getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__)


def wait_for_payment(txn_id: str, seconds: float = 15) -> bool:
    """The audit log may still be recording the payment: wait for it briefly."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        found = _conn().execute(
            "SELECT 1 FROM events WHERE txn_id = %s AND event_type = 'transaction.received'", (txn_id,)
        ).fetchone()
        if found:
            return True
        time.sleep(1)
    return False


def investigate(graph, txn_id: str, logger: LLMCallLogger) -> dict:
    return graph.invoke(
        {"txn_id": txn_id, "nudges": 0, "messages": [], "tool_calls_made": 0, "tools_used": [],
         "report": None, "report_tries": 0, "problems": []},
        config=run_config(PRODUCER, txn_id, logger) | {"recursion_limit": 30},
    )


def give_up(r, event, reason: str) -> None:
    """The investigation failed every try (e.g. the AI or a tool is down): hand it to a human.

    Without this, a flagged or held payment would be set aside with no final outcome."""
    if not (isinstance(event, TransactionDecided) and event.payload.decision in ("flag", "hold")):
        return
    bus.publish(r, CaseEscalated(
        event_id=derived_event_id(event.event_id, PRODUCER, "case.escalated"),
        published_at=now_utc(), txn_id=event.txn_id, occurred_at=event.occurred_at, producer=PRODUCER,
        payload=CaseEscalatedPayload(
            reason=f"The investigation could not finish ({reason[:300]}), so a human must decide.",
            based_on_event_id=event.event_id)))
    log.warning("txn %s  investigation gave up: ESCALATED to a human", event.txn_id[:8])


NUMBER = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)(%?)")
LONG_ID = re.compile(r"\b(?:card-)?[0-9a-f]{16,}\b")   # txn and card IDs are not facts to check


def tool_numbers(messages) -> list[float]:
    """Every number the tools (and the payment's facts) gave the AI."""
    texts = [str(m.content) for m in messages
             if isinstance(m, ToolMessage) or (isinstance(m, HumanMessage) and "Facts:" in str(m.content))]
    found = NUMBER.findall(LONG_ID.sub(" ", " ".join(texts)))
    return [float(n.replace(",", "")) for n, _ in found]


def ungrounded_numbers(report: dict, messages) -> list[str]:
    """Numbers in the report that no tool gave (allowing rounding, and 0.0141 = 1.41%)."""
    source = tool_numbers(messages)
    text = LONG_ID.sub(" ", " ".join([report["summary"], *report["evidence"]]))
    missing = []
    for raw, percent in NUMBER.findall(text):
        value = float(raw.replace(",", ""))
        places = len(raw.split(".")[1]) if "." in raw else 0
        candidates = source + ([s * 100 for s in source] if percent else [])
        if not any(abs(round(c, places) - value) < 1e-9 for c in candidates):
            missing.append(raw + percent)
    return missing


def guard_action(report: dict, messages) -> tuple[str, str | None]:
    """'no_further_action' releases a flagged payment without asking anyone, so the AI may
    choose it only with a likely_genuine verdict, every piece of evidence stating a number,
    and every number in its report traced to the tools. Otherwise the customer is asked
    instead (Part 3 · Step 3). This can't prove the AI's reasoning right; it only stops
    unchecked claims from releasing a payment on their own."""
    action = report["recommended_action"]
    if action != "no_further_action":
        return action, None
    if report["verdict"] != "likely_genuine":
        return "verify_with_customer", f"verdict {report['verdict']}"
    # Every piece of evidence must be a checkable number from the tools; facts in words only
    # ("shopped here weekly for years") can't be checked, so they can't release a payment.
    wordy = [e for e in report["evidence"] if not NUMBER.search(LONG_ID.sub(" ", e))]
    if wordy:
        return "verify_with_customer", "evidence without a checkable number: " + "; ".join(wordy)[:200]
    missing = ungrounded_numbers(report, messages)
    if missing:
        return "verify_with_customer", "numbers not found in the tool results: " + ", ".join(missing[:5])
    return action, None


def to_event(decided: TransactionDecided, result: dict, model: str):
    if result["problems"] or not result["report"]:
        return CaseEscalated(
            event_id=derived_event_id(decided.event_id, PRODUCER, "case.escalated"),
            published_at=now_utc(), txn_id=decided.txn_id, occurred_at=decided.occurred_at,
            producer=PRODUCER,
            payload=CaseEscalatedPayload(
                reason="The AI investigation report failed the checks twice: "
                       + "; ".join(result["problems"])[:400],
                based_on_event_id=decided.event_id),
        )
    r = result["report"]
    action, why = guard_action(r, result.get("messages", []))
    if why:
        log.warning("txn %s  AI suggested no_further_action, changed to %s (%s)",
                    decided.txn_id[:8], action, why)
    return InvestigationCompleted(
        event_id=derived_event_id(decided.event_id, PRODUCER, "investigation.completed"),
        published_at=now_utc(), txn_id=decided.txn_id, occurred_at=decided.occurred_at,
        payload=InvestigationCompletedPayload(
            verdict=r["verdict"], summary=r["summary"][:1000], evidence=r["evidence"][:6],
            tools_used=list(dict.fromkeys(result["tools_used"])),
            recommended_action=action, model=model,
            based_on_event_id=decided.event_id),
    )


def print_trace(result: dict) -> None:
    for message in result["messages"]:
        kind = type(message).__name__
        if kind == "AIMessage" and message.tool_calls:
            print(f"  AI asks for: {[c['name'] for c in message.tool_calls]}")
        elif kind == "ToolMessage":
            print(f"  tool result: {message.content[:160]}...")
        elif kind == "AIMessage":
            print(f"  AI: {message.content[:200]}")
    print(f"\n  tools used: {list(dict.fromkeys(result['tools_used']))}")
    print(f"  report tries: {result['report_tries']}, problems: {result['problems'] or 'none'}")
    print(f"  report: {json.dumps(result['report'], indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", nargs="?", const="latest", help="investigate one payment and print the trace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    graph, model = build_graph()
    logger = LLMCallLogger()

    if args.once:
        txn = args.once
        if txn == "latest":
            row = _conn().execute(
                "SELECT txn_id FROM events WHERE event_type = 'transaction.decided' "
                "AND decision IN ('flag', 'hold') ORDER BY published_at DESC LIMIT 1").fetchone()
            if row is None:
                raise SystemExit("No flagged or held payment yet: run a replay first.")
            txn = row[0]
        print(f"Investigating {txn} with {model_name(model)} ...\n")
        start = time.time()
        result = investigate(graph, txn, logger)
        print_trace(result)
        print(f"\n  took {time.time() - start:.1f} s (nothing was published: --once is a dry run)")
        return

    r = bus.connect()
    r.ping()

    def handle(event):
        if not isinstance(event, TransactionDecided):
            raise ValueError(f"Expected transaction.decided, got {event.event_type}")
        if event.payload.decision == "approve":
            return  # approved payments need no investigation
        if not wait_for_payment(event.txn_id):
            raise RuntimeError(f"payment {event.txn_id} is not in the audit log yet")
        start = time.time()
        result = investigate(graph, event.txn_id, logger)
        out = to_event(event, result, model_name(model))
        bus.publish(r, out)
        summary = (f"{out.payload.verdict} -> {out.payload.recommended_action}"
                   if isinstance(out, InvestigationCompleted) else "ESCALATED to a human")
        log.info("txn %s  %s  tools %s  (%.1f s)", event.txn_id[:8], summary,
                 list(dict.fromkeys(result["tools_used"])), time.time() - start)

    source = bus.stream_for("transaction.decided")
    log.info("Listening on %s (group %s) with %s. Press Ctrl + C to stop.", source, GROUP, model_name(model))
    try:
        bus.consume(r, source, GROUP, handle, batch_size=1, reclaim_after_ms=SLOW_RECLAIM_MS,
                    on_give_up=lambda event, reason: give_up(r, event, reason))
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
