"""The language models: the "brains" of the AI agents (Part 2 · Step 1).

chat_model() gives a LangChain chat model:
- "ollama" (default): Llama 3.1 8B running on this Mac through Ollama. Free and private.
- "anthropic": Claude through the API (optional; needs ANTHROPIC_API_KEY in .env).
Choose with LLM_PROVIDER in .env. Agent code never changes: no lock-in to one provider.

LLMCallLogger records every call (who, which payment, which model, how long,
how many tokens, success or error) in the llm_calls table: the "llm.called" record.

Self-check (Ollama running, model pulled):  python -m fraud_agents.shared.llm
"""

import logging
import os
import threading
import time
from typing import Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from fraud_agents.shared import config, db

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"  # Anthropic's low-cost tier; change in .env (CLAUDE_MODEL)

log = logging.getLogger("llm")

INSERT_CALL = """
INSERT INTO llm_calls (agent, txn_id, provider, model, latency_ms, input_tokens,
                       output_tokens, ok, error)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def chat_model(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """The chat model chosen in .env (LLM_PROVIDER), with temperature 0 for steady answers."""
    config.load_env()
    provider = provider or os.environ.get("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            temperature=0,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        config.require("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model or os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            temperature=0,
            max_tokens=1024,
        )
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}: use 'ollama' or 'anthropic'")


class LLMCallLogger(BaseCallbackHandler):
    """Writes one llm_calls row per model call. Never stops an agent if the database is down."""

    def __init__(self):
        self.started: dict[UUID, tuple[float, dict]] = {}
        self.conn = None
        self.lock = threading.Lock()  # safe if several threads share one logger

    def _write(self, row: tuple) -> None:
        with self.lock:
            try:
                if self.conn is None or self.conn.closed:
                    self.conn = db.connect()
                    db.create_schema(self.conn)
                self.conn.execute(INSERT_CALL, row)
            except Exception as error:  # the record is useful, but not worth crashing an agent for
                if self.conn is not None:
                    self.conn.close()
                self.conn = None
                log.warning("Could not record the LLM call: %s", error)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self.started[run_id] = (time.perf_counter(), metadata or {})

    def _finish(self, run_id, ok: bool, error: str | None, usage: dict | None) -> None:
        start, meta = self.started.pop(run_id, (time.perf_counter(), {}))
        usage = usage or {}
        self._write((
            meta.get("agent"), meta.get("txn_id"),
            meta.get("ls_provider"), meta.get("ls_model_name"),
            int((time.perf_counter() - start) * 1000),
            usage.get("input_tokens"), usage.get("output_tokens"),
            ok, error,
        ))

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = None
        try:
            usage = response.generations[0][0].message.usage_metadata
        except (AttributeError, IndexError):
            pass
        self._finish(run_id, True, None, usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, False, str(error)[:500], None)


def run_config(agent: str, txn_id: str | None, logger: LLMCallLogger) -> dict:
    """The extra settings passed with each call, so the record knows who asked and why."""
    return {"callbacks": [logger], "metadata": {"agent": agent, "txn_id": txn_id}}


# ---------- Self-check: python -m fraud_agents.shared.llm ----------

class Opinion(BaseModel):
    """A tiny fixed-shape answer, to test structured output."""

    verdict: Literal["likely_fraud", "likely_genuine", "unsure"]
    reason: str = Field(description="One short sentence")


def _self_check() -> None:
    from langchain_core.tools import tool

    @tool
    def card_history(card_id: str) -> str:
        """Look up how many payments a card made in the last 24 hours and their total amount."""
        return f"{card_id}: 14 payments in 24 hours, total $3,912.40"

    model = chat_model()
    logger = LLMCallLogger()
    cfg = run_config("self-check", "selfcheck-txn", logger)
    print(f"Model: {getattr(model, 'model', None) or getattr(model, 'model_name', None)}\n")

    t = time.perf_counter()
    answer = model.invoke("In one short sentence: why do banks hold suspicious payments?", config=cfg)
    print(f"1. Plain answer ({time.perf_counter() - t:.1f} s):\n   {answer.content}\n")

    t = time.perf_counter()
    reply = model.bind_tools([card_history]).invoke(
        "A payment on card card-7f3a looks odd. Check the card's recent history.", config=cfg
    )
    calls = [(c["name"], c["args"]) for c in reply.tool_calls]
    print(f"2. Tool calling ({time.perf_counter() - t:.1f} s): the model asked for {calls}")
    print("   OK: it chose the right tool." if calls and calls[0][0] == "card_history"
          else "   PROBLEM: it did not call the tool.")
    print()

    t = time.perf_counter()
    opinion = model.with_structured_output(Opinion).invoke(
        "A card made 14 payments totalling $3,912 in 24 hours, mostly online shopping at night, "
        "far from home. Give your opinion.", config=cfg
    )
    print(f"3. Structured answer ({time.perf_counter() - t:.1f} s): {opinion!r}\n")

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT agent, model, latency_ms, input_tokens, output_tokens, ok FROM llm_calls "
            "WHERE txn_id = 'selfcheck-txn' ORDER BY id DESC LIMIT 3"
        ).fetchall()
    print("4. Recorded in llm_calls (newest first):")
    for row in rows:
        print("  ", row)


if __name__ == "__main__":
    _self_check()
