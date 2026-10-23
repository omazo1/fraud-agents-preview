"""The run report (Part 3 · Step 3): the headline numbers for one run.

Reads only what the system recorded (the audit log, the AI-call log, the dead-letter
stream) for the payments of one run, and checks them against the dataset's answer key
(is_fraud in fraudTest.csv). The answer key is used HERE ONLY: the agents never see it.

The run is the one the Monitor started (▶ Start replay or ↺ Reset counters), unless
--since gives another start time. Writes data/processed/run_report_<start>.md and prints it.

Run from the project folder:  python -m fraud_agents.services.metrics.report
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraud_agents.shared import bus, db

ANSWER_KEY_FILE = Path("data/raw/fraudTest.csv")
OUT_DIR = Path("data/processed")
RUN_KEY = "monitor:run"   # written by the Monitor when a run starts
FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)
# Part 1 · Step 4: the model alone on the whole test file (555,719 payments) at the flag line (0.5).
OFFLINE = {"caught": 1427, "frauds": 2145, "false_alarms": 397, "genuine": 553574}


def run_start(since: str | None) -> tuple[datetime, str, datetime | None]:
    """The run's start (UTC), its label, and when the Monitor saw it complete (if it did)."""
    if since:
        start = datetime.fromisoformat(since)
        if start.tzinfo is None:
            raise SystemExit("--since needs a time zone, e.g. 2026-10-22T23:00:00+00:00")
        return start.astimezone(timezone.utc), "given with --since", None
    run = bus.connect().hgetall(RUN_KEY)
    if not run:
        raise SystemExit("No run found: start one from the Monitor, or pass --since 2026-10-22T23:00:00+00:00")
    done = datetime.fromisoformat(run["completed_at"]) if run.get("completed_at") else None
    return datetime.fromisoformat(run["started_at"]), run.get("label", ""), done


def wilson(k: int, n: int) -> str:
    """The 95% range for a rate measured as k out of n (Wilson interval)."""
    if not n:
        return "n/a"
    z, ph = 1.96, k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{max(0, centre - half):.1%}–{min(1, centre + half):.1%}"


def load(start: datetime, until: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The run's payments: those that ARRIVED between start and until, with all their events."""
    in_run = ("SELECT txn_id FROM events WHERE event_type = 'transaction.received' "
              "AND published_at >= %s AND published_at < %s")
    with db.connect() as conn:
        events = pd.DataFrame(conn.execute(
            "SELECT txn_id, event_type, producer, published_at, decision, score, body FROM events "
            f"WHERE txn_id IN ({in_run}) ORDER BY published_at", (start, until)).fetchall(),
            columns=["txn_id", "type", "producer", "at", "result", "score", "body"])
        calls = pd.DataFrame(conn.execute(
            "SELECT txn_id, agent, model, latency_ms, input_tokens, output_tokens, ok FROM llm_calls "
            f"WHERE called_at >= %s AND txn_id IN ({in_run})", (start, start, until)).fetchall(),
            columns=["txn_id", "agent", "model", "ms", "tokens_in", "tokens_out", "ok"])
    events["body"] = [b if isinstance(b, dict) else json.loads(b) for b in events["body"]]
    return events, calls


def pct(part: float, whole: float) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def secs(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "n/a"
    return f"median {s.median():.1f} s · 95th percentile {s.quantile(0.95):.1f} s · max {s.max():.1f} s"


def build(start: datetime, label: str, completed: datetime | None = None,
          until: datetime = FAR_FUTURE) -> str:
    events, calls = load(start, until)
    if events.empty:
        raise SystemExit("No payments arrived in this run yet.")
    key_df = pd.read_csv(ANSWER_KEY_FILE, usecols=["trans_num", "is_fraud"])
    key = dict(zip(key_df["trans_num"], key_df["is_fraud"]))

    first = events.drop_duplicates(["txn_id", "type"], keep="first").set_index(["txn_id", "type"])
    latest = events.drop_duplicates(["txn_id", "type"], keep="last").set_index(["txn_id", "type"])

    kinds = set(events["type"])

    def col(kind: str, field: str = "at", which: pd.DataFrame = first) -> pd.Series:
        """One column of the first (or latest) event of a kind, per payment; empty if none."""
        if kind not in kinds:
            return pd.Series(dtype="datetime64[ns, UTC]" if field == "at" else object)
        values = which.xs(kind, level="type")[field]
        return pd.to_datetime(values, utc=True) if field == "at" else values

    def at(kind: str) -> pd.Series:
        return col(kind)

    pay = first.xs("transaction.received", level="type")
    p = pd.DataFrame(index=pay.index)
    p["amount"] = [float(b["payload"]["amount"]) for b in pay["body"]]
    p["category"] = [b["payload"]["category"] for b in pay["body"]]
    p["arrived"] = pay["at"]
    p["decision"] = col("transaction.decided", "result")
    p["score"] = col("transaction.decided", "score")
    p["decided_at"] = col("transaction.decided")
    p["final"] = col("transaction.resolved", "result", latest)   # the latest one counts (analyst)
    p["final_at"] = col("transaction.resolved")   # the system's own outcome (before any analyst)
    p["truth"] = [key.get(t) for t in p.index]
    unknown = int(p["truth"].isna().sum())
    p = p[p["truth"].notna()]
    p["fraud"] = p["truth"] == 1
    p["stopped"] = p["decision"].isin(["flag", "hold"])
    decided = p[p["decision"].notna()]

    fraud, genuine = decided[decided["fraud"]], decided[~decided["fraud"]]
    tp, fn = int(fraud["stopped"].sum()), int((~fraud["stopped"]).sum())
    fp, tn = int(genuine["stopped"].sum()), int((~genuine["stopped"]).sum())

    def final_count(df, outcome):
        return int((df["final"] == outcome).sum())

    last_event = completed or events["at"].max()   # the Monitor's frozen completion time, if any
    in_progress = int((decided["stopped"] & decided["final"].isna()).sum())
    off_recall = OFFLINE["caught"] / OFFLINE["frauds"]
    off_false = OFFLINE["false_alarms"] / OFFLINE["genuine"]
    all_frauds = int(p["fraud"].sum())

    fast = (decided["decided_at"] - decided["arrived"]).dt.total_seconds()
    stopped = decided[decided["stopped"]]
    slow = (stopped["final_at"] - stopped["arrived"]).dt.total_seconds()

    def gap(a: str, b: str) -> pd.Series:
        return (at(b) - at(a)).dt.total_seconds()

    def payloads(kind: str) -> list[dict]:
        return [b["payload"] for b in events.loc[events["type"] == kind, "body"]]   # every one sent

    def tally(items: list) -> dict:
        return pd.Series(items, dtype=object).value_counts().to_dict()

    sent = payloads("notification.sent")
    texts, templates = len(sent), sum(x["used_template"] for x in sent)
    reports = [b["payload"] for b in col("investigation.completed", "body")]
    verdicts = tally([x["verdict"] for x in reports])
    actions = tally([x["recommended_action"] for x in reports])
    finished = col("verification.completed", "body")
    outcomes = tally([b["payload"]["outcome"] for b in finished])
    replied = finished[[b["payload"]["outcome"] != "no_reply" for b in finished]]   # a timeout is no reply
    replies = set(replied.index)
    ai_read = len(replies & set(calls.loc[calls["agent"] == "verification", "txn_id"]))
    answered = at("verification.completed")[lambda x: x.index.isin(replies)] - at("verification.requested")

    r = bus.connect()
    start_ms = int(start.timestamp() * 1000)
    end_ms = "+" if until == FAR_FUTURE else f"{int(until.timestamp() * 1000)}-0"
    dead = len(r.xrange(bus.DEAD_LETTER_STREAM, min=f"{start_ms}-0", max=end_ms))

    lines = [
        f"# Run report: {start:%Y-%m-%d %H:%M:%S} UTC",
        "",
        f"- **Run:** {label or 'unlabelled'}; started {start:%H:%M:%S} UTC.",
        f"- **Payments:** {len(p) + unknown} arrived, {len(decided)} decided by the fast lane"
        + (f" ({unknown} not in the answer key, left out)" if unknown else "") + ".",
        f"- **Duration:** {(last_event - start).total_seconds() / 60:.1f} min from the start to the "
        + ("moment the Monitor saw the run complete" if completed else "last event")
        + f" ({last_event:%H:%M:%S} UTC)." + (f" **{in_progress} case(s) still in progress.**" if in_progress else ""),
        f"- **Set-aside events (dead letters) during the run:** {dead}.",
        "",
        "## 1. Catching fraud (the first decision: flag or hold = caught)",
        "",
        "| | Fraud (answer key) | Genuine (answer key) |",
        "| --- | --- | --- |",
        f"| Flagged or held | **{tp}** caught | **{fp}** false alarms |",
        f"| Approved | **{fn}** missed | {tn} correctly approved |",
        "",
        f"- **Catch rate (recall): {pct(tp, tp + fn)}** of frauds ({tp} of {tp + fn}; 95% range {wilson(tp, tp + fn)}).",
        f"- **False-alarm rate: {pct(fp, fp + tn)}** of genuine payments were stopped ({fp} of {fp + tn};"
        f" 95% range {wilson(fp, fp + tn)}).",
        f"- **Precision in this run: {pct(tp, tp + fp)}** of flagged or held payments were fraud. *Not comparable"
        f" with real traffic:* this run is {pct(tp + fn, len(decided))} fraud, real card traffic about 0.4%,"
        " and precision falls when fraud is rarer.",
        f"- **Same model, same line as Part 1.** The first decision is the XGBoost model alone at 0.5, so the"
        f" fair comparison is with Part 1's test on the whole test file: catch rate {off_recall:.1%}"
        f" ({OFFLINE['caught']} of {OFFLINE['frauds']}), false-alarm rate {off_false:.2%}"
        f" ({OFFLINE['false_alarms']} of {OFFLINE['genuine']:,}). A difference inside the 95% range above is"
        " sampling noise, not a change in the model.",
        f"- By decision: {int((fraud['decision'] == 'hold').sum())} frauds held, {int((fraud['decision'] == 'flag').sum())} flagged;"
        f" {int((genuine['decision'] == 'hold').sum())} genuine held, {int((genuine['decision'] == 'flag').sum())} flagged.",
        "",
        "## 2. Final outcomes (after the AI agents, the customer and any analyst)",
        "",
        "| Final outcome | Fraud | Genuine |",
        "| --- | --- | --- |",
        f"| Blocked | **{final_count(fraud, 'blocked')}** | {final_count(genuine, 'blocked')} |",
        f"| Released | {final_count(fraud, 'released')} | {final_count(genuine, 'released')} |",
        f"| Sent to an analyst (waiting) | {final_count(fraud, 'sent_to_analyst')} | {final_count(genuine, 'sent_to_analyst')} |",
        f"| No final outcome yet | {int((fraud['stopped'] & fraud['final'].isna()).sum())} | {int((genuine['stopped'] & genuine['final'].isna()).sum())} |",
        "",
        f"- **Frauds blocked in the end: {final_count(fraud, 'blocked')} of {all_frauds}** ({pct(final_count(fraud, 'blocked'), all_frauds)} of all frauds in the run).",
        f"- Genuine payments blocked by mistake: {final_count(genuine, 'blocked')}.",
        "",
        "## 3. Speed",
        "",
        f"- **Fast lane** (payment arrived → first decision): {secs(fast)}.",
        f"- **Slow lane** (payment arrived → the system's outcome, flagged or held only; an analyst's later"
        f" decision is human time and not counted): {secs(slow)}.",
        f"  - Investigation (decision → AI report): {secs(gap('transaction.decided', 'investigation.completed'))}.",
        f"  - Customer (text sent → reply read; customers who replied): {secs(answered.dt.total_seconds())}.",
        f"  - Notification (final outcome → customer told): {secs(gap('transaction.resolved', 'notification.sent'))}.",
        "",
        "## 4. The AI's part",
        "",
    ]
    if calls.empty:
        lines.append("- No AI calls were recorded for this run.")
    else:
        g = calls.groupby("agent").agg(calls=("ms", "size"), failed=("ok", lambda s: int((~s).sum())),
                                       median_s=("ms", lambda s: s.median() / 1000),
                                       tokens_in=("tokens_in", "sum"), tokens_out=("tokens_out", "sum"))
        lines += ["| Agent | AI calls | Failed | Median time per call | Tokens in | Tokens out |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in g.itertuples():
            lines.append(f"| {row.Index} | {int(row.calls)} | {int(row.failed)} | {row.median_s:.1f} s | "
                         f"{int(row.tokens_in or 0):,} | {int(row.tokens_out or 0):,} |")
        lines += ["", f"- Models: {', '.join(sorted(calls['model'].dropna().unique()))}."]
    lines += [
        f"- **Investigation reports:** {len(reports)}; verdicts {verdicts or 'none'}; suggested {actions or 'none'}.",
        f"- **Customer checks:** {len(finished)} finished; outcomes {outcomes or 'none'}."
        f" Of the {len(replies)} that got a reply: read by rules {len(replies) - ai_read}, sent to the AI {ai_read}.",
        f"- **Texts telling the outcome:** {texts} (an analyst case gets two: 'reviewing', then the decision);"
        f" the AI's draft passed the rules {texts - templates}, the safe template was used {templates}.",
        "",
        "## 5. Frauds the system missed",
        "",
    ]
    missed = fraud[~fraud["stopped"]].sort_values("amount", ascending=False)
    if missed.empty:
        lines.append("None.")
    else:
        lines += ["| txn | Amount | Shop type | Score |", "| --- | --- | --- | --- |"]
        lines += [f"| {t[:8]} | ${row.amount:,.2f} | {row.category} | {row.score:.3f} |" for t, row in missed.iterrows()]
    lines += ["", "*The answer key is used only in this report and the Monitor's testing pages; the agents never see it.*"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Headline numbers for one run.")
    parser.add_argument("--since", help="start time (ISO, with time zone); default: the Monitor's current run")
    parser.add_argument("--until", help="end time (ISO, with time zone): only payments that arrived before it")
    args = parser.parse_args()
    start, label, completed = run_start(args.since)
    until = FAR_FUTURE
    if args.until:
        until = datetime.fromisoformat(args.until)
        if until.tzinfo is None:
            raise SystemExit("--until needs a time zone, e.g. 2026-10-22T23:30:00+00:00")
        until = until.astimezone(timezone.utc)
    report = build(start, label + (f" (payments that arrived before {until:%H:%M:%S} UTC)" if args.until else ""),
                   completed, until)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"run_report_{start:%Y%m%d_%H%M%S}.md"
    out.write_text(report)
    print(report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
