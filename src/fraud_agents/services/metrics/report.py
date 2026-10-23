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
BANK_PAYMENTS_PER_DAY = 1_000_000   # an illustration of "real volume" for section 1b
REAL_RATES = (0.001, 0.004, 0.005)  # the range of real card-fraud rates given in feedback #3


def offline_test() -> dict:
    """The live model's own test on the WHOLE test file (the real mix, 0.39% fraud) at the
    Decision agent's flag line, from the model's details file. These are the numbers that
    hold at the real fraud rate; a short replay's numbers are only a sample."""
    from fraud_agents.agents.decision.agent import load_rules
    from fraud_agents.agents.risk_scoring.agent import META_FILE
    meta = json.loads(META_FILE.read_text())
    flag_at = load_rules()["flag_at"]
    line = meta["test"]["at_threshold"][str(flag_at)]
    frauds = meta["test"]["fraud_rows"]
    return {"model": meta["model_version"], "flag_at": flag_at, "caught": line["caught"],
            "frauds": frauds, "false_alarms": line["false_alarms"],
            "genuine": meta["test"]["rows"] - frauds}


def precision_at(rate: float, recall: float, false_alarm_rate: float) -> float:
    """Of the payments stopped, the share that is fraud, if fraud is `rate` of all traffic."""
    return recall * rate / (recall * rate + false_alarm_rate * (1 - rate))


def real_rate_section(off: dict, run_fraud: int, run_total: int, test_days: float) -> list[str]:
    recall = off["caught"] / off["frauds"]
    far = off["false_alarms"] / off["genuine"]
    per_million = far * 1_000_000
    lo, hi = wilson_bounds(off["false_alarms"], off["genuine"])
    test_rate = off["frauds"] / (off["frauds"] + off["genuine"])
    per_day = (off["frauds"] + off["genuine"]) / test_days
    bank_frauds = BANK_PAYMENTS_PER_DAY * 0.004
    return [
        "## 1b. At the real fraud rate",
        "",
        f"- This run was **{pct(run_fraud, run_total)} fraud**. Real card traffic is about 0.1–0.5% fraud; this"
        f" dataset's test file is {test_rate:.2%} ({off['frauds']:,} of {off['frauds'] + off['genuine']:,}).",
        f"- The numbers that hold at the real rate come from the model's test on the **whole test file**"
        f" ({off['model']}, flag line {off['flag_at']}):",
        "",
        "| Catch rate | False alarms per million genuine payments | Precision at the test file's rate |",
        "| --- | --- | --- |",
        f"| {recall:.1%} ({off['caught']:,} of {off['frauds']:,}) | **{per_million:,.0f}** (95% range"
        f" {lo * 1e6:,.0f}–{hi * 1e6:,.0f}) | {precision_at(test_rate, recall, far):.1%} |",
        "",
        "- **Precision falls as fraud gets rarer:** "
        + "; ".join(f"at {r:.1%} fraud, {precision_at(r, recall, far):.0%} of stopped payments are fraud"
                    for r in REAL_RATES) + ".",
        f"- **At real volume:** a bank with {BANK_PAYMENTS_PER_DAY:,} card payments a day (0.4% fraud) would stop"
        f" about **{far * BANK_PAYMENTS_PER_DAY * (1 - 0.004):,.0f} genuine payments a day by mistake**, each one a soft"
        f" decline and a \"was this you?\" text, while catching about {recall * bank_frauds:,.0f} of its"
        f" {bank_frauds:,.0f} frauds. (This dataset's bank has about {per_day:,.0f} payments a day:"
        f" about {far * per_day:.1f} false alarms a day.)",
        "",
    ]


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


def wilson_bounds(k: int, n: int) -> tuple[float, float]:
    """The 95% range for a rate measured as k out of n (Wilson interval)."""
    z, ph = 1.96, k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson_bounds(k, n)
    return f"{lo:.1%}–{hi:.1%}"


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
    key_df = pd.read_csv(ANSWER_KEY_FILE, usecols=["trans_num", "is_fraud", "trans_date_trans_time"])
    key = dict(zip(key_df["trans_num"], key_df["is_fraud"]))
    span = pd.to_datetime(key_df["trans_date_trans_time"])
    test_days = (span.max() - span.min()).total_seconds() / 86400
    offline = offline_test()

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
    off_recall = offline["caught"] / offline["frauds"]
    off_false = offline["false_alarms"] / offline["genuine"]
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
        f"- **Precision in this run: {pct(tp, tp + fp)}** of flagged or held payments were fraud. "
        + (f"This run used about the real fraud rate ({pct(tp + fn, len(decided))}), but with only {tp + fn}"
           " frauds it is a small sample; section 1b has the numbers that hold at scale."
           if len(decided) and (tp + fn) / len(decided) < 0.01 else
           f"*Not comparable with real traffic:* this run is {pct(tp + fn, len(decided))} fraud, real card"
           " traffic about 0.1–0.5%, and precision falls when fraud is rarer (see section 1b)."),
        f"- **Same model, same line as its offline test.** The first decision is the XGBoost model alone at"
        f" {offline['flag_at']}, so the fair comparison is with its test on the whole test file: catch rate {off_recall:.1%}"
        f" ({offline['caught']} of {offline['frauds']}), false-alarm rate {off_false:.2%}"
        f" ({offline['false_alarms']} of {offline['genuine']:,}). A difference inside the 95% range above is"
        " sampling noise, not a change in the model.",
        f"- By decision: {int((fraud['decision'] == 'hold').sum())} frauds held, {int((fraud['decision'] == 'flag').sum())} flagged;"
        f" {int((genuine['decision'] == 'hold').sum())} genuine held, {int((genuine['decision'] == 'flag').sum())} flagged.",
        "",
        *real_rate_section(offline, tp + fn, len(decided), test_days),
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
