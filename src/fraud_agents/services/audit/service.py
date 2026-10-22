"""The audit log (Part 1 · Step 8): copies every event into PostgreSQL.

It listens to all three streams as its own consumer group ("audit"), so it gets
its own copy of every event without taking anything away from the agents.
Writing the same event twice is harmless: the event_id is the table's key.

Run from the project folder:  python -m fraud_agents.services.audit.service
Stop it with Ctrl + C.
"""

import logging
import os
import socket
import threading
import time

import psycopg

from fraud_agents.shared import bus, db

GROUP = "audit"
EVENT_TYPES = ["transaction.received", "risk.scored", "transaction.decided"]

INSERT = """
INSERT INTO events (event_id, event_type, txn_id, producer, occurred_at, published_at,
                    score, decision, body)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO NOTHING
"""

log = logging.getLogger("audit")


class Recorder:
    """Writes events to the database. If PostgreSQL is down, it waits and retries
    until it is back, so no event is ever skipped (the event stays unacknowledged
    in Redis meanwhile)."""

    def __init__(self):
        self.conn = None

    def record(self, event) -> None:
        payload = event.payload
        row = (
            event.event_id, event.event_type, event.txn_id, event.producer,
            event.occurred_at, event.published_at,
            getattr(payload, "score", None), getattr(payload, "decision", None),
            event.model_dump_json(),
        )
        wait = 1
        while True:
            try:
                if self.conn is None or self.conn.closed:
                    self.conn = db.connect()
                self.conn.execute(INSERT, row)
                break
            except psycopg.OperationalError as error:
                log.warning("PostgreSQL unavailable (%s); retrying in %d s", error, wait)
                self.conn = None
                time.sleep(wait)
                wait = min(wait * 2, 30)
        log.info("recorded %-20s txn %s", event.event_type, event.txn_id[:8])


def watch(event_type: str) -> None:
    """Follow one stream forever (runs in its own thread)."""
    recorder = Recorder()
    r = bus.connect()
    consumer = f"{socket.gethostname()}-{os.getpid()}-{event_type}"
    bus.consume(r, bus.stream_for(event_type), GROUP, recorder.record, consumer=consumer)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with db.connect() as conn:
        db.create_schema(conn)
    threads = [threading.Thread(target=watch, args=(t,), daemon=True, name=t) for t in EVENT_TYPES]
    for thread in threads:
        thread.start()
    log.info("Recording %s into PostgreSQL. Press Ctrl + C to stop.", ", ".join(EVENT_TYPES))
    try:
        while all(thread.is_alive() for thread in threads):
            time.sleep(1)
        log.error("A recording thread stopped; see the error above.")
        raise SystemExit(1)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
