"""The event bus: publish forms to Redis Streams and read them back safely.

One stream per event type, e.g. "events:transaction.received".
Every agent reads its input stream as a member of a consumer group, so:
- each event is handled by one copy of the agent;
- an event is only marked done (XACK) after it has been handled;
- an event left unfinished by a crash is picked up again later;
- an event that keeps failing goes to the "events:dead" stream instead of blocking the line;
- an event seen before is skipped (idempotency by event_id).
"""

import logging
import os
import socket
from collections.abc import Callable

import redis
from pydantic import ValidationError

from fraud_agents.shared.events import parse_event

log = logging.getLogger(__name__)

DEAD_LETTER_STREAM = "events:dead"
MAX_DELIVERIES = 3            # tries before an event goes to the dead-letter stream
RECLAIM_AFTER_MS = 60_000     # an event unfinished for 60 s is picked up again
SEEN_TTL_SECONDS = 7 * 24 * 3600


def stream_for(event_type: str) -> str:
    return f"events:{event_type}"


def connect() -> redis.Redis:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    # socket_timeout must be longer than the 5 s we wait for new events in consume().
    return redis.Redis.from_url(url, decode_responses=True, socket_timeout=30)


def publish(r: redis.Redis, event) -> str:
    """Add one form to the end of its stream. Returns the Redis entry ID."""
    return r.xadd(stream_for(event.event_type), {"event": event.model_dump_json()})


def ensure_group(r: redis.Redis, stream: str, group: str) -> None:
    """Create the consumer group (and the stream) if they don't exist yet."""
    try:
        r.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as error:
        if "BUSYGROUP" not in str(error):  # BUSYGROUP = it already exists
            raise


def _dead_letter(r, stream, group, entry_id, fields, reason):
    r.xadd(DEAD_LETTER_STREAM, {
        "from_stream": stream, "group": group, "entry_id": entry_id,
        "reason": reason[:500], "event": fields.get("event", ""),
    })
    r.xack(stream, group, entry_id)
    log.warning("Sent %s to %s: %s", entry_id, DEAD_LETTER_STREAM, reason[:200])


def _handle(r, stream, group, entry_id, fields, handler) -> None:
    try:
        event = parse_event(fields["event"])
    except (KeyError, ValidationError) as error:
        _dead_letter(r, stream, group, entry_id, fields, f"bad form: {error}")
        return

    seen_key = f"seen:{group}:{event.event_id}"
    if r.exists(seen_key):
        r.xack(stream, group, entry_id)  # already handled before: skip it
        log.info("Skipped duplicate %s", event.event_id)
        return

    handler(event)
    r.set(seen_key, 1, ex=SEEN_TTL_SECONDS)
    r.xack(stream, group, entry_id)


def consume(
    r: redis.Redis,
    stream: str,
    group: str,
    handler: Callable,
    consumer: str | None = None,
    block_ms: int = 5000,
    stop_after: int | None = None,
) -> int:
    """Read events from `stream` forever and pass each one to `handler`.

    `stop_after` (for tests) stops after that many events. Returns how many were handled.
    """
    consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
    ensure_group(r, stream, group)
    handled = 0
    while stop_after is None or handled < stop_after:
        try:
            # 1. Pick up events another copy left unfinished (e.g. it crashed).
            _, claimed, _ = r.xautoclaim(stream, group, consumer, RECLAIM_AFTER_MS, "0-0", count=10)
            # 2. Then read new events, waiting up to block_ms for one to arrive.
            batch = claimed
            if not batch:
                reply = r.xreadgroup(group, consumer, {stream: ">"}, count=10, block=block_ms)
                batch = reply[0][1] if reply else []
        except redis.ResponseError as error:
            if "NOGROUP" not in str(error) and "UNBLOCKED" not in str(error):
                raise
            ensure_group(r, stream, group)  # the stream was deleted (e.g. Redis was reset)
            continue

        for entry_id, fields in batch:
            info = r.xpending_range(stream, group, entry_id, entry_id, 1)
            tries = info[0]["times_delivered"] if info else 1
            if tries > MAX_DELIVERIES:
                # It keeps coming back (e.g. it crashed the whole agent): stop retrying it.
                _dead_letter(r, stream, group, entry_id, fields, f"delivered {tries} times")
                continue
            try:
                _handle(r, stream, group, entry_id, fields, handler)
                handled += 1
            except Exception as error:  # left un-acknowledged, so it will be retried
                log.exception("Failed on %s (try %d)", entry_id, tries)
                if tries >= MAX_DELIVERIES:
                    _dead_letter(r, stream, group, entry_id, fields, f"failed {tries} times: {error}")
            if stop_after is not None and handled >= stop_after:
                break
    return handled
