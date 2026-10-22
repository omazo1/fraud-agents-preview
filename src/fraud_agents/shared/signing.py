"""Webhook signatures: the wax seal on each payment message.

The sender and the gateway share one secret. The sender computes
HMAC-SHA256(secret, "<timestamp>.<body>") and puts it in the X-Signature header.
The gateway recomputes it. If the two don't match, the message was faked or changed.
The timestamp stops an old, copied message from being sent again later.
"""

import hashlib
import hmac
import time

MAX_AGE_SECONDS = 300  # messages older than 5 minutes are refused


def sign(secret: str, timestamp: str, body: bytes) -> str:
    message = timestamp.encode() + b"." + body
    return "sha256=" + hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp: str | None, signature: str | None, body: bytes) -> bool:
    if not timestamp or not signature:
        return False
    # Refuse odd-looking headers before doing any maths with them.
    if not (timestamp.isascii() and timestamp.isdigit() and len(timestamp) <= 12):
        return False
    if not signature.isascii():
        return False
    age = abs(time.time() - int(timestamp))
    if age > MAX_AGE_SECONDS:
        return False
    # compare_digest takes the same time whether or not the strings match (safer).
    return hmac.compare_digest(sign(secret, timestamp, body), signature)
