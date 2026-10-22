"""Card tokens: a stand-in for the card number, so the real number never travels.

Keyed with a secret (HMAC-SHA256 with CARD_TOKEN_KEY), so nobody can work back to
the card number by trying every possible number. The gateway and the history
loader use the same function, so the same card always gets the same token.
"""

import hashlib
import hmac

from fraud_agents.shared import config


def card_token(cc_num: str, key: str | None = None) -> str:
    key = key or config.require("CARD_TOKEN_KEY")
    digest = hmac.new(key.encode(), str(cc_num).encode(), hashlib.sha256).hexdigest()
    return "card-" + digest[:16]
