"""Settings from the environment, with the project's .env file as a fallback.

The .env file holds secrets (passwords, keys). It is never saved to Git.
"""

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """Copy KEY=VALUE lines from .env into the environment (existing values win)."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require(name: str) -> str:
    """Read a setting that must exist, with a clear message if it doesn't."""
    load_env()
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"Missing setting {name}. Add it to the .env file (see .env.example).")
    return value
