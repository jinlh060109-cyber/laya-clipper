"""Change values in the local .env file without disturbing anything else.

The web page's AI settings are saved here, next to the keys the user typed
in by hand. Comments, blank lines and unrelated values stay as they were.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(".env")


def update(values: dict[str, str | None], path: Path = ENV_PATH) -> None:
    """Set each key to its value (None removes it), in the file and in this process."""
    for key, value in values.items():
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError(f"{key} cannot contain a line break.")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in pending:
            value = pending.pop(key)
            if value is not None:
                out.append(f"{key}={value}")
            continue
        out.append(line)
    out += [f"{key}={value}" for key, value in pending.items() if value is not None]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    tmp.replace(path)
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
