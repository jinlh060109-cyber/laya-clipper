"""Skills built into the app: guidance written for the AI, sent with its
prompts. Each is a Markdown file with YAML front matter; the body is what
the AI reads."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent


@lru_cache(maxsize=None)
def body(name: str) -> str:
    """The skill's text without its front matter."""
    text = (HERE / f"{name}.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()
