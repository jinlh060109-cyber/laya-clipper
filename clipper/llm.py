from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

PRESETS = {
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.cn/v1",
}

SYSTEM = (
    "You write metadata for short video clips. Reply with JSON only, with keys "
    "title, hook and description. The title is under 60 characters and is not "
    "clickbait. The hook is one sentence that could be the first line on screen. "
    "The description is two sentences."
)


@dataclass(frozen=True)
class WriterConfig:
    base_url: str
    api_key: str
    model: str


def writer_from_env() -> WriterConfig | None:
    base = os.environ.get("WRITER_BASE_URL")
    model = os.environ.get("WRITER_MODEL")
    if not base or not model:
        return None
    key = os.environ.get("WRITER_API_KEY")
    if not key:
        raise ValueError("WRITER_BASE_URL is set but WRITER_API_KEY is missing.")
    return WriterConfig(base_url=PRESETS.get(base, base), api_key=key, model=model)


def build_payload(config: WriterConfig, system: str, user: str) -> dict:
    return {
        "model": config.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }


def parse_metadata(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(text)
    for field in ("title", "hook", "description"):
        if field not in data:
            raise ValueError(f"Writer response is missing '{field}': {text[:200]}")
    return {k: data[k] for k in ("title", "hook", "description")}


def write_metadata(config: WriterConfig, transcript_text: str, clip_format: str) -> dict:
    user = f"Clip format: {clip_format}\n\nTranscript:\n{transcript_text}"
    response = httpx.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}"},
        json=build_payload(config, SYSTEM, user),
        timeout=60.0,
    )
    response.raise_for_status()
    return parse_metadata(response.json()["choices"][0]["message"]["content"])
