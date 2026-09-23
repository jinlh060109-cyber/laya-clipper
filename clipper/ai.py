"""The AI that reads transcripts: Claude by default, or any OpenAI-compatible
server (DeepSeek, Kimi, OpenRouter, OpenAI, a local Ollama).

Configured from the environment (.env):
  ANTHROPIC_API_KEY          alone selects Claude (claude-opus-5)
  AI_PROVIDER                anthropic | openai | deepseek | kimi | openrouter | ollama | custom
  AI_MODEL                   model name (required except for Claude)
  AI_API_KEY                 key for the OpenAI-compatible providers (not needed for ollama)
  AI_BASE_URL                endpoint for `custom`, or to override a preset
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

CLAUDE_DEFAULT_MODEL = "claude-opus-5"
OPENAI_COMPATIBLE = {
    "openai": ("OpenAI", "https://api.openai.com/v1"),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1"),
    "kimi": ("Kimi (Moonshot)", "https://api.moonshot.cn/v1"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1"),
    "ollama": ("Ollama (local, free)", "http://localhost:11434/v1"),
    "custom": ("Custom OpenAI-compatible server", None),
}
PROVIDERS = ("anthropic", *OPENAI_COMPATIBLE)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
OPENAI_MAX_TOKENS = 8192


class AIError(RuntimeError):
    """The AI call failed or returned something unusable."""


@dataclass(frozen=True)
class AIConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model}


def config_from_env() -> AIConfig | None:
    provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    model = (os.environ.get("AI_MODEL") or "").strip()
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            return None
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown AI_PROVIDER {provider!r}. Valid: {', '.join(PROVIDERS)}.")

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("AI_PROVIDER is anthropic but ANTHROPIC_API_KEY is not set.")
        return AIConfig("anthropic", model or CLAUDE_DEFAULT_MODEL, key, None)

    if not model:
        raise ValueError(f"AI_PROVIDER is {provider}; set AI_MODEL to the model to use.")
    base = os.environ.get("AI_BASE_URL") or OPENAI_COMPATIBLE[provider][1]
    if not base:
        raise ValueError("AI_PROVIDER is custom; set AI_BASE_URL to the server's /v1 URL.")
    key = os.environ.get("AI_API_KEY") or None
    if key is None and provider != "ollama" and provider != "custom":
        raise ValueError(f"AI_PROVIDER is {provider}; set AI_API_KEY.")
    return AIConfig(provider, model, key, base.rstrip("/"))


def available_providers() -> list[dict]:
    """Every provider, with the one the environment selects marked active."""
    try:
        active = config_from_env()
    except ValueError:
        active = None
    labels = {"anthropic": "Claude (Anthropic)",
              **{pid: label for pid, (label, _) in OPENAI_COMPATIBLE.items()}}
    return [{"id": pid, "label": labels[pid],
             "active": bool(active and active.provider == pid),
             "model": active.model if active and active.provider == pid else None}
            for pid in PROVIDERS]


def parse_json(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIError(f"The AI did not answer with valid JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise AIError("The AI answered with JSON that is not an object.")
    return data


def _anthropic_client(config: AIConfig):
    import anthropic

    return anthropic.Anthropic(api_key=config.api_key)


def _claude(config: AIConfig, system: str, user: str, schema: dict,
            max_tokens: int) -> dict:
    client = _anthropic_client(config)
    # Streaming: a whole-video transcript is a long input, and a long answer
    # would otherwise risk the HTTP timeout.
    with client.beta.messages.stream(
        model=config.model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        reason = getattr(details, "explanation", None) or "no reason given"
        raise AIError(f"Claude declined this request ({reason}).")
    if message.stop_reason == "max_tokens":
        raise AIError("The AI's answer was too long and got cut off.")
    text = next((b.text for b in message.content if b.type == "text"), "")
    return parse_json(text)


def _claude_failure(exc: Exception) -> str:
    """A message that says what to do, most specific error first."""
    try:
        import anthropic
    except ImportError:
        return "The anthropic package is not installed: pip install anthropic"
    if isinstance(exc, anthropic.AuthenticationError):
        return "Claude rejected the API key. Check ANTHROPIC_API_KEY in .env."
    if isinstance(exc, anthropic.NotFoundError):
        return "Claude does not know that model. Check AI_MODEL in .env."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude is rate-limiting this key. Wait a minute and start again."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude returned an error ({exc.status_code}): {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach Claude. Check the internet connection."
    return f"The Claude request failed: {exc}"


def _openai_compatible(config: AIConfig, system: str, user: str, schema: dict,
                       max_tokens: int) -> dict:
    # Not every OpenAI-compatible server enforces a JSON schema, so the schema
    # is also spelled out in the instructions and the answer is validated.
    instructions = (f"{system}\n\nReply with one JSON object only, matching this "
                    f"JSON schema exactly:\n{json.dumps(schema)}")
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        response = httpx.post(
            f"{config.base_url}/chat/completions",
            headers=headers,
            json={"model": config.model,
                  "messages": [{"role": "system", "content": instructions},
                               {"role": "user", "content": user}],
                  "response_format": {"type": "json_object"},
                  # Claude takes 64000; most OpenAI-compatible servers cap at 8192.
                  "max_tokens": min(max_tokens, OPENAI_MAX_TOKENS)},
            timeout=600.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        # The OS's own text for this is often localized and garbled on Windows.
        raise AIError(f"Could not reach the {config.provider} server at {config.base_url}. "
                      f"Is it running, and is AI_BASE_URL right?") from exc
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise AIError(f"The {config.provider} request failed: {exc}") from exc
    return parse_json(content)


def complete_json(config: AIConfig, system: str, user: str, schema: dict,
                  max_tokens: int) -> dict:
    """One request, answered as a JSON object that follows `schema`."""
    if config.provider == "anthropic":
        try:
            return _claude(config, system, user, schema, max_tokens)
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK errors surface as one AIError
            raise AIError(_claude_failure(exc)) from exc
    return _openai_compatible(config, system, user, schema, max_tokens)
