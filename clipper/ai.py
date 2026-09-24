"""The AI that reads transcripts: Claude by default, or any OpenAI-compatible
server (Alibaba Cloud Token Plan, Model Studio, DeepSeek, Kimi, OpenRouter,
OpenAI, a local Ollama, or your own).

Configured from the environment (.env), which the web page's AI panel writes:
  AI_PROVIDER      one of PROVIDERS (default: anthropic when ANTHROPIC_API_KEY is set)
  AI_MODEL         the model; each provider has a default where one is known
  <provider key>   each provider's own key, e.g. ANTHROPIC_API_KEY,
                   ALIBABA_TOKEN_PLAN_API_KEY (AI_API_KEY also works, for any)
  AI_BASE_URL      the endpoint for `custom`, or to override a preset
  AI_THINKING      on: let models that reason before answering do so (slower;
                   off by default where the provider allows switching it off)
  AI_FALLBACK_MODEL  tried once when the model is rate-limited, overloaded or
                   times out (default: the provider's fast model, if it has one)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

CLAUDE_DEFAULT_MODEL = "claude-opus-5"
# kind "anthropic" goes through the Claude SDK; "openai" through /chat/completions.
PROVIDERS: dict[str, dict] = {
    "anthropic": {"label": "Claude (Anthropic)", "kind": "anthropic", "base_url": None,
                  "key_env": "ANTHROPIC_API_KEY", "default_model": CLAUDE_DEFAULT_MODEL,
                  "models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
                  "key_hint": "Starts with sk-ant-"},
    "alibaba_token_plan": {
        "label": "Alibaba Cloud Token Plan", "kind": "openai",
        "base_url": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "key_env": "ALIBABA_TOKEN_PLAN_API_KEY", "default_model": "qwen3.8-max",
        "models": ["qwen3.8-max", "qwen3.7-max", "qwen3.7-plus", "qwen3.8-flash", "qwen3.6-flash",
                   "deepseek-v4-pro", "deepseek-v4.1-flash", "glm-5.3", "glm-5.2"],
        "key_hint": "Token Plan key, starts with sk-sp- (Singapore region)",
        "no_thinking": {"enable_thinking": False}, "fallback_model": "qwen3.8-flash"},
    "alibaba": {"label": "Alibaba Cloud Model Studio (pay-as-you-go)", "kind": "openai",
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "key_env": "DASHSCOPE_API_KEY", "default_model": "qwen-plus",
                "models": ["qwen-plus", "qwen-max", "qwen-flash"],
                "key_hint": "Model Studio key (not a Token Plan key)",
                "no_thinking": {"enable_thinking": False}, "fallback_model": "qwen-flash"},
    "deepseek": {"label": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com/v1",
                 "key_env": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat",
                 "models": ["deepseek-chat"], "key_hint": ""},
    "kimi": {"label": "Kimi (Moonshot)", "kind": "openai", "base_url": "https://api.moonshot.cn/v1",
             "key_env": "MOONSHOT_API_KEY", "default_model": "", "models": [], "key_hint": ""},
    "openrouter": {"label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY", "default_model": "", "models": [],
                   "key_hint": ""},
    "openai": {"label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
               "key_env": "OPENAI_API_KEY", "default_model": "", "models": [], "key_hint": ""},
    "ollama": {"label": "Ollama (local, free)", "kind": "openai",
               "base_url": "http://localhost:11434/v1", "key_env": None,
               "default_model": "", "models": [], "key_hint": "No key needed"},
    "custom": {"label": "Custom OpenAI-compatible server", "kind": "openai", "base_url": None,
               "key_env": "AI_API_KEY", "default_model": "", "models": [],
               "key_hint": "Only if the server needs one"},
}
# AI_PROVIDER=none turns the AI off even when a key is present.
NO_AI = "none"
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


def _key(provider: str) -> str | None:
    env = PROVIDERS[provider]["key_env"]
    return (os.environ.get(env) if env else None) or os.environ.get("AI_API_KEY") or None


def config_for(provider: str, model: str | None = None) -> AIConfig:
    """The settings to call `provider`, with its key read from the environment."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown AI provider {provider!r}. Valid: {', '.join(PROVIDERS)}.")
    spec = PROVIDERS[provider]
    model = (model or os.environ.get("AI_MODEL") or spec["default_model"] or "").strip()
    if not model:
        raise ValueError(f"Choose a model for {spec['label']} (AI_MODEL).")
    key = _key(provider)
    if spec["kind"] == "anthropic":
        if not key:
            raise ValueError(f"{spec['label']} needs a key: set {spec['key_env']}.")
        return AIConfig(provider, model, key, None)
    base = os.environ.get("AI_BASE_URL") or spec["base_url"]
    if not base:
        raise ValueError(f"{spec['label']} needs a server address: set AI_BASE_URL.")
    if key is None and provider not in ("ollama", "custom"):
        raise ValueError(f"{spec['label']} needs a key: set {spec['key_env']}.")
    return AIConfig(provider, model, key, base.rstrip("/"))


def config_from_env() -> AIConfig | None:
    provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    if provider == NO_AI:
        return None
    if not provider:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        provider = "anthropic"
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown AI_PROVIDER {provider!r}. Valid: {', '.join(PROVIDERS)}.")
    return config_for(provider)


ANTHROPIC_VERSION = "2023-06-01"
# Listing endpoints also return models this app cannot use (it needs a chat
# model that writes JSON): images, video, speech, embeddings, moderation.
_NOT_CHAT = re.compile(
    r"(embed|rerank|tts|asr|whisper|transcri|audio|speech|realtime|image|dall-e|"
    r"sora|video|^wan\d|moderation|omni-moderation|vision-preview|ocr|paraformer|"
    r"cosyvoice|sambert|^auto$)", re.I)


def list_models(provider: str, api_key: str | None = None,
                base_url: str | None = None) -> list[dict]:
    """The chat models `provider` offers this key, newest or most relevant
    first, as [{"id", "name"}]. Asks the provider's own model-list API; a
    key typed on the page may be passed in before it is saved.

    Claude: GET /v1/models (paged). Model Studio: GET /api/v1/models filtered
    to text generation. Everything else: the OpenAI-style GET {base}/models."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown AI provider {provider!r}.")
    spec = PROVIDERS[provider]
    key = api_key or _key(provider)
    base = (base_url or (os.environ.get("AI_BASE_URL") if provider == "custom" else None)
            or spec["base_url"] or "").rstrip("/")
    if spec["kind"] == "openai" and not base:
        raise ValueError(f"{spec['label']} needs a server address first.")
    if spec["key_env"] and not key and provider not in ("ollama", "custom"):
        raise ValueError(f"Enter the {spec['label']} key to list its models.")
    try:
        if spec["kind"] == "anthropic":
            models = _list_claude(key)
        elif provider == "alibaba":
            models = _list_model_studio(base, key)
        else:
            models = _list_openai(base, key)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        why = ("the key was refused" if code in (401, 403)
               else "this server has no model list" if code == 404 else f"HTTP {code}")
        raise AIError(f"Could not list {spec['label']} models: {why}.") from exc
    except httpx.HTTPError as exc:
        raise AIError(f"Could not reach {spec['label']} to list its models.") from exc
    seen, out = set(), []
    for model in models:
        if model["id"] and model["id"] not in seen and not _NOT_CHAT.search(model["id"]):
            seen.add(model["id"])
            out.append(model)
    return out


def _get(url: str, headers: dict, params: dict | None = None) -> dict:
    response = httpx.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _list_claude(key: str | None) -> list[dict]:
    headers = {"x-api-key": key or "", "anthropic-version": ANTHROPIC_VERSION}
    models, after = [], None
    for _ in range(20):
        page = _get("https://api.anthropic.com/v1/models", headers,
                    {"limit": 1000, **({"after_id": after} if after else {})})
        models += [{"id": m["id"], "name": m.get("display_name") or m["id"]}
                   for m in page.get("data") or []]
        if not page.get("has_more"):
            break
        after = page.get("last_id")
    return models  # the API lists newest first


def _list_model_studio(base: str, key: str | None) -> list[dict]:
    # .../compatible-mode/v1 -> .../api/v1/models, the native catalogue that
    # can filter to text generation (TG).
    root = base.split("/compatible-mode")[0]
    headers = {"Authorization": f"Bearer {key}"}
    models = []
    for page_no in range(1, 21):
        page = _get(f"{root}/api/v1/models", headers,
                    {"capabilities": "TG", "page_no": page_no, "page_size": 100})
        batch = (page.get("output") or {}).get("models") or []
        models += [{"id": m.get("model") or "", "name": m.get("name") or m.get("model") or ""}
                   for m in batch]
        if len(batch) < 100:
            break
    return models


def _list_openai(base: str, key: str | None) -> list[dict]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = _get(f"{base}/models", headers).get("data") or []
    models = []
    for m in data:
        outputs = ((m.get("architecture") or {}).get("output_modalities"))  # OpenRouter
        if outputs is not None and outputs != ["text"]:
            continue
        models.append({"id": m.get("id") or "", "name": m.get("name") or m.get("id") or "",
                       "created": m.get("created") or 0})
    models.sort(key=lambda m: -int(m.pop("created") or 0))  # newest first
    return models


def available_providers() -> list[dict]:
    """Every provider for the settings page: whether its key is set (never the
    key itself), its models, and which one is in use."""
    try:
        active = config_from_env()
    except ValueError:
        active = None
    chosen = (os.environ.get("AI_PROVIDER") or "").strip().lower() or (active and active.provider)
    return [{"id": pid, "label": spec["label"], "kind": spec["kind"],
             "has_key": bool(_key(pid)) or spec["key_env"] is None,
             "key_env": spec["key_env"], "key_hint": spec["key_hint"],
             "needs_base_url": spec["base_url"] is None and spec["kind"] == "openai",
             "models": spec["models"], "default_model": spec["default_model"],
             "active": pid == chosen,
             "model": active.model if active and active.provider == pid else None}
            for pid, spec in PROVIDERS.items()]


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


RETRY_STATUS = (429, 500, 502, 503, 504)


def _thinking_off(provider: str) -> dict:
    """Extra request fields that stop a model reasoning before it answers.

    Qwen 3.8 on Model Studio thinks by default and writes about 100
    characters of reasoning a second before any JSON: a 99 s transcript took
    over 7 minutes. Switched off, the same request answered in 13 s with
    equally sensible clips."""
    if (os.environ.get("AI_THINKING") or "").strip().lower() in ("1", "on", "true", "yes"):
        return {}
    return dict(PROVIDERS.get(provider, {}).get("no_thinking") or {})


def fallback_model(config: AIConfig) -> str | None:
    model = (os.environ.get("AI_FALLBACK_MODEL")
             or PROVIDERS.get(config.provider, {}).get("fallback_model") or "").strip()
    return model if model and model != config.model else None


def _openai_compatible(config: AIConfig, system: str, user: str, schema: dict,
                       max_tokens: int) -> dict:
    try:
        return _openai_request(config, system, user, schema, max_tokens)
    except _Retryable as exc:
        backup = fallback_model(config)
        if backup is None:
            raise AIError(str(exc)) from exc
        try:
            return _openai_request(AIConfig(config.provider, backup, config.api_key,
                                            config.base_url), system, user, schema, max_tokens)
        except _Retryable as again:
            raise AIError(f"{exc} The fallback model {backup} failed too: {again}") from again


class _Retryable(AIError):
    """A failure another model might not have: rate limit, overload, timeout."""


def _openai_request(config: AIConfig, system: str, user: str, schema: dict,
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
                  "max_tokens": min(max_tokens, OPENAI_MAX_TOKENS),
                  **_thinking_off(config.provider)},
            timeout=600.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except httpx.ConnectError as exc:
        # The OS's own text for this is often localized and garbled on Windows.
        raise AIError(f"Could not reach the {config.provider} server at {config.base_url}. "
                      f"Is it running, and is AI_BASE_URL right?") from exc
    except httpx.TimeoutException as exc:
        raise _Retryable(f"{config.model} on {config.provider} did not answer in time.") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in RETRY_STATUS:
            raise _Retryable(f"{config.model} on {config.provider} is busy or rate-limited "
                             f"({exc.response.status_code}).") from exc
        raise AIError(f"The {config.provider} request failed: {exc}") from exc
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise AIError(f"The {config.provider} request failed: {exc}") from exc
    return parse_json(content)


def complete_json(config: AIConfig, system: str, user: str, schema: dict,
                  max_tokens: int) -> dict:
    """One request, answered as a JSON object that follows `schema`."""
    if PROVIDERS[config.provider]["kind"] == "anthropic":
        try:
            return _claude(config, system, user, schema, max_tokens)
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK errors surface as one AIError
            raise AIError(_claude_failure(exc)) from exc
    return _openai_compatible(config, system, user, schema, max_tokens)
