import json
import types

import pytest

from clipper import ai

ENV = ("AI_PROVIDER", "AI_MODEL", "AI_API_KEY", "AI_BASE_URL", "ANTHROPIC_API_KEY")
SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}},
          "required": ["x"], "additionalProperties": False}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)


def test_nothing_configured_means_no_ai():
    assert ai.config_from_env() is None


def test_an_anthropic_key_alone_selects_claude_with_the_default_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    config = ai.config_from_env()
    assert (config.provider, config.model) == ("anthropic", "claude-opus-5")


def test_claude_without_a_key_names_the_missing_variable(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ai.config_from_env()


def test_openai_compatible_presets_resolve_their_base_url(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deepseek")
    monkeypatch.setenv("AI_MODEL", "deepseek-chat")
    monkeypatch.setenv("AI_API_KEY", "k")
    config = ai.config_from_env()
    assert config.base_url == "https://api.deepseek.com/v1"


def test_ollama_runs_locally_without_a_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_MODEL", "qwen3")
    config = ai.config_from_env()
    assert config.base_url.startswith("http://localhost:11434") and config.api_key is None


def test_a_provider_other_than_claude_needs_a_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("AI_API_KEY", "k")
    with pytest.raises(ValueError, match="AI_MODEL"):
        ai.config_from_env()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "skynet")
    with pytest.raises(ValueError, match="skynet"):
        ai.config_from_env()


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def fake_claude(message, calls):
    def stream(**kwargs):
        calls.append(kwargs)
        return FakeStream(message)
    return types.SimpleNamespace(beta=types.SimpleNamespace(
        messages=types.SimpleNamespace(stream=stream)))


def _message(text, stop="end_turn"):
    return types.SimpleNamespace(stop_reason=stop, stop_details=None,
                                 content=[types.SimpleNamespace(type="text", text=text)])


CLAUDE = ai.AIConfig("anthropic", "claude-opus-5", "sk-ant-x", None)


def test_claude_returns_parsed_json_and_constrains_the_output(monkeypatch):
    calls = []
    monkeypatch.setattr(ai, "_anthropic_client",
                        lambda config: fake_claude(_message('{"x": 3}'), calls))
    assert ai.complete_json(CLAUDE, "sys", "user", SCHEMA, 1000) == {"x": 3}
    sent = calls[0]
    assert sent["model"] == "claude-opus-5" and sent["system"] == "sys"
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert sent["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in sent["betas"]


def test_a_refusal_is_an_error_not_an_empty_answer(monkeypatch):
    monkeypatch.setattr(ai, "_anthropic_client",
                        lambda config: fake_claude(_message("", stop="refusal"), []))
    with pytest.raises(ai.AIError, match="declined"):
        ai.complete_json(CLAUDE, "s", "u", SCHEMA, 1000)


def test_a_cut_off_answer_is_an_error(monkeypatch):
    monkeypatch.setattr(ai, "_anthropic_client",
                        lambda config: fake_claude(_message('{"x": 1', stop="max_tokens"), []))
    with pytest.raises(ai.AIError, match="too long"):
        ai.complete_json(CLAUDE, "s", "u", SCHEMA, 1000)


def test_openai_compatible_providers_get_the_schema_in_the_prompt(monkeypatch):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, body=json)
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '```json\n{"x": 2}\n```'}}]})

    monkeypatch.setattr(ai.httpx, "post", fake_post)
    config = ai.AIConfig("deepseek", "deepseek-chat", "k", "https://api.deepseek.com/v1")
    assert ai.complete_json(config, "sys", "user", SCHEMA, 500) == {"x": 2}
    assert sent["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert sent["body"]["response_format"] == {"type": "json_object"}
    assert json.dumps(SCHEMA) in sent["body"]["messages"][0]["content"]


def test_bad_json_from_the_model_is_an_ai_error():
    with pytest.raises(ai.AIError, match="JSON"):
        ai.parse_json("not json at all")


def test_available_providers_marks_the_configured_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    listed = {p["id"]: p for p in ai.available_providers()}
    assert listed["anthropic"]["active"] is True
    assert listed["ollama"]["active"] is False


def test_openai_compatible_requests_stay_within_common_output_limits(monkeypatch):
    """DeepSeek and most OpenAI-compatible servers reject max_tokens above
    8192; asking for Claude's 64000 would make every segmentation call fail."""
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(body=json)
        return types.SimpleNamespace(raise_for_status=lambda: None,
                                     json=lambda: {"choices": [{"message": {"content": '{"x": 1}'}}]})

    monkeypatch.setattr(ai.httpx, "post", fake_post)
    config = ai.AIConfig("deepseek", "deepseek-chat", "k", "https://api.deepseek.com/v1")
    ai.complete_json(config, "s", "u", SCHEMA, 64000)
    assert sent["body"]["max_tokens"] == 8192


def test_an_unreachable_ai_server_is_named_in_plain_words(monkeypatch):
    def refuse(url, headers, json, timeout):
        raise ai.httpx.ConnectError("[WinError 10061] (localized gibberish)")

    monkeypatch.setattr(ai.httpx, "post", refuse)
    config = ai.AIConfig("ollama", "qwen3", None, "http://localhost:11434/v1")
    with pytest.raises(ai.AIError) as caught:
        ai.complete_json(config, "s", "u", SCHEMA, 100)
    message = str(caught.value)
    assert "Could not reach" in message and "http://localhost:11434/v1" in message
    assert "WinError" not in message


def test_an_ai_server_error_status_is_reported_with_its_code(monkeypatch):
    request = ai.httpx.Request("POST", "http://x/v1/chat/completions")
    response = ai.httpx.Response(401, request=request, text='{"error": "bad key"}')

    def fake_post(url, headers, json, timeout):
        return response

    monkeypatch.setattr(ai.httpx, "post", fake_post)
    config = ai.AIConfig("deepseek", "deepseek-chat", "k", "http://x/v1")
    with pytest.raises(ai.AIError, match="401"):
        ai.complete_json(config, "s", "u", SCHEMA, 100)


ALL_KEYS = ("ALIBABA_TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY",
            "MOONSHOT_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_alibaba_token_plan_uses_its_own_endpoint_and_key(monkeypatch, no_keys):
    monkeypatch.setenv("AI_PROVIDER", "alibaba_token_plan")
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "sk-sp-abc")
    config = ai.config_from_env()
    assert config.base_url == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    assert config.api_key == "sk-sp-abc" and config.model == "qwen3.8-max"


def test_each_provider_keeps_its_own_key(monkeypatch, no_keys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "sk-sp-y")
    assert ai.config_for("anthropic").api_key == "sk-ant-x"
    assert ai.config_for("alibaba_token_plan", "kimi-k2.6").api_key == "sk-sp-y"
    assert ai.config_for("alibaba_token_plan", "kimi-k2.6").model == "kimi-k2.6"


def test_a_provider_without_its_key_names_the_variable(monkeypatch, no_keys):
    with pytest.raises(ValueError, match="ALIBABA_TOKEN_PLAN_API_KEY"):
        ai.config_for("alibaba_token_plan")


def test_the_old_shared_key_still_works(monkeypatch, no_keys):
    monkeypatch.setenv("AI_API_KEY", "legacy")
    assert ai.config_for("deepseek", "deepseek-chat").api_key == "legacy"


def test_the_provider_list_says_which_have_a_key_and_never_shows_it(monkeypatch, no_keys):
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "sk-sp-secret")
    monkeypatch.setenv("AI_PROVIDER", "alibaba_token_plan")
    listed = {p["id"]: p for p in ai.available_providers()}
    token_plan = listed["alibaba_token_plan"]
    assert token_plan["has_key"] and token_plan["active"]
    assert "qwen3.8-max" in token_plan["models"] and token_plan["key_env"] == "ALIBABA_TOKEN_PLAN_API_KEY"
    assert not listed["deepseek"]["has_key"]
    assert "sk-sp-secret" not in str(listed)
