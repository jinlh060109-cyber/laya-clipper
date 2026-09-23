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
