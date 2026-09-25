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


ALL_KEYS = ("ALIBABA_TOKEN_PLAN_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY",
            "MOONSHOT_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


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


def _post_recorder(monkeypatch, statuses):
    """Fake httpx.post answering with each status in turn; records the bodies."""
    sent = []

    def fake_post(url, headers, json, timeout):
        sent.append(json)
        status = statuses[len(sent) - 1]
        request = ai.httpx.Request("POST", url)
        body = {"choices": [{"message": {"content": '{"ok": true}'}}]} if status == 200 else {}
        return ai.httpx.Response(status, request=request, json=body)

    monkeypatch.setattr(ai.httpx, "post", fake_post)
    return sent


def test_qwen_answers_without_thinking_first_unless_asked(monkeypatch):
    monkeypatch.delenv("AI_THINKING", raising=False)
    sent = _post_recorder(monkeypatch, [200, 200])
    config = ai.AIConfig("alibaba_token_plan", "qwen3.8-max", "k", "http://x/v1")
    ai.complete_json(config, "s", "u", SCHEMA, 100)
    assert sent[0]["enable_thinking"] is False
    monkeypatch.setenv("AI_THINKING", "on")
    ai.complete_json(config, "s", "u", SCHEMA, 100)
    assert "enable_thinking" not in sent[1]


def test_other_providers_get_no_thinking_switch(monkeypatch):
    sent = _post_recorder(monkeypatch, [200])
    ai.complete_json(ai.AIConfig("deepseek", "deepseek-chat", "k", "http://x/v1"), "s", "u", SCHEMA, 100)
    assert "enable_thinking" not in sent[0]


def test_a_rate_limited_model_falls_back_once(monkeypatch):
    monkeypatch.delenv("AI_FALLBACK_MODEL", raising=False)
    sent = _post_recorder(monkeypatch, [429, 200])
    config = ai.AIConfig("alibaba_token_plan", "qwen3.8-max", "k", "http://x/v1")
    assert ai.complete_json(config, "s", "u", SCHEMA, 100) == {"ok": True}
    assert [body["model"] for body in sent] == ["qwen3.8-max", "qwen3.8-flash"]


def test_a_bad_key_is_not_retried_on_another_model(monkeypatch):
    sent = _post_recorder(monkeypatch, [401, 200])
    config = ai.AIConfig("alibaba_token_plan", "qwen3.8-max", "k", "http://x/v1")
    with pytest.raises(ai.AIError, match="401"):
        ai.complete_json(config, "s", "u", SCHEMA, 100)
    assert len(sent) == 1


def test_when_the_fallback_fails_too_both_are_named(monkeypatch):
    monkeypatch.setenv("AI_FALLBACK_MODEL", "backup-model")
    _post_recorder(monkeypatch, [503, 503])
    config = ai.AIConfig("deepseek", "deepseek-chat", "k", "http://x/v1")
    with pytest.raises(ai.AIError, match="backup-model failed too"):
        ai.complete_json(config, "s", "u", SCHEMA, 100)


def test_tool_results_with_frames_become_a_user_image_message_for_openai_style_models():
    from clipper.ai import AIConfig, ToolChat
    chat = ToolChat(AIConfig("openai", "gpt-x", "k", None), "sys", [])
    chat.reply([("c1", [{"type": "text", "text": "Frames:"}, {"type": "image", "jpeg": "QUJD"}]),
                ("c2", "plain")])
    assert chat.messages[0] == {"role": "tool", "tool_call_id": "c1", "content": "Frames:"}
    assert chat.messages[1]["content"] == "plain"
    image = chat.messages[2]["content"][1]
    assert chat.messages[2]["role"] == "user"
    assert image["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_tool_results_with_frames_stay_inside_the_tool_result_for_claude():
    from clipper.ai import AIConfig, ToolChat
    chat = ToolChat(AIConfig("anthropic", "claude-x", "k", None), "sys", [])
    chat.reply([("c1", [{"type": "text", "text": "Frames:"}, {"type": "image", "jpeg": "QUJD"}])])
    block = chat.messages[0]["content"][0]
    assert block["type"] == "tool_result" and block["content"][1]["source"]["data"] == "QUJD"


def test_a_model_that_refuses_images_carries_on_without_them(monkeypatch):
    from clipper import ai
    chat = ai.ToolChat(ai.AIConfig("openai", "gpt-x", "k", None), "sys", [])
    chat.reply([("c1", [{"type": "text", "text": "Frames:"}, {"type": "image", "jpeg": "QUJD"}])])
    seen = []

    def step(max_tokens):
        seen.append([m for m in chat.messages])
        if len(seen) == 1:
            raise ai.AIError("The openai request failed (400): model does not support image input")
        return "ok", []
    monkeypatch.setattr(chat, "_openai_step", step)
    assert chat.step() == ("ok", [])
    assert chat.vision is False
    assert "image_url" not in str(chat.messages)
    chat.reply([("c2", [{"type": "image", "jpeg": "QUJD"}])])
    assert "cannot see images" in chat.messages[-1]["content"]
