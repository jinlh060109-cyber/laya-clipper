import json
import pytest
from clipper.llm import (
    PRESETS, WriterConfig, build_payload, parse_metadata, writer_from_env,
)

def test_presets_cover_deepseek_and_kimi():
    assert "deepseek" in PRESETS
    assert "kimi" in PRESETS
    assert all(url.startswith("https://") for url in PRESETS.values())

def test_writer_from_env_returns_none_when_unconfigured(monkeypatch):
    for key in ("WRITER_BASE_URL", "WRITER_API_KEY", "WRITER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert writer_from_env() is None

def test_writer_from_env_builds_config(monkeypatch):
    monkeypatch.setenv("WRITER_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("WRITER_API_KEY", "sk-x")
    monkeypatch.setenv("WRITER_MODEL", "deepseek-chat")
    config = writer_from_env()
    assert config.model == "deepseek-chat"

def test_writer_from_env_expands_a_preset_name(monkeypatch):
    monkeypatch.setenv("WRITER_BASE_URL", "kimi")
    monkeypatch.setenv("WRITER_API_KEY", "sk-x")
    monkeypatch.setenv("WRITER_MODEL", "moonshot-v1-8k")
    assert writer_from_env().base_url == PRESETS["kimi"]

def test_writer_from_env_requires_a_key(monkeypatch):
    monkeypatch.setenv("WRITER_BASE_URL", "kimi")
    monkeypatch.delenv("WRITER_API_KEY", raising=False)
    monkeypatch.setenv("WRITER_MODEL", "m")
    with pytest.raises(ValueError, match="WRITER_API_KEY"):
        writer_from_env()

def test_build_payload_requests_json_output():
    config = WriterConfig("https://x/v1", "sk", "m")
    payload = build_payload(config, "sys", "user")
    assert payload["model"] == "m"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"

def test_parse_metadata_reads_the_three_fields():
    raw = json.dumps({"title": "T", "hook": "H", "description": "D"})
    assert parse_metadata(raw) == {"title": "T", "hook": "H", "description": "D"}

def test_parse_metadata_strips_markdown_fencing():
    """Several providers wrap JSON in a fenced block despite json_object mode."""
    raw = '```json\n{"title":"T","hook":"H","description":"D"}\n```'
    assert parse_metadata(raw)["title"] == "T"

def test_parse_metadata_rejects_missing_fields():
    with pytest.raises(ValueError, match="hook"):
        parse_metadata('{"title": "T", "description": "D"}')
