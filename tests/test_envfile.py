import os

import pytest

from clipper import envfile


@pytest.fixture(autouse=True)
def restore_environment():
    """update() writes os.environ too. monkeypatch.delenv on an absent key
    records nothing to undo, so the environment is snapshotted instead."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_updates_keep_every_other_line_and_comment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# my notes\nHF_TOKEN=hf_x\nAI_MODEL=old\n\nOTHER=1\n", encoding="utf-8")
    monkeypatch.delenv("AI_MODEL", raising=False)
    envfile.update({"AI_MODEL": "qwen3.8-max", "AI_PROVIDER": "alibaba_token_plan"}, env)
    assert env.read_text(encoding="utf-8") == (
        "# my notes\nHF_TOKEN=hf_x\nAI_MODEL=qwen3.8-max\n\nOTHER=1\n"
        "AI_PROVIDER=alibaba_token_plan\n")
    assert os.environ["AI_MODEL"] == "qwen3.8-max"


def test_none_removes_a_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AI_BASE_URL=http://x\nKEEP=1\n", encoding="utf-8")
    monkeypatch.setenv("AI_BASE_URL", "http://x")
    envfile.update({"AI_BASE_URL": None}, env)
    assert env.read_text(encoding="utf-8") == "KEEP=1\n"
    assert "AI_BASE_URL" not in os.environ


def test_a_missing_env_file_is_created(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    envfile.update({"ANTHROPIC_API_KEY": "sk-ant-1"}, env)
    assert env.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-ant-1\n"


def test_values_with_line_breaks_are_refused(tmp_path):

    with pytest.raises(ValueError, match="line"):
        envfile.update({"AI_MODEL": "a\nEVIL=1"}, tmp_path / ".env")
