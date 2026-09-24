import pytest
from clipper.run import Run, MissingArtifact

def test_missing_artifact_names_producing_stage(tmp_path):
    run = Run.create(tmp_path, "ep47")
    with pytest.raises(MissingArtifact) as err:
        run.read_json("transcript.json")
    assert "transcribe" in str(err.value)

def test_write_is_atomic_leaving_no_partial_file(tmp_path, monkeypatch):
    run = Run.create(tmp_path, "ep47")
    run.write_json("source.json", {"ok": True})
    monkeypatch.setattr(
        "json.dump", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    )
    with pytest.raises(ValueError):
        run.write_json("source.json", {"ok": False})
    assert run.read_json("source.json") == {"ok": True}

def test_open_raises_actionable_error_when_run_missing(tmp_path):
    with pytest.raises(MissingArtifact) as err:
        Run.open(tmp_path / "nonexistent")
    assert "clipper ingest" in str(err.value)
