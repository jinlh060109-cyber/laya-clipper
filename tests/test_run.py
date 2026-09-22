import pytest
from clipper.run import Run, MissingArtifact

def test_create_makes_run_directory(tmp_path):
    run = Run.create(tmp_path, "ep47")
    assert run.root == tmp_path / "ep47"
    assert run.root.is_dir()

def test_write_then_read_roundtrips(tmp_path):
    run = Run.create(tmp_path, "ep47")
    run.write_json("source.json", {"duration": 5400.0})
    assert run.read_json("source.json") == {"duration": 5400.0}

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

def test_clips_dir_is_created_on_demand(tmp_path):
    run = Run.create(tmp_path, "ep47")
    assert run.clips_dir().is_dir()
