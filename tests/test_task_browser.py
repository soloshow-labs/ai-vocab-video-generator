import json
import os

import pytest

from ai_vocab_video_generator.domain import GenerationRequest, JobStatus, WordEntry
from ai_vocab_video_generator.storage import JobStorage


def _job(storage, topic="Fruit", word="apple"):
    entries = [WordEntry(english=word)]
    paths = storage.create_job(GenerationRequest(topic=topic, entries=entries))
    storage.update_manifest(
        paths.job_id, entries=[entry.model_dump(mode="json") for entry in entries]
    )
    return paths


def _videos(storage, paths):
    directory = paths.artifacts / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    refs = [f"artifacts/videos/video-{i:04d}.mp4" for i in (1, 2)]
    for index, ref in enumerate(refs):
        (paths.root / ref).write_bytes(f"version-{index + 1}".encode())
    (directory / "uncommitted.mp4").write_bytes(b"partial")
    storage.update_manifest(
        paths.job_id, status="complete", artifacts={"video": refs[-1], "videos": refs}
    )
    return refs


def test_recent_jobs_sorted_bounded_read_only_and_no_media_reads(tmp_path, monkeypatch):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    older = _job(storage)
    newer = _job(storage, "", "pear")
    _videos(storage, older)
    os.utime(older.manifest, (100, 100))
    os.utime(newer.manifest, (200, 200))
    before = {path: path.read_bytes() for path in (older.manifest, newer.manifest)}

    def no_font_reads(*_args):
        pytest.fail("The task list must not read fonts or video contents")

    monkeypatch.setattr(storage, "_validate_manifest_fonts", no_font_reads)
    jobs = storage.list_recent_jobs(limit=1)
    assert len(jobs) == 1
    assert jobs[0].job_id == newer.job_id
    assert jobs[0].title == "pear"
    assert jobs[0].word_count == 1
    assert jobs[0].status is JobStatus.QUEUED
    assert jobs[0].updated_at == 200
    assert storage.list_recent_jobs(limit=2)[1].status is JobStatus.COMPLETE
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize(
    "damage",
    [
        "legacy",
        "malformed",
        "secret",
        "private_path",
        "wrong_id",
        "manifest_symlink",
        "job_symlink",
    ],
)
def test_recent_jobs_skip_invalid_or_private_entries(tmp_path, damage):
    storage = JobStorage(tmp_path / "storage", active_secrets=("fixture-credential-value",))
    good = _job(storage)
    bad = _job(storage)
    payload = json.loads(bad.manifest.read_text())
    if damage == "job_symlink":
        moved = tmp_path / "external-job"
        bad.root.rename(moved)
        bad.root.symlink_to(moved, target_is_directory=True)
    elif damage == "manifest_symlink":
        moved = tmp_path / "external-manifest.json"
        bad.manifest.rename(moved)
        bad.manifest.symlink_to(moved)
    elif damage == "malformed":
        bad.manifest.write_text("{")
    else:
        if damage == "legacy":
            payload["schema_version"] = 2
        elif damage == "secret":
            payload["request"]["topic"] = "fixture-credential-value"
        elif damage == "private_path":
            payload["request"]["topic"] = "/Users/example/private"
        else:
            payload["job_id"] = "0" * 32
        bad.manifest.write_text(json.dumps(payload))
    assert [job.job_id for job in storage.list_recent_jobs()] == [good.job_id]


def test_recent_jobs_missing_root_does_not_create_it(tmp_path):
    storage = JobStorage(tmp_path / "absent", active_secrets=())
    assert storage.list_recent_jobs() == []
    assert not storage.root.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cache", []),
        ("cache", None),
        ("cache", {"cards": []}),
        ("local_materials", None),
        ("local_materials", {}),
        ("pinned_materials", None),
        ("pinned_materials", [None]),
        ("pinned_materials", [{"entry_index": 0, "asset": None}]),
        ("pinned_materials", [{"entry_index": 0, "asset": []}]),
        ("background_music", None),
        ("background_music", []),
    ],
)
def test_invalid_manifest_containers_are_rejected_without_hiding_valid_tasks(
    tmp_path, field, value
):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    good, bad = _job(storage), _job(storage)
    payload = json.loads(bad.manifest.read_text())
    target = payload if field == "cache" else payload["request"]
    target[field] = value
    bad.manifest.write_text(json.dumps(payload))
    before = {path: path.read_bytes() for path in (good.manifest, bad.manifest)}

    assert [job.job_id for job in storage.list_recent_jobs()] == [good.job_id]
    with pytest.raises(ValueError):
        storage.load_manifest(bad.job_id)
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize("field", ["background_music", "pinned_materials"])
def test_missing_required_request_container_is_a_safe_validation_error(tmp_path, field):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    paths = _job(storage)
    payload = json.loads(paths.manifest.read_text())
    del payload["request"][field]
    paths.manifest.write_text(json.dumps(payload))
    assert storage.list_recent_jobs() == []
    with pytest.raises(ValueError):
        storage.load_manifest(paths.job_id)


@pytest.mark.parametrize(("field", "value"), [("kind", []), ("source", {}), ("source_id", 3)])
def test_invalid_assignment_scalar_types_are_safe_validation_errors(tmp_path, field, value):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    paths = _job(storage)
    payload = json.loads(paths.manifest.read_text())
    assignment = {
        "path": "artifacts/materials/0.png",
        "fingerprint": "a" * 64,
        "kind": "image",
        "source": "remote",
        "start_offset_seconds": None,
    }
    assignment[field] = value
    payload["material_assignments"] = {"0": assignment}
    paths.manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        storage.load_manifest(paths.job_id)
    assert storage.list_recent_jobs() == []


@pytest.mark.parametrize("depth", [600, 1500])
def test_deep_manifest_is_skipped_and_rejected_without_changing_files(tmp_path, depth):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    good = _job(storage)
    bad = _job(storage)
    raw = '{"schema_version":3,"cache":' + "[" * depth + "0" + "]" * depth + "}"
    bad.manifest.write_text(raw)
    before = good.manifest.read_bytes()

    assert [job.job_id for job in storage.list_recent_jobs()] == [good.job_id]
    with pytest.raises(ValueError):
        storage.load_manifest(bad.job_id)
    assert good.manifest.read_bytes() == before
    assert bad.manifest.read_text() == raw


def test_video_versions_only_declared_files_newest_first_and_read_only(tmp_path):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    paths = _job(storage)
    refs = _videos(storage, paths)
    before = paths.manifest.read_bytes()
    versions = storage.list_video_versions(paths.job_id)
    assert [version.reference for version in versions] == refs[::-1]
    assert versions[0].is_current
    assert not versions[1].is_current
    assert storage.read_video_version(paths.job_id, refs[0]) == b"version-1"
    assert paths.manifest.read_bytes() == before
    with pytest.raises(ValueError):
        storage.read_video_version(paths.job_id, "artifacts/videos/uncommitted.mp4")
    with pytest.raises(ValueError):
        storage.read_video_version(paths.job_id, "../../outside.mp4")


def test_video_version_revalidates_file_at_read_time(tmp_path):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    paths = _job(storage)
    refs = _videos(storage, paths)
    storage.list_video_versions(paths.job_id)
    version = paths.root / refs[0]
    external = tmp_path / "outside.mp4"
    version.rename(external)
    version.symlink_to(external)
    with pytest.raises(ValueError):
        storage.read_video_version(paths.job_id, refs[0])


def test_manifest_non_regular_file_is_rejected_before_open(tmp_path, monkeypatch):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    paths = _job(storage)
    paths.manifest.rename(tmp_path / "saved-manifest.json")
    paths.manifest.mkdir()

    def unexpected_open(*_args, **_kwargs):
        pytest.fail("Non-regular manifests must be rejected before opening")

    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(ValueError):
        storage.load_manifest(paths.job_id)


def test_oversized_manifest_is_skipped_and_cannot_be_loaded(tmp_path):
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    paths = _job(storage)
    with paths.manifest.open("r+b") as handle:
        handle.truncate(4 * 1024 * 1024 + 1)
    assert storage.list_recent_jobs() == []
    with pytest.raises(ValueError):
        storage.load_manifest(paths.job_id)
