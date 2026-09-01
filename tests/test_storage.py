import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from PIL import Image, ImageFont

import ai_vocab_video_generator.rendering.cards as cards_module
import ai_vocab_video_generator.storage as storage_module
from ai_vocab_video_generator.domain import (
    BackgroundMusicSettings,
    GenerationRequest,
    MaterialAsset,
    MaterialKind,
    PinnedMaterial,
    WordEntry,
)
from ai_vocab_video_generator.errors import ConfigurationError, JobBusyError
from ai_vocab_video_generator.rendering.cards import CardRenderer
from ai_vocab_video_generator.storage import JobStorage


def _file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _request(tmp_path: Path) -> GenerationRequest:
    background = tmp_path / "background.png"
    apple = tmp_path / "apple.png"
    banana = tmp_path / "banana.jpg"
    Image.new("RGB", (8, 8), "white").save(background)
    Image.new("RGB", (8, 8), "red").save(apple)
    Image.new("RGB", (8, 8), "yellow").save(banana)
    return GenerationRequest(
        topic="fruit",
        entries=[WordEntry(english="apple", phonetic="/ˈæp.əl/", chinese="苹果")],
        background_image=background,
        local_materials=[apple, banana],
    )


def _system_font_path() -> Path:
    candidates = (
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/System/Library/Fonts/Symbol.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        font = ImageFont.truetype(str(candidate), size=12)
        del font
        return candidate
    pytest.skip("No small system font is available for the storage acceptance test.")


def test_create_job_snapshots_uploads_and_writes_schema_v3(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")

    paths = storage.create_job(_request(tmp_path))

    manifest = storage.load_manifest(paths.job_id)
    assert paths.root.is_dir()
    assert paths.inputs.is_dir()
    assert paths.artifacts.is_dir()
    assert manifest["schema_version"] == 3
    assert manifest["job_id"] == paths.job_id
    assert manifest["status"] == "queued"
    assert manifest["request"]["background_image"].startswith("inputs/")
    assert all(value.startswith("inputs/") for value in manifest["request"]["local_materials"])
    serialized = json.dumps(manifest).lower()
    assert "api_key" not in serialized
    assert str(tmp_path) not in serialized

    loaded = storage.load_request(paths.job_id)
    assert loaded.background_image is not None
    assert loaded.background_image.read_bytes() == (tmp_path / "background.png").read_bytes()
    assert [path.read_bytes() for path in loaded.local_materials] == [
        (tmp_path / "apple.png").read_bytes(),
        (tmp_path / "banana.jpg").read_bytes(),
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_job_storage_enforces_owner_only_permissions(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")

    paths = storage.create_job(_request(tmp_path))

    assert stat.S_IMODE(storage.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.inputs.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.manifest.stat().st_mode) == 0o600
    for path in paths.inputs.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


@pytest.mark.parametrize("failure_point", ["second-copy", "manifest-write"])
def test_create_job_failure_never_publishes_a_partial_job(
    tmp_path: Path, failure_point: str
) -> None:
    class FailingStorage(JobStorage):
        copies = 0

        @staticmethod
        def _copy_input(source: Path, destination: Path) -> Path:
            FailingStorage.copies += 1
            if failure_point == "second-copy" and FailingStorage.copies == 2:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"partial-private-copy")
                raise OSError("copy failed")
            return JobStorage._copy_input(source, destination)

        @staticmethod
        def _write_json(destination: Path, payload: dict[str, object]) -> None:
            if failure_point == "manifest-write":
                raise OSError("manifest failed")
            JobStorage._write_json(destination, payload)

    storage = FailingStorage(tmp_path / "jobs")

    with pytest.raises(OSError):
        storage.create_job(_request(tmp_path))

    assert storage.root.is_dir()
    assert list(storage.root.iterdir()) == []


def test_replacement_batch_rolls_back_every_copy_when_a_later_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(first)
    Image.new("RGB", (8, 8), "blue").save(second)
    real_copy = storage._copy_input
    calls = 0

    def fail_second(source: Path, destination: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"partial-private-copy")
            raise OSError("copy failed")
        return real_copy(source, destination)

    monkeypatch.setattr(storage, "_copy_input", fail_second)

    with pytest.raises(OSError):
        storage.snapshot_replacements(paths.job_id, {0: first, 1: second})

    assert not (paths.inputs / "replacements").exists()


def test_replacement_batch_rejects_undecodable_media_before_copying(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    invalid = _file(tmp_path / "invalid.png", b"not-an-image")

    with pytest.raises(ConfigurationError, match="replacement"):
        storage.snapshot_replacements(paths.job_id, {0: invalid})

    assert not (paths.inputs / "replacements").exists()


def test_create_job_snapshots_custom_fonts_and_loads_only_job_relative_paths(
    tmp_path: Path,
) -> None:
    source_font = _system_font_path()
    request = _request(tmp_path)
    request.english_text.font_path = source_font
    request.question.font_path = source_font
    storage = JobStorage(tmp_path / "jobs")

    paths = storage.create_job(request)

    manifest = storage.load_manifest(paths.job_id)
    english_font = manifest["request"]["english_text"]["font_path"]
    question_font = manifest["request"]["question"]["font_path"]
    assert english_font == question_font
    assert english_font.startswith("inputs/fonts/")
    assert str(source_font) not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(manifest)

    loaded = storage.load_request(paths.job_id)
    assert loaded.english_text.font_path is not None
    assert loaded.english_text.font_path == loaded.question.font_path
    assert loaded.english_text.font_path.is_relative_to(paths.inputs / "fonts")
    assert loaded.english_text.font_path.read_bytes() == source_font.read_bytes()
    rendered_font = ImageFont.truetype(str(loaded.english_text.font_path), size=12)
    del rendered_font


@pytest.mark.parametrize("suffix", [".woff", ".bin"])
def test_create_job_rejects_unsupported_custom_font_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    bad_font = _file(tmp_path / f"font{suffix}", b"not-a-font")
    request = _request(tmp_path)
    request.english_text.font_path = bad_font

    with pytest.raises(ConfigurationError, match="font"):
        JobStorage(tmp_path / "jobs").create_job(request)


def test_create_job_rejects_an_oversized_custom_font_before_copying(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.ttf"
    with oversized.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    request = _request(tmp_path)
    request.english_text.font_path = oversized

    with pytest.raises(ConfigurationError, match="font"):
        JobStorage(tmp_path / "jobs").create_job(request)


def test_create_job_snapshots_music_and_pinned_materials(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    music = _file(tmp_path / "track.mp3", b"music")
    pinned = _file(tmp_path / "pinned.png", b"pinned")
    request = GenerationRequest(
        entries=[WordEntry(english="apple")],
        background_music=BackgroundMusicSettings(enabled=True, path=music),
        pinned_materials=[
            PinnedMaterial(
                entry_index=0,
                asset=MaterialAsset(path=pinned, kind=MaterialKind.IMAGE, source_id="preview"),
            )
        ],
    )

    paths = storage.create_job(request)

    manifest = storage.load_manifest(paths.job_id)
    assert manifest["request"]["background_music"]["path"].startswith("inputs/music/")
    pinned_path = manifest["request"]["pinned_materials"][0]["asset"]["path"]
    assert pinned_path == "inputs/pins/000.png"
    assert (paths.root / pinned_path).read_bytes() == b"pinned"
    assert str(tmp_path) not in json.dumps(manifest)

    loaded = storage.load_request(paths.job_id)
    assert loaded.background_music.path is not None
    assert loaded.background_music.path.read_bytes() == b"music"
    assert loaded.pinned_materials[0].asset.path.read_bytes() == b"pinned"


def test_update_status_preserves_existing_manifest_data(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))

    storage.update_manifest(paths.job_id, status="failed", error="provider unavailable")

    manifest = storage.load_manifest(paths.job_id)
    assert manifest["request"]["topic"] == "fruit"
    assert manifest["status"] == "failed"
    assert manifest["error"] == "provider unavailable"


def test_job_lock_rejects_a_concurrent_mutation(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))

    with (
        storage.lock(paths.job_id),
        pytest.raises(JobBusyError, match="already running"),
        storage.lock(paths.job_id, timeout=0),
    ):
        pass


def test_load_request_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    manifest = storage.load_manifest(paths.job_id)
    manifest["request"]["background_image"] = "../outside.png"
    with pytest.raises(ValueError, match="job directory"):
        storage.replace_manifest(paths.job_id, manifest)


def test_replace_manifest_rejects_unknown_request_keys_before_writing(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    original = paths.manifest.read_bytes()
    manifest = storage.load_manifest(paths.job_id)
    manifest["request"]["api_key"] = "synthetic-private-value"

    with pytest.raises(ValueError, match="unknown") as caught:
        storage.replace_manifest(paths.job_id, manifest)

    assert "synthetic-private-value" not in str(caught.value)
    assert paths.manifest.read_bytes() == original


@pytest.mark.parametrize(
    "topic",
    [
        "study /Users/alice/private",
        "study %2FUsers%2Falice%2Fprivate",
        "study %2525252FUsers%2525252Falice%2525252Fprivate",
        "路径：/Users/alice/private",
        "study C:\\Users\\alice\\private",
        r"study \\server\share\private",
        "study file:///Users/alice/private",
    ],
    ids=[
        "posix",
        "encoded-posix",
        "nested-encoded-posix",
        "chinese-prefix",
        "windows",
        "unc",
        "file-url",
    ],
)
def test_create_job_rejects_personal_absolute_paths_in_persisted_text(
    tmp_path: Path, topic: str
) -> None:
    storage = JobStorage(tmp_path / "jobs")

    with pytest.raises(ValueError):
        storage.create_job(GenerationRequest(topic=topic))


def test_job_storage_rejects_active_secret_in_request_and_manifest_entries(
    tmp_path: Path,
) -> None:
    active_secret = "SyntheticActiveSecretValue0001"
    storage = JobStorage(tmp_path / "jobs", active_secrets=(active_secret,))

    with pytest.raises(ValueError):
        storage.create_job(GenerationRequest(topic=f"study {active_secret}"))

    paths = storage.create_job(
        GenerationRequest(entries=[WordEntry(english="apple", phonetic="/apple/")])
    )
    manifest = storage.load_manifest(paths.job_id)
    original = paths.manifest.read_bytes()
    manifest["entries"] = [{"english": "apple", "phonetic": "/apple/", "chinese": active_secret}]

    with pytest.raises(ValueError):
        storage.replace_manifest(paths.job_id, manifest)

    assert paths.manifest.read_bytes() == original


@pytest.mark.parametrize(
    ("active_secret", "source_id"),
    [
        ("SyntheticProviderSecretValue0001", "SyntheticProviderSecretValue0001"),
        (
            "SyntheticProviderSecretValue0001",
            "prefix-SyntheticProviderSecretValue0001-suffix",
        ),
        ("ABCD1234EFGH5678", "ABCD-1234-EFGH-5678"),
        ("1234567", "1234567"),
        ("unrelated-active-secret", "sk-syntheticcredential123"),
    ],
    ids=["exact", "substring", "canonicalized", "short", "secret-pattern"],
)
def test_replace_manifest_rejects_active_secret_in_material_source_id_without_writing(
    tmp_path: Path,
    active_secret: str,
    source_id: str,
) -> None:
    storage = JobStorage(tmp_path / "jobs", active_secrets=(active_secret,))
    paths = storage.create_job(_request(tmp_path))
    manifest = storage.load_manifest(paths.job_id)
    original = paths.manifest.read_bytes()
    material_path = manifest["request"]["local_materials"][0]
    manifest["material_assignments"] = {
        "0": {
            "path": material_path,
            "fingerprint": hashlib.sha256((paths.root / material_path).read_bytes()).hexdigest(),
            "source": "local",
            "kind": "image",
            "start_offset_seconds": None,
            "source_id": source_id,
        }
    }

    with pytest.raises(ValueError):
        storage.replace_manifest(paths.job_id, manifest)

    assert paths.manifest.read_bytes() == original


def test_load_manifest_rejects_an_existing_active_secret_material_source_id(
    tmp_path: Path,
) -> None:
    active_secret = "SyntheticProviderSecretValue0001"
    storage = JobStorage(tmp_path / "jobs", active_secrets=(active_secret,))
    paths = storage.create_job(_request(tmp_path))
    manifest = storage.load_manifest(paths.job_id)
    material_path = manifest["request"]["local_materials"][0]
    manifest["material_assignments"] = {
        "0": {
            "path": material_path,
            "fingerprint": hashlib.sha256((paths.root / material_path).read_bytes()).hexdigest(),
            "source": "local",
            "kind": "image",
            "start_offset_seconds": None,
            "source_id": active_secret,
        }
    }
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        storage.load_manifest(paths.job_id)


def test_job_storage_loads_environment_secrets_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_secret = "EfbXDjUvEnvironmentValue0001"
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", active_secret)
    storage = JobStorage(tmp_path / "jobs")

    with pytest.raises(ValueError):
        storage.create_job(GenerationRequest(topic=f"study {active_secret}"))


def test_create_job_rejects_bearer_credentials_in_persisted_text(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")

    with pytest.raises(ValueError):
        storage.create_job(GenerationRequest(topic="Bearer synthetic-credential-value"))


def test_manifest_privacy_allows_ordinary_security_words_and_ipa(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(
        GenerationRequest(
            topic="password vocabulary",
            entries=[WordEntry(english="password", phonetic="/apple/", chinese="密码")],
        )
    )

    assert storage.load_request(paths.job_id).entries[0].phonetic == "/apple/"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.__setitem__("unknown_private", "value"),
        lambda manifest: manifest["artifacts"].update(
            {"video": "../outside.mp4", "videos": ["../outside.mp4"]}
        ),
        lambda manifest: manifest["cache"].update({"api_key": "private-value"}),
        lambda manifest: manifest["warnings"].append("failed at /Users/alice/private"),
        lambda manifest: manifest.__setitem__("error", "api_key=private-value"),
    ],
    ids=["unknown-top", "artifact-traversal", "secret-cache-key", "warning-path", "error-secret"],
)
def test_replace_manifest_rejects_hostile_metadata_without_writing(
    tmp_path: Path, mutate: object
) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    original = paths.manifest.read_bytes()
    manifest = storage.load_manifest(paths.job_id)
    mutate(manifest)  # type: ignore[operator]

    with pytest.raises(ValueError):
        storage.replace_manifest(paths.job_id, manifest)

    assert paths.manifest.read_bytes() == original


def test_schema_v2_manifest_is_rejected_without_writing(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    valid_v3_manifest = json.loads(json.dumps(manifest))
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (8, 8), "green").save(replacement)
    manifest["schema_version"] = 2
    original_bytes = json.dumps(manifest, indent=2).encode()
    paths.manifest.write_bytes(original_bytes)

    for operation in (
        lambda: storage.load_manifest(paths.job_id),
        lambda: storage.load_request(paths.job_id),
        lambda: storage.update_manifest(paths.job_id, status="running"),
        lambda: storage.replace_manifest(paths.job_id, valid_v3_manifest),
        lambda: storage.snapshot_replacements(paths.job_id, {0: replacement}),
    ):
        with pytest.raises(ValueError, match="unsupported manifest version"):
            operation()
        assert paths.manifest.read_bytes() == original_bytes
        assert not (paths.inputs / "replacements").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "base_url",
            "https://session-user:session-password@example.test/v1",
        ),
        (
            "font_path",
            "/private/account/custom-font.ttf",
        ),
    ],
    ids=["credential-url", "absolute-font"],
)
def test_replace_manifest_rejects_new_private_request_values_without_writing(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    original = paths.manifest.read_bytes()
    manifest = storage.load_manifest(paths.job_id)
    if field == "base_url":
        manifest["request"]["vocabulary"]["base_url"] = value
    else:
        manifest["request"]["english_text"]["font_path"] = value

    with pytest.raises(ValueError):
        storage.replace_manifest(paths.job_id, manifest)

    assert paths.manifest.read_bytes() == original


@pytest.mark.parametrize(
    "case",
    [
        "bogus-suffix",
        "missing",
        "directory",
        "symlink-escape",
        "oversized",
        "corrupt",
        "wrong-digest",
    ],
)
def test_replace_manifest_rejects_untrusted_job_font_files_without_writing(
    tmp_path: Path,
    case: str,
) -> None:
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(_request(tmp_path))
    font_root = paths.inputs / "fonts"
    font_root.mkdir(parents=True, exist_ok=True)
    source_font = _system_font_path()
    suffix = source_font.suffix.casefold()
    relative_font: str
    if case == "bogus-suffix":
        candidate = font_root / "bogus.bin"
        candidate.write_bytes(b"not-a-font")
    elif case == "missing":
        candidate = font_root / ("a" * 64 + ".ttf")
    elif case == "directory":
        candidate = font_root / ("d" * 64 + ".ttf")
        candidate.mkdir()
    elif case == "symlink-escape":
        digest = hashlib.sha256(source_font.read_bytes()).hexdigest()
        candidate = font_root / f"{digest}{suffix}"
        try:
            candidate.symlink_to(source_font)
        except OSError:
            pytest.skip("File symlinks are unavailable in this test environment.")
    elif case == "oversized":
        candidate = font_root / ("b" * 64 + ".ttf")
        with candidate.open("wb") as handle:
            handle.truncate(64 * 1024 * 1024 + 1)
    elif case == "corrupt":
        candidate = font_root / ("c" * 64 + ".ttf")
        candidate.write_bytes(b"not-a-decodable-font")
    else:
        candidate = font_root / ("0" * 64 + suffix)
        candidate.write_bytes(source_font.read_bytes())
    relative_font = str(candidate.relative_to(paths.root))
    original = paths.manifest.read_bytes()
    manifest = storage.load_manifest(paths.job_id)
    manifest["request"]["english_text"]["font_path"] = relative_font

    with pytest.raises(ValueError, match="Saved job font is invalid"):
        storage.replace_manifest(paths.job_id, manifest)

    assert paths.manifest.read_bytes() == original


def test_replace_manifest_accepts_and_resolves_a_valid_snapshotted_font(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request.english_text.font_path = _system_font_path()
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(request)
    manifest = storage.load_manifest(paths.job_id)
    manifest["status"] = "running"

    storage.replace_manifest(paths.job_id, manifest)

    loaded = storage.load_request(paths.job_id)
    assert storage.load_manifest(paths.job_id)["status"] == "running"
    assert loaded.english_text.font_path is not None
    assert loaded.english_text.font_path.is_relative_to(paths.inputs / "fonts")


def test_font_open_rejects_a_symlink_swap_between_metadata_check_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    source_font = _system_font_path()
    request.english_text.font_path = source_font
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(request)
    manifest = storage.load_manifest(paths.job_id)
    relative_font = manifest["request"]["english_text"]["font_path"]
    candidate = paths.root / relative_font
    real_open = storage_module.os.open
    swapped = False

    def swap_before_font_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == candidate.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            candidate.unlink()
            candidate.symlink_to(source_font)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage_module.os, "open", swap_before_font_open)

    with pytest.raises(ValueError, match="Saved job font is invalid"):
        storage.load_request(paths.job_id)

    assert swapped is True


def test_loaded_request_renders_only_verified_font_bytes_after_path_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    request.english_text.font_path = _system_font_path()
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(request)
    loaded = storage.load_request(paths.job_id)
    assert loaded.english_text.font_path is not None
    loaded.english_text.font_path.write_bytes(b"swapped-after-load")
    real_truetype = cards_module.ImageFont.truetype
    received_font_objects: list[object] = []

    def require_immutable_font(font, *args, **kwargs):
        received_font_objects.append(font)
        if isinstance(font, (str, Path)):
            pytest.fail("Renderer reopened a mutable font pathname.")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(cards_module.ImageFont, "truetype", require_immutable_font)

    rendered_font = CardRenderer._font(loaded.english_text, 18)

    assert rendered_font is not None
    assert received_font_objects


@pytest.mark.parametrize("root_kind", ["storage-root", "job-root"])
def test_saved_font_loading_rejects_symlinked_roots(
    tmp_path: Path,
    root_kind: str,
) -> None:
    request = _request(tmp_path)
    request.english_text.font_path = _system_font_path()
    actual_root = tmp_path / "actual-jobs"
    storage = JobStorage(actual_root)
    paths = storage.create_job(request)
    if root_kind == "storage-root":
        linked_root = tmp_path / "linked-jobs"
        try:
            linked_root.symlink_to(actual_root, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks are unavailable in this test environment.")
        reader = JobStorage(linked_root)
    else:
        moved_job = actual_root / f"{paths.job_id}-real"
        paths.root.rename(moved_job)
        try:
            paths.root.symlink_to(moved_job, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks are unavailable in this test environment.")
        reader = storage

    with pytest.raises(ValueError, match="Saved job is invalid"):
        reader.load_request(paths.job_id)
