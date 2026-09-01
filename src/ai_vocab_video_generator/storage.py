"""Atomic filesystem-backed job storage with immutable input snapshots."""

import copy
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, get_args
from urllib.parse import unquote
from uuid import uuid4

from filelock import FileLock, Timeout
from PIL import ImageFont
from pydantic import BaseModel

from ai_vocab_video_generator.domain import (
    GenerationRequest,
    JobStatus,
    MaterialKind,
    VocabularySettings,
    WordEntry,
)
from ai_vocab_video_generator.errors import (
    ConfigurationError,
    JobBusyError,
    ProviderError,
    UploadSizeError,
)
from ai_vocab_video_generator.media_limits import (
    MAX_LOCAL_AUDIO_BYTES,
    MAX_LOCAL_IMAGE_BYTES,
    MAX_LOCAL_VIDEO_BYTES,
)
from ai_vocab_video_generator.private_fs import (
    copy_private_file,
    ensure_private_directory,
    mark_private_file,
    write_private_bytes,
)
from ai_vocab_video_generator.providers.images import probe_material

_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_MAX_FONT_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000
_FONT_SUFFIXES = frozenset({".otc", ".otf", ".ttc", ".ttf"})
_FONT_STYLE_FIELDS = ("question", "english_text", "phonetic_text", "chinese_text")
_SAFE_FONT_DESCRIPTOR_OPEN = (
    os.open in os.supports_dir_fd
    and getattr(os, "O_NOFOLLOW", 0) != 0
    and getattr(os, "O_DIRECTORY", 0) != 0
)
_IMPORT_SECRET_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(?:sk-[a-z0-9_-]{8,}|akia[0-9a-z]{16}|"
    r"bearer\s+[a-z0-9._~-]{8,}|"
    r"(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]|-----begin)",
    re.IGNORECASE,
)
_SCHEMA_VERSION = 3
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "status",
        "request",
        "entries",
        "material_assignments",
        "cache",
        "artifacts",
        "warnings",
        "error",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {"path", "fingerprint", "source", "kind", "start_offset_seconds", "source_id"}
)
_ARTIFACT_FIELDS = frozenset({"video", "videos"})
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?token|authorization|credential|password|secret|token)",
    re.IGNORECASE,
)
_ABSOLUTE_TEXT_PATH_PATTERN = re.compile(
    r"(?:^|[\s(\[{'\"\uFF08,;:\uFF0C\uFF1B\uFF1A])(?:file://)?(?:/(?!/)[^\s,;]+|"
    r"[A-Za-z]:[\\/][^\s,;]+|\\\\[^\\/\s]+[\\/][^\s,;]+)",
    re.IGNORECASE,
)


def safe_source_id(value: str | None) -> str | None:
    """Keep only bounded opaque identifiers in persisted manifests."""
    if (
        value is not None
        and len(value) <= 120
        and value.replace("_", "").replace("-", "").isalnum()
    ):
        return value
    return None


@dataclass(frozen=True, slots=True)
class JobPaths:
    job_id: str
    root: Path
    inputs: Path
    artifacts: Path
    manifest: Path
    lock_file: Path


@dataclass(frozen=True, slots=True)
class RecentJob:
    job_id: str
    title: str
    word_count: int
    status: JobStatus
    updated_at: float


@dataclass(frozen=True, slots=True)
class VideoVersion:
    reference: str
    modified_at: float
    is_current: bool


class JobStorage:
    def __init__(self, root: Path, *, active_secrets: tuple[str, ...] | None = None) -> None:
        self._configured_root = root.absolute()
        self.root = root.resolve()
        if active_secrets is None:
            from ai_vocab_video_generator.config import SecretSettings

            active_secrets = SecretSettings().values()
        self._source_id_secrets = tuple(
            dict.fromkeys(secret for secret in active_secrets if secret)
        )
        self._active_secrets = tuple(
            secret for secret in self._source_id_secrets if len(secret) >= 8
        )

    def create_job(self, request: GenerationRequest) -> JobPaths:
        return self._create_job(request, entries=[])

    def _create_job(
        self,
        request: GenerationRequest,
        *,
        entries: list[WordEntry],
    ) -> JobPaths:
        if self._configured_root.is_symlink():
            raise ConfigurationError("The storage directory must not be a symlink.")
        self._validate_input_files(request)
        job_id = uuid4().hex
        final_paths = self.paths(job_id)
        paths = self._staging_paths(job_id)
        created_files: list[Path] = []
        published = False
        try:
            ensure_private_directory(self.root)
            ensure_private_directory(paths.root)
            ensure_private_directory(paths.inputs)
            ensure_private_directory(paths.artifacts)

            saved = request.model_copy(deep=True)
            if request.background_image is not None:
                background_destination = (
                    paths.inputs / f"background{self._suffix(request.background_image)}"
                )
                created_files.append(background_destination)
                saved.background_image = self._copy_input(
                    request.background_image,
                    background_destination,
                )
            saved.local_materials = []
            for index, source in enumerate(request.local_materials):
                destination = paths.inputs / "materials" / f"{index:03d}{self._suffix(source)}"
                created_files.append(destination)
                saved.local_materials.append(self._copy_input(source, destination))
            if request.background_music.path is not None:
                music_destination = (
                    paths.inputs
                    / "music"
                    / f"{uuid4().hex}{self._suffix(request.background_music.path)}"
                )
                created_files.append(music_destination)
                saved.background_music.path = self._copy_input(
                    request.background_music.path,
                    music_destination,
                )
            saved.pinned_materials = []
            for pinned in request.pinned_materials:
                pin_destination = (
                    paths.inputs
                    / "pins"
                    / f"{pinned.entry_index:03d}{self._suffix(pinned.asset.path)}"
                )
                created_files.append(pin_destination)
                copied_pin = self._copy_input(pinned.asset.path, pin_destination)
                saved.pinned_materials.append(
                    pinned.model_copy(
                        update={
                            "asset": pinned.asset.model_copy(
                                update={
                                    "path": copied_pin,
                                    "source_id": safe_source_id(pinned.asset.source_id),
                                }
                            )
                        }
                    )
                )
            self._snapshot_fonts(saved, paths, created_files)
            saved.job_seed = (
                request.job_seed if request.job_seed is not None else secrets.randbits(63)
            )
            request_payload = saved.model_dump(mode="json")
            request_payload["background_image"] = self._relative_or_none(
                paths, saved.background_image
            )
            request_payload["local_materials"] = [
                str(source.relative_to(paths.root)) for source in saved.local_materials
            ]
            request_payload["background_music"]["path"] = self._relative_or_none(
                paths, saved.background_music.path
            )
            for index, pinned in enumerate(saved.pinned_materials):
                request_payload["pinned_materials"][index]["asset"]["path"] = str(
                    pinned.asset.path.relative_to(paths.root)
                )
            for field in _FONT_STYLE_FIELDS:
                request_payload[field]["font_path"] = self._relative_or_none(
                    paths, getattr(saved, field).font_path
                )
            manifest: dict[str, object] = {
                "schema_version": _SCHEMA_VERSION,
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "request": request_payload,
                "entries": [entry.model_dump(mode="json") for entry in entries],
                "material_assignments": {},
                "cache": {},
                "artifacts": {"video": None, "videos": []},
                "warnings": [],
                "error": None,
            }
            manifest = self._normalize_manifest(manifest)
            self._validate_manifest_paths(paths, manifest)
            created_files.append(paths.manifest)
            self._write_json(paths.manifest, manifest)
            os.replace(paths.root, final_paths.root)
            published = True
            return final_paths
        except BaseException:
            cleanup_paths = final_paths if published else paths
            cleanup_files = (
                [
                    final_paths.root / created_file.relative_to(paths.root)
                    for created_file in created_files
                ]
                if published
                else created_files
            )
            self._cleanup_staging(cleanup_paths, cleanup_files)
            raise

    def _staging_paths(self, job_id: str) -> JobPaths:
        staging_root = self.root / f".job-{job_id}-{uuid4().hex}"
        return JobPaths(
            job_id=job_id,
            root=staging_root,
            inputs=staging_root / "inputs",
            artifacts=staging_root / "artifacts",
            manifest=staging_root / "manifest.json",
            lock_file=staging_root / ".job.lock",
        )

    @staticmethod
    def _cleanup_staging(paths: JobPaths, created_files: list[Path]) -> None:
        for created_file in reversed(created_files):
            with suppress(OSError):
                created_file.unlink(missing_ok=True)
        for directory in (
            paths.inputs / "materials",
            paths.inputs / "music",
            paths.inputs / "pins",
            paths.inputs / "fonts",
            paths.inputs,
            paths.artifacts,
            paths.root,
        ):
            with suppress(OSError):
                directory.rmdir()

    def paths(self, job_id: str) -> JobPaths:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("Invalid job identifier.")
        job_root = self.root / job_id
        return JobPaths(
            job_id=job_id,
            root=job_root,
            inputs=job_root / "inputs",
            artifacts=job_root / "artifacts",
            manifest=job_root / "manifest.json",
            lock_file=job_root / ".job.lock",
        )

    def load_manifest(self, job_id: str) -> dict[str, Any]:
        paths, normalized = self._load_metadata(job_id)
        self._validate_manifest_fonts(paths, normalized["request"])
        return normalized

    def _load_metadata(self, job_id: str) -> tuple[JobPaths, dict[str, Any]]:
        paths = self.paths(job_id)
        self._validate_job_root(paths)
        payload = self._read_manifest(paths)
        normalized = self._normalize_manifest(payload)
        if normalized["job_id"] != job_id:
            raise ValueError("Saved job manifest identifier is invalid.")
        self._validate_manifest_paths(paths, normalized)
        return paths, normalized

    def list_recent_jobs(self, *, limit: int = 20) -> list[RecentJob]:
        """Read current-schema summaries without loading font or media contents."""
        if not 1 <= limit <= 100:
            raise ValueError("The task list limit must be between 1 and 100.")
        if not self.root.exists():
            return []
        if self._configured_root.is_symlink() or self._configured_root.resolve() != self.root:
            raise ValueError("Saved job storage is invalid.")
        candidates: list[tuple[float, str]] = []
        with os.scandir(self.root) as directories:
            for directory in directories:
                if not _JOB_ID_PATTERN.fullmatch(directory.name):
                    continue
                try:
                    if not directory.is_dir(follow_symlinks=False):
                        continue
                    manifest_path = Path(directory.path) / "manifest.json"
                    details = manifest_path.lstat()
                    if stat.S_ISREG(details.st_mode) and details.st_size <= _MAX_MANIFEST_BYTES:
                        candidates.append((details.st_mtime, directory.name))
                except OSError:
                    continue
        jobs: list[RecentJob] = []
        for updated_at, job_id in sorted(candidates, reverse=True):
            try:
                _, manifest = self._load_metadata(job_id)
                request = manifest["request"]
                entries = manifest["entries"] or request["entries"]
                title = request["topic"] or ", ".join(entry["english"] for entry in entries[:3])
                jobs.append(
                    RecentJob(
                        job_id, title[:100], len(entries), JobStatus(manifest["status"]), updated_at
                    )
                )
            except (OSError, ValueError, ConfigurationError):
                continue
            if len(jobs) == limit:
                break
        return jobs

    def list_video_versions(self, job_id: str) -> list[VideoVersion]:
        paths, manifest = self._load_metadata(job_id)
        artifacts = manifest["artifacts"]
        return [
            VideoVersion(
                reference, (paths.root / reference).stat().st_mtime, reference == artifacts["video"]
            )
            for reference in reversed(list(dict.fromkeys(artifacts["videos"])))
        ]

    def read_video_version(self, job_id: str, reference: str) -> bytes:
        """Revalidate the declared artifact when previewing, never accept arbitrary paths."""
        paths, manifest = self._load_metadata(job_id)
        if reference not in manifest["artifacts"]["videos"]:
            raise ValueError("The video version is not part of this task.")
        descriptor = os.open(
            paths.root / reference,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("The video version is invalid.")
            return handle.read()

    def load_request(self, job_id: str) -> GenerationRequest:
        paths = self.paths(job_id)
        self._validate_job_root(paths)
        raw_manifest = self._read_manifest(paths)
        manifest = self._normalize_manifest(raw_manifest)
        if manifest["job_id"] != job_id:
            raise ValueError("Saved job manifest identifier is invalid.")
        self._validate_manifest_paths(paths, manifest)
        verified_fonts = self._validate_manifest_fonts(paths, manifest["request"])
        payload = dict(manifest["request"])
        payload["background_image"] = self._resolve_job_path(paths, payload.get("background_image"))
        payload["local_materials"] = [
            self._resolve_job_path(paths, value) for value in payload.get("local_materials", [])
        ]
        background_music = dict(payload["background_music"])
        background_music["path"] = self._resolve_job_path(paths, background_music.get("path"))
        payload["background_music"] = background_music
        pinned_materials: list[dict[str, Any]] = []
        for pinned in payload["pinned_materials"]:
            normalized_pin = dict(pinned)
            asset = dict(normalized_pin["asset"])
            asset["path"] = self._resolve_job_path(paths, asset.get("path"))
            normalized_pin["asset"] = asset
            pinned_materials.append(normalized_pin)
        payload["pinned_materials"] = pinned_materials
        verified_font_paths: dict[str, Path] = {}
        for field in _FONT_STYLE_FIELDS:
            style = dict(payload[field])
            font_path = style.get("font_path")
            if font_path is not None:
                verified_font_paths[field] = paths.root / font_path
            style["font_path"] = None
            payload[field] = style
        request = GenerationRequest.model_validate(payload)
        for field, contents in verified_fonts.items():
            style = getattr(request, field)
            style.font_path = verified_font_paths[field]
            style.bind_verified_font_bytes(contents)
        return request

    def update_manifest(self, job_id: str, **changes: Any) -> None:
        manifest = self.load_manifest(job_id)
        manifest.update(changes)
        self.replace_manifest(job_id, manifest)

    def replace_manifest(self, job_id: str, manifest: dict[str, Any]) -> None:
        paths = self.paths(job_id)
        self._validate_job_root(paths)
        normalized = self._normalize_manifest(manifest)
        if normalized["job_id"] != job_id:
            raise ValueError("Saved job manifest identifier is invalid.")
        self._validate_manifest_paths(paths, normalized)
        self._validate_manifest_fonts(paths, normalized["request"])
        self._normalize_manifest(self._read_manifest(paths))
        self._write_json(paths.manifest, normalized)

    def snapshot_replacements(
        self,
        job_id: str,
        replacements: Mapping[int, Path],
    ) -> dict[int, Path]:
        paths = self.paths(job_id)
        self._validate_job_root(paths)
        self._normalize_manifest(self._read_manifest(paths))
        missing = [source for source in replacements.values() if not source.is_file()]
        if missing:
            raise ValueError("A replacement material file does not exist.")
        try:
            for source in replacements.values():
                probe_material(source)
        except ProviderError:
            raise ConfigurationError(
                "A replacement material is invalid or exceeds its safety limit."
            ) from None
        batch = uuid4().hex[:12]
        staged: list[Path] = []
        published: list[Path] = []
        destinations: dict[int, Path] = {}
        try:
            for index, source in replacements.items():
                destination = (
                    paths.inputs / "replacements" / f"{batch}-{index:03d}{self._suffix(source)}"
                )
                temporary = destination.with_name(f".{destination.name}.staged")
                staged.append(temporary)
                self._copy_input(source, temporary)
                destinations[index] = destination
            for destination in destinations.values():
                temporary = destination.with_name(f".{destination.name}.staged")
                os.replace(temporary, destination)
                staged.remove(temporary)
                published.append(destination)
            return destinations
        except BaseException:
            for candidate in reversed(staged):
                with suppress(OSError):
                    candidate.unlink(missing_ok=True)
            for candidate in reversed(published):
                with suppress(OSError):
                    candidate.unlink(missing_ok=True)
            replacement_dir = paths.inputs / "replacements"
            with suppress(OSError):
                replacement_dir.rmdir()
            raise

    @contextmanager
    def lock(self, job_id: str, timeout: float = 0) -> Iterator[None]:
        paths = self.paths(job_id)
        self._validate_job_root(paths)
        lock = FileLock(paths.lock_file)
        try:
            lock.acquire(timeout=timeout)
            if paths.lock_file.exists():
                mark_private_file(paths.lock_file)
        except Timeout as exc:
            raise JobBusyError("A generation for this job is already running.") from exc
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _validate_input_files(request: GenerationRequest) -> None:
        sources = list(request.local_materials)
        if request.background_image is not None:
            sources.append(request.background_image)
        if request.background_music.path is not None:
            sources.append(request.background_music.path)
        sources.extend(pinned.asset.path for pinned in request.pinned_materials)
        if any(not source.is_file() for source in sources):
            raise ConfigurationError("A selected local input file does not exist.")
        try:
            if request.background_image is not None:
                background = probe_material(request.background_image)
                if background.kind is not MaterialKind.IMAGE:
                    raise ValueError
            for source in request.local_materials:
                probe_material(source)
            for pinned in request.pinned_materials:
                size_limit = (
                    MAX_LOCAL_VIDEO_BYTES
                    if pinned.asset.kind is MaterialKind.VIDEO
                    else MAX_LOCAL_IMAGE_BYTES
                )
                size = pinned.asset.path.stat().st_size
                if size > size_limit:
                    raise UploadSizeError(size, size_limit)
            if request.background_music.path is not None:
                size = request.background_music.path.stat().st_size
                if size > MAX_LOCAL_AUDIO_BYTES:
                    raise UploadSizeError(size, MAX_LOCAL_AUDIO_BYTES)
        except UploadSizeError:
            raise
        except (OSError, ProviderError, ValueError):
            raise ConfigurationError(
                "A selected media input is invalid or exceeds its safety limit."
            ) from None
        for field in _FONT_STYLE_FIELDS:
            font_path = getattr(request, field).font_path
            if font_path is not None:
                JobStorage._validate_font(font_path)

    @staticmethod
    def _validate_font(source: Path) -> None:
        JobStorage._read_valid_font(source)

    @staticmethod
    def _read_valid_font(source: Path) -> bytes:
        try:
            with source.open("rb") as handle:
                contents = handle.read(_MAX_FONT_BYTES + 1)
            JobStorage._validate_font_contents(contents, source.suffix.casefold())
        except (OSError, ValueError):
            raise ConfigurationError(
                "A custom font must be a valid TTF, TTC, OTF, or OTC file no larger than 64 MiB."
            ) from None
        return contents

    @staticmethod
    def _validate_font_contents(contents: bytes, suffix: str) -> None:
        if suffix not in _FONT_SUFFIXES or not contents or len(contents) > _MAX_FONT_BYTES:
            raise ValueError
        font = ImageFont.truetype(BytesIO(contents), size=12)
        del font

    @classmethod
    def _snapshot_fonts(
        cls, request: GenerationRequest, paths: JobPaths, created_files: list[Path]
    ) -> None:
        snapshots: dict[tuple[str, str], Path] = {}
        for field in _FONT_STYLE_FIELDS:
            style = getattr(request, field)
            source = style.font_path
            if source is None:
                continue
            contents = cls._read_valid_font(source)
            digest = hashlib.sha256(contents).hexdigest()
            suffix = source.suffix.casefold()
            destination = snapshots.get((digest, suffix))
            if destination is None:
                destination = paths.inputs / "fonts" / f"{digest}{suffix}"
                created_files.append(destination)
                write_private_bytes(destination, contents)
                destination = destination.resolve()
                snapshots[(digest, suffix)] = destination
            style.font_path = destination

    def _normalize_manifest(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Manifest payload must be an object.")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("This job uses an unsupported manifest version.")
        try:
            normalized = copy.deepcopy(payload)
        except RecursionError:
            raise ValueError("Manifest nesting is too deep.") from None
        request = normalized.get("request")
        if not isinstance(request, dict):
            raise ValueError("Manifest request must be an object.")
        self._canonicalize_v3_manifest(normalized)
        self._validate_manifest_privacy(normalized["request"])
        return normalized

    def _canonicalize_v3_manifest(self, manifest: dict[str, Any]) -> None:
        unknown_top = set(manifest) - _MANIFEST_FIELDS
        if unknown_top:
            raise ValueError("Manifest contains an unknown field.")
        if not _MANIFEST_FIELDS.issubset(manifest):
            raise ValueError("Manifest is missing a required field.")
        job_id = manifest.get("job_id")
        if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None:
            raise ValueError("Manifest job identifier is invalid.")
        try:
            manifest["status"] = JobStatus(str(manifest.get("status"))).value
        except ValueError:
            raise ValueError("Manifest job status is invalid.") from None
        request = manifest["request"]
        JobStorage._reject_model_unknown_fields(request, GenerationRequest)
        JobStorage._validate_request_containers(request)
        entries = manifest.get("entries")
        if not isinstance(entries, list) or len(entries) > 50:
            raise ValueError("Manifest entries must be a bounded list.")
        try:
            manifest["entries"] = [
                WordEntry.model_validate(entry).model_dump(mode="json") for entry in entries
            ]
        except Exception:
            raise ValueError("Manifest vocabulary entries are invalid.") from None
        self._validate_safe_manifest_value(
            manifest["entries"], depth=0, active_secrets=self._active_secrets
        )
        assignments = manifest.get("material_assignments")
        if not isinstance(assignments, dict):
            raise ValueError("Manifest material assignments must be an object.")
        for key, assignment in assignments.items():
            if not isinstance(key, str) or not key.isdigit() or not isinstance(assignment, dict):
                raise ValueError("Manifest material assignment is invalid.")
            if set(assignment) - _ASSIGNMENT_FIELDS or not {
                "path",
                "fingerprint",
                "source",
                "kind",
                "start_offset_seconds",
            }.issubset(assignment):
                raise ValueError("Manifest material assignment is invalid.")
            kind = assignment.get("kind")
            if not isinstance(kind, str) or kind not in {kind.value for kind in MaterialKind}:
                raise ValueError("Manifest material assignment is invalid.")
            fingerprint = assignment.get("fingerprint")
            if (
                not isinstance(fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
            ):
                raise ValueError("Manifest material assignment is invalid.")
            source = assignment.get("source")
            if not isinstance(source, str) or source not in {
                "local",
                "remote",
                "pin",
                "replacement",
            }:
                raise ValueError("Manifest material assignment is invalid.")
            offset = assignment.get("start_offset_seconds")
            if offset is not None and (
                not isinstance(offset, (int, float)) or isinstance(offset, bool) or offset < 0
            ):
                raise ValueError("Manifest material assignment is invalid.")
            source_id = assignment.get("source_id")
            self._validate_source_id_secret(source_id)
            if source_id is not None and safe_source_id(source_id) != source_id:
                raise ValueError("Manifest material assignment is invalid.")
            self._validate_safe_manifest_value(
                source_id,
                depth=0,
                active_secrets=self._active_secrets,
                field_name="source_id",
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACT_FIELDS:
            raise ValueError("Manifest artifacts are invalid.")
        if not isinstance(artifacts.get("videos"), list):
            raise ValueError("Manifest artifacts are invalid.")
        video = artifacts.get("video")
        if video is not None and not isinstance(video, str):
            raise ValueError("Manifest artifacts are invalid.")
        cache = manifest["cache"]
        if not isinstance(cache, dict):
            raise ValueError("Manifest cache must be an object.")
        if "cards" in cache and not isinstance(cache["cards"], dict):
            raise ValueError("Manifest card cache must be an object.")
        self._validate_safe_manifest_value(cache, depth=0, active_secrets=self._active_secrets)
        warnings = manifest.get("warnings")
        if not isinstance(warnings, list) or len(warnings) > 100:
            raise ValueError("Manifest warnings are invalid.")
        for warning in warnings:
            JobStorage._validate_safe_manifest_text(warning)
        error = manifest.get("error")
        if error is not None:
            JobStorage._validate_safe_manifest_text(error)

    @staticmethod
    def _validate_request_containers(request: dict[str, Any]) -> None:
        """Check container shapes before path resolution or value validation."""
        if not isinstance(request.get("local_materials", []), list):
            raise ValueError("Manifest local materials must be a list.")
        if not isinstance(request.get("background_music"), dict):
            raise ValueError("Manifest background music must be an object.")
        pins = request.get("pinned_materials")
        if not isinstance(pins, list):
            raise ValueError("Manifest pinned materials must be a list.")
        for pin in pins:
            if not isinstance(pin, dict) or not isinstance(pin.get("asset"), dict):
                raise ValueError("Manifest pinned material asset must be an object.")

    def _validate_source_id_secret(self, source_id: object) -> None:
        if source_id is None:
            return
        if not isinstance(source_id, str):
            raise ValueError("Manifest material assignment is invalid.")
        canonical_source_id = source_id.replace("_", "").replace("-", "")
        for secret in self._source_id_secrets:
            canonical_secret = secret.replace("_", "").replace("-", "")
            if len(secret) >= 8:
                contains_secret = secret in source_id or (
                    bool(canonical_secret) and canonical_secret in canonical_source_id
                )
            else:
                contains_secret = source_id == secret or (
                    bool(canonical_secret) and canonical_source_id == canonical_secret
                )
            if contains_secret:
                raise ValueError("Manifest material assignment is invalid.")

    @staticmethod
    def _reject_model_unknown_fields(payload: object, model: type[BaseModel]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Manifest request settings must be objects.")
        unknown = set(payload) - set(model.model_fields)
        if unknown:
            raise ValueError("Manifest request contains an unknown field.")
        for name, field in model.model_fields.items():
            if name not in payload:
                continue
            nested_models = JobStorage._nested_model_types(field.annotation)
            if not nested_models:
                continue
            value = payload[name]
            if isinstance(value, list):
                for item in value:
                    JobStorage._reject_model_unknown_fields(item, nested_models[0])
            elif isinstance(value, dict):
                JobStorage._reject_model_unknown_fields(value, nested_models[0])

    @staticmethod
    def _nested_model_types(annotation: object) -> list[type[BaseModel]]:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        result: list[type[BaseModel]] = []
        for argument in get_args(annotation):
            result.extend(JobStorage._nested_model_types(argument))
        return result

    @staticmethod
    def _validate_safe_manifest_text(value: object) -> None:
        if (
            not isinstance(value, str)
            or len(value) > 500
            or _IMPORT_SECRET_PATTERN.search(value)
            or _ABSOLUTE_TEXT_PATH_PATTERN.search(value)
        ):
            raise ValueError("Manifest text metadata is invalid.")

    @staticmethod
    def _decoded_text_variants(value: str) -> tuple[str, ...]:
        variants = [value]
        decoded = value
        while True:
            decoded_value = unquote(decoded)
            if decoded_value == decoded:
                break
            variants.append(decoded_value)
            decoded = decoded_value
        return tuple(variants)

    @staticmethod
    def _validate_safe_manifest_value(
        value: object,
        *,
        depth: int,
        active_secrets: tuple[str, ...] = (),
        field_name: str | None = None,
    ) -> None:
        if depth > 8:
            raise ValueError("Manifest cache metadata is invalid.")
        if value is None or isinstance(value, (bool, int, float)):
            return
        if isinstance(value, str):
            if len(value) > 1000:
                raise ValueError("Manifest cache metadata is invalid.")
            variants = JobStorage._decoded_text_variants(value)
            contains_secret = any(
                _IMPORT_SECRET_PATTERN.search(candidate)
                or any(secret in candidate for secret in active_secrets)
                for candidate in variants
            )
            contains_path = any(
                _ABSOLUTE_TEXT_PATH_PATTERN.search(candidate) for candidate in variants
            )
            if field_name == "phonetic" and not any(
                re.search(
                    r"(?:/(?:Users|home|private|tmp|var|Volumes|root|etc|opt)/|"
                    r"[A-Za-z]:[\\/]|\\\\|file://)",
                    candidate,
                    re.IGNORECASE,
                )
                for candidate in variants
            ):
                contains_path = False
            if contains_secret or contains_path:
                raise ValueError("Manifest cache metadata is invalid.")
            return
        if isinstance(value, list):
            if len(value) > 1000:
                raise ValueError("Manifest cache metadata is invalid.")
            for item in value:
                JobStorage._validate_safe_manifest_value(
                    item,
                    depth=depth + 1,
                    active_secrets=active_secrets,
                    field_name=field_name,
                )
            return
        if isinstance(value, dict):
            if len(value) > 1000:
                raise ValueError("Manifest cache metadata is invalid.")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 240 or _SENSITIVE_KEY_PATTERN.search(key):
                    raise ValueError("Manifest cache metadata is invalid.")
                JobStorage._validate_safe_manifest_value(
                    item,
                    depth=depth + 1,
                    active_secrets=active_secrets,
                    field_name=key,
                )
            return
        raise ValueError("Manifest cache metadata is invalid.")

    def _validate_manifest_privacy(self, request: dict[str, Any]) -> None:
        self._validate_safe_manifest_value(request, depth=0, active_secrets=self._active_secrets)
        vocabulary = request.get("vocabulary")
        if not isinstance(vocabulary, dict):
            raise ValueError("Manifest vocabulary settings must be an object.")
        VocabularySettings.model_validate(vocabulary)
        for field in _FONT_STYLE_FIELDS:
            style = request.get(field)
            if not isinstance(style, dict):
                raise ValueError("Manifest text style settings must be objects.")
            font_path = style.get("font_path")
            if font_path is None:
                continue
            if not isinstance(font_path, str):
                raise ValueError("Manifest font path must be a relative string.")
            parts = Path(font_path).parts
            if (
                Path(font_path).is_absolute()
                or re.match(r"^[A-Za-z]:[\\/]", font_path)
                or len(parts) != 3
                or parts[:2] != ("inputs", "fonts")
                or parts[2] in {"", ".", ".."}
            ):
                raise ValueError("Manifest font path must reference a snapshotted job font.")

    def _validate_manifest_paths(self, paths: JobPaths, manifest: dict[str, Any]) -> None:
        request = manifest["request"]
        background = request.get("background_image")
        if background is not None:
            self._validate_job_file_reference(paths, background, ("inputs/background",))
        for value in request.get("local_materials", []):
            self._validate_job_file_reference(paths, value, ("inputs/materials/",))
        music = request.get("background_music", {}).get("path")
        if music is not None:
            self._validate_job_file_reference(paths, music, ("inputs/music/",))
        for pin in request.get("pinned_materials", []):
            self._validate_job_file_reference(
                paths, pin.get("asset", {}).get("path"), ("inputs/pins/", "inputs/replacements/")
            )
        for assignment in manifest.get("material_assignments", {}).values():
            self._validate_job_file_reference(
                paths,
                assignment.get("path"),
                (
                    "artifacts/materials/",
                    "inputs/materials/",
                    "inputs/pins/",
                    "inputs/replacements/",
                ),
            )
        artifacts = manifest.get("artifacts", {})
        videos = artifacts.get("videos", [])
        for value in videos:
            self._validate_job_file_reference(paths, value, ("artifacts/videos/",))
        current_video = artifacts.get("video")
        if current_video is not None:
            self._validate_job_file_reference(paths, current_video, ("artifacts/videos/",))
        cards = manifest.get("cache", {}).get("cards", {})
        if isinstance(cards, dict):
            for card_group in cards.values():
                if not isinstance(card_group, dict):
                    raise ValueError("Manifest card cache is invalid.")
                for value in card_group.values():
                    self._validate_job_file_reference(paths, value, ("artifacts/cards/",))
        manifest["request"] = self._canonical_request_values(paths, request)

    def _canonical_request_values(self, paths: JobPaths, request: dict[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(request)
        payload["background_image"] = self._resolve_job_path(paths, payload.get("background_image"))
        payload["local_materials"] = [
            self._resolve_job_path(paths, value) for value in payload.get("local_materials", [])
        ]
        background_music = dict(payload["background_music"])
        background_music["path"] = self._resolve_job_path(paths, background_music.get("path"))
        payload["background_music"] = background_music
        pinned_materials: list[dict[str, Any]] = []
        for pin in payload["pinned_materials"]:
            normalized_pin = dict(pin)
            asset = dict(normalized_pin["asset"])
            asset["path"] = self._resolve_job_path(paths, asset.get("path"))
            normalized_pin["asset"] = asset
            pinned_materials.append(normalized_pin)
        payload["pinned_materials"] = pinned_materials
        font_paths = {field: request[field].get("font_path") for field in _FONT_STYLE_FIELDS}
        for field in _FONT_STYLE_FIELDS:
            style = dict(payload[field])
            style["font_path"] = None
            payload[field] = style
        try:
            canonical = GenerationRequest.model_validate(payload).model_dump(mode="json")
        except Exception:
            raise ValueError("Manifest request values are invalid.") from None
        canonical["background_image"] = request.get("background_image")
        canonical["local_materials"] = list(request.get("local_materials", []))
        canonical["background_music"]["path"] = request["background_music"].get("path")
        for index, pin in enumerate(request["pinned_materials"]):
            canonical["pinned_materials"][index]["asset"]["path"] = pin["asset"]["path"]
        for field, font_path in font_paths.items():
            canonical[field]["font_path"] = font_path
        return canonical

    @staticmethod
    def _validate_job_file_reference(
        paths: JobPaths,
        value: object,
        prefixes: tuple[str, ...],
    ) -> None:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("Manifest input path must stay inside the job directory.")
        lexical = Path(value)
        if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
            raise ValueError("Manifest input path must stay inside the job directory.")
        normalized = lexical.as_posix()
        if not any(
            normalized.startswith(prefix)
            if prefix.endswith("/")
            else normalized == prefix or normalized.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            raise ValueError("Manifest input path must stay inside the job directory.")
        try:
            unresolved = paths.root
            for part in lexical.parts:
                unresolved /= part
                if unresolved.is_symlink():
                    raise ValueError
            root = paths.root.resolve(strict=True)
            candidate = (paths.root / lexical).resolve(strict=True)
            candidate.relative_to(root)
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError
        except (OSError, RuntimeError, ValueError):
            raise ValueError("Manifest input path must stay inside the job directory.") from None

    def _validate_job_root(self, paths: JobPaths) -> None:
        try:
            if self._configured_root.is_symlink() or paths.root.is_symlink():
                raise ValueError
            storage_root = self.root.resolve(strict=True)
            if self._configured_root.resolve(strict=True) != storage_root:
                raise ValueError
            resolved_job = paths.root.resolve(strict=True)
            if resolved_job.parent != storage_root or not resolved_job.is_dir():
                raise ValueError
        except (OSError, RuntimeError, ValueError):
            raise ValueError("Saved job is invalid.") from None

    def _validate_manifest_fonts(
        self,
        paths: JobPaths,
        request: dict[str, Any],
    ) -> dict[str, bytes]:
        verified: dict[str, bytes] = {}
        for field in _FONT_STYLE_FIELDS:
            font_path = request[field].get("font_path")
            if font_path is None:
                continue
            try:
                candidate = paths.root / font_path
                contents = self._read_job_font(paths, candidate)
                expected_digest = candidate.stem.casefold()
                if (
                    re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
                    or hashlib.sha256(contents).hexdigest() != expected_digest
                ):
                    raise ValueError
                verified[field] = contents
            except (ConfigurationError, OSError, RuntimeError, TypeError, ValueError):
                raise ValueError("Saved job font is invalid.") from None
        return verified

    def _read_job_font(self, paths: JobPaths, candidate: Path) -> bytes:
        suffix = candidate.suffix.casefold()
        if suffix not in _FONT_SUFFIXES:
            raise ValueError
        if not _SAFE_FONT_DESCRIPTOR_OPEN:
            return self._read_job_font_fallback(paths, candidate)
        descriptors: list[int] = []
        try:
            directory_flags = (
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            )
            file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            storage_fd = os.open(self.root, directory_flags)
            descriptors.append(storage_fd)
            job_fd = os.open(paths.job_id, directory_flags, dir_fd=storage_fd)
            descriptors.append(job_fd)
            inputs_fd = os.open("inputs", directory_flags, dir_fd=job_fd)
            descriptors.append(inputs_fd)
            fonts_fd = os.open("fonts", directory_flags, dir_fd=inputs_fd)
            descriptors.append(fonts_fd)
            font_fd = os.open(candidate.name, file_flags, dir_fd=fonts_fd)
            descriptors.append(font_fd)
            metadata = os.fstat(font_fd)
            if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_FONT_BYTES:
                raise ValueError
            contents = self._read_bounded_descriptor(font_fd)
            self._validate_font_contents(contents, suffix)
            return contents
        finally:
            for descriptor in reversed(descriptors):
                with suppress(OSError):
                    os.close(descriptor)

    def _read_job_font_fallback(self, paths: JobPaths, candidate: Path) -> bytes:
        font_root = paths.inputs / "fonts"
        if paths.inputs.is_symlink() or font_root.is_symlink() or candidate.is_symlink():
            raise ValueError
        root = paths.root.resolve(strict=True)
        resolved_font_root = font_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        resolved.relative_to(resolved_font_root)
        if not resolved.is_file():
            raise ValueError
        return self._read_valid_font(resolved)

    @staticmethod
    def _read_bounded_descriptor(descriptor: int) -> bytes:
        chunks: list[bytes] = []
        remaining = _MAX_FONT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        contents = b"".join(chunks)
        if not contents or len(contents) > _MAX_FONT_BYTES:
            raise ValueError
        return contents

    @staticmethod
    def _read_manifest(paths: JobPaths) -> object:
        details = paths.manifest.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("Saved job manifest is invalid or too large.")
        descriptor = os.open(
            paths.manifest,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            details = os.fstat(handle.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size > _MAX_MANIFEST_BYTES:
                raise ValueError("Saved job manifest is invalid or too large.")
            raw_bytes = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError("Saved job manifest is too large.")
        try:
            return json.loads(raw_bytes)
        except RecursionError:
            raise ValueError("Manifest nesting is too deep.") from None
        except json.JSONDecodeError as exc:
            raise ValueError("Manifest JSON is malformed.") from exc

    @staticmethod
    def _suffix(source: Path) -> str:
        suffix = source.suffix.lower()
        return suffix if suffix and len(suffix) <= 10 else ".bin"

    @staticmethod
    def _copy_input(source: Path, destination: Path) -> Path:
        copy_private_file(source, destination)
        return destination.resolve()

    @staticmethod
    def _relative_or_none(paths: JobPaths, value: Path | None) -> str | None:
        return str(value.relative_to(paths.root)) if value is not None else None

    @staticmethod
    def _resolve_job_path(paths: JobPaths, value: object) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Manifest input path must be a relative string.")
        candidate = (paths.root / value).resolve()
        try:
            candidate.relative_to(paths.root)
        except ValueError as exc:
            raise ValueError("Manifest input path escapes the job directory.") from exc
        return candidate

    @staticmethod
    def _write_json(destination: Path, payload: dict[str, Any] | dict[str, object]) -> None:
        ensure_private_directory(destination.parent)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            mark_private_file(destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
