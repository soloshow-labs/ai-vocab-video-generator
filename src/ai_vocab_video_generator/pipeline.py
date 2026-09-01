"""Manifest-driven vocabulary video generation and regeneration workflow."""

import contextlib
import hashlib
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ai_vocab_video_generator.domain import (
    BackgroundMusicSettings,
    GenerationRequest,
    GenerationResult,
    JobStatus,
    MaterialAsset,
    MaterialKind,
    MaterialSource,
    NarrationTrackSettings,
    PhoneticMode,
    PipelineProgress,
    PipelineStage,
    RenderSettings,
    SelectionMode,
    WordEntry,
)
from ai_vocab_video_generator.errors import (
    ApplicationError,
    ConfigurationError,
    SpeechGenerationError,
    SpeechTrack,
)
from ai_vocab_video_generator.providers.base import (
    ImageProvider,
    ImageSelectionContext,
    SpeechProvider,
    VocabularyProvider,
)
from ai_vocab_video_generator.providers.images import LocalImageProvider, seeded_video_start_offset
from ai_vocab_video_generator.rendering.audio import (
    COUNTDOWN_CUE_FILENAME,
    create_countdown_wav,
)
from ai_vocab_video_generator.rendering.cards import CardLayers
from ai_vocab_video_generator.rendering.video import (
    MaterialVideoOverlay,
    VideoSegment,
)
from ai_vocab_video_generator.storage import JobPaths, JobStorage, safe_source_id

ProgressCallback = Callable[[PipelineProgress], None]
_CARD_RENDERER_VERSION = "3"


class CardRendererProtocol(Protocol):
    def render_answer(
        self,
        entry: WordEntry,
        background: Path | None,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path: ...

    def render_question(
        self,
        question: str,
        background: Path | None,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path: ...

    def render_answer_layers(
        self,
        entry: WordEntry,
        background: Path | None,
        request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers: ...

    def render_question_layers(
        self,
        question: str,
        background: Path | None,
        request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers: ...


class VideoComposerProtocol(Protocol):
    def compose(
        self,
        segments: Sequence[VideoSegment],
        destination: Path,
        *,
        render: RenderSettings,
        music: BackgroundMusicSettings,
    ) -> Path: ...


class GenerationPipeline:
    def __init__(
        self,
        *,
        storage: JobStorage,
        vocabulary_provider: VocabularyProvider | None,
        image_provider: ImageProvider,
        speech_provider: SpeechProvider,
        card_renderer: CardRendererProtocol,
        video_composer: VideoComposerProtocol,
    ) -> None:
        self._storage = storage
        self._vocabulary = vocabulary_provider
        self._images = image_provider
        self._speech = speech_provider
        self._cards = card_renderer
        self._composer = video_composer

    def run(
        self,
        request: GenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        if request.background_image is None or not request.background_image.is_file():
            raise ConfigurationError("Select a valid background image before generation.")
        paths = self._storage.create_job(request)
        with self._storage.lock(paths.job_id):
            saved_request = self._storage.load_request(paths.job_id)
            return self._run_job(paths, saved_request, on_progress=on_progress)

    def regenerate(
        self,
        job_id: str,
        replacements: Mapping[int, Path] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        replacement_inputs = dict(replacements or {})
        paths = self._storage.paths(job_id)
        with self._storage.lock(job_id):
            manifest = self._storage.load_manifest(job_id)
            cached_entries = [
                WordEntry.model_validate(item) for item in manifest.get("entries", [])
            ]
            if not cached_entries:
                raise ValueError("The job has no saved vocabulary entries.")
            invalid = sorted(
                index for index in replacement_inputs if index not in range(len(cached_entries))
            )
            if invalid:
                raise ValueError(f"Replacement entry index is out of range: {invalid[0]}")
            snapshots = self._storage.snapshot_replacements(job_id, replacement_inputs)
            request = self._storage.load_request(job_id)
            return self._run_job(
                paths,
                request,
                cached_entries=cached_entries,
                replacements=snapshots,
                on_progress=on_progress,
            )

    def _run_job(
        self,
        paths: JobPaths,
        request: GenerationRequest,
        *,
        cached_entries: list[WordEntry] | None = None,
        replacements: Mapping[int, Path] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        temporary: Path | None = None
        rollback = (
            {
                "material_assignments": self._storage.load_manifest(paths.job_id)[
                    "material_assignments"
                ]
            }
            if replacements
            else {}
        )
        try:
            self._storage.update_manifest(paths.job_id, status=JobStatus.RUNNING.value, error=None)
            self._emit(on_progress, PipelineStage.PREPARING, 0, "Preparing job")
            entries, vocabulary_fingerprint = self._entries(request, cached_entries)
            self._validate_pinned_materials(request, entries)
            manifest = self._storage.load_manifest(paths.job_id)
            cache = dict(manifest.get("cache", {}))
            cache["vocabulary"] = vocabulary_fingerprint
            self._storage.update_manifest(
                paths.job_id,
                entries=[entry.model_dump(mode="json") for entry in entries],
                cache=cache,
            )
            self._emit(on_progress, PipelineStage.VOCABULARY, 10, "Vocabulary is ready")

            materials = self._prepare_materials(paths, request, entries, replacements or {})
            self._emit(on_progress, PipelineStage.IMAGES, 30, "Vocabulary materials are ready")

            question_audio, answer_audio = self._synthesize_speech(paths, request, entries)
            self._emit(on_progress, PipelineStage.SPEECH, 60, "Narration is ready")

            segments = self._render_cards(
                paths, request, entries, materials, question_audio, answer_audio
            )
            self._emit(on_progress, PipelineStage.CARDS, 80, "Vocabulary cards are ready")

            destination = self._next_video_path(paths)
            temporary = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
            self._emit(on_progress, PipelineStage.COMPOSING, 90, "Composing video")
            self._composer.compose(
                segments,
                temporary,
                render=request.render,
                music=request.background_music,
            )
            os.replace(temporary, destination)
            relative_video = str(destination.relative_to(paths.root))
            manifest = self._storage.load_manifest(paths.job_id)
            cache = dict(manifest.get("cache", {}))
            cache["composition"] = self._composition_fingerprint(request, segments, materials)
            cache["video_sha256"] = self._file_fingerprint(destination)
            artifacts = dict(manifest.get("artifacts", {}))
            videos = list(artifacts.get("videos", []))
            videos.append(relative_video)
            artifacts.update({"video": relative_video, "videos": videos})
            saved_request = dict(manifest["request"])
            if replacements:
                pins = {pin["entry_index"]: pin for pin in saved_request["pinned_materials"]}
                for index, replacement_path in replacements.items():
                    asset = materials[index]
                    if asset is not None:
                        pins[index] = {
                            "entry_index": index,
                            "asset": {
                                **asset.model_dump(mode="json"),
                                "path": str(replacement_path.relative_to(paths.root)),
                            },
                        }
                saved_request["pinned_materials"] = [pins[index] for index in sorted(pins)]
            self._storage.update_manifest(
                paths.job_id,
                status=JobStatus.COMPLETE.value,
                cache=cache,
                artifacts=artifacts,
                request=saved_request,
                error=None,
            )
            self._emit(on_progress, PipelineStage.COMPLETE, 100, "Video is ready")
            return GenerationResult(
                job_id=paths.job_id,
                status=JobStatus.COMPLETE,
                video_path=destination,
                manifest_path=paths.manifest,
            )
        except ApplicationError as exc:
            self._discard_partial_video(temporary)
            self._storage.update_manifest(
                paths.job_id,
                status=JobStatus.FAILED.value,
                error=exc.safe_message,
                **rollback,
            )
            raise
        except (KeyboardInterrupt, SystemExit):
            self._discard_partial_video(temporary)
            self._storage.update_manifest(
                paths.job_id,
                status=JobStatus.FAILED.value,
                error="Video generation was interrupted.",
                **rollback,
            )
            raise
        except Exception as exc:
            self._discard_partial_video(temporary)
            message = "Video generation failed unexpectedly."
            self._storage.update_manifest(
                paths.job_id,
                status=JobStatus.FAILED.value,
                error=message,
                **rollback,
            )
            raise ApplicationError(message, diagnostic=type(exc).__name__) from exc

    def _entries(
        self,
        request: GenerationRequest,
        cached_entries: list[WordEntry] | None,
    ) -> tuple[list[WordEntry], str]:
        fingerprint = self._fingerprint(
            {
                "topic": request.topic,
                "count": request.word_count,
                "entries": [entry.model_dump(mode="json") for entry in request.entries],
                "phonetic_mode": request.phonetic_mode,
                "provider": request.vocabulary.model_dump(mode="json"),
            }
        )
        if cached_entries:
            entries = cached_entries
        elif request.entries:
            entries = request.entries
        else:
            if self._vocabulary is None:
                raise ConfigurationError("Configure an LLM or enter vocabulary manually.")
            entries = self._vocabulary.generate(request.topic, request.word_count)
        if request.phonetic_mode is PhoneticMode.DISABLED:
            entries = [entry.model_copy(update={"phonetic": ""}) for entry in entries]
        elif request.phonetic_mode is PhoneticMode.AUTOMATIC:
            if any(not entry.phonetic for entry in entries):
                if self._vocabulary is None:
                    raise ConfigurationError("Configure an LLM to complete automatic phonetics.")
                entries = self._vocabulary.complete_phonetics(entries)
            entries = [
                entry.model_copy(update={"phonetic": entry.phonetic.replace(".", "")})
                for entry in entries
            ]
        return entries, fingerprint

    def _prepare_materials(
        self,
        paths: JobPaths,
        request: GenerationRequest,
        entries: list[WordEntry],
        replacements: Mapping[int, Path],
    ) -> list[MaterialAsset | None]:
        if not request.material.enabled:
            self._storage.update_manifest(paths.job_id, material_assignments={})
            return [None for _ in entries]
        manifest = self._storage.load_manifest(paths.job_id)
        previous: dict[str, dict[str, Any]] = dict(manifest.get("material_assignments", {}))
        assignments: dict[str, dict[str, Any]] = {}
        result: list[MaterialAsset | None] = []
        seed = request.job_seed or 0
        pins = {pin.entry_index: pin.asset for pin in request.pinned_materials}
        material_provider = (
            LocalImageProvider(request.local_materials)
            if request.material.source is MaterialSource.LOCAL and request.local_materials
            else self._images
        )
        for index, entry in enumerate(entries):
            if index in replacements:
                material = replacements[index]
                asset = MaterialAsset(path=material, kind=self._material_kind(material))
                fingerprint = self._file_fingerprint(material)
                source = "replacement"
            elif index in pins:
                asset = pins[index]
                fingerprint = self._fingerprint(
                    {
                        "pin_hash": self._file_fingerprint(asset.path),
                        "entry": entry.model_dump(mode="json"),
                        "seed": seed,
                        "index": index,
                        "kind": asset.kind.value,
                    }
                )
                source = "pin"
            else:
                fingerprint = self._material_acquisition_fingerprint(request, entry, index)
                cached = previous.get(str(index), {})
                cached_path = self._relative_path(paths, cached.get("path"))
                if (
                    cached_path is not None
                    and cached.get("fingerprint") == fingerprint
                    and self._usable(cached_path)
                ):
                    asset = MaterialAsset(
                        path=cached_path,
                        kind=self._cached_material_kind(cached, cached_path),
                        source_id=(
                            cached["source_id"]
                            if isinstance(cached.get("source_id"), str)
                            else None
                        ),
                    )
                    source = str(cached.get("source", "provider"))
                else:
                    destination_stem = (
                        paths.artifacts / "materials" / f"{index:03d}-{fingerprint[:12]}"
                    )
                    asset = material_provider.fetch(
                        request.material_queries.get(index, entry.english)
                        if request.material.source is MaterialSource.REMOTE
                        else entry.english,
                        destination_stem,
                        request.canvas.aspect,
                        ImageSelectionContext(
                            entry_index=index,
                            pool_size=request.material.pool_size,
                            mode=request.material.selection_mode,
                            seed=seed,
                        ),
                    )
                    source = request.material.source.value
            start_offset_seconds = None
            if asset.kind is MaterialKind.VIDEO:
                duration = self._video_duration(asset.path)
                start_offset_seconds = seeded_video_start_offset(
                    duration, seed=seed, entry_index=index
                )
            assignments[str(index)] = {
                "path": str(asset.path.relative_to(paths.root)),
                "fingerprint": fingerprint,
                "source": source,
                "kind": asset.kind.value,
                "start_offset_seconds": start_offset_seconds,
            }
            source_id = safe_source_id(asset.source_id)
            if source_id is not None:
                assignments[str(index)]["source_id"] = source_id
            result.append(asset)
        warnings = list(manifest.get("warnings", []))
        for warning in getattr(material_provider, "warnings", ()):
            if warning not in warnings:
                warnings.append(warning)
        self._storage.update_manifest(
            paths.job_id,
            material_assignments=assignments,
            warnings=warnings,
        )
        return result

    @classmethod
    def _material_acquisition_fingerprint(
        cls,
        request: GenerationRequest,
        entry: WordEntry,
        index: int,
    ) -> str:
        selection: dict[str, object] = {
            "source": request.material.source.value,
            "selection_mode": request.material.selection_mode.value,
            "seed": request.job_seed or 0,
            "index": index,
        }
        if request.material.source is MaterialSource.REMOTE:
            selection.update(
                {
                    "query": " ".join(
                        request.material_queries.get(index, entry.english).split()
                    ).casefold(),
                    "remote_provider": request.material.remote_provider.value,
                    "pool_size": request.material.pool_size,
                    "aspect": request.canvas.aspect.value,
                }
            )
        elif request.local_materials:
            context = ImageSelectionContext(
                entry_index=index,
                pool_size=request.material.pool_size,
                mode=request.material.selection_mode,
                seed=request.job_seed or 0,
            )
            if context.mode is SelectionMode.SEQUENTIAL:
                source_index = context.entry_index % len(request.local_materials)
            else:
                source_index = random.Random(context.seed + context.entry_index).randrange(
                    len(request.local_materials)
                )
            source = request.local_materials[source_index]
            selection["local_source_sha256"] = cls._file_fingerprint(source)
            selection["local_source_suffix"] = source.suffix.lower()
        return cls._fingerprint(selection)

    @staticmethod
    def _validate_pinned_materials(
        request: GenerationRequest,
        entries: list[WordEntry],
    ) -> None:
        invalid = sorted(
            pin.entry_index
            for pin in request.pinned_materials
            if pin.entry_index not in range(len(entries))
        )
        if invalid:
            raise ConfigurationError(f"Pinned material entry index is out of range: {invalid[0]}")

    def _synthesize_speech(
        self,
        paths: JobPaths,
        request: GenerationRequest,
        entries: list[WordEntry],
    ) -> tuple[tuple[Path, ...], list[tuple[Path, ...]]]:
        question_tracks: tuple[Path, ...] = ()
        manifest = self._storage.load_manifest(paths.job_id)
        cache = dict(manifest.get("cache", {}))
        previous_speech = cache.get("speech", {})
        previous_question: str | None = None
        previous_answers: Mapping[str, object] = {}
        if isinstance(previous_speech, Mapping):
            question = previous_speech.get("question")
            previous_question = question if isinstance(question, str) else None
            nested_answers = previous_speech.get("answers")
            previous_answers = (
                nested_answers if isinstance(nested_answers, Mapping) else previous_speech
            )
        answer_cache: dict[str, str] = {}
        speech_cache: dict[str, object] = {"answers": answer_cache}
        if request.question.enabled:
            countdown = paths.artifacts / "audio" / COUNTDOWN_CUE_FILENAME
            if not self._usable(countdown):
                create_countdown_wav(countdown)
            settings = request.narration.question
            if settings.enabled and settings.repeats > 0:
                question = paths.artifacts / "audio" / "question.mp3"
                question_fingerprint = self._speech_fingerprint(request.question_text, settings)
                if previous_question != question_fingerprint or not self._usable(question):
                    self._synthesize_track(
                        request.question_text,
                        question,
                        settings,
                        track="question",
                    )
                speech_cache["question"] = question_fingerprint
                question_tracks = (countdown,) + (question,) * settings.repeats
            else:
                question_tracks = (countdown,)

        answers: list[tuple[Path, ...]] = []
        for index, entry in enumerate(entries):
            paths_in_order: list[Path] = []
            track_specs: tuple[tuple[SpeechTrack, str, NarrationTrackSettings], ...] = (
                ("zh", entry.chinese, request.narration.chinese),
                ("fast", entry.english, request.narration.fast_english),
                ("slow", entry.english, request.narration.slow_english),
            )
            for name, text, settings in track_specs:
                if not settings.enabled or settings.repeats <= 0 or not text:
                    continue
                destination = paths.artifacts / "audio" / f"{index:03d}-{name}.mp3"
                fingerprint = self._speech_fingerprint(text, settings)
                cache_key = f"{index}:{name}"
                if previous_answers.get(cache_key) != fingerprint or not self._usable(destination):
                    self._synthesize_track(
                        text,
                        destination,
                        settings,
                        track=name,
                        word_index=index,
                    )
                answer_cache[cache_key] = fingerprint
                paths_in_order.extend(destination for _ in range(settings.repeats))
            answers.append(tuple(paths_in_order))
        cache["speech"] = speech_cache
        self._storage.update_manifest(paths.job_id, cache=cache)
        return question_tracks, answers

    def _synthesize_track(
        self,
        text: str,
        destination: Path,
        settings: NarrationTrackSettings,
        *,
        track: SpeechTrack,
        word_index: int | None = None,
    ) -> Path:
        try:
            return self._speech.synthesize(
                text,
                destination,
                voice=settings.voice,
                rate=settings.rate_value,
                volume=settings.volume_value,
            )
        except SpeechGenerationError as exc:
            raise exc.with_context(track, word_index) from exc

    def _render_cards(
        self,
        paths: JobPaths,
        request: GenerationRequest,
        entries: list[WordEntry],
        materials: list[MaterialAsset | None],
        question_audio: tuple[Path, ...],
        answer_audio: list[tuple[Path, ...]],
    ) -> list[VideoSegment]:
        segments: list[VideoSegment] = []
        card_cache: dict[str, dict[str, str]] = {}
        manifest = self._storage.load_manifest(paths.job_id)
        assignments = manifest.get("material_assignments", {})
        for index, (entry, asset) in enumerate(zip(entries, materials, strict=True)):
            material = asset.path if asset is not None else None
            material_hash = self._file_fingerprint(material) if material is not None else "disabled"
            common = {
                "renderer_version": _CARD_RENDERER_VERSION,
                "canvas": request.canvas.model_dump(mode="json"),
                "material": request.material.model_dump(mode="json"),
                "material_hash": material_hash,
                "entry": entry.model_dump(mode="json"),
            }
            item_cache: dict[str, str] = {}
            material_video = None
            if asset is not None and asset.kind is MaterialKind.VIDEO:
                assignment = assignments.get(str(index), {})
                offset = assignment.get("start_offset_seconds")
                if not isinstance(offset, (int, float)):
                    raise ValueError("Video material assignment is missing its start offset.")
                material_video = MaterialVideoOverlay(asset, request.material, float(offset))
            if request.question.enabled:
                question_fingerprint = self._fingerprint(
                    {
                        **common,
                        "question": request.question.model_dump(mode="json"),
                        "text": request.question_text,
                    }
                )
                progress_style = request.progress if request.progress.enabled else None
                if material_video is None:
                    question_card = (
                        paths.artifacts
                        / "cards"
                        / f"{index:03d}-question-{question_fingerprint[:12]}.png"
                    )
                    if not self._usable(question_card):
                        self._cards.render_question(
                            request.question_text,
                            request.background_image,
                            material,
                            request,
                            question_card,
                        )
                    item_cache["question"] = str(question_card.relative_to(paths.root))
                    segments.append(
                        VideoSegment(
                            question_card,
                            question_audio,
                            progress_style=progress_style,
                        )
                    )
                else:
                    question_base = (
                        paths.artifacts
                        / "cards"
                        / f"{index:03d}-question-{question_fingerprint[:12]}-base.png"
                    )
                    question_foreground = question_base.with_name(
                        question_base.name.replace("-base.png", "-foreground.png")
                    )
                    if not self._usable(question_base) or not self._usable(question_foreground):
                        self._cards.render_question_layers(
                            request.question_text,
                            request.background_image,
                            request,
                            question_base,
                            question_foreground,
                        )
                    item_cache["question_base"] = str(question_base.relative_to(paths.root))
                    item_cache["question_foreground"] = str(
                        question_foreground.relative_to(paths.root)
                    )
                    segments.append(
                        VideoSegment(
                            question_base,
                            question_audio,
                            progress_style=progress_style,
                            foreground_path=question_foreground,
                            material_video=material_video,
                        )
                    )

            answer_fingerprint = self._fingerprint(
                {
                    **common,
                    "question": request.question.model_dump(mode="json"),
                    "question_text": request.question_text,
                    "english": request.english_text.model_dump(mode="json"),
                    "phonetic": request.phonetic_text.model_dump(mode="json"),
                    "chinese": request.chinese_text.model_dump(mode="json"),
                }
            )
            if material_video is None:
                answer_card = (
                    paths.artifacts / "cards" / f"{index:03d}-answer-{answer_fingerprint[:12]}.png"
                )
                if not self._usable(answer_card):
                    self._cards.render_answer(
                        entry,
                        request.background_image,
                        material,
                        request,
                        answer_card,
                    )
                item_cache["answer"] = str(answer_card.relative_to(paths.root))
                segments.append(VideoSegment(answer_card, answer_audio[index]))
            else:
                answer_base = (
                    paths.artifacts
                    / "cards"
                    / f"{index:03d}-answer-{answer_fingerprint[:12]}-base.png"
                )
                answer_foreground = answer_base.with_name(
                    answer_base.name.replace("-base.png", "-foreground.png")
                )
                if not self._usable(answer_base) or not self._usable(answer_foreground):
                    self._cards.render_answer_layers(
                        entry,
                        request.background_image,
                        request,
                        answer_base,
                        answer_foreground,
                    )
                item_cache["answer_base"] = str(answer_base.relative_to(paths.root))
                item_cache["answer_foreground"] = str(answer_foreground.relative_to(paths.root))
                segments.append(
                    VideoSegment(
                        answer_base,
                        answer_audio[index],
                        foreground_path=answer_foreground,
                        material_video=material_video,
                    )
                )
            card_cache[str(index)] = item_cache
        manifest = self._storage.load_manifest(paths.job_id)
        cache = dict(manifest.get("cache", {}))
        cache["cards"] = card_cache
        self._storage.update_manifest(paths.job_id, cache=cache)
        return segments

    def _next_video_path(self, paths: JobPaths) -> Path:
        manifest = self._storage.load_manifest(paths.job_id)
        artifacts = manifest.get("artifacts", {})
        version = len(artifacts.get("videos", [])) + 1
        return paths.artifacts / "videos" / f"video-{version:04d}.mp4"

    @staticmethod
    def _discard_partial_video(path: Path | None) -> None:
        if path is None:
            return
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    @staticmethod
    def _speech_fingerprint(text: str, settings: NarrationTrackSettings) -> str:
        return GenerationPipeline._fingerprint(
            {"text": text, "settings": settings.model_dump(mode="json")}
        )

    @staticmethod
    def _composition_fingerprint(
        request: GenerationRequest,
        segments: Sequence[VideoSegment],
        materials: Sequence[MaterialAsset | None],
    ) -> str:
        music = request.background_music
        music_payload: dict[str, object] = {"enabled": music.enabled}
        if music.enabled:
            assert music.path is not None
            music_payload.update(
                {
                    "hash": GenerationPipeline._file_fingerprint(music.path),
                    "volume_percent": music.volume_percent,
                    "ducking_percent": music.ducking_percent,
                }
            )
        question_audio: dict[str, object] = {"enabled": request.question.enabled}
        if request.question.enabled:
            question_audio["countdown"] = True
            settings = request.narration.question
            if settings.enabled and settings.repeats > 0:
                question_audio["speech"] = GenerationPipeline._speech_fingerprint(
                    request.question_text, settings
                )
        segment_payload: list[dict[str, object]] = []
        for segment in segments:
            item: dict[str, object] = {
                "card_sha256": GenerationPipeline._file_fingerprint(segment.image_path),
                "audio_sha256": [
                    GenerationPipeline._file_fingerprint(path) for path in segment.audio_paths
                ],
                "duration": segment.duration,
                "progress": (
                    segment.progress_style.model_dump(mode="json")
                    if segment.progress_style is not None
                    else None
                ),
            }
            if segment.foreground_path is not None:
                item["foreground_sha256"] = GenerationPipeline._file_fingerprint(
                    segment.foreground_path
                )
            if segment.material_video is not None:
                item["material_video"] = {
                    "sha256": GenerationPipeline._file_fingerprint(
                        segment.material_video.asset.path
                    ),
                    "kind": segment.material_video.asset.kind.value,
                    "start_offset_seconds": segment.material_video.start_offset_seconds,
                    "style": segment.material_video.style.model_dump(mode="json"),
                }
            segment_payload.append(item)
        return GenerationPipeline._fingerprint(
            {
                "render": request.render.model_dump(mode="json"),
                "music": music_payload,
                "question_audio": question_audio,
                "segments": segment_payload,
                "materials": [
                    (
                        {
                            "sha256": GenerationPipeline._file_fingerprint(material.path),
                            "kind": material.kind.value,
                            "source_id": safe_source_id(material.source_id),
                        }
                        if material is not None
                        else None
                    )
                    for material in materials
                ],
            }
        )

    @staticmethod
    def _fingerprint(payload: object) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _material_kind(path: Path) -> MaterialKind:
        if path.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}:
            return MaterialKind.VIDEO
        return MaterialKind.IMAGE

    @staticmethod
    def _cached_material_kind(
        assignment: Mapping[str, Any],
        path: Path,
    ) -> MaterialKind:
        value = assignment.get("kind")
        try:
            if isinstance(value, str):
                return MaterialKind(value)
            return GenerationPipeline._material_kind(path)
        except ValueError:
            return GenerationPipeline._material_kind(path)

    @staticmethod
    def _video_duration(path: Path) -> float:
        from moviepy.editor import VideoFileClip  # type: ignore[import-untyped]

        clip = VideoFileClip(str(path), audio=False)
        try:
            duration = float(clip.duration)
        finally:
            clip.close()
        if duration <= 0:
            raise ValueError("Material video duration must be positive.")
        return duration

    @staticmethod
    def _relative_path(paths: JobPaths, value: object) -> Path | None:
        if not isinstance(value, str):
            return None
        candidate = (paths.root / value).resolve()
        try:
            candidate.relative_to(paths.root)
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _usable(path: Path | None) -> bool:
        return path is not None and path.is_file() and path.stat().st_size > 0

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        stage: PipelineStage,
        percent: int,
        message: str,
    ) -> None:
        if callback is not None:
            callback(PipelineProgress(stage=stage, percent=percent, message=message))
