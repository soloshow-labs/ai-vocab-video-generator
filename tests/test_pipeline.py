import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import ai_vocab_video_generator.pipeline as pipeline_module
from ai_vocab_video_generator.domain import (
    BackgroundMusicSettings,
    GenerationRequest,
    MaterialAsset,
    MaterialKind,
    MaterialSource,
    NarrationSettings,
    PhoneticMode,
    PinnedMaterial,
    PipelineProgress,
    PipelineStage,
    RenderSettings,
    VideoAspect,
    WordEntry,
)
from ai_vocab_video_generator.errors import ConfigurationError, ProviderError
from ai_vocab_video_generator.pipeline import GenerationPipeline
from ai_vocab_video_generator.providers.base import ImageSelectionContext
from ai_vocab_video_generator.providers.images import LocalImageProvider
from ai_vocab_video_generator.rendering.cards import CardLayers
from ai_vocab_video_generator.rendering.video import VideoSegment
from ai_vocab_video_generator.storage import JobStorage


class FakeVocabularyProvider:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.complete_calls = 0

    def generate(self, topic: str, count: int) -> list[WordEntry]:
        del topic
        self.generate_calls += 1
        entries = [
            WordEntry(english="apple", phonetic="/apple/", chinese="苹果"),
            WordEntry(english="banana", phonetic="/banana/", chinese="香蕉"),
        ]
        return entries[:count]

    def complete_phonetics(self, entries: Sequence[WordEntry]) -> list[WordEntry]:
        self.complete_calls += 1
        return [entry.model_copy(update={"phonetic": f"/{entry.english}/"}) for entry in entries]


class FakeImageProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[ImageSelectionContext] = []

    @property
    def warnings(self) -> tuple[str, ...]:
        return ()

    def fetch(
        self,
        query: str,
        destination_stem: Path,
        aspect: VideoAspect,
        context: ImageSelectionContext | None = None,
    ) -> MaterialAsset:
        del query, aspect
        assert context is not None
        self.calls += 1
        self.contexts.append(context)
        destination = destination_stem.with_suffix(".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"image-{context.entry_index}".encode())
        return MaterialAsset(path=destination, kind=MaterialKind.IMAGE)


class FakeSpeechProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, str, str]] = []
        self.fail = fail

    def synthesize(
        self,
        text: str,
        destination: Path,
        *,
        voice: str,
        rate: str,
        volume: str = "+0%",
    ) -> Path:
        self.calls.append((text, voice, rate, volume))
        if self.fail:
            raise ProviderError("Speech generation failed.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"audio")
        return destination


class FakeCardRenderer:
    def __init__(self) -> None:
        self.answer_calls: Counter[int] = Counter()
        self.question_calls: Counter[int] = Counter()
        self.answer_layer_calls: Counter[int] = Counter()
        self.question_layer_calls: Counter[int] = Counter()
        self.answer_materials: dict[int, bytes] = {}

    @staticmethod
    def _index(destination: Path) -> int:
        return int(destination.name.split("-", 1)[0])

    def render_answer(
        self,
        entry: WordEntry,
        background: Path | None,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path:
        del entry, background, request
        index = self._index(destination)
        self.answer_calls[index] += 1
        assert material is not None
        self.answer_materials[index] = material.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"answer")
        return destination

    def render_question(
        self,
        question: str,
        background: Path | None,
        material: Path | None,
        request: GenerationRequest,
        destination: Path,
    ) -> Path:
        del question, background, material, request
        index = self._index(destination)
        self.question_calls[index] += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"question")
        return destination

    def render_answer_layers(
        self,
        entry: WordEntry,
        background: Path | None,
        request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers:
        del entry, background, request
        index = self._index(base_destination)
        self.answer_layer_calls[index] += 1
        base_destination.parent.mkdir(parents=True, exist_ok=True)
        base_destination.write_bytes(b"answer-base")
        foreground_destination.write_bytes(b"answer-foreground")
        return CardLayers(base_destination, foreground_destination)

    def render_question_layers(
        self,
        question: str,
        background: Path | None,
        request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers:
        del question, background, request
        index = self._index(base_destination)
        self.question_layer_calls[index] += 1
        base_destination.parent.mkdir(parents=True, exist_ok=True)
        base_destination.write_bytes(b"question-base")
        foreground_destination.write_bytes(b"question-foreground")
        return CardLayers(base_destination, foreground_destination)


class FakeComposer:
    def __init__(self) -> None:
        self.calls = 0
        self.last_segments: Sequence[VideoSegment] = []
        self.last_render: RenderSettings | None = None
        self.last_music: BackgroundMusicSettings | None = None

    def compose(
        self,
        segments: Sequence[VideoSegment],
        destination: Path,
        *,
        render: RenderSettings,
        music: BackgroundMusicSettings,
    ) -> Path:
        self.calls += 1
        self.last_segments = segments
        self.last_render = render
        self.last_music = music
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"video-{self.calls}".encode())
        return destination


def _file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _write_video(destination: Path, *, duration: float = 1.0) -> Path:
    moviepy = pytest.importorskip("moviepy.editor")
    np = pytest.importorskip("numpy")

    def make_frame(_time: float) -> Any:
        return np.full((90, 80, 3), (255, 0, 0), dtype=np.uint8)

    clip = moviepy.VideoClip(make_frame=make_frame, duration=duration)
    try:
        clip.write_videofile(
            str(destination),
            fps=20,
            codec="libx264",
            audio=False,
            logger=None,
            threads=1,
        )
    finally:
        clip.close()
    return destination


def _request(
    tmp_path: Path,
    *,
    entries: list[WordEntry] | None = None,
    count: int = 1,
) -> GenerationRequest:
    background = tmp_path / "background.png"
    material = tmp_path / "material.png"
    Image.new("RGB", (8, 8), "white").save(background)
    Image.new("RGB", (8, 8), "blue").save(material)
    request = GenerationRequest(
        topic="fruit" if entries is None else "",
        word_count=count,
        entries=entries or [],
        background_image=background,
        local_materials=[material],
    )
    request.question.enabled = True
    request.narration.chinese.enabled = True
    request.narration.chinese.repeats = 1
    request.narration.fast_english.repeats = 2
    request.narration.slow_english.repeats = 0
    return request


def _pipeline(
    tmp_path: Path,
    *,
    speech: FakeSpeechProvider | None = None,
) -> tuple[Any, ...]:
    vocabulary = FakeVocabularyProvider()
    images = FakeImageProvider()
    speech_provider = speech or FakeSpeechProvider()
    renderer = FakeCardRenderer()
    composer = FakeComposer()
    pipeline = GenerationPipeline(
        storage=JobStorage(tmp_path / "jobs"),
        vocabulary_provider=vocabulary,
        image_provider=images,
        speech_provider=speech_provider,
        card_renderer=renderer,
        video_composer=composer,
    )
    return pipeline, vocabulary, images, speech_provider, renderer, composer


def test_material_search_query_is_saved_used_and_only_invalidates_image_cache(
    tmp_path, monkeypatch
):
    pipeline, _, images, speech, _, _ = _pipeline(tmp_path)
    queries = []
    original = images.fetch

    def fetch(query, *args, **kwargs):
        queries.append(query)
        return original(query, *args, **kwargs)

    monkeypatch.setattr(images, "fetch", fetch)
    request = _request(tmp_path, entries=[WordEntry(english="bank", chinese="河岸")])
    request.phonetic_mode = PhoneticMode.DISABLED
    request.material.source = MaterialSource.REMOTE
    request = GenerationRequest.model_validate(
        {**request.model_dump(), "material_queries": {0: "river bank"}}
    )
    result = pipeline.run(request)
    assert queries == ["river bank"]
    storage = JobStorage(tmp_path / "jobs")
    assert storage.load_request(result.job_id).material_queries == {0: "river bank"}
    assert any(call[0] == "bank" for call in speech.calls)
    assert all(call[0] != "river bank" for call in speech.calls)
    pipeline.regenerate(result.job_id)
    assert queries == ["river bank"]
    speech_count = len(speech.calls)
    manifest = storage.load_manifest(result.job_id)
    manifest["request"]["material_queries"] = {"0": "bank building"}
    storage.update_manifest(result.job_id, request=manifest["request"])
    pipeline.regenerate(result.job_id)
    assert queries == ["river bank", "bank building"]
    assert len(speech.calls) == speech_count


def test_pipeline_synthesizes_and_repeats_the_independent_question_track(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, _images, speech, _renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.narration.question.voice = "en-GB-SoniaNeural"
    request.narration.question.repeats = 2
    request.narration.question.rate = 5
    request.narration.question.volume = -10

    pipeline.run(request)

    assert [call for call in speech.calls if call[0] == request.question_text] == [
        (request.question_text, "en-GB-SoniaNeural", "+5%", "-10%")
    ]
    assert [path.name for path in composer.last_segments[0].audio_paths] == [
        "countdown-soft-chime-v1.wav",
        "question.mp3",
        "question.mp3",
    ]


@pytest.mark.parametrize(
    ("enabled", "repeats"),
    [(False, 2), (True, 0)],
    ids=["disabled", "zero-repeats"],
)
def test_pipeline_keeps_countdown_when_question_narration_is_inactive(
    tmp_path: Path,
    enabled: bool,
    repeats: int,
) -> None:
    pipeline, _vocabulary, _images, speech, _renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.narration.question.enabled = enabled
    request.narration.question.repeats = repeats

    pipeline.run(request)

    assert [path.name for path in composer.last_segments[0].audio_paths] == [
        "countdown-soft-chime-v1.wav"
    ]
    assert all(call[0] != request.question_text for call in speech.calls)


@pytest.mark.parametrize("custom_repeats", [None, 3])
def test_default_english_plays_once_and_saved_custom_repeats_are_preserved(
    tmp_path: Path, custom_repeats: int | None
) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.question.enabled = False
    request.narration = NarrationSettings()
    if custom_repeats is not None:
        request.narration.fast_english.repeats = custom_repeats
    expected = 1 if custom_repeats is None else custom_repeats

    result = pipeline.run(request)
    assert [path.name for path in composer.last_segments[0].audio_paths] == [
        "000-fast.mp3"
    ] * expected
    pipeline.regenerate(result.job_id)
    assert len(composer.last_segments[0].audio_paths) == expected


def test_pipeline_does_not_reuse_a_stale_beep_countdown(tmp_path: Path) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, composer = _pipeline(tmp_path)
    result = pipeline.run(_request(tmp_path))
    stale_countdown = result.manifest_path.parent / "artifacts" / "audio" / "countdown.wav"
    stale_countdown.write_bytes(b"stale-beep")

    pipeline.regenerate(result.job_id)

    countdown = composer.last_segments[0].audio_paths[0]
    assert countdown.name == "countdown-soft-chime-v1.wav"
    assert countdown.read_bytes() != b"stale-beep"


def test_question_narration_changes_invalidate_only_question_speech_and_composition(
    tmp_path: Path,
) -> None:
    pipeline, vocabulary, images, speech, renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    result = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first_manifest = storage.load_manifest(result.job_id)
    first_speech = first_manifest["cache"]["speech"]
    first_composition = first_manifest["cache"]["composition"]
    first_materials = first_manifest["material_assignments"]

    first_manifest["request"]["narration"]["question"]["voice"] = "en-GB-SoniaNeural"
    storage.replace_manifest(result.job_id, first_manifest)
    pipeline.regenerate(result.job_id)
    second_manifest = storage.load_manifest(result.job_id)
    second_speech = second_manifest["cache"]["speech"]

    assert len(speech.calls) == 4
    assert [call for call in speech.calls if call[0] == request.question_text] == [
        (request.question_text, "en-US-JennyNeural", "-20%", "+0%"),
        (request.question_text, "en-GB-SoniaNeural", "-20%", "+0%"),
    ]
    assert second_speech["question"] != first_speech["question"]
    assert second_speech["answers"] == first_speech["answers"]
    assert second_manifest["cache"]["composition"] != first_composition
    assert second_manifest["material_assignments"] == first_materials
    assert vocabulary.generate_calls == 1
    assert images.calls == 1
    assert renderer.question_calls == Counter({0: 1})
    assert renderer.answer_calls == Counter({0: 1})
    assert composer.calls == 2


def test_pipeline_orders_segments_audio_and_reuses_cached_artifacts(tmp_path: Path) -> None:
    pipeline, vocabulary, images, speech, _renderer, composer = _pipeline(tmp_path)
    events: list[PipelineProgress] = []

    result = pipeline.run(_request(tmp_path), on_progress=events.append)

    assert result.status.value == "complete"
    assert result.video_path is not None and result.video_path.read_bytes() == b"video-1"
    assert result.video_path.name == "video-0001.mp4"
    assert [event.percent for event in events] == sorted(event.percent for event in events)
    assert list(dict.fromkeys(event.stage for event in events)) == [
        PipelineStage.PREPARING,
        PipelineStage.VOCABULARY,
        PipelineStage.IMAGES,
        PipelineStage.SPEECH,
        PipelineStage.CARDS,
        PipelineStage.COMPOSING,
        PipelineStage.COMPLETE,
    ]
    assert vocabulary.generate_calls == 1
    assert images.calls == 1
    assert len(speech.calls) == 3
    assert len(composer.last_segments) == 2
    assert [path.name for path in composer.last_segments[0].audio_paths] == [
        "countdown-soft-chime-v1.wav",
        "question.mp3",
    ]
    assert [path.name for path in composer.last_segments[1].audio_paths] == [
        "000-zh.mp3",
        "000-fast.mp3",
        "000-fast.mp3",
    ]
    assert composer.last_segments[0].progress_style is not None
    assert composer.last_segments[1].progress_style is None
    assert composer.last_render == RenderSettings()
    assert composer.last_music == BackgroundMusicSettings()

    regenerated = pipeline.regenerate(result.job_id)

    assert regenerated.job_id == result.job_id
    assert regenerated.video_path is not None
    assert regenerated.video_path.name == "video-0002.mp4"
    assert result.video_path.is_file()
    assert vocabulary.generate_calls == 1
    assert images.calls == 1
    assert len(speech.calls) == 3
    assert composer.calls == 2


def test_renderer_version_change_invalidates_cached_cards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, _vocabulary, _images, _speech, renderer, _composer = _pipeline(tmp_path)
    result = pipeline.run(_request(tmp_path))
    assert renderer.question_calls == Counter({0: 1})
    assert renderer.answer_calls == Counter({0: 1})

    monkeypatch.setattr(
        pipeline_module,
        "_CARD_RENDERER_VERSION",
        "test-new-renderer",
        raising=False,
    )
    pipeline.regenerate(result.job_id)

    assert renderer.question_calls == Counter({0: 2})
    assert renderer.answer_calls == Counter({0: 2})


def test_changing_only_fps_reuses_upstream_artifacts_and_invalidates_composition(
    tmp_path: Path,
) -> None:
    pipeline, vocabulary, images, speech, renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.render.fps = 15

    first = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first_manifest = storage.load_manifest(first.job_id)
    first_composition = first_manifest["cache"]["composition"]
    first_cards = first_manifest["cache"]["cards"]
    first_speech = first_manifest["cache"]["speech"]
    first_materials = first_manifest["material_assignments"]
    first_manifest["request"]["render"]["fps"] = 30
    storage.replace_manifest(first.job_id, first_manifest)

    second = pipeline.regenerate(first.job_id)
    second_manifest = storage.load_manifest(first.job_id)

    assert composer.last_render == RenderSettings(fps=30)
    assert composer.calls == 2
    assert first.video_path is not None and first.video_path.is_file()
    assert second.video_path is not None and second.video_path.name == "video-0002.mp4"
    assert vocabulary.generate_calls == 1
    assert images.calls == 1
    assert len(speech.calls) == 3
    assert renderer.question_calls == Counter({0: 1})
    assert renderer.answer_calls == Counter({0: 1})
    assert second_manifest["cache"]["cards"] == first_cards
    assert second_manifest["cache"]["speech"] == first_speech
    assert second_manifest["material_assignments"] == first_materials
    assert second_manifest["cache"]["composition"] != first_composition


def test_music_content_volume_and_ducking_invalidate_only_composition_cache(
    tmp_path: Path,
) -> None:
    pipeline, vocabulary, images, speech, renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.background_music = BackgroundMusicSettings(
        enabled=True,
        path=_file(tmp_path / "music.wav", b"music-a"),
    )

    result = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first_manifest = storage.load_manifest(result.job_id)
    compositions = [first_manifest["cache"]["composition"]]
    upstream_cache = {
        key: value
        for key, value in first_manifest["cache"].items()
        if key not in {"composition", "video_sha256"}
    }
    material_assignments = first_manifest["material_assignments"]

    first_manifest["request"]["background_music"]["volume_percent"] = 20
    storage.replace_manifest(result.job_id, first_manifest)
    pipeline.regenerate(result.job_id)
    volume_manifest = storage.load_manifest(result.job_id)
    compositions.append(volume_manifest["cache"]["composition"])

    volume_manifest["request"]["background_music"]["ducking_percent"] = 40
    storage.replace_manifest(result.job_id, volume_manifest)
    pipeline.regenerate(result.job_id)
    ducking_manifest = storage.load_manifest(result.job_id)
    compositions.append(ducking_manifest["cache"]["composition"])

    saved_music = (
        result.manifest_path.parent / ducking_manifest["request"]["background_music"]["path"]
    )
    saved_music.write_bytes(b"music-b")
    pipeline.regenerate(result.job_id)
    final_manifest = storage.load_manifest(result.job_id)
    compositions.append(final_manifest["cache"]["composition"])

    assert len(set(compositions)) == 4
    assert {
        key: value
        for key, value in final_manifest["cache"].items()
        if key not in {"composition", "video_sha256"}
    } == upstream_cache
    assert final_manifest["material_assignments"] == material_assignments
    assert composer.last_music is not None
    assert composer.last_music.volume_percent == 20
    assert composer.last_music.ducking_percent == 40
    assert composer.calls == 4
    assert vocabulary.generate_calls == 1
    assert images.calls == 1
    assert len(speech.calls) == 3
    assert renderer.question_calls == Counter({0: 1})
    assert renderer.answer_calls == Counter({0: 1})


def test_pipeline_persists_and_renders_local_material_with_its_real_suffix(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    material = tmp_path / "material.png"
    Image.new("RGB", (32, 32), "white").save(background)
    Image.new("RGB", (12, 12), "blue").save(material)
    renderer = FakeCardRenderer()
    pipeline = GenerationPipeline(
        storage=JobStorage(tmp_path / "jobs"),
        vocabulary_provider=FakeVocabularyProvider(),
        image_provider=LocalImageProvider(material),
        speech_provider=FakeSpeechProvider(),
        card_renderer=renderer,
        video_composer=FakeComposer(),
    )
    request = GenerationRequest(
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
        background_image=background,
        local_materials=[material],
    )
    request.narration.chinese.enabled = True
    request.narration.chinese.repeats = 1
    request.narration.fast_english.repeats = 0
    request.narration.slow_english.repeats = 0

    result = pipeline.run(request)
    manifest = JobStorage(tmp_path / "jobs").load_manifest(result.job_id)
    material_path = result.manifest_path.parent / manifest["material_assignments"]["0"]["path"]

    assert result.status.value == "complete"
    assert material_path.suffix == ".png"
    assert material_path.is_file()
    assert renderer.answer_materials[0] == material.read_bytes()


def test_pipeline_uses_snapshotted_preview_bytes_without_remote_refetch(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, images, _speech, renderer, _composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
    )
    original_pin = _file(tmp_path / "remote-preview.jpg", b"preview")
    request.pinned_materials = [
        PinnedMaterial(
            entry_index=0,
            asset=MaterialAsset(
                path=original_pin,
                kind=MaterialKind.IMAGE,
                source_id="pexels-42",
            ),
        )
    ]
    request.job_seed = 314159

    result = pipeline.run(request)
    manifest = JobStorage(tmp_path / "jobs").load_manifest(result.job_id)
    assignment = manifest["material_assignments"]["0"]

    assert images.calls == 0
    assert renderer.answer_materials[0] == b"preview"
    assert assignment["path"] == "inputs/pins/000.jpg"
    assert assignment["source"] == "pin"
    assert assignment["source_id"] == "pexels-42"
    assert assignment["kind"] == "image"
    assert assignment["start_offset_seconds"] is None
    assert str(original_pin) not in json.dumps(manifest)

    pipeline.regenerate(result.job_id)

    assert images.calls == 0
    assert renderer.answer_materials[0] == b"preview"


def test_material_style_change_reuses_acquired_bytes_but_rerenders_card(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, images, _speech, renderer, _composer = _pipeline(tmp_path)
    result = pipeline.run(_request(tmp_path))
    storage = JobStorage(tmp_path / "jobs")
    manifest = storage.load_manifest(result.job_id)
    first_assignment = manifest["material_assignments"]["0"].copy()
    manifest["request"]["material"]["width"] += 2
    storage.replace_manifest(result.job_id, manifest)

    pipeline.regenerate(result.job_id)
    updated = storage.load_manifest(result.job_id)

    assert images.calls == 1
    assert updated["material_assignments"]["0"]["fingerprint"] == first_assignment["fingerprint"]
    assert renderer.answer_calls == Counter({0: 2})


def test_local_source_byte_mutation_invalidates_material_acquisition(tmp_path: Path) -> None:
    pipeline, _vocabulary, images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.material.source = MaterialSource.LOCAL
    Image.new("RGB", (12, 12), "red").save(request.local_materials[0])
    result = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first = storage.load_manifest(result.job_id)["material_assignments"]["0"]["fingerprint"]
    saved_source = storage.load_request(result.job_id).local_materials[0]
    Image.new("RGB", (12, 12), "blue").save(saved_source)

    pipeline.regenerate(result.job_id)
    second = storage.load_manifest(result.job_id)["material_assignments"]["0"]["fingerprint"]

    assert images.calls == 0
    assert first != second


def test_composition_audit_changes_when_cached_card_bytes_change(tmp_path: Path) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, _composer = _pipeline(tmp_path)
    result = pipeline.run(_request(tmp_path))
    storage = JobStorage(tmp_path / "jobs")
    manifest = storage.load_manifest(result.job_id)
    first_identity = manifest["cache"]["composition"]
    card = storage.paths(result.job_id).root / manifest["cache"]["cards"]["0"]["answer"]
    card.write_bytes(b"changed-card-bytes")

    pipeline.regenerate(result.job_id)
    updated = storage.load_manifest(result.job_id)

    assert updated["cache"]["composition"] != first_identity
    assert updated["cache"]["video_sha256"] == GenerationPipeline._file_fingerprint(
        storage.paths(result.job_id).root / updated["artifacts"]["video"]
    )


def test_pipeline_rejects_pin_index_before_fetching_materials(tmp_path: Path) -> None:
    pipeline, _vocabulary, images, _speech, _renderer, composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
    )
    request.pinned_materials = [
        PinnedMaterial(
            entry_index=1,
            asset=MaterialAsset(
                path=_file(tmp_path / "orphan-pin.png", b"orphan"),
                kind=MaterialKind.IMAGE,
            ),
        )
    ]

    with pytest.raises(ConfigurationError, match="Pinned material entry index is out of range: 1"):
        pipeline.run(request)

    assert images.calls == 0
    assert composer.calls == 0


def test_changing_snapshotted_pin_bytes_invalidates_material_and_card_cache(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, images, _speech, renderer, _composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
    )
    request.pinned_materials = [
        PinnedMaterial(
            entry_index=0,
            asset=MaterialAsset(
                path=_file(tmp_path / "preview.png", b"preview-a"),
                kind=MaterialKind.IMAGE,
            ),
        )
    ]

    result = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first_manifest = storage.load_manifest(result.job_id)
    first_fingerprint = first_manifest["material_assignments"]["0"]["fingerprint"]
    saved_pin = storage.load_request(result.job_id).pinned_materials[0].asset.path
    saved_pin.write_bytes(b"preview-b")

    pipeline.regenerate(result.job_id)
    second_manifest = storage.load_manifest(result.job_id)

    assert images.calls == 0
    assert renderer.answer_calls == Counter({0: 2})
    assert renderer.answer_materials[0] == b"preview-b"
    assert second_manifest["material_assignments"]["0"]["fingerprint"] != first_fingerprint


def test_pin_assignment_omits_unsafe_source_identifiers(tmp_path: Path) -> None:
    pipeline, *_ = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
    )
    request.pinned_materials = [
        PinnedMaterial(
            entry_index=0,
            asset=MaterialAsset(
                path=_file(tmp_path / "preview.jpg", b"preview"),
                kind=MaterialKind.IMAGE,
                source_id="https://example.test/image?api_key=secret",
            ),
        )
    ]

    result = pipeline.run(request)
    manifest = JobStorage(tmp_path / "jobs").load_manifest(result.job_id)

    assert "source_id" not in manifest["material_assignments"]["0"]
    assert "api_key" not in json.dumps(manifest)


def test_changing_pin_seed_changes_material_fingerprint_without_refetch(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
    )
    request.pinned_materials = [
        PinnedMaterial(
            entry_index=0,
            asset=MaterialAsset(
                path=_file(tmp_path / "preview.png", b"same-preview"),
                kind=MaterialKind.IMAGE,
            ),
        )
    ]
    request.job_seed = 10

    result = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first_manifest = storage.load_manifest(result.job_id)
    first_fingerprint = first_manifest["material_assignments"]["0"]["fingerprint"]
    first_manifest["request"]["job_seed"] = 11
    storage.replace_manifest(result.job_id, first_manifest)

    pipeline.regenerate(result.job_id)
    second_manifest = storage.load_manifest(result.job_id)

    assert images.calls == 0
    assert second_manifest["material_assignments"]["0"]["fingerprint"] != first_fingerprint


def test_pipeline_layers_video_material_with_stable_literal_offset(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    background = tmp_path / "background.png"
    Image.new("RGB", (160, 90), "white").save(background)
    material = _write_video(tmp_path / "material.mp4")
    renderer = FakeCardRenderer()
    composer = FakeComposer()
    pipeline = GenerationPipeline(
        storage=JobStorage(tmp_path / "jobs"),
        vocabulary_provider=FakeVocabularyProvider(),
        image_provider=LocalImageProvider(material),
        speech_provider=FakeSpeechProvider(),
        card_renderer=renderer,
        video_composer=composer,
    )
    request = GenerationRequest(
        entries=[WordEntry(english="apple", phonetic="/apple/", chinese="苹果")],
        background_image=background,
        local_materials=[material],
        job_seed=41,
    )
    request.question.enabled = True
    request.narration.chinese.enabled = True
    request.narration.chinese.repeats = 1
    request.narration.fast_english.repeats = 0
    request.narration.slow_english.repeats = 0

    result = pipeline.run(request)
    first_segments = list(composer.last_segments)
    manifest = JobStorage(tmp_path / "jobs").load_manifest(result.job_id)
    assignment = manifest["material_assignments"]["0"]

    assert assignment["kind"] == "video"
    assert assignment["start_offset_seconds"] == pytest.approx(0.38102068999577143)
    assert renderer.question_layer_calls == Counter({0: 1})
    assert renderer.answer_layer_calls == Counter({0: 1})
    assert len(first_segments) == 2
    assert all(segment.foreground_path is not None for segment in first_segments)
    assert all(segment.material_video is not None for segment in first_segments)
    assert all(
        segment.material_video is not None
        and segment.material_video.start_offset_seconds == pytest.approx(0.38102068999577143)
        for segment in first_segments
    )

    pipeline.regenerate(result.job_id)

    assert all(
        segment.material_video is not None
        and segment.material_video.start_offset_seconds == pytest.approx(0.38102068999577143)
        for segment in composer.last_segments
    )

    replacement = _write_video(tmp_path / "replacement.mp4", duration=0.5)
    pipeline.regenerate(result.job_id, replacements={0: replacement})
    replaced_manifest = JobStorage(tmp_path / "jobs").load_manifest(result.job_id)

    assert replaced_manifest["material_assignments"]["0"]["start_offset_seconds"] == pytest.approx(
        0.19051034499788572
    )
    assert all(
        segment.material_video is not None
        and segment.material_video.start_offset_seconds == pytest.approx(0.19051034499788572)
        for segment in composer.last_segments
    )


def test_automatic_mode_completes_missing_phonetics_once(tmp_path: Path) -> None:
    pipeline, vocabulary, _images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(tmp_path, entries=[WordEntry(english="apple", chinese="苹果")])
    request.phonetic_mode = PhoneticMode.AUTOMATIC

    result = pipeline.run(request)
    pipeline.regenerate(result.job_id)

    assert vocabulary.complete_calls == 1


def test_automatic_mode_omits_ipa_syllable_breaks_from_saved_entries(tmp_path: Path) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="passport", phonetic="/ˈpæs.pɔːrt/", chinese="护照")],
    )
    request.phonetic_mode = PhoneticMode.AUTOMATIC

    result = pipeline.run(request)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"] == [
        {"english": "passport", "phonetic": "/ˈpæspɔːrt/", "chinese": "护照"}
    ]


def test_manual_mode_preserves_user_entered_ipa_syllable_breaks(tmp_path: Path) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[WordEntry(english="passport", phonetic="/ˈpæs.pɔːrt/", chinese="护照")],
    )
    request.phonetic_mode = PhoneticMode.MANUAL

    result = pipeline.run(request)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"] == [
        {"english": "passport", "phonetic": "/ˈpæs.pɔːrt/", "chinese": "护照"}
    ]


def test_disabled_mode_removes_provider_phonetics_from_saved_entries(tmp_path: Path) -> None:
    pipeline, vocabulary, _images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    request.phonetic_mode = PhoneticMode.DISABLED

    result = pipeline.run(request)
    pipeline.regenerate(result.job_id)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"] == [{"english": "apple", "phonetic": "", "chinese": "苹果"}]
    assert vocabulary.generate_calls == 1
    assert vocabulary.complete_calls == 0


def test_regeneration_replaces_one_material_and_preserves_previous_video(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, images, _speech, renderer, _composer = _pipeline(tmp_path)
    request = _request(
        tmp_path,
        entries=[
            WordEntry(english="apple", phonetic="/apple/", chinese="苹果"),
            WordEntry(english="banana", phonetic="/banana/", chinese="香蕉"),
        ],
        count=2,
    )
    first = pipeline.run(request)
    storage = JobStorage(tmp_path / "jobs")
    first_audit = storage.load_manifest(first.job_id)["cache"]["composition"]
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (8, 8), "purple").save(replacement)

    second = pipeline.regenerate(first.job_id, replacements={0: replacement})
    second_audit = storage.load_manifest(first.job_id)["cache"]["composition"]

    assert second.video_path != first.video_path
    assert first.video_path is not None and first.video_path.is_file()
    assert images.calls == 2
    assert renderer.answer_calls == Counter({0: 2, 1: 1})
    assert renderer.answer_materials[0] == replacement.read_bytes()
    assert renderer.answer_materials[1] == b"image-1"
    assert second_audit != first_audit


@pytest.mark.parametrize("source", ["pin", "remote", "local"])
def test_repeated_regeneration_retains_successful_replacements(tmp_path, source):
    pipeline, _vocabulary, _images, _speech, _renderer, _composer = _pipeline(tmp_path)
    request = _request(tmp_path, count=2)
    if source == "pin":
        request.pinned_materials = [
            PinnedMaterial(
                entry_index=0,
                asset=MaterialAsset(path=request.local_materials[0], kind=MaterialKind.IMAGE),
            )
        ]
    elif source == "local":
        request.material.source = MaterialSource.LOCAL
    first = pipeline.run(request)
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (8, 8), "purple").save(replacement)
    second = pipeline.regenerate(first.job_id, replacements={0: replacement})
    third = pipeline.regenerate(first.job_id, replacements={1: request.background_image})
    storage = JobStorage(tmp_path / "jobs")

    def assigned_bytes(index):
        assignment = storage.load_manifest(first.job_id)["material_assignments"][str(index)]
        return (storage.paths(first.job_id).root / assignment["path"]).read_bytes()

    assert assigned_bytes(0) == replacement.read_bytes()
    assert assigned_bytes(1) == request.background_image.read_bytes()
    fourth = pipeline.regenerate(first.job_id)
    assert assigned_bytes(0) == replacement.read_bytes()
    assert assigned_bytes(1) == request.background_image.read_bytes()
    assert all(result.video_path.is_file() for result in [first, second, third, fourth])
    assert len({result.video_path for result in [first, second, third, fourth]}) == 4


def test_failed_regeneration_does_not_commit_a_new_replacement(tmp_path, monkeypatch):
    pipeline, _vocabulary, _images, _speech, _renderer, composer = _pipeline(tmp_path)
    request = _request(tmp_path)
    first = pipeline.run(request)
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (8, 8), "purple").save(replacement)
    pipeline.regenerate(first.job_id, replacements={0: replacement})
    compose = composer.compose

    def fail(*args, **kwargs):
        raise ProviderError("Test composition failed.")

    monkeypatch.setattr(composer, "compose", fail)
    with pytest.raises(ProviderError):
        pipeline.regenerate(first.job_id, replacements={0: request.background_image})
    monkeypatch.setattr(composer, "compose", compose)
    pipeline.regenerate(first.job_id)
    storage = JobStorage(tmp_path / "jobs")
    assignment = storage.load_manifest(first.job_id)["material_assignments"]["0"]
    assert (
        storage.paths(first.job_id).root / assignment["path"]
    ).read_bytes() == replacement.read_bytes()


def test_regeneration_validates_all_replacement_indices_before_state_changes(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, composer = _pipeline(tmp_path)
    result = pipeline.run(_request(tmp_path))
    manifest_path = result.manifest_path
    before = manifest_path.read_bytes()
    replacement = _file(tmp_path / "replacement.png", b"purple")

    with pytest.raises(ValueError, match="index"):
        pipeline.regenerate(result.job_id, replacements={4: replacement})

    assert manifest_path.read_bytes() == before
    assert composer.calls == 1


def test_pipeline_requires_a_background_before_creating_a_job(tmp_path: Path) -> None:
    pipeline, *_ = _pipeline(tmp_path)
    request = GenerationRequest(entries=[WordEntry(english="apple", phonetic="/apple/")])

    with pytest.raises(ConfigurationError, match="background"):
        pipeline.run(request)

    assert not (tmp_path / "jobs").exists()


def test_pipeline_marks_failure_and_preserves_completed_provider_artifacts(
    tmp_path: Path,
) -> None:
    pipeline, _vocabulary, _images, _speech, _renderer, _composer = _pipeline(
        tmp_path, speech=FakeSpeechProvider(fail=True)
    )

    with pytest.raises(ProviderError, match="Speech generation failed"):
        pipeline.run(_request(tmp_path))

    job_root = next((tmp_path / "jobs").iterdir())
    manifest = JobStorage(tmp_path / "jobs").load_manifest(job_root.name)
    assert manifest["status"] == "failed"
    assert manifest["error"] == "Speech generation failed."
    material_paths = list((job_root / "artifacts/materials").iterdir())
    assert len(material_paths) == 1
    assert material_paths[0].read_bytes() == b"image-0"


@pytest.mark.parametrize(
    ("filename", "track", "index", "zh_part", "en_part"),
    [
        ("question.mp3", "question", None, "问题片段朗读", "question narration"),
        ("001-zh.mp3", "zh", 1, "中文朗读", "Chinese narration"),
        ("001-fast.mp3", "fast", 1, "英语快读", "English narration"),
        ("001-slow.mp3", "slow", 1, "英语慢读", "slow English narration"),
    ],
)
def test_speech_failures_identify_track_and_word_in_both_languages_and_manifest(
    tmp_path, monkeypatch, filename, track, index, zh_part, en_part
):
    import aiohttp

    import ai_vocab_video_generator.providers.tts as tts_module
    from ai_vocab_video_generator.i18n import Locale
    from ai_vocab_video_generator.providers.tts import EdgeSpeechProvider
    from ai_vocab_video_generator.webui import _safe_message

    monkeypatch.setattr(tts_module, "sleep", lambda _: None)

    class SelectiveCommunicator:
        async def save(self, target):
            if Path(target).name == filename:
                raise aiohttp.ClientConnectorError(None, ConnectionResetError(54, "private"))
            Path(target).write_bytes(b"audio")

    pipeline, *_ = _pipeline(tmp_path)
    pipeline._speech = EdgeSpeechProvider(lambda *_: SelectiveCommunicator())
    request = _request(tmp_path, count=2)
    request.narration.slow_english.enabled = True
    request.narration.slow_english.repeats = 1
    with pytest.raises(ProviderError) as caught:
        pipeline.run(request)
    error = caught.value
    assert getattr(error, "track", None) == track
    assert getattr(error, "word_index", -1) == index
    assert getattr(error, "attempts", None) == 3
    zh = _safe_message(error, Locale.ZH_CN)
    en = _safe_message(error, Locale.EN_US)
    assert zh_part in zh and en_part in en
    if index is not None:
        assert "第 2 个单词" in zh and "word 2" in en
    else:
        assert "第" not in zh and "word" not in en
    job_root = next((tmp_path / "jobs").iterdir())
    manifest = JobStorage(tmp_path / "jobs").load_manifest(job_root.name)
    assert manifest["status"] == "failed"
    assert en_part in manifest["error"]
    assert "private" not in manifest["error"]
    assert not (job_root / "artifacts/audio" / filename).exists()


def test_pipeline_marks_interrupted_composition_failed_and_removes_partial_video(
    tmp_path: Path,
) -> None:
    class InterruptingComposer(FakeComposer):
        def compose(
            self,
            segments: Sequence[VideoSegment],
            destination: Path,
            *,
            render: RenderSettings,
            music: BackgroundMusicSettings,
        ) -> Path:
            del segments, render, music
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"partial video")
            raise KeyboardInterrupt

    storage = JobStorage(tmp_path / "jobs")
    pipeline = GenerationPipeline(
        storage=storage,
        vocabulary_provider=FakeVocabularyProvider(),
        image_provider=FakeImageProvider(),
        speech_provider=FakeSpeechProvider(),
        card_renderer=FakeCardRenderer(),
        video_composer=InterruptingComposer(),
    )

    with pytest.raises(KeyboardInterrupt):
        pipeline.run(_request(tmp_path))

    job_root = next((tmp_path / "jobs").iterdir())
    manifest = storage.load_manifest(job_root.name)
    assert manifest["status"] == "failed"
    assert manifest["error"] == "Video generation was interrupted."
    assert list((job_root / "artifacts" / "videos").glob("*.partial.mp4")) == []


def test_manifest_records_nonsecret_fingerprints_and_active_video(tmp_path: Path) -> None:
    pipeline, *_ = _pipeline(tmp_path)
    result = pipeline.run(_request(tmp_path))

    manifest = JobStorage(tmp_path / "jobs").load_manifest(result.job_id)

    assert len(manifest["cache"]["vocabulary"]) == 64
    assert manifest["artifacts"]["video"].endswith("video-0001.mp4")
    assert manifest["artifacts"]["videos"] == [manifest["artifacts"]["video"]]
    assert isinstance(manifest["material_assignments"], Mapping)
