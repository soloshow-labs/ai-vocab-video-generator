from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_vocab_video_generator.domain import (
    MAX_TOPIC_LENGTH,
    MAX_VOCABULARY_ENTRIES,
    AnchorOffsets,
    BackgroundMusicSettings,
    GenerationRequest,
    MaterialFitMode,
    PhoneticMode,
    RenderSettings,
    VideoAspect,
    VocabularySettings,
    WordEntry,
)


def test_word_entry_normalizes_whitespace() -> None:
    entry = WordEntry(english="  apple  ", phonetic=" /ˈæp.əl/ ", chinese=" 苹果 ")

    assert entry.english == "apple"
    assert entry.phonetic == "/ˈæp.əl/"
    assert entry.chinese == "苹果"


def test_material_queries_normalize_and_reject_invalid_indices():
    request = GenerationRequest(
        entries=[WordEntry(english="bank")], material_queries={0: " river   bank "}
    )
    assert request.material_queries == {0: "river bank"}
    for queries in ({-1: "bank"}, {1: "bank"}, {0: "x" * 121}):
        with pytest.raises(ValidationError):
            GenerationRequest(entries=[WordEntry(english="bank")], material_queries=queries)
    request = GenerationRequest(entries=[WordEntry(english="bank")], material_queries={0: " "})
    assert request.material_queries == {}


def test_word_entry_rejects_blank_english_text() -> None:
    with pytest.raises(ValidationError):
        WordEntry(english="   ", chinese="空")


def test_video_aspect_has_deterministic_resolution() -> None:
    assert VideoAspect.PORTRAIT.resolution == (1080, 1920)
    assert VideoAspect.LANDSCAPE.resolution == (1920, 1080)


def test_generation_request_requires_topic_or_entries() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(topic="", entries=[])


def test_generation_request_bounds_topic_and_entry_count() -> None:
    GenerationRequest(topic="x" * MAX_TOPIC_LENGTH)
    GenerationRequest(entries=[WordEntry(english="word") for _ in range(MAX_VOCABULARY_ENTRIES)])

    with pytest.raises(ValidationError):
        GenerationRequest(topic="x" * (MAX_TOPIC_LENGTH + 1))
    with pytest.raises(ValidationError):
        GenerationRequest(
            entries=[WordEntry(english="word") for _ in range(MAX_VOCABULARY_ENTRIES + 1)]
        )


def test_portrait_defaults_match_visible_controls() -> None:
    request = GenerationRequest(entries=[WordEntry(english="test", chinese="测试")])

    assert request.canvas.model_dump(mode="json") == {
        "aspect": "portrait",
        "width": 1080,
        "height": 1920,
    }
    assert request.phonetic_mode is PhoneticMode.AUTOMATIC
    assert request.question.enabled is False
    assert request.question.fill_color == "#000000"
    assert request.question.stroke_color == "#FFFFFF"
    assert request.question.font_size == 80
    assert request.english_text.font_size == 100
    assert request.phonetic_text.font_size == 90
    assert request.chinese_text.font_size == 80
    for style in (request.english_text, request.phonetic_text, request.chinese_text):
        assert style.fill_color == "#000000"
        assert style.stroke_color == "#FFFFFF"
        assert style.stroke_width == 1.5
    assert request.material.shape.value == "circle"
    assert request.progress.start_color == "#FFA500"
    assert request.progress.end_color == "#ADFF2F"
    assert request.narration.chinese.repeats == 0
    assert request.narration.chinese.enabled is False
    assert request.narration.chinese.voice == "zh-CN-XiaoxiaoNeural"
    assert request.narration.fast_english.repeats == 1
    assert request.narration.fast_english.enabled is True
    assert request.narration.fast_english.rate == -20
    assert request.narration.slow_english.repeats == 0
    assert request.narration.slow_english.enabled is False


def test_landscape_defaults_move_every_layout_element() -> None:
    request = GenerationRequest.with_aspect_defaults(
        VideoAspect.LANDSCAPE,
        entries=[WordEntry(english="test", chinese="测试")],
    )

    assert (request.canvas.width, request.canvas.height) == (1920, 1080)
    assert request.question.offsets.top == 50
    assert request.material.offsets.top == 200
    assert request.english_text.offsets.top == 670
    assert request.phonetic_text.offsets.top == 770
    assert request.chinese_text.offsets.top == 880
    assert request.progress.offsets.top == 880


def test_anchor_resolves_all_edges_and_rejects_empty_layout_box() -> None:
    assert AnchorOffsets().resolve((1080, 1920), (180, 100)) == (450, 910)
    assert AnchorOffsets(left=20, top=30).resolve((1080, 1920), (180, 100)) == (20, 30)
    assert AnchorOffsets(right=40, bottom=50).resolve((1080, 1920), (180, 100)) == (
        860,
        1770,
    )
    assert AnchorOffsets(left=20, right=40, top=30, bottom=50).resolve(
        (1080, 1920), (180, 100)
    ) == (440, 900)

    with pytest.raises(ValueError, match="layout box"):
        AnchorOffsets(left=600, right=600).resolve((1080, 1920), (100, 100))


def test_generation_request_requires_a_positive_answer_repeat() -> None:
    request = GenerationRequest(entries=[WordEntry(english="test", chinese="测试")])
    request.narration.chinese.repeats = 0
    request.narration.fast_english.repeats = 0
    request.narration.slow_english.repeats = 0

    with pytest.raises(ValidationError, match="narration"):
        GenerationRequest.model_validate(request.model_dump())


def test_schema_v3_media_defaults_and_bounds() -> None:
    request = GenerationRequest(entries=[WordEntry(english="apple")])

    assert request.render.fps == 24
    assert request.background_music.enabled is False
    assert request.background_music.volume_percent == 12
    assert request.background_music.ducking_percent == 65
    assert request.material.fit_mode is MaterialFitMode.COVER
    assert request.narration.question.voice == "en-US-JennyNeural"
    assert request.narration.question.repeats == 1
    with pytest.raises(ValidationError):
        RenderSettings(fps=61)


def test_enabled_music_requires_an_existing_supported_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="music"):
        BackgroundMusicSettings(enabled=True, path=tmp_path / "missing.mp3")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://session-user:session-password@example.test/v1",
        "https://example.test/v1?api_key=not-a-live-value",
        "https://example.test/v1?access-token=not-a-live-value",
        "https://example.test/v1?client_secret=not-a-live-value",
        "https://example.test/v1?x-api-key=not-a-live-value",
        "https://example.test/v1?redirect=sk-" + "A" * 48,
    ],
    ids=[
        "userinfo",
        "api-key-query",
        "access-token-query",
        "client-secret-query",
        "x-api-key-query",
        "token-query-value",
    ],
)
def test_vocabulary_settings_reject_credential_bearing_base_urls(base_url: str) -> None:
    with pytest.raises(ValidationError, match="credentials"):
        VocabularySettings(base_url=base_url)


def test_vocabulary_settings_retains_nonsecret_compatible_query_options() -> None:
    settings = VocabularySettings(base_url="https://example.test/openai/v1?api-version=2026-01-01")

    assert settings.base_url.endswith("api-version=2026-01-01")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.test/v1?authkey=opaque-private-value",
        "https://example.test/v1?session=opaque-private-value",
    ],
    ids=["unknown-auth-query", "opaque-session-query"],
)
def test_vocabulary_settings_rejects_unapproved_query_parameters(base_url: str) -> None:
    with pytest.raises(ValidationError, match="query parameters"):
        VocabularySettings(base_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.test/v1/sk%2Dproj%2D" + "P" * 40,
        "https://example.test/v1#api_key=sk-" + "F" * 48,
        "https://example.test/v1#client_secret=not-a-live-value",
        "https://example.test/v1/api_key/not-a-live-value",
    ],
    ids=[
        "encoded-path-token",
        "fragment-key-and-token",
        "fragment-key-marker",
        "path-key-marker",
    ],
)
def test_vocabulary_settings_rejects_secret_tokens_outside_the_query(
    base_url: str,
) -> None:
    with pytest.raises(ValidationError, match="credentials"):
        VocabularySettings(base_url=base_url)


def test_vocabulary_settings_rejects_every_fragment_without_echoing_it() -> None:
    private_marker = "opaque-private-fragment"
    base_url = f"https://example.test/v1#session={private_marker}"

    with pytest.raises(ValidationError, match="fragments") as caught:
        VocabularySettings(base_url=base_url)

    assert private_marker not in str(caught.value)


def test_vocabulary_settings_rejects_nested_fragment_assignment_without_echoing_it() -> None:
    private_marker = "private-" + "round-three-marker"
    base_url = f"https://example.test/v1#redirect=api_key={private_marker}"

    with pytest.raises(ValidationError) as caught:
        VocabularySettings(base_url=base_url)

    for rendered in (str(caught.value), repr(caught.value)):
        assert private_marker not in rendered
        assert base_url not in rendered
        assert "input_value" not in rendered
        assert "api_key=" not in rendered


def test_generation_request_does_not_reintroduce_nested_url_input_in_errors() -> None:
    private_marker = "private-" + "nested-request-marker"
    base_url = f"https://example.test/v1#redirect=api_key={private_marker}"

    with pytest.raises(ValidationError) as caught:
        GenerationRequest(
            entries=[WordEntry(english="apple")],
            vocabulary={"base_url": base_url},
        )

    for rendered in (str(caught.value), repr(caught.value)):
        assert private_marker not in rendered
        assert base_url not in rendered
        assert "input_value" not in rendered


def test_vocabulary_settings_allows_benign_sk_projection_model_path() -> None:
    base_url = "https://example.test/models/sk-projection-techniques-explained"

    assert VocabularySettings(base_url=base_url).base_url == base_url
