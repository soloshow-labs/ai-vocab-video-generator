from ai_vocab_video_generator.providers.edge_voices import (
    EDGE_TTS_VOICE_BY_NAME,
    EDGE_TTS_VOICES,
    edge_voices_for_language,
)


def test_catalog_contains_every_current_chinese_and_english_edge_voice() -> None:
    chinese = edge_voices_for_language("zh")
    english = edge_voices_for_language("en")

    assert len(chinese) == 14
    assert len(english) == 47
    assert len(EDGE_TTS_VOICES) == 61
    assert len(EDGE_TTS_VOICE_BY_NAME) == len(EDGE_TTS_VOICES)
    assert "zh-CN-XiaoxiaoNeural" in EDGE_TTS_VOICE_BY_NAME
    assert "en-US-JennyNeural" in EDGE_TTS_VOICE_BY_NAME


def test_voice_display_name_removes_only_locale_and_neural_suffix() -> None:
    assert EDGE_TTS_VOICE_BY_NAME["en-US-AvaMultilingualNeural"].display_name == ("AvaMultilingual")
