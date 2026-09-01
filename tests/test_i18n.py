import pytest

from ai_vocab_video_generator.i18n import (
    DEFAULT_LOCALE,
    Locale,
    catalog_keys,
    translate,
)


def test_translation_catalogs_have_identical_keys() -> None:
    assert DEFAULT_LOCALE is Locale.ZH_CN
    assert catalog_keys(Locale.ZH_CN) == catalog_keys(Locale.EN_US)
    assert translate(Locale.ZH_CN, "generate_video") == "生成视频"
    assert translate(Locale.EN_US, "generate_video") == "Generate Video"


def test_complete_media_and_import_catalog_is_bilingual() -> None:
    expected = {
        "render_settings",
        "fps",
        "fit_mode",
        "candidate_pool_size",
        "contain",
        "cover",
        "stretch",
        "video_upload_help",
        "material_uploads_mixed",
        "background_music_settings",
        "background_music_enabled",
        "background_music_file",
        "music_volume",
        "music_ducking",
        "question_narration",
        "material_search",
        "material_search_again",
        "material_current_word",
        "material_use",
        "material_selected",
        "material_upload_help",
        "progress_vocabulary",
        "progress_materials",
        "progress_narration",
        "progress_cards",
        "progress_composing",
        "progress_complete",
    }

    assert expected <= catalog_keys(Locale.ZH_CN)
    assert expected <= catalog_keys(Locale.EN_US)


def test_every_settings_panel_has_bilingual_visible_guidance() -> None:
    expected = {
        "basic_settings_help",
        "topic_settings_help",
        "background_settings_help",
        "phonetic_settings_help",
        "video_settings_help",
        "material_source_help",
        "selection_mode_help",
        "aspect_help",
        "fps_help",
        "candidate_pool_size_help",
        "question_settings_help",
        "progress_settings_help",
        "material_settings_help",
        "fit_mode_help",
        "narration_settings_help",
        "question_narration_settings_help",
        "text_settings_help",
        "background_music_settings_help",
        "regeneration_settings_help",
    }

    assert expected <= catalog_keys(Locale.ZH_CN)
    assert expected <= catalog_keys(Locale.EN_US)


def test_unknown_translation_key_is_a_programmer_error() -> None:
    with pytest.raises(KeyError, match="missing-key"):
        translate(Locale.ZH_CN, "missing-key")
