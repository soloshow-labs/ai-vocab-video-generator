"""Bilingual Streamlit presentation layer."""

import hashlib
import json
import re
import secrets
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlparse
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageDraw
from pydantic import SecretStr, ValidationError
from streamlit.runtime.uploaded_file_manager import UploadedFile

from ai_vocab_video_generator import __version__
from ai_vocab_video_generator.config import AppSettings, LLMPreset, LLMSettings
from ai_vocab_video_generator.domain import (
    MAX_SCRIPT_LENGTH,
    MAX_TOPIC_LENGTH,
    AnchorOffsets,
    BackgroundMusicSettings,
    CanvasSettings,
    GenerationRequest,
    MaterialFitMode,
    MaterialKind,
    MaterialShape,
    MaterialSource,
    MaterialStyle,
    NarrationSettings,
    NarrationTrackSettings,
    PhoneticMode,
    PinnedMaterial,
    PipelineProgress,
    PipelineStage,
    ProgressBarStyle,
    RemoteMaterialProvider,
    RenderSettings,
    SelectionMode,
    TextElementStyle,
    VideoAspect,
    VocabularySettings,
    WordEntry,
)
from ai_vocab_video_generator.errors import (
    ApplicationError,
    SpeechGenerationError,
    SpeechTrack,
    UploadSizeError,
)
from ai_vocab_video_generator.i18n import DEFAULT_LOCALE, Locale, translate
from ai_vocab_video_generator.media_limits import (
    MAX_LOCAL_AUDIO_BYTES,
    MAX_LOCAL_IMAGE_BYTES,
    MAX_LOCAL_VIDEO_BYTES,
    MIB,
)
from ai_vocab_video_generator.pipeline import GenerationPipeline
from ai_vocab_video_generator.preview import CandidateGallery, WordMaterialState, remote_search_key
from ai_vocab_video_generator.private_fs import (
    ensure_private_directory,
    mark_private_file,
    write_private_bytes,
)
from ai_vocab_video_generator.providers.asr import FunASRTranscriptionProvider
from ai_vocab_video_generator.providers.base import (
    ImageProvider,
    ImageSelectionContext,
    VocabularyProvider,
)
from ai_vocab_video_generator.providers.edge_voices import (
    EDGE_TTS_VOICE_BY_NAME,
    edge_voices_for_language,
)
from ai_vocab_video_generator.providers.images import (
    LocalImageProvider,
    PexelsImageProvider,
    PixabayImageProvider,
    RemoteImageCandidate,
    _candidate_index,
    extract_seeded_video_frame,
    probe_material,
)
from ai_vocab_video_generator.providers.llm import OpenAICompatibleVocabularyProvider
from ai_vocab_video_generator.providers.tts import EdgeSpeechProvider
from ai_vocab_video_generator.rendering.cards import CardRenderer
from ai_vocab_video_generator.rendering.video import VideoComposer
from ai_vocab_video_generator.script import (
    ScriptFormatError,
    parse_vocabulary_script,
    serialize_vocabulary_script,
)
from ai_vocab_video_generator.storage import JobStorage

_APP_STYLES = """
<style>
.st-key-material_gallery [class*="st-key-candidate_card_"] [data-testid="stImage"] img {
    aspect-ratio: 3 / 2;
    object-fit: cover;
}
.st-key-material_gallery [class*="st-key-candidate_card_selected_"] {
    outline: 2px solid #0068c9;
    border-radius: 0.5rem;
}
.st-key-material_gallery [class*="st-key-candidate_card_selected_"] button:disabled {
    background: #0068c9;
    color: white;
    opacity: 1;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 2.25rem;
    padding-bottom: 3rem;
}
[data-testid="stHeadingWithActionElements"] h1 {
    line-height: 1.25;
    overflow: visible;
    padding-top: 0.2rem;
    padding-bottom: 0.35rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0.5rem;
}
span.stMarkdownColoredText[style*="189, 64, 67"] {
    color: #ff4b4b !important;
    font-weight: 600;
}
[data-testid="stExpander"] summary {
    cursor: pointer;
}
[data-testid="stExpander"] summary p {
    font-weight: 700;
    text-decoration: none !important;
}
[data-testid="stExpander"] summary span.stMarkdownColoredText[style*="0, 84, 163"] {
    color: #0068c9 !important;
    text-decoration: none !important;
}
[data-testid="stExpander"] summary:hover p {
    color: #ff1f2d !important;
}
[data-testid="stExpander"] summary:hover
span.stMarkdownColoredText[style*="0, 84, 163"] {
    color: #0068c9 !important;
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.25rem;
    }
}
</style>
"""


class UploadedData(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def _t(locale: Locale, key: str) -> str:
    return translate(locale, key)


_PROGRESS_TRANSLATION_KEYS = {
    PipelineStage.PREPARING: "preparing",
    PipelineStage.VOCABULARY: "progress_vocabulary",
    PipelineStage.IMAGES: "progress_materials",
    PipelineStage.SPEECH: "progress_narration",
    PipelineStage.CARDS: "progress_cards",
    PipelineStage.COMPOSING: "progress_composing",
    PipelineStage.COMPLETE: "progress_complete",
}


def _localized_progress_message(locale: Locale, progress: PipelineProgress) -> str:
    return _t(locale, _PROGRESS_TRANSLATION_KEYS[progress.stage])


def _close_provider(provider: object) -> None:
    close = getattr(provider, "close", None)
    if callable(close):
        with suppress(OSError, RuntimeError):
            close()


def _stable_translated_choice(
    value: object,
    choices: dict[str, str],
    *,
    default: str,
) -> str:
    text = str(value)
    if text in choices:
        return text
    for candidate_locale in Locale:
        for stable_value, translation_key in choices.items():
            if text == _t(candidate_locale, translation_key):
                return stable_value
    return default


def _normalize_translated_widget_state() -> None:
    translated_choices = {
        "aspect": (
            {aspect.value: aspect.value for aspect in VideoAspect},
            VideoAspect.PORTRAIT.value,
        ),
        "material_source": (
            {
                MaterialSource.LOCAL.value: "local_uploads",
                MaterialSource.REMOTE.value: "remote_search",
            },
            MaterialSource.REMOTE.value,
        ),
        "selection_mode": (
            {mode.value: mode.value for mode in SelectionMode},
            SelectionMode.SEQUENTIAL.value,
        ),
        "material_shape": (
            {shape.value: shape.value for shape in MaterialShape},
            MaterialShape.CIRCLE.value,
        ),
        "material_fit_mode": (
            {mode.value: mode.value for mode in MaterialFitMode},
            MaterialFitMode.COVER.value,
        ),
    }
    for widget_key, (choices, default) in translated_choices.items():
        if widget_key in st.session_state:
            st.session_state[widget_key] = _stable_translated_choice(
                st.session_state[widget_key], choices, default=default
            )


def _normalize_word_index_state(widget_key: str, labels: list[str]) -> None:
    value = st.session_state.get(widget_key, 0)
    if isinstance(value, str):
        index = labels.index(value) if value in labels else 0
    elif isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(labels):
        index = value
    else:
        index = 0
    # Selectboxes send labels over the wire. Explicit assignment both repairs
    # stale labels and sends the current label after the vocabulary changes.
    st.session_state[widget_key] = index


def _on_locale_changed() -> None:
    for widget_key in (
        "aspect_widget",
        "material_source_widget",
        "selection_mode_widget",
        "material_shape_widget",
        "material_fit_mode_widget",
    ):
        st.session_state.pop(widget_key, None)


def _initialize_state(settings: AppSettings) -> None:
    if "draft_job_seed" not in st.session_state:
        st.session_state["draft_job_seed"] = secrets.randbits(63) or 1
    initial_preset = LLMPreset(str(st.session_state.get("llm_preset", settings.llm.preset.value)))
    initial_base_url = str(st.session_state.get("llm_base_url", settings.llm.base_url))
    defaults: dict[str, Any] = {
        "locale": DEFAULT_LOCALE.value,
        "llm_preset": settings.llm.preset.value,
        "llm_base_url": settings.llm.base_url,
        "llm_model": settings.llm.model,
        "auto_phonetic": True,
        "manual_phonetic": False,
        "aspect": VideoAspect.PORTRAIT.value,
        "question_text": "What is this?",
        "question_top": 240,
        "question_font_size": 80,
        "material_width": 648,
        "material_height": 648,
        "material_top": 384,
        "english_top": 1052,
        "english_font_size": 100,
        "phonetic_top": 1252,
        "phonetic_font_size": 90,
        "chinese_top": 1422,
        "chinese_font_size": 80,
        "progress_top": 1422,
        "script_text": "",
        "fps": 24,
        "material_fit_mode": MaterialFitMode.COVER.value,
        "material_shape": MaterialShape.CIRCLE.value,
        "material_pool_size": 8,
        "material_source": MaterialSource.REMOTE.value,
        "selection_mode": SelectionMode.SEQUENTIAL.value,
        "music_enabled": False,
        "music_volume": 12,
        "music_ducking": 65,
        "llm_key_input": "",
        "llm_credentials": {},
        "active_llm_preset": initial_preset.value,
        "active_llm_credential_slot": _llm_credential_slot(initial_preset, initial_base_url),
        "provider_credentials": {},
        "active_remote_provider": str(
            st.session_state.get("remote_provider", RemoteMaterialProvider.PEXELS.value)
        ),
        "task_id": str(st.session_state.get("last_job_id", "")),
        "music_upload_shadow": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    _normalize_translated_widget_state()


def _on_llm_preset_changed() -> None:
    previous = str(st.session_state.get("active_llm_credential_slot", LLMPreset.OPENAI.value))
    credentials = dict(st.session_state.get("llm_credentials", {}))
    credentials[previous] = str(st.session_state.get("llm_key_input", ""))
    preset = LLMPreset(str(st.session_state["llm_preset"]))
    defaults = LLMSettings.for_preset(preset)
    slot = _llm_credential_slot(preset, defaults.base_url)
    st.session_state["llm_credentials"] = credentials
    st.session_state["active_llm_preset"] = preset.value
    st.session_state["active_llm_credential_slot"] = slot
    st.session_state["llm_key_input"] = str(credentials.get(slot, ""))
    st.session_state["llm_base_url"] = defaults.base_url
    st.session_state["llm_model"] = defaults.model


def _llm_credential_slot(preset: LLMPreset, base_url: str) -> str:
    if preset is not LLMPreset.CUSTOM:
        return preset.value
    try:
        parsed = urlparse(base_url.strip())
        host = (parsed.hostname or "invalid").casefold()
        port_value = parsed.port
    except ValueError:
        return "custom:invalid"
    port = f":{port_value}" if port_value is not None else ""
    return f"custom:{parsed.scheme.casefold()}://{host}{port}"


def _on_llm_base_url_changed() -> None:
    preset = LLMPreset(str(st.session_state.get("llm_preset", LLMPreset.OPENAI.value)))
    if preset is not LLMPreset.CUSTOM:
        return
    previous = str(st.session_state.get("active_llm_credential_slot", preset.value))
    credentials = dict(st.session_state.get("llm_credentials", {}))
    credentials[previous] = str(st.session_state.get("llm_key_input", ""))
    slot = _llm_credential_slot(preset, str(st.session_state.get("llm_base_url", "")))
    st.session_state["llm_credentials"] = credentials
    st.session_state["active_llm_credential_slot"] = slot
    st.session_state["llm_key_input"] = str(credentials.get(slot, ""))


def _remember_provider_credential(provider: str) -> None:
    credentials = dict(st.session_state.get("provider_credentials", {}))
    credentials[provider] = str(st.session_state.get(f"{provider}_key_input", ""))
    st.session_state["provider_credentials"] = credentials


def _remember_llm_credential() -> None:
    preset = str(st.session_state.get("active_llm_credential_slot", LLMPreset.OPENAI.value))
    credentials = dict(st.session_state.get("llm_credentials", {}))
    credentials[preset] = str(st.session_state.get("llm_key_input", ""))
    st.session_state["llm_credentials"] = credentials


def _on_remote_provider_changed() -> None:
    previous = str(
        st.session_state.get("active_remote_provider", RemoteMaterialProvider.PEXELS.value)
    )
    _remember_provider_credential(previous)
    st.session_state["active_remote_provider"] = str(st.session_state["remote_provider"])


def _remember_conditional_value(widget_key: str, shadow_key: str) -> None:
    st.session_state[shadow_key] = st.session_state[widget_key]


def _restore_conditional_value(widget_key: str, shadow_key: str) -> None:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[shadow_key]


def _on_music_upload_changed() -> None:
    st.session_state["music_upload_shadow"] = st.session_state.get("music_upload_widget")


def _on_auto_phonetic_changed() -> None:
    if st.session_state.get("auto_phonetic"):
        st.session_state["manual_phonetic"] = False


def _on_manual_phonetic_changed() -> None:
    if st.session_state.get("manual_phonetic"):
        st.session_state["auto_phonetic"] = False


def _on_aspect_changed() -> None:
    aspect = VideoAspect(
        _stable_translated_choice(
            st.session_state["aspect_widget"],
            {candidate.value: candidate.value for candidate in VideoAspect},
            default=VideoAspect.PORTRAIT.value,
        )
    )
    st.session_state["aspect"] = aspect.value
    portrait = aspect is VideoAspect.PORTRAIT
    values = {
        "question_top": 240 if portrait else 50,
        "question_font_size": 80,
        "question_style_top": 240 if portrait else 50,
        "question_style_font_size": 80,
        "material_width": 648 if portrait else 400,
        "material_height": 648 if portrait else 400,
        "material_top": 384 if portrait else 200,
        "english_top": 1052 if portrait else 670,
        "english_font_size": 100,
        "phonetic_top": 1252 if portrait else 770,
        "phonetic_font_size": 90 if portrait else 80,
        "chinese_top": 1422 if portrait else 880,
        "chinese_font_size": 80 if portrait else 60,
        "progress_top": 1422 if portrait else 880,
    }
    st.session_state.update(values)


def _phonetic_mode() -> PhoneticMode:
    if st.session_state.get("manual_phonetic"):
        return PhoneticMode.MANUAL
    if st.session_state.get("auto_phonetic"):
        return PhoneticMode.AUTOMATIC
    return PhoneticMode.DISABLED


def _available_secret(entered: str, configured: SecretStr | None) -> SecretStr | None:
    if entered:
        return SecretStr(entered)
    return configured


def _storage_secrets(settings: AppSettings) -> tuple[str, ...]:
    values = list(settings.secrets.values())
    for state_key in ("llm_credentials", "provider_credentials"):
        stored = st.session_state.get(state_key, {})
        if isinstance(stored, dict):
            values.extend(str(value) for value in stored.values() if value)
    values.extend(
        str(st.session_state.get(key, ""))
        for key in ("llm_key_input", "pexels_key_input", "pixabay_key_input")
        if st.session_state.get(key)
    )
    return tuple(values)


def _job_storage(settings: AppSettings) -> JobStorage:
    return JobStorage(settings.storage_dir, active_secrets=_storage_secrets(settings))


def _save_upload(
    upload: UploadedData,
    root: Path,
    fallback_suffix: str,
    *,
    allowed_suffixes: frozenset[str] | None = None,
) -> Path:
    candidate = Path(upload.name).suffix.lower()
    suffix = (
        candidate
        if candidate and (allowed_suffixes is None or candidate in allowed_suffixes)
        else fallback_suffix
    )
    size_limit = (
        MAX_LOCAL_VIDEO_BYTES
        if suffix in {".mp4", ".mov", ".m4v", ".webm"}
        else MAX_LOCAL_AUDIO_BYTES
        if suffix in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
        else MAX_LOCAL_IMAGE_BYTES
    )
    declared_size = getattr(upload, "size", None)
    if isinstance(declared_size, int) and declared_size > size_limit:
        raise UploadSizeError(declared_size, size_limit)
    contents = upload.getvalue()
    if len(contents) > size_limit:
        raise UploadSizeError(len(contents), size_limit)
    destination = root / "_session_uploads" / f"{uuid4().hex}{suffix}"
    return write_private_bytes(destination, contents)


def _save_uploads(
    uploads: Sequence[UploadedData] | None,
    root: Path,
    fallback_suffix: str = ".jpg",
    *,
    allowed_suffixes: frozenset[str] | None = None,
) -> list[Path]:
    selected = list(uploads or [])
    if len(selected) > 50:
        raise ValueError("Select at most 50 uploaded files.")
    declared_total = sum(
        size for upload in selected if isinstance((size := getattr(upload, "size", None)), int)
    )
    if declared_total > 512 * 1024 * 1024:
        raise ValueError("The selected uploads exceed the aggregate size limit.")
    return [
        _save_upload(
            upload,
            root,
            fallback_suffix,
            allowed_suffixes=allowed_suffixes,
        )
        for upload in selected
    ]


def _qualified_label(locale: Locale, prefix_key: str, suffix_key: str) -> str:
    separator = "" if locale == Locale.ZH_CN else " "
    return f"{_t(locale, prefix_key)}{separator}{_t(locale, suffix_key)}"


def _parenthetical_label(locale: Locale, label: str, detail: str) -> str:
    if locale == Locale.ZH_CN:
        return f"{label} ({detail})"
    return f"{label} ({detail})"


def _offset_fields(
    prefix: str,
    locale: Locale,
    *,
    top: int | None,
    label_prefix_key: str | None = None,
    bottom: int | None = None,
    left: int | None = None,
    right: int | None = None,
    disabled: bool = False,
) -> dict[str, int | None]:
    columns = st.columns(4)
    defaults = (top, bottom, left, right)
    names = ("top", "bottom", "left", "right")
    result: dict[str, int | None] = {}
    for column, name, default in zip(columns, names, defaults, strict=True):
        with column:
            key = f"{prefix}_{name}"
            arguments: dict[str, Any] = {
                "min_value": -1,
                "max_value": 4096,
                "step": 1,
                "key": key,
                "help": "-1 = 自动 / auto",
                "disabled": disabled,
            }
            if key not in st.session_state:
                arguments["value"] = default if default is not None else -1
            label = _t(locale, name)
            if label_prefix_key is not None:
                label = _qualified_label(locale, label_prefix_key, f"{name}_margin_suffix")
            raw = st.number_input(label, **arguments)
        result[name] = None if raw < 0 else int(raw)
    return result


def _panel_heading(
    locale: Locale,
    key: str,
    *,
    help_key: str | None = None,
    show_help: bool = True,
) -> None:
    """Render a compact, always-open section heading."""
    st.markdown(f"**{_t(locale, key)}**")
    if show_help:
        st.caption(_t(locale, help_key or f"{key}_help"))


def _field_help(locale: Locale, key: str) -> None:
    """Keep high-value setting guidance visible without overloading field labels."""
    st.caption(_t(locale, f"{key}_help"))


def _text_style_fields(
    prefix: str,
    locale: Locale,
    *,
    enabled: bool,
    font_path: str,
    font_size: int,
    fill_color: str,
    stroke_color: str,
    top: int,
    label_prefix_key: str,
    show_enabled: bool = True,
) -> dict[str, Any]:
    values: dict[str, Any] = {"enabled": enabled}
    if show_enabled:
        values["enabled"] = st.checkbox(
            _t(locale, "enabled"), value=enabled, key=f"{prefix}_enabled"
        )
    controls_disabled = not bool(values["enabled"])
    values["font_path"] = st.text_input(
        _qualified_label(locale, label_prefix_key, "font_suffix"),
        value=font_path,
        key=f"{prefix}_font_path",
        disabled=controls_disabled,
    )
    font_size_key = f"{prefix}_font_size"
    font_size_arguments: dict[str, Any] = {
        "min_value": 8,
        "max_value": 400,
        "step": 1,
        "key": font_size_key,
        "disabled": controls_disabled,
    }
    if font_size_key not in st.session_state:
        font_size_arguments["value"] = font_size
    values["font_size"] = st.slider(
        _qualified_label(locale, label_prefix_key, "font_size_suffix"),
        **font_size_arguments,
    )
    first, second = st.columns([0.3, 0.7])
    with first:
        values["fill_color"] = st.color_picker(
            _qualified_label(locale, label_prefix_key, "font_color_suffix"),
            value=fill_color,
            key=f"{prefix}_fill_color",
            disabled=controls_disabled,
        )
    with second:
        values["weight"] = st.slider(
            _qualified_label(locale, label_prefix_key, "font_weight_suffix"),
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key=f"{prefix}_weight",
            disabled=controls_disabled,
        )
    first, second = st.columns([0.3, 0.7])
    with first:
        values["stroke_color"] = st.color_picker(
            _qualified_label(locale, label_prefix_key, "stroke_color_suffix"),
            value=stroke_color,
            key=f"{prefix}_stroke_color",
            disabled=controls_disabled,
        )
    with second:
        values["stroke_width"] = st.slider(
            _qualified_label(locale, label_prefix_key, "stroke_weight_suffix"),
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            key=f"{prefix}_stroke_width",
            disabled=controls_disabled,
        )
    values["offsets"] = _offset_fields(
        prefix,
        locale,
        top=top,
        label_prefix_key=label_prefix_key,
        disabled=controls_disabled,
    )
    return values


def _edge_voice_label(short_name: str, locale: Locale) -> str:
    voice = EDGE_TTS_VOICE_BY_NAME.get(short_name)
    if voice is None:
        return short_name
    gender = (
        _t(locale, "voice_gender_female")
        if voice.gender == "Female"
        else _t(locale, "voice_gender_male")
    )
    return f"{voice.locale} · {voice.display_name} · {gender}"


def _narration_panel(
    title_key: str,
    prefix: str,
    locale: Locale,
    settings: AppSettings,
    *,
    voice: str,
    repeats: int,
    rate: int,
    sample: str,
    field_prefix_key: str,
    voice_language: Literal["zh", "en"],
    enabled: bool = True,
    enabled_label_key: str = "enabled",
    help_key: str = "narration_settings_help",
    available: bool = True,
    show_heading: bool = True,
) -> dict[str, Any]:
    if show_heading:
        _panel_heading(locale, title_key, help_key=help_key)
    else:
        st.caption(_t(locale, help_key))
    enabled = st.checkbox(
        _t(locale, enabled_label_key),
        value=enabled,
        key=f"{prefix}_enabled",
        disabled=not available,
    )
    controls_disabled = not available or not enabled
    voice_names = [voice.short_name for voice in edge_voices_for_language(voice_language)]
    current_voice = str(st.session_state.get(f"{prefix}_voice", voice))
    if current_voice not in voice_names:
        voice_names.insert(0, current_voice)
    values: dict[str, Any] = {
        "enabled": enabled,
        "repeats": st.number_input(
            _qualified_label(locale, field_prefix_key, "narration_repeats_suffix"),
            min_value=0,
            max_value=10,
            value=repeats,
            step=1,
            key=f"{prefix}_repeats",
            disabled=controls_disabled,
        ),
        "voice": st.selectbox(
            _qualified_label(locale, field_prefix_key, "narration_voice_suffix"),
            options=voice_names,
            index=voice_names.index(voice) if voice in voice_names else 0,
            format_func=lambda short_name: _edge_voice_label(short_name, locale),
            key=f"{prefix}_voice",
            disabled=controls_disabled,
        ),
    }
    values["volume"] = st.slider(
        _qualified_label(locale, field_prefix_key, "narration_volume_suffix"),
        min_value=-100,
        max_value=100,
        value=0,
        key=f"{prefix}_volume",
        disabled=controls_disabled,
    )
    values["rate"] = st.slider(
        _qualified_label(locale, field_prefix_key, "narration_rate_suffix"),
        min_value=-100,
        max_value=100,
        value=rate,
        key=f"{prefix}_rate",
        disabled=controls_disabled,
    )
    if st.button(
        _t(locale, "play_voice"),
        key=f"{prefix}_play",
        disabled=controls_disabled,
        help=_t(locale, "play_voice_help"),
    ):
        try:
            track = NarrationTrackSettings.model_validate(values)
            destination = settings.storage_dir / "_session_audio" / f"{prefix}-{uuid4().hex}.mp3"
            EdgeSpeechProvider().synthesize(
                sample,
                destination,
                voice=track.voice,
                rate=track.rate_value,
                volume=track.volume_value,
            )
            st.audio(str(destination))
        except (ApplicationError, OSError, ValidationError) as exc:
            if isinstance(exc, SpeechGenerationError):
                speech_tracks: dict[str, SpeechTrack] = {
                    "chinese_narration": "zh",
                    "fast_narration": "fast",
                    "slow_narration": "slow",
                    "question_narration": "question",
                }
                exc = exc.with_context(speech_tracks[prefix])
            st.error(_safe_message(exc, locale))
    return values


_PRIVATE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:])(?:/(?:Users|home|private|tmp|var|Volumes)/[^\s,;]+|"
    r"[A-Za-z]:[\\/][^\s,;]+|\\\\[^\s\\/]+[\\/][^\s,;]+)",
    re.IGNORECASE,
)
_SECRET_LIKE_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{6,}|(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=])",
    re.IGNORECASE,
)
_LOCALIZED_ERROR_KEYS = {
    "Speech generation failed.": "error_speech_generic",
    "The vocabulary provider rejected the API key. Check the configured credential.": (
        "error_llm_auth"
    ),
    "The vocabulary provider request timed out. Try again and check provider availability.": (
        "error_llm_timeout"
    ),
    "The vocabulary provider is unavailable. Check that the service is running and reachable.": (
        "error_llm_unavailable"
    ),
    "The configured vocabulary model or API endpoint was not found. "
    "Check the model name and provider URL.": "error_llm_not_found",
    "The image provider rejected the API key. Check the configured credential.": (
        "error_image_auth"
    ),
    "The image provider connection test timed out. Try again and check the network.": (
        "error_image_timeout"
    ),
    "The image provider is unavailable. Check the network and try again.": (
        "error_image_unavailable"
    ),
    "The image provider returned an invalid response.": "error_image_invalid",
    "The selected upload exceeds the size limit.": "error_upload_too_large",
    "The selected material image cannot be decoded.": "error_material_image_decode",
    "The selected material video cannot be decoded.": "error_material_video_decode",
    "The selected material file type is not supported.": "error_material_type",
    "Video generation was interrupted.": "error_generation_interrupted",
}


def _safe_message(exc: Exception, locale: Locale | None = None) -> str:
    """Return field-level feedback without echoing hostile input values."""
    if isinstance(exc, SpeechGenerationError) and locale is not None:
        reason = _t(locale, f"error_speech_{exc.reason}")
        track = _t(locale, f"error_speech_track_{exc.track}") if exc.track else ""
        if exc.word_index is not None and exc.track:
            return _t(locale, "error_speech_word").format(
                number=exc.word_index + 1, track=track, reason=reason, attempts=exc.attempts
            )
        if exc.track:
            return _t(locale, "error_speech_track").format(
                track=track, reason=reason, attempts=exc.attempts
            )
        return _t(locale, "error_speech_summary").format(reason=reason, attempts=exc.attempts)
    if isinstance(exc, UploadSizeError) and locale is not None:
        return _t(locale, "error_upload_size_details").format(
            actual=exc.size_bytes / MIB, limit=exc.limit_bytes / MIB
        )
    if isinstance(exc, ValidationError):
        fields = {
            str(part)
            for error in exc.errors(include_input=False, include_url=False)
            for part in error.get("loc", ())
        }
        if "font_path" in fields:
            return "Select an existing font file."
        if "path" in fields:
            return "Select an existing supported media file."
        return "Fix the invalid input values."
    message = exc.safe_message if isinstance(exc, ApplicationError) else str(exc)
    if locale is not None:
        translation_key = _LOCALIZED_ERROR_KEYS.get(message)
        if translation_key is not None:
            message = _t(locale, translation_key)
    if _PRIVATE_PATH_PATTERN.search(message) or _SECRET_LIKE_PATTERN.search(message):
        return "The operation could not be completed safely."
    return message


def _style(values: dict[str, Any]) -> TextElementStyle:
    font_value = str(values["font_path"]).strip()
    return TextElementStyle(
        enabled=bool(values["enabled"]),
        font_path=Path(font_value).expanduser() if font_value else None,
        font_size=int(values["font_size"]),
        fill_color=str(values["fill_color"]),
        weight=float(values["weight"]),
        stroke_color=str(values["stroke_color"]),
        stroke_width=float(values["stroke_width"]),
        offsets=AnchorOffsets.model_validate(values["offsets"]),
    )


def _track(values: dict[str, Any]) -> NarrationTrackSettings:
    return NarrationTrackSettings(
        enabled=bool(values["enabled"]),
        repeats=int(values["repeats"]),
        voice=str(values["voice"]),
        volume=int(values["volume"]),
        rate=int(values["rate"]),
    )


def _load_request_into_state(request: GenerationRequest, job_id: str) -> None:
    """Load editable values while retaining immutable job-snapshot fallbacks."""
    st.session_state.pop("music_upload_widget", None)
    for key in list(st.session_state):
        if str(key).startswith(
            ("word_material_", "material_override_", "material_query_", "preview_card_type_")
        ) or key in {
            "material_word_index",
            "preview_word_index",
            "preview_card_type",
            "last_preview_path",
            "last_preview_key",
        }:
            st.session_state.pop(key, None)
    st.session_state.update(
        {
            "loaded_job_id": job_id,
            "task_id": job_id,
            "loaded_background_path": (
                str(request.background_image) if request.background_image is not None else ""
            ),
            "loaded_material_paths": [str(path) for path in request.local_materials],
            "loaded_music_path": (
                str(request.background_music.path)
                if request.background_music.path is not None
                else ""
            ),
            "loaded_pinned_materials": [
                pinned.model_dump(mode="json") for pinned in request.pinned_materials
            ],
            "loaded_job_seed": request.job_seed,
            "loaded_entries": [entry.model_dump(mode="json") for entry in request.entries],
            "word_material_states": {
                index: WordMaterialState(
                    identity=" ".join(entry.english.split()).casefold(),
                    search_query=request.material_queries.get(index, ""),
                )
                for index, entry in enumerate(request.entries)
            },
            "topic": request.topic,
            "word_count": request.word_count,
            "script_text": serialize_vocabulary_script(request.entries, request.phonetic_mode),
            "loaded_script_text": serialize_vocabulary_script(
                request.entries, request.phonetic_mode
            ),
            "auto_phonetic": request.phonetic_mode is PhoneticMode.AUTOMATIC,
            "manual_phonetic": request.phonetic_mode is PhoneticMode.MANUAL,
            "aspect": request.canvas.aspect.value,
            "aspect_widget": request.canvas.aspect.value,
            "remote_provider": request.material.remote_provider.value,
            "material_source": request.material.source.value,
            "material_source_widget": request.material.source.value,
            "selection_mode": request.material.selection_mode.value,
            "selection_mode_widget": request.material.selection_mode.value,
            "question_enabled": request.question.enabled,
            "question_text": request.question_text,
            "question_text_widget": request.question_text,
            "progress_enabled": request.progress.enabled,
            "progress_width": request.progress.width,
            "progress_height": request.progress.height,
            "progress_start_color": request.progress.start_color,
            "progress_end_color": request.progress.end_color,
            "material_enabled": request.material.enabled,
            "material_width": request.material.width,
            "material_height": request.material.height,
            "material_shape": request.material.shape.value,
            "material_shape_widget": request.material.shape.value,
            "material_fit_mode": request.material.fit_mode.value,
            "material_fit_mode_widget": request.material.fit_mode.value,
            "material_pool_size": request.material.pool_size,
            "material_pool_size_widget": request.material.pool_size,
            "fps": request.render.fps,
            "music_enabled": request.background_music.enabled,
            "music_volume": request.background_music.volume_percent,
            "music_volume_widget": request.background_music.volume_percent,
            "music_ducking": request.background_music.ducking_percent,
            "music_ducking_widget": request.background_music.ducking_percent,
            "music_upload_shadow": None,
        }
    )

    def load_offsets(prefix: str, offsets: AnchorOffsets) -> None:
        for name in ("top", "bottom", "left", "right"):
            value = getattr(offsets, name)
            st.session_state[f"{prefix}_{name}"] = value if value is not None else -1

    def load_style(prefix: str, style: TextElementStyle) -> None:
        st.session_state[f"{prefix}_enabled"] = style.enabled
        st.session_state[f"{prefix}_font_path"] = str(style.font_path or "")
        st.session_state[f"{prefix}_font_size"] = style.font_size
        st.session_state[f"{prefix}_fill_color"] = style.fill_color
        st.session_state[f"{prefix}_weight"] = style.weight
        st.session_state[f"{prefix}_stroke_color"] = style.stroke_color
        st.session_state[f"{prefix}_stroke_width"] = style.stroke_width
        load_offsets(prefix, style.offsets)

    def load_track(prefix: str, track: NarrationTrackSettings) -> None:
        st.session_state[f"{prefix}_enabled"] = track.enabled
        st.session_state[f"{prefix}_repeats"] = track.repeats
        st.session_state[f"{prefix}_voice"] = track.voice
        st.session_state[f"{prefix}_volume"] = track.volume
        st.session_state[f"{prefix}_rate"] = track.rate

    load_style("question_style", request.question)
    load_style("english", request.english_text)
    load_style("phonetic", request.phonetic_text)
    load_style("chinese", request.chinese_text)
    load_offsets("progress", request.progress.offsets)
    load_offsets("material", request.material.offsets)
    load_track("chinese_narration", request.narration.chinese)
    load_track("fast_narration", request.narration.fast_english)
    load_track("slow_narration", request.narration.slow_english)
    load_track("question_narration", request.narration.question)


def _build_request(
    form: dict[str, Any],
    entries: list[WordEntry],
    background: Path | None,
    local_materials: list[Path],
) -> GenerationRequest:
    llm = LLMSettings(
        preset=LLMPreset(str(form["llm_preset"])),
        base_url=str(form["llm_base_url"]),
        model=str(form["llm_model"]),
    )
    return GenerationRequest(
        topic=str(form["topic"]),
        word_count=int(form["word_count"]),
        entries=entries,
        phonetic_mode=form["phonetic_mode"],
        canvas=CanvasSettings.for_aspect(VideoAspect(str(form["aspect"]))),
        question_text=str(form["question_text"]),
        question=_style(form["question"]),
        progress=ProgressBarStyle(
            enabled=bool(form["progress"]["enabled"]),
            width=int(form["progress"]["width"]),
            height=int(form["progress"]["height"]),
            start_color=str(form["progress"]["start_color"]),
            end_color=str(form["progress"]["end_color"]),
            offsets=AnchorOffsets.model_validate(form["progress"]["offsets"]),
        ),
        material=MaterialStyle(
            enabled=bool(form["material"]["enabled"]),
            width=int(form["material"]["width"]),
            height=int(form["material"]["height"]),
            shape=MaterialShape(str(form["material"]["shape"])),
            fit_mode=MaterialFitMode(str(form["material"]["fit_mode"])),
            source=MaterialSource(str(form["material_source"])),
            remote_provider=RemoteMaterialProvider(str(form["remote_provider"])),
            selection_mode=SelectionMode(str(form["selection_mode"])),
            pool_size=int(form["material"]["pool_size"]),
            offsets=AnchorOffsets.model_validate(form["material"]["offsets"]),
        ),
        english_text=_style(form["english_text"]),
        phonetic_text=_style(form["phonetic_text"]),
        chinese_text=_style(form["chinese_text"]),
        narration=NarrationSettings(
            chinese=_track(form["chinese_narration"]),
            fast_english=_track(form["fast_narration"]),
            slow_english=_track(form["slow_narration"]),
            question=_track(form["question_narration"]),
        ),
        render=RenderSettings.model_validate(form["render"]),
        background_music=BackgroundMusicSettings.model_validate(form["background_music"]),
        vocabulary=VocabularySettings(
            preset=llm.preset.value,
            base_url=llm.base_url,
            model=llm.model,
        ),
        background_image=background,
        local_materials=local_materials,
        pinned_materials=form.get("pinned_materials", []),
        material_queries=form.get("material_queries", {}),
        job_seed=form.get("job_seed"),
    )


def _is_loopback_url(value: str) -> bool:
    return urlparse(value).hostname in {"localhost", "127.0.0.1", "::1"}


def _llm_provider(
    form: dict[str, Any],
    settings: AppSettings,
) -> VocabularyProvider | None:
    secret = _llm_secret(form, settings)
    if secret is None:
        return None
    preset = LLMPreset(str(form["llm_preset"]))
    is_ollama = preset is LLMPreset.OLLAMA
    is_moonshot_k26 = preset is LLMPreset.MOONSHOT and str(form["llm_model"]).strip() == "kimi-k2.6"
    return OpenAICompatibleVocabularyProvider(
        api_key=secret,
        base_url=str(form["llm_base_url"]),
        model=str(form["llm_model"]),
        strict_json_schema=is_ollama,
        reasoning_effort="none" if is_ollama else None,
        thinking_mode="disabled" if is_moonshot_k26 else None,
    )


def _llm_secret(form: dict[str, Any], settings: AppSettings) -> SecretStr | None:
    preset = LLMPreset(str(form["llm_preset"]))
    secret = _available_secret(str(form["llm_key"]), settings.secrets.for_preset(preset))
    local_endpoint = preset is LLMPreset.OLLAMA or _is_loopback_url(str(form["llm_base_url"]))
    if secret is None and local_endpoint:
        secret = SecretStr("local-session-key")
    return secret


def _image_provider(
    form: dict[str, Any],
    settings: AppSettings,
    local_materials: list[Path],
) -> ImageProvider:
    if not form["material"]["enabled"]:
        return LocalImageProvider([])
    if form["material_source"] == MaterialSource.LOCAL.value:
        return LocalImageProvider(local_materials)
    return _remote_image_provider(form, settings)


def _remote_image_provider(
    form: dict[str, Any],
    settings: AppSettings,
) -> PexelsImageProvider | PixabayImageProvider:
    provider = RemoteMaterialProvider(str(form["remote_provider"]))
    if provider is RemoteMaterialProvider.PEXELS:
        key = _available_secret(str(form["pexels_key"]), settings.secrets.pexels_api_key)
        if key is None:
            raise ValueError("Pexels API key is required.")
        return PexelsImageProvider(key)
    key = _available_secret(str(form["pixabay_key"]), settings.secrets.pixabay_api_key)
    if key is None:
        raise ValueError("Pixabay API key is required.")
    return PixabayImageProvider(key)


def _script_fingerprint(script: str, mode: PhoneticMode, form: dict[str, Any]) -> str:
    payload = {
        "script": script,
        "mode": mode.value,
        "preset": form["llm_preset"],
        "base_url": form["llm_base_url"],
        "model": form["llm_model"],
        "prompt_version": "v1",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _open_folder(path: Path) -> None:
    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform == "win32":
        command = ["explorer", str(path)]
    else:
        command = ["xdg-open", str(path)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _create_preview_tile(destination: Path) -> Path:
    ensure_private_directory(destination.parent)
    image = Image.new("RGB", (512, 512), "#E2E8F0")
    draw = ImageDraw.Draw(image)
    draw.ellipse((156, 106, 356, 306), fill="#94A3B8")
    draw.rectangle((116, 306, 396, 406), fill="#64748B")
    image.save(destination, format="PNG")
    image.close()
    return mark_private_file(destination)


def _preview_still_for_material(
    source: Path,
    preview_root: Path,
    *,
    seed: int,
    entry_index: int = 0,
) -> Path:
    asset = probe_material(source)
    if asset.kind is MaterialKind.IMAGE:
        return source
    return extract_seeded_video_frame(
        source,
        preview_root / f"frame-{uuid4().hex}.png",
        seed=seed,
        entry_index=entry_index,
    )


def _result_preview_width(aspect: object) -> int:
    try:
        video_aspect = VideoAspect(str(aspect))
    except ValueError:
        video_aspect = VideoAspect.PORTRAIT
    return 280 if video_aspect is VideoAspect.PORTRAIT else 480


def _display_image_preview(image: str | Path | bytes, aspect: object) -> None:
    st.image(image, width=_result_preview_width(aspect))


def _invalidate_card_preview() -> None:
    st.session_state.pop("last_preview_path", None)
    st.session_state.pop("last_preview_key", None)
    st.session_state.pop("last_preview_placeholder", None)


def _card_preview_key(
    form: dict[str, Any],
    entries: list[WordEntry],
    background_upload: UploadedFile | None,
    background: Path | None,
    material_uploads: Sequence[UploadedFile],
    local_materials: list[Path],
    overrides: dict[int, UploadedData],
    search_key: str | None,
    *,
    entry_index: int = 0,
    card_type: str = "answer",
) -> str:
    def upload_identity(upload: UploadedData | None) -> str | None:
        if upload is None:
            return None
        return str(getattr(upload, "file_id", "")) or hashlib.sha256(upload.getvalue()).hexdigest()

    def file_identity(path: Path | None) -> tuple[str, int, int] | None:
        if path is None:
            return None
        try:
            stat = path.stat()
            return str(path), stat.st_size, stat.st_mtime_ns
        except OSError:
            return str(path), 0, 0

    payload = {
        "visuals": {
            key: form[key]
            for key in (
                "aspect",
                "question",
                "question_text",
                "material",
                "english_text",
                "phonetic_text",
                "chinese_text",
                "phonetic_mode",
                "material_source",
                "selection_mode",
                "job_seed",
            )
        },
        "entries": [
            entry.model_dump(mode="json") for entry in entries[entry_index : entry_index + 1]
        ],
        "entry_index": entry_index,
        "card_type": card_type,
        "search": search_key,
        "background": upload_identity(background_upload) or file_identity(background),
        "uploads": [upload_identity(upload) for upload in material_uploads],
        "local": [file_identity(path) for path in local_materials],
        "override": upload_identity(overrides.get(entry_index)),
        "pins": [
            file_identity(pin.asset.path)
            for pin in form.get("pinned_materials", [])
            if pin.entry_index == entry_index
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _display_result(locale: Locale) -> None:
    last_video = st.session_state.get("last_video_path")
    if not last_video or not Path(str(last_video)).is_file():
        return
    video_path = Path(str(last_video))
    try:
        contents = video_path.read_bytes()
        st.subheader(_t(locale, "result"))
        job_id = str(st.session_state.get("last_job_id", ""))
        if job_id:
            st.caption(_t(locale, "task_id"))
            st.code(job_id, language=None)
            st.caption(_t(locale, "result_task_help"))
        st.caption(f"{_t(locale, 'result_version')} · {video_path.name}")
        preview_width = _result_preview_width(
            st.session_state.get(
                "last_video_aspect",
                st.session_state.get("aspect", VideoAspect.PORTRAIT.value),
            )
        )
        with st.container(horizontal_alignment="center"):
            st.video(contents, width=preview_width)
            st.download_button(
                _t(locale, "download"),
                data=contents,
                file_name=f"{job_id}-{video_path.name}" if job_id else video_path.name,
                mime="video/mp4",
            )
    except OSError as exc:
        st.error(_safe_message(exc, locale))


def _clear_regeneration_task() -> None:
    loaded = st.session_state.pop("regeneration_task", None)
    if loaded is not None:
        shadow_key = f"history_version_{loaded['id']}"
        st.session_state.pop(shadow_key, None)
        for locale in Locale:
            st.session_state.pop(f"{shadow_key}_{locale.value}", None)


def _invalidate_recent_tasks() -> None:
    st.session_state.pop("recent_tasks_cache", None)


def _clear_task_selection() -> None:
    _clear_regeneration_task()
    st.session_state["recent_task"] = None
    for locale in Locale:
        st.session_state.pop(f"recent_task_{locale.value}", None)


def _choose_recent_task(widget_key: str) -> None:
    task_id = st.session_state.get(widget_key)
    st.session_state["recent_task"] = task_id
    if task_id:
        st.session_state["task_id"] = task_id
        _clear_regeneration_task()
        st.session_state["load_recent_task"] = True


def _display_recent_tasks(settings: AppSettings, locale: Locale) -> None:
    selection_column, refresh_column = st.columns([4, 1], vertical_alignment="bottom")
    with refresh_column:
        st.button(
            _t(locale, "refresh_tasks"),
            key="refresh_recent_tasks",
            on_click=_invalidate_recent_tasks,
            use_container_width=True,
        )
    identity = (
        str(settings.storage_dir.absolute()),
        hashlib.sha256(json.dumps(_storage_secrets(settings)).encode()).hexdigest(),
    )
    cached = st.session_state.get("recent_tasks_cache")
    if cached is None or cached[0] != identity:
        try:
            jobs = _job_storage(settings).list_recent_jobs()
        except (ApplicationError, OSError, ValueError):
            st.warning(_t(locale, "recent_tasks_unavailable"))
            jobs = []
        st.session_state["recent_tasks_cache"] = (identity, jobs)
    else:
        jobs = cached[1]
    by_id = {job.job_id: job for job in jobs}
    widget_key = f"recent_task_{locale.value}"
    if st.session_state.get("recent_task") not in by_id:
        st.session_state.pop("recent_task", None)
        # Send an explicit reset so the browser also clears its displayed label.
        st.session_state[widget_key] = None
    selected = st.session_state.get("recent_task")
    options = list(by_id)

    def label(job_id: str) -> str:
        job = by_id[job_id]
        return _t(locale, "recent_task_option").format(
            title=job.title or _t(locale, "untitled_task"),
            count=job.word_count,
            status=_t(locale, f"task_status_{job.status.value}"),
            time=datetime.fromtimestamp(job.updated_at).astimezone().strftime("%m-%d %H:%M %Z"),
            task_id=job_id[:8],
        )

    with selection_column:
        st.selectbox(
            _t(locale, "recent_tasks"),
            options,
            index=options.index(selected) if selected in by_id else None,
            placeholder=_t(locale, "recent_tasks_placeholder")
            if jobs
            else _t(locale, "recent_tasks_empty"),
            format_func=label,
            key=widget_key,
            on_change=_choose_recent_task,
            args=(widget_key,),
            disabled=not jobs,
            help=_t(locale, "recent_tasks_help"),
        )


def _display_video_history(settings: AppSettings, locale: Locale, loaded: dict[str, Any]) -> None:
    st.markdown(f"**{_t(locale, 'video_history')}**")
    try:
        storage = _job_storage(settings)
        versions = storage.list_video_versions(loaded["id"])
        if not versions:
            st.caption(_t(locale, "video_history_empty"))
            return
        by_reference = {version.reference: version for version in versions}
        shadow_key = f"history_version_{loaded['id']}"
        key = f"{shadow_key}_{locale.value}"
        if st.session_state.get(shadow_key) not in by_reference:
            st.session_state[shadow_key] = versions[0].reference
            # Explicit assignment also updates the browser's selected label.
            st.session_state[key] = versions[0].reference

        def label(reference: str) -> str:
            version = by_reference[reference]
            timestamp = (
                datetime.fromtimestamp(version.modified_at)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M %Z")
            )
            current = f" · {_t(locale, 'video_history_current')}" if version.is_current else ""
            return f"{Path(reference).name} · {timestamp}{current}"

        reference = st.selectbox(
            _t(locale, "video_history_version"),
            list(by_reference),
            index=list(by_reference).index(st.session_state[shadow_key]),
            format_func=label,
            key=key,
            on_change=_remember_conditional_value,
            args=(key, shadow_key),
        )
        st.caption(_t(locale, "video_history_help"))
        contents = storage.read_video_version(loaded["id"], reference)
        with st.container(horizontal_alignment="center"):
            st.video(contents, width=_result_preview_width(loaded["aspect"]))
            st.download_button(
                _t(locale, "download_history"),
                data=contents,
                file_name=f"{loaded['id']}-{Path(reference).name}",
                mime="video/mp4",
                key=f"download_history_{loaded['id']}",
            )
    except (ApplicationError, ValueError, OSError):
        st.warning(_t(locale, "video_history_unavailable"))


def _display_regeneration(settings: AppSettings, locale: Locale) -> None:
    with st.expander(_t(locale, "regeneration_settings"), expanded=False):
        st.caption(_t(locale, "regeneration_settings_help"))
        _display_recent_tasks(settings, locale)
        id_column, load_column = st.columns([4, 1], vertical_alignment="bottom")
        with id_column:
            task_id = st.text_input(
                _t(locale, "task_id"), key="task_id", on_change=_clear_task_selection
            ).strip()
        with load_column:
            load = st.button(
                _t(locale, "load_task"),
                key="load_regeneration_task",
                disabled=not task_id,
                use_container_width=True,
            )
        load_recent = st.session_state.pop("load_recent_task", False)
        if load or load_recent:
            _clear_regeneration_task()
            try:
                storage = _job_storage(settings)
                manifest = storage.load_manifest(task_id)
                request = storage.load_request(task_id)
                saved_entries = [WordEntry.model_validate(item) for item in manifest["entries"]]
                if not saved_entries:
                    st.error(_t(locale, "task_no_vocabulary"))
                else:
                    st.session_state["regeneration_task"] = {
                        "id": task_id,
                        "entries": saved_entries,
                        "aspect": request.canvas.aspect.value,
                        "materials_enabled": request.material.enabled,
                        "token": uuid4().hex,
                    }
            except (ApplicationError, ValidationError, ValueError, OSError):
                st.error(_t(locale, "task_load_failed"))

        loaded = st.session_state.get("regeneration_task")
        if loaded is not None and loaded["id"] != task_id:
            _clear_regeneration_task()
            loaded = None
        replacement_uploads: list[UploadedFile] = []
        replacement_indices: list[int] = []
        if loaded is None:
            st.info(_t(locale, "task_load_required"))
        else:
            saved_entries = loaded["entries"]
            st.success(
                _t(locale, "task_loaded").format(task_id=loaded["id"], count=len(saved_entries))
            )
            st.dataframe(
                [
                    {
                        _t(locale, "replacement_index"): index + 1,
                        "English": entry.english,
                        _t(locale, "saved_chinese"): entry.chinese,
                        _t(locale, "saved_phonetic"): entry.phonetic,
                    }
                    for index, entry in enumerate(saved_entries)
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(_t(locale, "saved_task_help"))
            if not loaded["materials_enabled"]:
                st.info(_t(locale, "task_materials_disabled"))
            else:
                replacement_uploads = (
                    st.file_uploader(
                        _t(locale, "replacement_images"),
                        type=("png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v", "webm"),
                        accept_multiple_files=True,
                        key=f"replacement_uploads_{loaded['token']}",
                        max_upload_size=MAX_LOCAL_VIDEO_BYTES // MIB,
                    )
                    or []
                )
                st.caption(_t(locale, "mixed_upload_limits"))

            def word_label(index: int) -> str:
                entry = saved_entries[index]
                label = _t(locale, "replacement_word_option").format(
                    number=index + 1, word=entry.english
                )
                return f"{label} · {entry.chinese}" if entry.chinese else label

            replacement_indices = [
                st.selectbox(
                    f"{_t(locale, 'replacement_word')} · {upload.name}",
                    range(len(saved_entries)),
                    index=min(index, len(saved_entries) - 1),
                    format_func=word_label,
                    key=f"replacement_word_{loaded['token']}_{upload.file_id}",
                )
                for index, upload in enumerate(replacement_uploads)
            ]
        duplicate_indices = len(replacement_indices) != len(set(replacement_indices))
        if duplicate_indices:
            st.error(_t(locale, "duplicate_replacements"))
        regenerate = st.button(
            _t(locale, "regenerate"),
            disabled=loaded is None or duplicate_indices,
            use_container_width=True,
        )
        if regenerate and loaded is not None:
            try:
                replacement_paths = _save_uploads(
                    replacement_uploads,
                    settings.storage_dir,
                    allowed_suffixes=frozenset(
                        {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".m4v", ".webm"}
                    ),
                )
                replacements = dict(zip(replacement_indices, replacement_paths, strict=True))
                pipeline = GenerationPipeline(
                    storage=_job_storage(settings),
                    vocabulary_provider=None,
                    image_provider=LocalImageProvider([]),
                    speech_provider=EdgeSpeechProvider(),
                    card_renderer=CardRenderer(),
                    video_composer=VideoComposer(),
                )
                progress_bar = st.progress(0, text=_t(locale, "preparing"))

                def report_regeneration(progress: PipelineProgress) -> None:
                    progress_bar.progress(
                        progress.percent, text=_localized_progress_message(locale, progress)
                    )

                result = pipeline.regenerate(
                    loaded["id"], replacements=replacements, on_progress=report_regeneration
                )
                assert result.video_path is not None
                st.session_state["last_video_path"] = str(result.video_path)
                st.session_state["last_video_aspect"] = loaded["aspect"]
                st.session_state["last_job_id"] = result.job_id
                _invalidate_recent_tasks()
                st.session_state.pop(f"history_version_{loaded['id']}", None)
                st.success(_t(locale, "regeneration_complete"))
            except (ApplicationError, ValidationError, ValueError, OSError) as exc:
                st.error(_safe_message(exc, locale))
        if loaded is not None:
            _display_video_history(settings, locale, loaded)
            task_directory = JobStorage(settings.storage_dir).paths(loaded["id"]).root
            st.caption(_t(locale, "task_directory"))
            if st.button(_t(locale, "open_folder"), disabled=not task_directory.is_dir()):
                _open_folder(task_directory)


def main() -> None:
    st.set_page_config(page_title="AI Vocab Video Generator", page_icon="🎬", layout="wide")
    st.markdown(_APP_STYLES, unsafe_allow_html=True)
    settings = AppSettings.from_toml()
    _initialize_state(settings)
    pending_request = st.session_state.pop("pending_loaded_request", None)
    pending_job_id = st.session_state.pop("pending_loaded_job_id", None)
    if isinstance(pending_request, GenerationRequest) and isinstance(pending_job_id, str):
        _load_request_into_state(pending_request, pending_job_id)
    locale = Locale(str(st.session_state.get("locale", DEFAULT_LOCALE.value)))
    st.title(f"{_t(locale, 'title')} v{__version__}")
    st.caption(_t(locale, "caption"))

    form: dict[str, Any] = {}
    with st.expander(_t(locale, "basic_settings"), expanded=False):
        _field_help(locale, "basic_settings")
        language_column, llm_column, provider_column = st.columns(3)
        with language_column:
            st.selectbox(
                _t(locale, "language"),
                options=[Locale.ZH_CN.value, Locale.EN_US.value],
                format_func=lambda value: "简体中文" if value == Locale.ZH_CN.value else "English",
                key="locale",
                on_change=_on_locale_changed,
            )
        with llm_column:
            current_llm_preset = str(st.session_state.get("llm_preset", LLMPreset.OPENAI.value))
            form["llm_preset"] = st.selectbox(
                _t(locale, "llm_preset"),
                options=[preset.value for preset in LLMPreset],
                format_func=lambda value: {
                    "openai": "OpenAI",
                    "deepseek": "DeepSeek",
                    "moonshot": "Moonshot",
                    "qwen": "Qwen",
                    "ollama": "Ollama",
                    "custom": "Custom",
                }[value],
                key="llm_preset",
                on_change=_on_llm_preset_changed,
                help=_t(locale, f"llm_{current_llm_preset}_help"),
            )
            st.info(_t(locale, f"llm_{form['llm_preset']}_setup"))
            form["llm_key"] = st.text_input(
                _t(locale, f"api_key_{form['llm_preset']}"),
                type="password",
                key="llm_key_input",
                on_change=_remember_llm_credential,
            )
            _remember_llm_credential()
            form["llm_base_url"] = st.text_input(
                _t(locale, "base_url"),
                key="llm_base_url",
                on_change=_on_llm_base_url_changed,
            )
            form["llm_model"] = st.text_input(_t(locale, "model"), key="llm_model")
            if st.button(
                _t(locale, "test_llm_connection"),
                key="test_llm_connection",
                use_container_width=True,
                help=_t(locale, "test_llm_connection_help"),
            ):
                llm_for_test = _llm_provider(form, settings)
                if llm_for_test is None:
                    st.error(_t(locale, "missing_llm_key"))
                else:
                    try:
                        llm_for_test.check_connection()
                        st.success(_t(locale, "llm_connection_success"))
                    except (ApplicationError, OSError, ValueError) as exc:
                        st.error(_safe_message(exc, locale))
                    finally:
                        _close_provider(llm_for_test)
        with provider_column:
            form["remote_provider"] = st.selectbox(
                _t(locale, "remote_provider"),
                options=[provider.value for provider in RemoteMaterialProvider],
                format_func=lambda value: "Pexels" if value == "pexels" else "Pixabay",
                key="remote_provider",
                on_change=_on_remote_provider_changed,
                help=_t(locale, "remote_provider_help"),
            )
            form["pexels_key"] = ""
            form["pixabay_key"] = ""
            selected_provider = str(form["remote_provider"])
            credential_key = f"{selected_provider}_key_input"
            credentials = dict(st.session_state.get("provider_credentials", {}))
            st.session_state.setdefault(credential_key, str(credentials.get(selected_provider, "")))
            entered_provider_key = st.text_input(
                _parenthetical_label(
                    locale,
                    _t(locale, f"{selected_provider}_key"),
                    _t(locale, f"{selected_provider}_key_help"),
                ),
                type="password",
                key=credential_key,
                on_change=_remember_provider_credential,
                args=(selected_provider,),
            )
            form[f"{selected_provider}_key"] = entered_provider_key
            _remember_provider_credential(selected_provider)
            if st.button(
                _t(locale, "test_image_connection"),
                key="test_image_connection",
                use_container_width=True,
                help=_t(locale, "test_image_connection_help"),
            ):
                image_for_test: PexelsImageProvider | PixabayImageProvider | None = None
                try:
                    image_for_test = _remote_image_provider(form, settings)
                    image_for_test.check_connection()
                    st.success(_t(locale, "image_connection_success"))
                except ValueError:
                    st.error(_t(locale, "missing_provider_key"))
                except (ApplicationError, OSError) as exc:
                    st.error(_safe_message(exc, locale))
                finally:
                    if image_for_test is not None:
                        _close_provider(image_for_test)

    with st.container(border=True):
        _panel_heading(locale, "topic_settings")
        topic_column, count_column = st.columns([0.78, 0.22])
        with topic_column:
            form["topic"] = st.text_input(
                _t(locale, "topic"), key="topic", max_chars=MAX_TOPIC_LENGTH
            )
        with count_column:
            word_count_arguments: dict[str, Any] = {
                "min_value": 1,
                "max_value": 50,
                "key": "word_count",
                "help": _t(locale, "word_count_help"),
            }
            if "word_count" not in st.session_state:
                word_count_arguments["value"] = 10
            form["word_count"] = st.number_input(_t(locale, "word_count"), **word_count_arguments)

        generate_column, voice_column = st.columns([0.62, 0.38], vertical_alignment="center")
        with generate_column:
            generate_script = st.button(
                _t(locale, "generate_script"), type="primary", use_container_width=True
            )
            if generate_script:
                provider = _llm_provider(form, settings)
                if provider is None:
                    st.error(_t(locale, "missing_llm_key"))
                elif not str(form["topic"]).strip():
                    st.error(_t(locale, "missing_topic"))
                else:
                    try:
                        generated = provider.generate(str(form["topic"]), int(form["word_count"]))
                        current_mode = _phonetic_mode()
                        visible_script = serialize_vocabulary_script(generated, current_mode)
                        st.session_state["script_text"] = visible_script
                        st.session_state["script_cache_entries"] = [
                            entry.model_dump(mode="json") for entry in generated
                        ]
                        st.session_state["script_cache_key"] = _script_fingerprint(
                            visible_script, current_mode, form
                        )
                    except (ApplicationError, OSError, ValueError) as exc:
                        st.error(_safe_message(exc, locale))
                    finally:
                        _close_provider(provider)
        with voice_column:
            recording = st.audio_input(_t(locale, "record_topic"), key="topic_recording")
            if recording is not None and st.button(
                _t(locale, "transcribe"), use_container_width=True
            ):
                try:
                    audio_path = _save_upload(recording, settings.storage_dir, ".wav")
                    transcription = FunASRTranscriptionProvider(
                        model_cache=settings.model_cache_dir
                    ).transcribe(audio_path)
                    st.session_state["topic"] = transcription
                except (ApplicationError, OSError, ValueError) as exc:
                    st.error(_safe_message(exc, locale))

        _panel_heading(locale, "phonetic_settings")
        phonetic_columns = st.columns(2)
        with phonetic_columns[0]:
            st.checkbox(
                _t(locale, "automatic_phonetic"),
                key="auto_phonetic",
                on_change=_on_auto_phonetic_changed,
            )
        with phonetic_columns[1]:
            st.checkbox(
                _t(locale, "manual_phonetic"),
                key="manual_phonetic",
                on_change=_on_manual_phonetic_changed,
            )
        st.caption(_t(locale, "phonetic_disabled_help"))
        form["phonetic_mode"] = _phonetic_mode()

        script = st.text_area(
            _t(locale, "script"),
            height=280,
            key="script_text",
            max_chars=MAX_SCRIPT_LENGTH,
        )

    material_uploads = None
    material_override_uploads: dict[int, Any] = {}
    with st.container(border=True):
        _panel_heading(locale, "canvas_material_settings")
        canvas_column, material_column = st.columns(2)
        with canvas_column:
            _panel_heading(locale, "video_settings", show_help=False)
            _restore_conditional_value("aspect_widget", "aspect")
            form["aspect"] = st.selectbox(
                _t(locale, "aspect"),
                options=[aspect.value for aspect in VideoAspect],
                format_func=lambda value: _t(
                    locale, "portrait" if value == "portrait" else "landscape"
                ),
                key="aspect_widget",
                on_change=_on_aspect_changed,
                help=_t(locale, "aspect_help"),
            )
            st.session_state["aspect"] = form["aspect"]
            st.caption(_t(locale, "background_settings_help"))
            background_upload = st.file_uploader(
                _t(locale, "background_image"),
                type=("png", "jpg", "jpeg", "webp"),
                key="background_upload",
                max_upload_size=MAX_LOCAL_IMAGE_BYTES // MIB,
                help=_t(locale, "background_settings_help"),
            )
            st.caption(_t(locale, "image_upload_limits"))
        with material_column:
            _panel_heading(locale, "material_settings")
            material_enabled = st.checkbox(
                _t(locale, "show_material"), value=True, key="material_enabled"
            )
            if material_enabled:
                _restore_conditional_value("material_source_widget", "material_source")
                form["material_source"] = st.selectbox(
                    _t(locale, "material_source"),
                    options=[source.value for source in MaterialSource],
                    format_func=lambda value: _t(
                        locale, "local_uploads" if value == "local" else "remote_search"
                    ),
                    key="material_source_widget",
                    help=_t(locale, "material_source_help"),
                )
                st.session_state["material_source"] = form["material_source"]
                if form["material_source"] == MaterialSource.LOCAL.value:
                    material_uploads = st.file_uploader(
                        _t(locale, "material_uploads_mixed"),
                        type=("png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v", "webm"),
                        accept_multiple_files=True,
                        help=_t(locale, "video_upload_help"),
                        key="material_uploads",
                        max_upload_size=MAX_LOCAL_VIDEO_BYTES // MIB,
                    )
                    st.caption(_t(locale, "mixed_upload_limits"))
                else:
                    st.session_state.pop("material_uploads", None)
            else:
                form["material_source"] = str(st.session_state["material_source"])
                st.session_state.pop("material_uploads", None)

    with st.expander(_t(locale, "lesson_flow_settings"), expanded=False):
        question_tab, question_narration_tab, progress_tab = st.tabs(
            [
                _t(locale, "question_settings"),
                _t(locale, "question_narration_settings"),
                _t(locale, "progress_settings"),
            ]
        )
        with question_tab:
            st.caption(_t(locale, "question_settings_help"))
            question_enabled = st.checkbox(
                _t(locale, "show_question"), value=False, key="question_enabled"
            )
            if question_enabled:
                _restore_conditional_value("question_text_widget", "question_text")
                form["question_text"] = st.text_input(
                    _t(locale, "question_text"),
                    key="question_text_widget",
                    on_change=_remember_conditional_value,
                    args=("question_text_widget", "question_text"),
                )
                st.session_state["question_text"] = form["question_text"]
            else:
                form["question_text"] = str(st.session_state["question_text"])
            form["question"] = _text_style_fields(
                "question_style",
                locale,
                enabled=question_enabled,
                font_path=str(settings.font_path or ""),
                font_size=int(st.session_state["question_font_size"]),
                fill_color="#000000",
                stroke_color="#FFFFFF",
                top=int(st.session_state["question_top"]),
                label_prefix_key="question_label",
                show_enabled=False,
            )
            form["question"]["enabled"] = question_enabled
        with question_narration_tab:
            form["question_narration"] = _narration_panel(
                "question_narration_settings",
                "question_narration",
                locale,
                settings,
                voice="en-US-JennyNeural",
                repeats=1,
                rate=-20,
                sample="What is this?",
                field_prefix_key="question_label",
                voice_language="en",
                enabled_label_key="question_narration",
                help_key="question_narration_settings_help",
                available=question_enabled,
                show_heading=False,
            )
        with progress_tab:
            st.caption(_t(locale, "progress_settings_help"))
            progress_enabled = st.checkbox(
                _t(locale, "show_progress"),
                value=True,
                key="progress_enabled",
                disabled=not question_enabled,
            )
            progress_controls_disabled = not question_enabled or not progress_enabled
            progress_dimensions = st.columns(2)
            with progress_dimensions[0]:
                progress_width = st.number_input(
                    _qualified_label(locale, "progress_label", "width_suffix"),
                    min_value=1,
                    max_value=3840,
                    value=756,
                    key="progress_width",
                    disabled=progress_controls_disabled,
                )
                start_color = st.color_picker(
                    _qualified_label(locale, "progress_label", "start_color_suffix"),
                    value="#FFA500",
                    key="progress_start_color",
                    disabled=progress_controls_disabled,
                )
            with progress_dimensions[1]:
                progress_height = st.number_input(
                    _qualified_label(locale, "progress_label", "height_suffix"),
                    min_value=1,
                    max_value=400,
                    value=20,
                    key="progress_height",
                    disabled=progress_controls_disabled,
                )
                end_color = st.color_picker(
                    _qualified_label(locale, "progress_label", "end_color_suffix"),
                    value="#ADFF2F",
                    key="progress_end_color",
                    disabled=progress_controls_disabled,
                )
            progress_offsets = _offset_fields(
                "progress",
                locale,
                top=int(st.session_state["progress_top"]),
                label_prefix_key="progress_label",
                disabled=progress_controls_disabled,
            )
            form["progress"] = {
                "enabled": progress_enabled,
                "width": progress_width,
                "height": progress_height,
                "start_color": start_color,
                "end_color": end_color,
                "offsets": progress_offsets,
            }

    with st.expander(_t(locale, "narration_group_settings"), expanded=False):
        chinese_tab, fast_tab, slow_tab = st.tabs(
            [
                _t(locale, "chinese_narration"),
                _t(locale, "fast_english_narration"),
                _t(locale, "slow_english_narration"),
            ]
        )
        with chinese_tab:
            form["chinese_narration"] = _narration_panel(
                "chinese_narration",
                "chinese_narration",
                locale,
                settings,
                voice="zh-CN-XiaoxiaoNeural",
                repeats=0,
                rate=0,
                sample="苹果",
                field_prefix_key="chinese_label",
                voice_language="zh",
                enabled=False,
                show_heading=False,
            )
        with fast_tab:
            form["fast_narration"] = _narration_panel(
                "fast_english_narration",
                "fast_narration",
                locale,
                settings,
                voice="en-US-JennyNeural",
                repeats=1,
                rate=-20,
                sample="apple",
                field_prefix_key="fast_english_label",
                voice_language="en",
                show_heading=False,
            )
        with slow_tab:
            form["slow_narration"] = _narration_panel(
                "slow_english_narration",
                "slow_narration",
                locale,
                settings,
                voice="en-US-JennyNeural",
                repeats=0,
                rate=0,
                sample="apple",
                field_prefix_key="slow_english_label",
                voice_language="en",
                enabled=False,
                show_heading=False,
            )

    material_pool_size = int(st.session_state["material_pool_size"])
    with st.expander(_t(locale, "visual_style_settings"), expanded=False):
        english_tab, phonetic_tab, chinese_text_tab, material_style_tab = st.tabs(
            [
                _t(locale, "english_text"),
                _t(locale, "phonetic_text"),
                _t(locale, "chinese_text"),
                _t(locale, "material_settings"),
            ]
        )
        with english_tab:
            st.caption(_t(locale, "text_settings_help"))
            form["english_text"] = _text_style_fields(
                "english",
                locale,
                enabled=True,
                font_path=str(settings.font_path or ""),
                font_size=int(st.session_state["english_font_size"]),
                fill_color="#000000",
                stroke_color="#FFFFFF",
                top=int(st.session_state["english_top"]),
                label_prefix_key="english_label",
            )
        with phonetic_tab:
            st.caption(_t(locale, "text_settings_help"))
            form["phonetic_text"] = _text_style_fields(
                "phonetic",
                locale,
                enabled=True,
                font_path=str(settings.font_path or ""),
                font_size=int(st.session_state["phonetic_font_size"]),
                fill_color="#000000",
                stroke_color="#FFFFFF",
                top=int(st.session_state["phonetic_top"]),
                label_prefix_key="phonetic_label",
            )
        with chinese_text_tab:
            st.caption(_t(locale, "text_settings_help"))
            form["chinese_text"] = _text_style_fields(
                "chinese",
                locale,
                enabled=True,
                font_path=str(settings.font_path or ""),
                font_size=int(st.session_state["chinese_font_size"]),
                fill_color="#000000",
                stroke_color="#FFFFFF",
                top=int(st.session_state["chinese_top"]),
                label_prefix_key="chinese_label",
            )
        with material_style_tab:
            st.caption(_t(locale, "material_settings_help"))
            _restore_conditional_value("selection_mode_widget", "selection_mode")
            form["selection_mode"] = st.selectbox(
                _t(locale, "selection_mode"),
                options=[mode.value for mode in SelectionMode],
                format_func=lambda value: _t(locale, value),
                key="selection_mode_widget",
                help=_t(locale, "selection_mode_help"),
                disabled=not material_enabled,
            )
            st.session_state["selection_mode"] = form["selection_mode"]
            if form["material_source"] == MaterialSource.REMOTE.value:
                _restore_conditional_value("material_pool_size_widget", "material_pool_size")
                material_pool_size = int(
                    st.number_input(
                        _t(locale, "candidate_pool_size"),
                        min_value=1,
                        max_value=20,
                        step=1,
                        key="material_pool_size_widget",
                        on_change=_remember_conditional_value,
                        args=("material_pool_size_widget", "material_pool_size"),
                        help=_t(locale, "candidate_pool_size_help"),
                        disabled=not material_enabled,
                    )
                )
                st.session_state["material_pool_size"] = material_pool_size
            material_dimensions = st.columns(2)
            with material_dimensions[0]:
                material_width = st.number_input(
                    _qualified_label(locale, "material_label", "width_suffix"),
                    min_value=1,
                    max_value=3840,
                    key="material_width",
                    disabled=not material_enabled,
                )
                _restore_conditional_value("material_shape_widget", "material_shape")
                material_shape = st.selectbox(
                    _qualified_label(locale, "material_label", "shape_suffix"),
                    options=[shape.value for shape in MaterialShape],
                    format_func=lambda value: _t(locale, value),
                    key="material_shape_widget",
                    disabled=not material_enabled,
                )
                st.session_state["material_shape"] = material_shape
            with material_dimensions[1]:
                material_height = st.number_input(
                    _qualified_label(locale, "material_label", "height_suffix"),
                    min_value=1,
                    max_value=3840,
                    key="material_height",
                    disabled=not material_enabled,
                )
                _restore_conditional_value("material_fit_mode_widget", "material_fit_mode")
                material_fit_mode = st.selectbox(
                    _t(locale, "fit_mode"),
                    options=[mode.value for mode in MaterialFitMode],
                    format_func=lambda value: _t(locale, value),
                    key="material_fit_mode_widget",
                    help=_t(locale, "fit_mode_help"),
                    disabled=not material_enabled,
                )
                st.session_state["material_fit_mode"] = material_fit_mode
            material_offsets = _offset_fields(
                "material",
                locale,
                top=int(st.session_state["material_top"]),
                label_prefix_key="material_label",
                disabled=not material_enabled,
            )
            form["material"] = {
                "enabled": material_enabled,
                "width": material_width,
                "height": material_height,
                "shape": material_shape,
                "fit_mode": material_fit_mode,
                "pool_size": material_pool_size,
                "offsets": material_offsets,
            }

    music_upload = None
    with st.expander(_t(locale, "audio_output_settings"), expanded=False):
        music_tab, output_tab = st.tabs(
            [_t(locale, "background_music_settings"), _t(locale, "render_settings")]
        )
        with music_tab:
            st.caption(_t(locale, "background_music_settings_help"))
            music_enabled = st.checkbox(_t(locale, "background_music_enabled"), key="music_enabled")
            if music_enabled:
                visible_music_upload = st.file_uploader(
                    _t(locale, "background_music_file"),
                    type=("mp3", "wav", "m4a", "aac", "ogg"),
                    key="music_upload_widget",
                    max_upload_size=MAX_LOCAL_AUDIO_BYTES // MIB,
                    on_change=_on_music_upload_changed,
                )
                if visible_music_upload is not None:
                    st.session_state["music_upload_shadow"] = visible_music_upload
                music_upload = visible_music_upload or st.session_state.get("music_upload_shadow")
                _restore_conditional_value("music_volume_widget", "music_volume")
                music_volume = st.slider(
                    _t(locale, "music_volume"),
                    min_value=0,
                    max_value=100,
                    key="music_volume_widget",
                    on_change=_remember_conditional_value,
                    args=("music_volume_widget", "music_volume"),
                )
                _restore_conditional_value("music_ducking_widget", "music_ducking")
                music_ducking = st.slider(
                    _t(locale, "music_ducking"),
                    min_value=0,
                    max_value=100,
                    key="music_ducking_widget",
                    on_change=_remember_conditional_value,
                    args=("music_ducking_widget", "music_ducking"),
                )
                st.session_state["music_volume"] = int(music_volume)
                st.session_state["music_ducking"] = int(music_ducking)
            else:
                music_upload = None
                music_volume = int(st.session_state["music_volume"])
                music_ducking = int(st.session_state["music_ducking"])
            form["background_music"] = {
                "enabled": music_enabled,
                "path": None,
                "volume_percent": music_volume,
                "ducking_percent": music_ducking,
            }
        with output_tab:
            form["render"] = {
                "fps": st.number_input(
                    _t(locale, "fps"),
                    min_value=12,
                    max_value=60,
                    step=1,
                    key="fps",
                    help=_t(locale, "fps_help"),
                )
            }

    entries: list[WordEntry] = []
    script_error = ""
    if script.strip():
        try:
            entries = parse_vocabulary_script(script, form["phonetic_mode"])
        except (ScriptFormatError, ValidationError) as exc:
            script_error = _safe_message(exc, locale)
            st.error(script_error)

    cache_key = _script_fingerprint(script, form["phonetic_mode"], form)
    if st.session_state.get("script_cache_key") == cache_key:
        entries = [
            WordEntry.model_validate(item)
            for item in st.session_state.get("script_cache_entries", [])
        ]
    if script == st.session_state.get("loaded_script_text"):
        entries = [
            WordEntry.model_validate(item) for item in st.session_state.get("loaded_entries", [])
        ]

    loaded_background_value = str(st.session_state.get("loaded_background_path", ""))
    loaded_background = Path(loaded_background_value) if loaded_background_value else None
    if loaded_background is not None and not loaded_background.is_file():
        loaded_background = None
    loaded_materials = [
        Path(str(value))
        for value in st.session_state.get("loaded_material_paths", [])
        if Path(str(value)).is_file()
    ]
    loaded_music_value = str(st.session_state.get("loaded_music_path", ""))
    loaded_music = Path(loaded_music_value) if loaded_music_value else None
    if loaded_music is not None and not loaded_music.is_file():
        loaded_music = None
    effective_background = loaded_background
    effective_local_materials = (
        loaded_materials
        if form["material"]["enabled"] and form["material_source"] == MaterialSource.LOCAL.value
        else []
    )
    form["background_music"]["path"] = loaded_music
    loaded_pins = [
        PinnedMaterial.model_validate(item)
        for item in st.session_state.get("loaded_pinned_materials", [])
    ]
    form["pinned_materials"] = (
        loaded_pins
        if form["material"]["enabled"]
        and form["material_source"] == MaterialSource.REMOTE.value
        and not material_uploads
        else []
    )
    existing_pins = {pin.entry_index: pin for pin in form["pinned_materials"]}
    loaded_job_seed = st.session_state.get("loaded_job_seed")
    form["job_seed"] = (
        int(loaded_job_seed)
        if loaded_job_seed is not None
        else int(st.session_state["draft_job_seed"])
    )

    material_states = cast(
        dict[int, WordMaterialState], st.session_state.setdefault("word_material_states", {})
    )
    if entries or not script.strip():
        for word_index in list(material_states):
            if word_index >= len(entries):
                material_states.pop(word_index)
                st.session_state.pop(f"material_override_{word_index}", None)
                st.session_state.pop(f"material_query_{word_index}", None)
        for word_index, word in enumerate(entries):
            identity = " ".join(word.english.split()).casefold()
            if (
                word_index not in material_states
                or material_states[word_index].identity != identity
            ):
                material_states[word_index] = WordMaterialState(identity=identity)
                st.session_state.pop(f"material_override_{word_index}", None)
                st.session_state.pop(f"material_query_{word_index}", None)
    remote_search_keys: dict[int, str] = {}
    if (
        entries
        and form["material"]["enabled"]
        and form["material_source"] == MaterialSource.REMOTE.value
    ):
        remote_choice = RemoteMaterialProvider(str(form["remote_provider"]))
        preview_secret = (
            _available_secret(str(form["pexels_key"]), settings.secrets.pexels_api_key)
            if remote_choice is RemoteMaterialProvider.PEXELS
            else _available_secret(str(form["pixabay_key"]), settings.secrets.pixabay_api_key)
        )
        if preview_secret is not None:
            credential_digest = hashlib.sha256(
                preview_secret.get_secret_value().encode("utf-8")
            ).hexdigest()
            preview_material_style = MaterialStyle.model_validate(
                {
                    **form["material"],
                    "source": form["material_source"],
                    "remote_provider": form["remote_provider"],
                    "selection_mode": form["selection_mode"],
                }
            )
            remote_search_keys = {
                index: remote_search_key(
                    entry,
                    VideoAspect(str(form["aspect"])),
                    preview_material_style,
                    remote_choice.value,
                    credential_digest,
                )
                for index, entry in enumerate(entries)
            }

        for word_index, key in remote_search_keys.items():
            material_states[word_index].sync_search(key)

        with (
            st.container(key="material_gallery"),
            st.expander(_t(locale, "word_material_review"), expanded=False),
        ):
            st.caption(_t(locale, "word_material_review_help"))
            st.caption(_t(locale, "material_overview_help"))

            def choose_word(word_index: int) -> None:
                st.session_state["material_word_index"] = word_index

            with st.container(height=190 if len(entries) > 12 else "content", border=False):
                for start in range(0, len(entries), 4):
                    overview_columns = st.columns(4)
                    for word_index in range(start, min(start + 4, len(entries))):
                        with overview_columns[word_index - start]:
                            status_name = material_states[word_index].review_status(
                                has_saved_pin=word_index in existing_pins
                            )
                            st.button(
                                f"{word_index + 1}. {entries[word_index].english} · "
                                f"{_t(locale, 'material_status_' + status_name)}",
                                key=f"material_overview_{word_index}",
                                on_click=choose_word,
                                args=(word_index,),
                                use_container_width=True,
                            )
            material_word_labels = [
                f"{i + 1}. {word.english} · {word.chinese}" for i, word in enumerate(entries)
            ]
            _normalize_word_index_state("material_word_index", material_word_labels)
            navigation = st.columns([5, 1.3, 1.3, 0.8], vertical_alignment="bottom")
            with navigation[0]:
                index = st.selectbox(
                    _t(locale, "material_current_word"),
                    range(len(entries)),
                    format_func=lambda i: material_word_labels[i],
                    key="material_word_index",
                )

            def move_word(offset: int) -> None:
                st.session_state["material_word_index"] = index + offset

            with navigation[1]:
                st.button(
                    _t(locale, "material_previous_word"),
                    key="material_previous_word",
                    disabled=index == 0,
                    on_click=move_word,
                    args=(-1,),
                    use_container_width=True,
                )
            with navigation[2]:
                st.button(
                    _t(locale, "material_next_word"),
                    key="material_next_word",
                    disabled=index == len(entries) - 1,
                    on_click=move_word,
                    args=(1,),
                    use_container_width=True,
                )
            with navigation[3]:
                st.caption(f"{index + 1} / {len(entries)}")
            entry = entries[index]
            state = material_states[index]
            gallery = state.gallery
            query_widget = f"material_query_{index}"
            st.session_state.setdefault(query_widget, state.search_query or entry.english)

            def change_query() -> None:
                state.set_query(str(st.session_state[query_widget]))

            search_row = st.columns([5, 1.5], vertical_alignment="bottom")
            with search_row[0]:
                st.text_input(
                    _t(locale, "material_search_query"),
                    key=query_widget,
                    max_chars=120,
                    on_change=change_query,
                    help=_t(locale, "material_search_query_help"),
                )
            with search_row[1]:
                search_clicked = st.button(
                    _t(locale, "material_search_again" if gallery else "material_search"),
                    key=f"search_candidates_{index}",
                    disabled=index not in remote_search_keys,
                    use_container_width=True,
                    help=_t(locale, "material_search_help"),
                )
            st.caption(
                f"{remote_choice.value.title()} · "
                + _t(locale, "material_candidate_count").format(
                    count=len(gallery.candidates) if gallery else form["material"]["pool_size"]
                )
            )

            def choose_candidate(candidate: RemoteImageCandidate, *, manual: bool) -> None:
                provider = cast(
                    PexelsImageProvider | PixabayImageProvider,
                    _image_provider(form, settings, effective_local_materials),
                )
                try:
                    root = settings.storage_dir / "_session_previews" / f"selected-{uuid4().hex}"
                    asset = provider.download_candidate(candidate, root / "material")
                    # Switch only after the full-size image has been safely downloaded.
                    state.select(candidate, asset, manual=manual)
                    st.session_state.pop(f"material_override_{index}", None)
                    _invalidate_card_preview()
                finally:
                    _close_provider(provider)

            if search_clicked:
                search_succeeded = False
                candidate_provider = cast(
                    PexelsImageProvider | PixabayImageProvider,
                    _image_provider(form, settings, effective_local_materials),
                )
                try:
                    with st.spinner(_t(locale, "material_searching")):
                        candidates = candidate_provider.search(
                            state.search_query or entry.english,
                            VideoAspect(str(form["aspect"])),
                            pool_size=preview_material_style.pool_size,
                        )
                        root = settings.storage_dir / "_session_previews" / f"gallery-{uuid4().hex}"
                        thumbnails: list[Path | None] = []
                        for candidate_index, candidate in enumerate(candidates):
                            try:
                                thumbnail = candidate_provider.download_candidate(
                                    candidate, root / str(candidate_index), thumbnail=True
                                )
                                thumbnails.append(thumbnail.path)
                            except (ApplicationError, OSError, ValueError):
                                thumbnails.append(None)
                        gallery = CandidateGallery(candidates, tuple(thumbnails))
                        state.gallery = gallery
                        # An explicit choice or upload survives a repeated search unchanged.
                        if (
                            candidates
                            and state.selection is None
                            and state.upload is None
                            and index not in existing_pins
                        ):
                            context = ImageSelectionContext(
                                entry_index=index,
                                pool_size=preview_material_style.pool_size,
                                mode=preview_material_style.selection_mode,
                                seed=int(form["job_seed"]),
                            )
                            choose_candidate(
                                candidates[_candidate_index(len(candidates), context)], manual=False
                            )
                        search_succeeded = True
                except (ApplicationError, OSError, ValidationError, ValueError) as exc:
                    st.error(_safe_message(exc, locale))
                finally:
                    _close_provider(candidate_provider)
                if search_succeeded:
                    st.rerun()

            status = st.empty()
            if gallery is not None and not gallery.candidates:
                st.warning(_t(locale, "remote_material_missing"))
            if gallery:
                for row_start in range(0, len(gallery.candidates), 4):
                    columns = st.columns(4)
                    for offset, column in enumerate(columns):
                        candidate_index = row_start + offset
                        if candidate_index >= len(gallery.candidates):
                            break
                        candidate = gallery.candidates[candidate_index]
                        selection = state.selection
                        selected = (
                            state.upload is None
                            and selection is not None
                            and selection.candidate == candidate
                        )
                        card_key = (
                            f"candidate_card_selected_{index}_{candidate_index}"
                            if selected
                            else f"candidate_card_{index}_{candidate_index}"
                        )
                        with column, st.container(border=True, key=card_key):
                            thumbnail_path = gallery.thumbnails[candidate_index]
                            if thumbnail_path is not None and thumbnail_path.is_file():
                                st.image(str(thumbnail_path), width="stretch")
                            else:
                                st.caption(_t(locale, "material_thumbnail_failed"))
                            st.caption(
                                _t(locale, "material_candidate").format(number=candidate_index + 1)
                            )
                            if st.button(
                                _t(locale, "material_selected" if selected else "material_use"),
                                key=f"use_candidate_{index}_{candidate_index}",
                                disabled=selected,
                                use_container_width=True,
                            ):
                                try:
                                    choose_candidate(candidate, manual=True)
                                    st.rerun()
                                except (
                                    ApplicationError,
                                    OSError,
                                    ValidationError,
                                    ValueError,
                                ) as exc:
                                    st.error(_safe_message(exc, locale))
            else:
                st.caption(_t(locale, "material_search_prompt"))

            st.divider()
            st.markdown(f"**{_t(locale, 'local_material_override')}**")
            local_override = st.file_uploader(
                _t(locale, "material_upload_prompt"),
                type=("png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v", "webm"),
                key=f"material_override_{index}",
                max_upload_size=MAX_LOCAL_VIDEO_BYTES // MIB,
            )
            st.caption(_t(locale, "mixed_upload_limits"))
            if local_override is not None:
                previous_upload = state.upload
                previous_id = getattr(previous_upload, "file_id", None)
                if previous_upload is None or previous_id != getattr(
                    local_override, "file_id", None
                ):
                    try:
                        validation_path = _save_upload(
                            local_override,
                            settings.storage_dir / "_session_previews" / "uploads",
                            ".jpg",
                            allowed_suffixes=frozenset(
                                {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".m4v", ".webm"}
                            ),
                        )
                        probe_material(validation_path)
                    except (ApplicationError, OSError, ValidationError, ValueError) as exc:
                        st.error(_safe_message(exc, locale))
                        st.session_state.pop(f"material_override_{index}", None)
                    else:
                        state.set_upload(local_override)
                        _invalidate_card_preview()
                        st.rerun()
            active_upload = state.upload
            if active_upload is not None:
                contents = active_upload.getvalue()
                if Path(active_upload.name).suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}:
                    st.video(contents, width=240)
                else:
                    st.image(contents, width=240)
                st.caption(active_upload.name)

                def remove_upload() -> None:
                    state.set_upload(None)
                    st.session_state.pop(f"material_override_{index}", None)
                    _invalidate_card_preview()

                st.button(
                    _t(locale, "material_remove_upload"),
                    key=f"remove_material_override_{index}",
                    on_click=remove_upload,
                )
                status.info(_t(locale, "local_material_selected"))
            elif state.selection is not None:
                selection = state.selection
                candidate_number = None
                if gallery and selection:
                    candidate_number = next(
                        (
                            i + 1
                            for i, item in enumerate(gallery.candidates)
                            if item == selection.candidate
                        ),
                        None,
                    )
                mode = _t(
                    locale, "material_manual" if selection and selection.manual else "material_auto"
                )
                status.info(
                    _t(locale, "material_using_candidate").format(
                        number=candidate_number, mode=mode
                    )
                    if candidate_number is not None
                    else _t(locale, "remote_material_pinned")
                )
                if candidate_number is None:
                    st.image(str(state.selection.asset.path), width=240)
            elif index in existing_pins:
                pin = existing_pins[index]
                status.info(_t(locale, "remote_material_pinned"))
                if pin.asset.kind is MaterialKind.VIDEO:
                    st.video(str(pin.asset.path), width=240)
                else:
                    st.image(str(pin.asset.path), width=240)
            else:
                status.info(_t(locale, "remote_material_automatic"))
            st.caption(_t(locale, "material_upload_help").format(word=entry.english))
            st.caption(_t(locale, "material_untouched_help"))
        material_override_uploads = {
            index: word_state.upload
            for index, word_state in material_states.items()
            if word_state.upload is not None
        }

        pins_by_index = dict(existing_pins)
        pins_by_index.update(
            {
                index: PinnedMaterial(entry_index=index, asset=word_state.selection.asset)
                for index, word_state in material_states.items()
                if word_state.selection is not None
            }
        )
        for index in material_override_uploads:
            pins_by_index.pop(index, None)
        form["pinned_materials"] = [pins_by_index[index] for index in sorted(pins_by_index)]

    form["material_queries"] = {
        index: state.search_query
        for index, state in material_states.items()
        if index < len(entries) and state.search_query
    }

    preview_choices = entries or [WordEntry(chinese="测试", english="test", phonetic="/test/")]
    preview_word_labels = (
        [f"{i + 1}. {word.english} · {word.chinese}" for i, word in enumerate(entries)]
        if entries
        else [_t(locale, "preview_sample_word")]
    )
    _normalize_word_index_state("preview_word_index", preview_word_labels)
    card_types = ["answer", "question"] if form["question"]["enabled"] else ["answer"]
    if st.session_state.get("preview_card_type", "answer") not in card_types:
        st.session_state["preview_card_type"] = "answer"
    # Radio values are serialized as labels. Rebuild the widget per locale so
    # the browser cannot retain the previous language's selected label.
    card_type_widget = f"preview_card_type_{locale.value}"
    if st.session_state.get(card_type_widget, "answer") not in card_types:
        st.session_state[card_type_widget] = "answer"
    preview_controls = st.columns([3, 2], vertical_alignment="bottom")
    with preview_controls[0]:
        preview_index = st.selectbox(
            _t(locale, "preview_word"),
            range(len(preview_choices)),
            format_func=lambda i: preview_word_labels[i],
            key="preview_word_index",
            disabled=not entries,
        )
    with preview_controls[1]:
        preview_card_type = st.radio(
            _t(locale, "preview_card_type"),
            card_types,
            index=card_types.index(st.session_state.get("preview_card_type", "answer")),
            format_func=lambda kind: _t(locale, f"preview_{kind}"),
            key=card_type_widget,
            on_change=_remember_conditional_value,
            args=(card_type_widget, "preview_card_type"),
            horizontal=True,
            help=_t(locale, "preview_question_help"),
        )
    st.session_state["preview_card_type"] = preview_card_type
    st.caption(_t(locale, "preview_help"))

    current_preview_key = _card_preview_key(
        form,
        entries,
        background_upload,
        effective_background,
        material_uploads or [],
        effective_local_materials,
        material_override_uploads,
        remote_search_keys.get(preview_index),
        entry_index=preview_index,
        card_type=preview_card_type,
    )
    if st.session_state.get("last_preview_key") != current_preview_key:
        _invalidate_card_preview()

    blockers: list[str] = []
    if not entries and not str(form["topic"]).strip():
        blockers.append(_t(locale, "missing_content"))
    if background_upload is None and effective_background is None:
        blockers.append(_t(locale, "missing_background"))
    llm_needed = (not entries and bool(str(form["topic"]).strip())) or (
        form["phonetic_mode"] is PhoneticMode.AUTOMATIC
        and any(not entry.phonetic for entry in entries)
    )
    if llm_needed and _llm_secret(form, settings) is None:
        blockers.append(_t(locale, "missing_llm_key"))
    if form["material"]["enabled"]:
        if (
            form["material_source"] == MaterialSource.LOCAL.value
            and not material_uploads
            and not effective_local_materials
        ):
            blockers.append(_t(locale, "missing_materials"))
        if form["material_source"] == MaterialSource.REMOTE.value:
            remote_choice = RemoteMaterialProvider(str(form["remote_provider"]))
            provider_key = (
                _available_secret(str(form["pexels_key"]), settings.secrets.pexels_api_key)
                if remote_choice is RemoteMaterialProvider.PEXELS
                else _available_secret(str(form["pixabay_key"]), settings.secrets.pixabay_api_key)
            )
            if provider_key is None:
                blockers.append(_t(locale, "missing_provider_key"))
    if form["background_music"]["enabled"] and music_upload is None and loaded_music is None:
        blockers.append(_t(locale, "missing_music"))
    if script_error:
        blockers.append(_t(locale, "script_invalid"))
    try:
        validation_form = form
        if music_upload is not None:
            validation_form = {
                **form,
                "background_music": {**form["background_music"], "enabled": False},
            }
        _build_request(
            validation_form,
            entries or [WordEntry(english="preview")],
            effective_background,
            effective_local_materials,
        )
    except (ValueError, ValidationError) as exc:
        blockers.append(_safe_message(exc, locale))
    for blocker in dict.fromkeys(blockers):
        st.info(blocker)

    preview_column, generate_column = st.columns([0.3, 0.7])
    with preview_column:
        preview = st.button(_t(locale, "preview"), use_container_width=True)
    with generate_column:
        generate = st.button(
            _t(locale, "generate_video"),
            type="primary",
            disabled=bool(blockers),
            use_container_width=True,
        )
    if generate:
        try:
            background_path = (
                _save_upload(
                    background_upload,
                    settings.storage_dir,
                    ".jpg",
                    allowed_suffixes=frozenset({".png", ".jpg", ".jpeg", ".webp"}),
                )
                if background_upload is not None
                else effective_background
            )
            assert background_path is not None
            local_paths = (
                _save_uploads(
                    material_uploads,
                    settings.storage_dir,
                    allowed_suffixes=frozenset(
                        {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".m4v", ".webm"}
                    ),
                )
                if material_uploads
                else effective_local_materials
            )
            music_path = (
                _save_upload(
                    music_upload,
                    settings.storage_dir,
                    ".mp3",
                    allowed_suffixes=frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg"}),
                )
                if music_upload is not None
                else loaded_music
            )
            form["background_music"]["path"] = music_path
            if material_override_uploads:
                override_items = sorted(material_override_uploads.items())
                override_paths = _save_uploads(
                    [upload for _entry_index, upload in override_items],
                    settings.storage_dir,
                    allowed_suffixes=frozenset(
                        {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".mp4",
                            ".mov",
                            ".m4v",
                            ".webm",
                        }
                    ),
                )
                override_pins = [
                    PinnedMaterial(entry_index=entry_index, asset=probe_material(path))
                    for (entry_index, _upload), path in zip(
                        override_items, override_paths, strict=True
                    )
                ]
                overridden_indices = set(material_override_uploads)
                form["pinned_materials"] = [
                    pin
                    for pin in form.get("pinned_materials", [])
                    if pin.entry_index not in overridden_indices
                ] + override_pins
            request = _build_request(form, entries, background_path, local_paths)
            image_provider = _image_provider(form, settings, local_paths)
            try:
                vocabulary_provider = _llm_provider(form, settings)
                pipeline = GenerationPipeline(
                    storage=_job_storage(settings),
                    vocabulary_provider=vocabulary_provider,
                    image_provider=image_provider,
                    speech_provider=EdgeSpeechProvider(),
                    card_renderer=CardRenderer(),
                    video_composer=VideoComposer(),
                )
                progress_bar = st.progress(0, text=_t(locale, "preparing"))
                log_placeholder = st.empty()
                logs: list[str] = []

                def report(progress: PipelineProgress) -> None:
                    message = _localized_progress_message(locale, progress)
                    progress_bar.progress(progress.percent, text=message)
                    logs.append(message)
                    del logs[:-20]
                    log_placeholder.code("\n".join(logs))

                try:
                    result = pipeline.run(request, on_progress=report)
                finally:
                    if vocabulary_provider is not None:
                        _close_provider(vocabulary_provider)
            finally:
                _close_provider(image_provider)
            assert result.video_path is not None
            st.session_state["last_video_path"] = str(result.video_path)
            st.session_state["last_video_aspect"] = request.canvas.aspect.value
            st.session_state["last_job_id"] = result.job_id
            _invalidate_recent_tasks()
            _clear_task_selection()
            st.session_state["task_id"] = result.job_id
            st.success(_t(locale, "generation_complete"))
        except (ApplicationError, OSError, ValidationError, ValueError) as exc:
            st.error(_safe_message(exc, locale))

    result_container = st.container()
    _display_regeneration(settings, locale)
    with result_container:
        _display_result(locale)

    if preview:
        try:
            preview_entries = preview_choices
            preview_root = settings.storage_dir / "_session_previews"
            preview_background = (
                _save_upload(
                    background_upload,
                    preview_root,
                    ".jpg",
                    allowed_suffixes=frozenset({".png", ".jpg", ".jpeg", ".webp"}),
                )
                if background_upload is not None
                else effective_background
            )
            if preview_background is None:
                raise ValueError(_t(locale, "missing_background"))
            background_asset = probe_material(preview_background)
            if background_asset.kind is not MaterialKind.IMAGE:
                raise ValueError("The preview background must be a supported image.")
            preview_background = background_asset.path
            selected_override = material_override_uploads.get(preview_index)
            selected_pin = next(
                (
                    pin
                    for pin in form.get("pinned_materials", [])
                    if pin.entry_index == preview_index
                ),
                None,
            )
            local_pool = material_uploads or effective_local_materials
            local_index = (
                _candidate_index(
                    len(local_pool),
                    ImageSelectionContext(
                        entry_index=preview_index,
                        pool_size=int(form["material"]["pool_size"]),
                        mode=SelectionMode(str(form["selection_mode"])),
                        seed=int(form["job_seed"]),
                    ),
                )
                if local_pool
                else 0
            )
            placeholder = False
            if selected_override is not None:
                preview_source = _save_upload(
                    selected_override,
                    preview_root,
                    ".jpg",
                    allowed_suffixes=frozenset(
                        {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".mp4",
                            ".mov",
                            ".m4v",
                            ".webm",
                        }
                    ),
                )
            elif selected_pin is not None:
                preview_source = selected_pin.asset.path
            elif material_uploads:
                preview_source = _save_upload(
                    material_uploads[local_index],
                    preview_root,
                    ".jpg",
                    allowed_suffixes=frozenset(
                        {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".mp4",
                            ".mov",
                            ".m4v",
                            ".webm",
                        }
                    ),
                )
            elif effective_local_materials:
                preview_source = effective_local_materials[local_index]
            else:
                preview_source = _create_preview_tile(preview_root / f"tile-{uuid4().hex}.png")
                placeholder = bool(form["material"]["enabled"])
            preview_material = _preview_still_for_material(
                preview_source,
                preview_root,
                seed=int(form.get("job_seed") or 0),
                entry_index=preview_index,
            )
            if music_upload is not None:
                form["background_music"]["path"] = _save_upload(
                    music_upload,
                    preview_root,
                    ".mp3",
                    allowed_suffixes=frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg"}),
                )
            request = _build_request(
                form,
                preview_entries,
                preview_background,
                [preview_material],
            )
            destination = preview_root / f"preview-{uuid4().hex}.png"
            renderer = CardRenderer()
            if preview_card_type == "question":
                renderer.render_question(
                    request.question_text,
                    preview_background,
                    preview_material if request.material.enabled else None,
                    request,
                    destination,
                )
            else:
                renderer.render_answer(
                    preview_entries[preview_index],
                    preview_background,
                    preview_material if request.material.enabled else None,
                    request,
                    destination,
                )
            st.session_state["last_preview_placeholder"] = placeholder
            st.session_state["last_preview_path"] = str(destination)
            if current_preview_key is not None:
                st.session_state["last_preview_key"] = current_preview_key
            else:
                st.session_state.pop("last_preview_key", None)
        except (ApplicationError, OSError, ValidationError, ValueError) as exc:
            st.error(_safe_message(exc, locale))
    preview_path = st.session_state.get("last_preview_path")
    if preview_path and Path(str(preview_path)).is_file():
        try:
            if st.session_state.get("last_preview_placeholder", False):
                st.info(_t(locale, "preview_placeholder"))
            _display_image_preview(Path(str(preview_path)), form["aspect"])
        except OSError as exc:
            st.error(_safe_message(exc, locale))


if __name__ == "__main__":
    main()
