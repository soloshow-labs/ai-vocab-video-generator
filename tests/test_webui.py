import json
import math
import secrets
import struct
import wave
from collections.abc import Sequence
from pathlib import Path, PosixPath
from typing import Any, ClassVar

import pytest
from PIL import Image
from pydantic import ValidationError
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

import ai_vocab_video_generator.pipeline as pipeline_module
import ai_vocab_video_generator.providers.images as images_module
import ai_vocab_video_generator.providers.llm as llm_module
import ai_vocab_video_generator.webui as webui_module
from ai_vocab_video_generator.config import AppSettings, LLMPreset
from ai_vocab_video_generator.domain import (
    AnchorOffsets,
    GenerationRequest,
    GenerationResult,
    JobStatus,
    MaterialAsset,
    MaterialFitMode,
    MaterialKind,
    MaterialSource,
    PinnedMaterial,
    PipelineProgress,
    PipelineStage,
    RenderSettings,
    SelectionMode,
    TextElementStyle,
    VideoAspect,
    VocabularySettings,
    WordEntry,
)
from ai_vocab_video_generator.errors import ApplicationError, ProviderError
from ai_vocab_video_generator.i18n import Locale, translate
from ai_vocab_video_generator.providers.base import ImageSelectionContext
from ai_vocab_video_generator.providers.images import (
    LocalImageProvider,
    RemoteImageCandidate,
    seeded_video_start_offset,
)
from ai_vocab_video_generator.rendering.cards import CardLayers
from ai_vocab_video_generator.rendering.video import (
    MaterialVideoOverlay,
    VideoComposer,
    VideoSegment,
)
from ai_vocab_video_generator.storage import JobStorage
from ai_vocab_video_generator.webui import (
    _APP_STYLES,
    _llm_credential_slot,
    _llm_provider,
    _localized_progress_message,
    _result_preview_width,
    _safe_message,
    _save_upload,
)

WEBUI = Path(__file__).parents[1] / "src" / "ai_vocab_video_generator" / "webui.py"
PUBLIC_WEBUI = Path(__file__).parents[1] / "streamlit_app.py"


def _without_credentials(monkeypatch) -> None:
    for name in (
        "AIVVG_OPENAI_API_KEY",
        "AIVVG_DEEPSEEK_API_KEY",
        "AIVVG_MOONSHOT_API_KEY",
        "AIVVG_QWEN_API_KEY",
        "AIVVG_CUSTOM_API_KEY",
        "AIVVG_PEXELS_API_KEY",
        "AIVVG_PIXABAY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_public_demo_is_session_isolated_and_never_uses_operator_keys(monkeypatch) -> None:
    monkeypatch.setenv("AIVVG_OPENAI_API_KEY", "operator-secret")
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "operator-image-secret")

    first = AppTest.from_file(str(PUBLIC_WEBUI)).run(timeout=15)
    second = AppTest.from_file(str(PUBLIC_WEBUI)).run(timeout=15)

    assert not first.exception
    assert not second.exception
    assert (
        first.session_state["_public_demo_storage_dir"]
        != second.session_state["_public_demo_storage_dir"]
    )
    assert "aivvg-public-demo" in first.session_state["_public_demo_storage_dir"]
    assert first.text_input(key="llm_key_input").value == ""
    assert first.selectbox(key="llm_preset").options == [
        "OpenAI",
        "DeepSeek",
        "Moonshot",
        "Qwen",
    ]
    assert first.text_input(key="llm_base_url").disabled
    assert first.text_input(key="llm_model").disabled
    assert first.number_input(key="word_count").value == 5
    assert first.number_input(key="material_pool_size_widget").value == 4
    assert not first.get("audio_input")
    assert "重新生成已有任务" not in [item.label for item in first.expander]
    assert any("公开体验版" in str(item.value) for item in first.info)

    first.session_state["llm_base_url"] = "https://untrusted.example/v1"
    first.session_state["llm_model"] = "untrusted-model"
    first.run(timeout=15)
    assert first.text_input(key="llm_base_url").value == "https://api.openai.com/v1"
    assert first.text_input(key="llm_model").value == "gpt-5.6-terra"

    six_words = "\n".join(f"单词{index}\nword{index}\n/word{index}/" for index in range(1, 7))
    first.text_area(key="script_text").set_value(six_words).run(timeout=15)

    assert any("最多支持 5 个单词" in str(item.value) for item in first.info)
    assert next(button for button in first.button if button.label == "生成视频").disabled


def _ready_public_demo(monkeypatch, tmp_path: Path) -> AppTest:
    _without_credentials(monkeypatch)
    monkeypatch.setattr(webui_module.tempfile, "gettempdir", lambda: str(tmp_path / "cloud-tmp"))
    _patch_generation_fakes(monkeypatch)
    monkeypatch.setattr(webui_module, "GenerationPipeline", _FakePipeline)
    background = _image(tmp_path / "background.png", "white")
    app = AppTest.from_file(str(PUBLIC_WEBUI))
    for key, value in {
        "script_text": "苹果\napple\n/ˈæp.əl/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "loaded_background_path": str(background),
        "material_enabled": False,
    }.items():
        app.session_state[key] = value
    app.run(timeout=15)
    assert not app.exception
    assert not next(button for button in app.button if button.label == "生成视频").disabled
    return app


def test_public_demo_generation_lock_blocks_contention_and_releases_after_success(
    monkeypatch, tmp_path: Path
) -> None:
    app = _ready_public_demo(monkeypatch, tmp_path)
    lock = webui_module._PUBLIC_DEMO_GENERATION_LOCK
    assert lock.acquire(blocking=False)
    try:
        next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    finally:
        lock.release()

    assert not app.exception
    assert _FakePipeline.run_requests == []
    assert any("当前已有视频正在生成" in str(item.value) for item in app.warning)

    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    assert not app.exception
    assert len(_FakePipeline.run_requests) == 1
    assert lock.acquire(blocking=False)
    lock.release()


def test_public_demo_generation_lock_releases_after_failure(monkeypatch, tmp_path: Path) -> None:
    class FailingPipeline(_FakePipeline):
        def run(self, request: GenerationRequest, on_progress: Any = None) -> GenerationResult:
            del request, on_progress
            raise ProviderError("Synthetic generation failure.")

    app = _ready_public_demo(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_module, "GenerationPipeline", FailingPipeline)
    monkeypatch.setattr(webui_module, "GenerationPipeline", FailingPipeline)

    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    assert not app.exception
    assert any("Synthetic generation failure" in str(item.value) for item in app.error)
    lock = webui_module._PUBLIC_DEMO_GENERATION_LOCK
    assert lock.acquire(blocking=False)
    lock.release()


def test_public_demo_storage_budget_blocks_new_work(tmp_path: Path) -> None:
    storage_dir = tmp_path / "aivvg-public-demo" / ("a" * 32)
    storage_dir.mkdir(parents=True)
    (storage_dir / "result.mp4").write_bytes(b"1234")

    assert webui_module._public_demo_storage_has_capacity(
        storage_dir, budget_bytes=5, minimum_free_bytes=0
    )
    assert not webui_module._public_demo_storage_has_capacity(
        storage_dir, budget_bytes=4, minimum_free_bytes=0
    )


def test_public_demo_download_hides_the_internal_task_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(webui_module.tempfile, "gettempdir", lambda: str(tmp_path / "cloud-tmp"))
    downloads: list[tuple[str, bytes]] = []
    original_download = webui_module.st.download_button

    def capture_download(*args: Any, **kwargs: Any) -> Any:
        downloads.append((kwargs["file_name"], kwargs["data"]))
        return original_download(*args, **kwargs)

    monkeypatch.setattr(webui_module.st, "download_button", capture_download)
    video = tmp_path / "video-0001.mp4"
    video.write_bytes(b"public-result")
    app = AppTest.from_file(str(PUBLIC_WEBUI))
    app.session_state["last_video_path"] = str(video)
    app.session_state["last_job_id"] = "a" * 32

    app.run(timeout=15)

    assert not app.exception
    assert downloads == [("vocabulary-video.mp4", b"public-result")]
    assert not any(item.value == "a" * 32 for item in app.code)


@pytest.mark.parametrize(
    ("stage", "zh_text", "en_text"),
    [
        (PipelineStage.PREPARING, "正在准备生成任务", "Preparing job"),
        (PipelineStage.VOCABULARY, "单词信息已准备完成", "Vocabulary is ready"),
        (PipelineStage.IMAGES, "单词素材已准备完成", "Vocabulary materials are ready"),
        (PipelineStage.SPEECH, "朗读音频已准备完成", "Narration is ready"),
        (PipelineStage.CARDS, "单词画面已准备完成", "Vocabulary cards are ready"),
        (PipelineStage.COMPOSING, "正在合成视频", "Composing video"),
        (PipelineStage.COMPLETE, "视频已生成完成", "Video is ready"),
    ],
)
def test_pipeline_progress_is_localized_by_stage(
    stage: PipelineStage,
    zh_text: str,
    en_text: str,
) -> None:
    progress = PipelineProgress(stage=stage, percent=0, message="internal message")

    assert _localized_progress_message(Locale.ZH_CN, progress) == zh_text
    assert _localized_progress_message(Locale.EN_US, progress) == en_text


@pytest.mark.parametrize(
    ("aspect", "expected_width"),
    [
        (VideoAspect.PORTRAIT, 280),
        (VideoAspect.LANDSCAPE, 480),
        ("invalid", 280),
    ],
)
def test_result_preview_width_matches_video_aspect(aspect: object, expected_width: int) -> None:
    assert _result_preview_width(aspect) == expected_width


@pytest.mark.parametrize(
    ("aspect", "expected_width"),
    [(VideoAspect.PORTRAIT, 280), (VideoAspect.LANDSCAPE, 480)],
)
def test_image_preview_is_bounded_by_video_aspect(
    monkeypatch,
    tmp_path: Path,
    aspect: VideoAspect,
    expected_width: int,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        webui_module.st,
        "image",
        lambda image, **kwargs: calls.append((image, kwargs)),
    )
    display_preview = getattr(webui_module, "_display_image_preview", None)

    assert display_preview is not None
    image = _image(tmp_path / "preview.png")
    display_preview(image, aspect)

    assert calls == [(image, {"width": expected_width})]


def _image(path: Path, color: str = "blue") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (24, 24), color) as image:
        image.save(path)
    return path


def _write_color_video(destination: Path) -> Path:
    moviepy = pytest.importorskip("moviepy.editor")
    np = pytest.importorskip("numpy")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def make_frame(time: float) -> Any:
        color = (255, 0, 0) if time < 0.5 else (0, 0, 255)
        return np.full((48, 64, 3), color, dtype=np.uint8)

    clip = moviepy.VideoClip(make_frame=make_frame, duration=1.0)
    try:
        clip.write_videofile(
            str(destination),
            fps=12,
            codec="libx264",
            audio=False,
            logger=None,
            threads=1,
        )
    finally:
        clip.close()
    return destination


def _write_sine_wav(destination: Path, duration: float = 0.5) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44_100
    frames = b"".join(
        struct.pack(
            "<h",
            round(0.25 * 32_767 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate)),
        )
        for index in range(round(duration * sample_rate))
    )
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return destination


def test_oversized_upload_is_rejected_before_materializing_its_bytes(tmp_path: Path) -> None:
    class OversizedUpload:
        name = "private-video.mp4"
        size = 129 * 1024 * 1024

        def getvalue(self) -> bytes:
            raise AssertionError("oversized upload bytes must not be materialized")

    with pytest.raises(ValueError, match="size limit"):
        _save_upload(
            OversizedUpload(),  # type: ignore[arg-type]
            tmp_path / "storage",
            ".mp4",
            allowed_suffixes=frozenset({".mp4"}),
        )

    assert not (tmp_path / "storage" / "_session_uploads").exists()


class _FakePipeline:
    run_requests: ClassVar[list[GenerationRequest]] = []
    regenerated_job_ids: ClassVar[list[str]] = []
    regenerated_replacements: ClassVar[list[dict[int, Path]]] = []

    def __init__(self, **dependencies: Any) -> None:
        self.storage = dependencies["storage"]

    def run(self, request: GenerationRequest, on_progress: Any = None) -> GenerationResult:
        del on_progress
        type(self).run_requests.append(request)
        paths = self.storage.create_job(request)
        video = paths.artifacts / "fake.mp4"
        video.write_bytes(b"fake-video")
        return GenerationResult(
            job_id=paths.job_id,
            status=JobStatus.COMPLETE,
            video_path=video,
            manifest_path=paths.manifest,
        )

    def regenerate(
        self,
        job_id: str,
        replacements: Any = None,
        on_progress: Any = None,
    ) -> GenerationResult:
        del on_progress
        type(self).regenerated_job_ids.append(job_id)
        type(self).regenerated_replacements.append(dict(replacements or {}))
        paths = self.storage.paths(job_id)
        video = paths.artifacts / "regenerated.mp4"
        video.write_bytes(b"regenerated-video")
        return GenerationResult(
            job_id=job_id,
            status=JobStatus.COMPLETE,
            video_path=video,
            manifest_path=paths.manifest,
        )


class _FakeRemoteProvider:
    fetches: ClassVar[list[tuple[str, int, int]]] = []
    searches: ClassVar[list[tuple[str, int]]] = []
    downloads: ClassVar[list[tuple[str, bool]]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def search(
        self, query: str, _aspect: Any, *, pool_size: int
    ) -> tuple[RemoteImageCandidate, ...]:
        type(self).searches.append((query, pool_size))
        return tuple(
            RemoteImageCandidate(
                f"{query}-{i}",
                f"https://images.pexels.com/{query}-{i}.jpg",
                f"https://images.pexels.com/{query}-{i}-thumb.jpg",
            )
            for i in range(pool_size)
        )

    def download_candidate(
        self, candidate: RemoteImageCandidate, destination_stem: Path, *, thumbnail: bool = False
    ) -> MaterialAsset:
        type(self).downloads.append((str(candidate.source_id), thumbnail))
        destination = _image(destination_stem.with_suffix(".jpg"), "orange")
        return MaterialAsset(
            path=destination, kind=MaterialKind.IMAGE, source_id=candidate.source_id
        )

    def fetch(
        self,
        query: str,
        destination_stem: Path,
        _aspect: Any,
        context: Any = None,
    ) -> MaterialAsset:
        assert context is not None
        type(self).fetches.append((query, context.seed, context.pool_size))
        destination = _image(destination_stem.with_suffix(".jpg"), "orange")
        return MaterialAsset(
            path=destination,
            kind=MaterialKind.IMAGE,
            source_id=f"remote-{context.seed}",
        )


class _ManifestCardRenderer:
    def render_answer_layers(
        self,
        _entry: WordEntry,
        _background: Path | None,
        _request: GenerationRequest,
        base_destination: Path,
        foreground_destination: Path,
    ) -> CardLayers:
        base_destination.parent.mkdir(parents=True, exist_ok=True)
        base_destination.write_bytes(b"base")
        foreground_destination.write_bytes(b"foreground")
        return CardLayers(base_destination, foreground_destination)


class _ManifestComposer:
    def compose(
        self,
        _segments: Sequence[VideoSegment],
        destination: Path,
        *,
        render: RenderSettings,
        music: Any,
    ) -> Path:
        del render, music
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"manifest-video")
        return destination


class _ManifestSpeechProvider:
    def synthesize(
        self,
        _text: str,
        destination: Path,
        *,
        voice: str,
        rate: str,
        volume: str = "+0%",
    ) -> Path:
        del voice, rate, volume
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"speech")
        return destination


def _patch_generation_fakes(monkeypatch) -> None:
    _FakePipeline.run_requests = []
    _FakePipeline.regenerated_job_ids = []
    _FakePipeline.regenerated_replacements = []
    _FakeRemoteProvider.fetches = []
    _FakeRemoteProvider.searches = []
    _FakeRemoteProvider.downloads = []
    monkeypatch.setattr(pipeline_module, "GenerationPipeline", _FakePipeline)
    monkeypatch.setattr(images_module, "PexelsImageProvider", _FakeRemoteProvider)


def test_webui_defaults_to_simplified_chinese_and_restores_section_order(monkeypatch) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    assert app.title[0].value.startswith("AI 单词视频生成器 v")
    assert app.caption[0].value == "一键生成带朗读的单词学习视频。"
    assert "padding-top: 2.25rem" in _APP_STYLES
    assert "color: #ff4b4b !important" in _APP_STYLES
    assert "text-decoration: underline" not in _APP_STYLES
    assert "text-decoration: none !important" in _APP_STYLES
    assert "summary:hover p" in _APP_STYLES
    assert "details[open] > summary" not in _APP_STYLES
    assert [item.label for item in app.expander] == [
        "基础设置 (:blue[点击展开])",
        "问答与进度",
        "朗读设置",
        "画面样式",
        "音频与输出",
        "重新生成已有任务",
    ]
    assert [item.label for item in app.tabs] == [
        "问题设置",
        "问题朗读设置",
        "进度条设置",
        "中文朗读",
        "快速英语朗读",
        "慢速英语朗读",
        "英文文本",
        "音标文本",
        "中文文本",
        "图片素材设置",
        "背景音乐",
        "渲染设置",
    ]
    headings = [item.value for item in app.markdown if item.value.startswith("**")]
    assert headings == [
        "**主题与单词信息**",
        "**音标设置**",
        "**画面与素材**",
        "**视频设置**",
        "**图片素材设置**",
    ]
    assert next(item for item in app.checkbox if item.label == "自动音标").value is True
    assert next(item for item in app.checkbox if item.label == "手动音标").value is False
    assert next(item for item in app.checkbox if item.label == "启用问题片段").value is False
    assert next(item for item in app.checkbox if item.label == "启用进度条").value is True
    assert sum(item.label == "启用" for item in app.checkbox) == 6


def test_chinese_controls_keep_specific_section_names_instead_of_generic_labels(
    monkeypatch,
) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    labels = {
        *(item.label for item in app.text_input),
        *(item.label for item in app.number_input),
        *(item.label for item in app.slider),
        *(item.label for item in app.color_picker),
        *(item.label for item in app.selectbox),
    }
    for expected in (
        "问题字体",
        "问题字体大小",
        "问题字体颜色",
        "问题描边颜色",
        "问题上边距",
        "进度条宽度",
        "进度条开始颜色",
        "图片素材宽度",
        "图片素材形状",
        "中文朗读次数",
        "中文朗读声音",
        "英文快读朗读次数",
        "英文快读朗读声音",
        "英文字体",
        "英文字体大小",
        "英文上边距",
    ):
        assert expected in labels


def test_question_dependency_is_visible_and_preserves_common_input(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    question = next(item for item in app.checkbox if item.label == "启用问题片段")
    progress = next(item for item in app.checkbox if item.label == "启用进度条")
    question_narration = next(item for item in app.checkbox if item.label == "启用问题朗读")
    captions = "\n".join(str(item.value) for item in app.caption)
    assert question.value is False
    assert progress.disabled is True
    assert question_narration.disabled is True
    assert not [item for item in app.text_input if item.key == "question_text_widget"]
    assert "问题文本、问题朗读和进度条均不生效" in captions
    assert "仅在“启用问题片段”后生效" in captions

    question.check().run(timeout=15)
    question_text = next(item for item in app.text_input if item.key == "question_text_widget")
    assert question_text.value == "What is this?"
    assert next(item for item in app.checkbox if item.label == "启用进度条").disabled is False
    assert next(item for item in app.checkbox if item.label == "启用问题朗读").disabled is False

    question_text.set_value("Which word is this?").run(timeout=15)
    next(item for item in app.checkbox if item.label == "启用问题片段").uncheck().run(timeout=15)
    assert not [item for item in app.text_input if item.key == "question_text_widget"]
    next(item for item in app.checkbox if item.label == "启用问题片段").check().run(timeout=15)
    assert (
        next(item for item in app.text_input if item.key == "question_text_widget").value
        == "Which word is this?"
    )


def test_material_dependency_hides_common_inputs_and_preserves_source(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    source = next(item for item in app.selectbox if item.label == "素材来源")
    source.set_value("local").run(timeout=15)
    assert any(item.key == "material_uploads" for item in app.file_uploader)

    next(item for item in app.checkbox if item.label == "启用图片素材").uncheck().run(timeout=15)
    assert not [item for item in app.selectbox if item.label == "素材来源"]
    assert not [item for item in app.file_uploader if item.key == "material_uploads"]
    assert next(item for item in app.number_input if item.key == "material_width").disabled

    next(item for item in app.checkbox if item.label == "启用图片素材").check().run(timeout=15)
    assert next(item for item in app.selectbox if item.label == "素材来源").value == "local"
    assert any(item.key == "material_uploads" for item in app.file_uploader)


def test_low_frequency_controls_use_single_level_sections_and_named_tabs(monkeypatch) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    assert [item.label for item in app.expander][1:] == [
        "问答与进度",
        "朗读设置",
        "画面样式",
        "音频与输出",
        "重新生成已有任务",
    ]
    assert "渲染设置" in [item.label for item in app.tabs]
    assert "图片素材设置" in [item.label for item in app.tabs]
    assert next(item for item in app.number_input if item.key == "fps").value == 24
    assert (
        next(item for item in app.number_input if item.key == "material_pool_size_widget").value
        == 8
    )


def test_video_field_explanations_use_help_icons_without_stretching_the_columns(
    monkeypatch,
) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    controls = {item.key: item for item in (*app.selectbox, *app.number_input) if item.key}
    assert "本地上传使用你选择的图片或视频" in controls["material_source_widget"].help
    assert "顺序会按词条轮换素材" in controls["selection_mode_widget"].help
    assert "竖屏固定输出 1080 × 1920" in controls["aspect_widget"].help
    assert "最终 H.264 视频支持 12–60 FPS" in controls["fps"].help
    assert "远程搜索会为每个单词请求" in controls["material_pool_size_widget"].help
    assert "覆盖填满可能裁切素材边缘" in controls["material_fit_mode_widget"].help
    captions = "\n".join(str(item.value) for item in app.caption)
    assert "顺序会按词条轮换素材" not in captions
    assert "竖屏固定输出 1080 × 1920" not in captions
    assert "覆盖填满可能裁切素材边缘" not in captions


def test_critical_guidance_is_attached_to_its_control_labels(monkeypatch) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    assert app.expander[0].label == "基础设置 (:blue[点击展开])"
    llm = next(item for item in app.selectbox if item.key == "llm_preset")
    remote = next(item for item in app.selectbox if item.key == "remote_provider")
    assert llm.options == [
        "OpenAI",
        "DeepSeek",
        "Moonshot",
        "Qwen",
        "Ollama",
        "Custom",
    ]
    assert "接口地址和模型会自动填入" in llm.help
    assert "当前 Pexels 和 Pixabay 只提供静态图片" in remote.help
    assert next(item for item in app.text_input if item.key == "llm_key_input").label == (
        "API 密钥 (:red[必填]；[获取 OpenAI API 密钥](https://platform.openai.com/api-keys))"
    )
    assert next(item for item in app.text_input if item.key == "pexels_key_input").label == (
        "Pexels API 密钥 (:red[必填]；[申请 Pexels API 密钥](https://www.pexels.com/api/))"
    )
    assert next(item for item in app.text_input if item.key == "topic").label == (
        "视频主题 (输入主题后，可让 :red[AI 自动生成]单词信息)"
    )
    assert next(item for item in app.text_area if item.key == "script_text").label == (
        "单词信息 (:blue[可由 AI 根据主题生成，也可按当前音标模式手动编辑])"
    )
    assert (
        "仅用于 AI 根据主题生成单词信息"
        in next(item for item in app.number_input if item.key == "word_count").help
    )
    assert translate(Locale.ZH_CN, "record_topic") == "语音输入主题 (可选)"
    assert any(button.label == "使用 AI 根据主题生成单词信息" for button in app.button)
    assert "主题用于让:red[大模型自动生成]单词列表" not in "\n".join(
        str(item.value) for item in app.caption
    )


def test_basic_settings_can_check_llm_and_image_connections(monkeypatch) -> None:
    class ConnectionLLM:
        checks = 0

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def check_connection(self) -> None:
            type(self).checks += 1

        def close(self) -> None:
            pass

    class ConnectionImages:
        checks = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def check_connection(self) -> None:
            type(self).checks += 1

        def close(self) -> None:
            pass

    _without_credentials(monkeypatch)
    monkeypatch.setenv("AIVVG_OPENAI_API_KEY", "test-only-openai-key")
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "test-only-pexels-key")
    monkeypatch.setattr(llm_module, "OpenAICompatibleVocabularyProvider", ConnectionLLM)
    monkeypatch.setattr(images_module, "PexelsImageProvider", ConnectionImages)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    next(button for button in app.button if button.label == "测试大模型连接").click().run(
        timeout=15
    )
    assert ConnectionLLM.checks == 1
    assert "大模型连接成功，当前密钥、接口地址和模型均可用。" in [
        item.value for item in app.success
    ]
    next(button for button in app.button if button.label == "测试图片服务连接").click().run(
        timeout=15
    )

    assert ConnectionImages.checks == 1
    assert "图片服务连接成功，当前密钥可以正常搜索图片。" in [item.value for item in app.success]


def test_english_connection_controls_and_edge_tts_help_are_localized(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["locale"] = "en-US"

    app.run(timeout=15)

    assert any(button.label == "Test LLM Connection" for button in app.button)
    assert any(button.label == "Test Image Service" for button in app.button)
    play_voice = next(button for button in app.button if button.label == "Play Voice")
    assert "Edge TTS connection" in play_voice.help


def test_webui_exposes_complete_media_defaults_after_existing_advanced_sections(
    monkeypatch,
) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    tabs = [item.label for item in app.tabs]
    assert tabs.index("问题朗读设置") < tabs.index("背景音乐")
    assert tabs.index("中文文本") < tabs.index("背景音乐")
    assert next(item for item in app.number_input if item.label == "帧率 (FPS)").value == 24
    assert next(item for item in app.selectbox if item.label == "素材填充方式").value == "cover"
    assert next(item for item in app.checkbox if item.label == "启用背景音乐").value is False
    assert next(item for item in app.checkbox if item.label == "启用问题朗读").value is True
    question_repeats = next(
        item for item in app.number_input if item.key == "question_narration_repeats"
    )
    assert question_repeats.value == 1
    assert not [item for item in app.file_uploader if item.key == "material_uploads"]
    next(item for item in app.selectbox if item.label == "素材来源").set_value("local").run(
        timeout=15
    )
    assert any("图片和视频" in item.help for item in app.file_uploader if item.help)
    assert any(item.label == "素材图片和视频" for item in app.file_uploader)


def test_english_locale_has_the_same_advanced_controls(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["locale"] = "en-US"

    app.run(timeout=15)

    assert not app.exception
    assert app.title[0].value.startswith("AI Vocab Video Generator v")
    assert [item.label for item in app.expander] == [
        "Basic Settings (:blue[click to expand])",
        "Question and Progress",
        "Narration Settings",
        "Visual Styles",
        "Audio and Output",
        "Regenerate Existing Task",
    ]
    assert [item.label for item in app.tabs] == [
        "Question Settings",
        "Question Narration Settings",
        "Progress Bar Settings",
        "Chinese Narration",
        "Fast English Narration",
        "Slow English Narration",
        "English Text",
        "Phonetic Text",
        "Chinese Text",
        "Image Material Settings",
        "Background Music",
        "Render Settings",
    ]
    headings = [item.value for item in app.markdown if item.value.startswith("**")]
    assert "**Canvas and Materials**" in headings
    assert any(item.label == "Generate Video" for item in app.button)
    assert any(item.label == "Preview" for item in app.button)
    assert "Background Music" in [item.label for item in app.tabs]
    assert "**Task Import**" not in headings
    assert any(item.label == "Frame Rate (FPS)" for item in app.number_input)
    assert any(item.label == "Material Fill Mode" for item in app.selectbox)
    assert any(item.label == "Question Narration" for item in app.checkbox)
    assert next(item for item in app.text_input if item.key == "pexels_key_input").label == (
        "Pexels API Key (:red[required]; [request a Pexels API key](https://www.pexels.com/api/))"
    )
    guidance = "\n".join(
        [*(str(item.value) for item in app.caption)]
        + [
            str(value)
            for item in (*app.selectbox, *app.number_input, *app.text_input)
            for value in (getattr(item, "help", ""), item.label)
            if value
        ]
    )
    for expected in (
        "appears only during each question segment",
        "keeps the countdown cue",
        "Replacement materials are optional",
    ):
        assert expected in guidance


def test_settings_guidance_is_visible_and_canvas_size_is_not_independently_editable(
    monkeypatch,
) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    assert any(item.label == "素材来源" for item in app.selectbox)
    assert any(item.label == "素材分配方式" for item in app.selectbox)
    assert any(item.label == "输出比例" for item in app.selectbox)
    assert any(item.label == "素材填充方式" for item in app.selectbox)
    assert not any(item.label in {"画布宽度", "画布高度"} for item in app.number_input)
    guidance = "\n".join(
        [*(str(item.value) for item in app.caption)]
        + [
            str(value)
            for item in (*app.selectbox, *app.number_input, *app.text_input)
            for value in (getattr(item, "help", ""), item.label)
            if value
        ]
    )
    for expected in (
        "界面输入的密钥只保留在当前会话",
        ":red[AI 自动生成]",
        "本地上传使用你选择的图片或视频",
        "顺序会按词条轮换素材",
        "竖屏固定输出 1080 × 1920",
        "覆盖填满可能裁切素材边缘",
        "只在每个问题片段中显示",
        "仍会保留倒计时提示音",
        "替换素材可选",
    ):
        assert expected in guidance


def test_locale_switch_preserves_session_inputs_with_stable_widget_keys(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    recording = UploadedFile(
        UploadedFileRec(
            "topic-recording",
            "topic.wav",
            "audio/wav",
            b"RIFF-session-recording",
        ),
        FileURLs(),
    )
    app = AppTest.from_file(str(WEBUI))
    app.session_state["topic_recording"] = recording
    app.run(timeout=15)
    next(item for item in app.text_input if item.key == "llm_key_input").set_value(
        "session-llm-key"
    )
    next(item for item in app.text_input if item.key == "pexels_key_input").set_value(
        "session-pexels-key"
    )
    next(item for item in app.text_input if item.label == "任务 ID").set_value("a" * 32)
    assert app.session_state["topic_recording"].name == "topic.wav"
    next(item for item in app.selectbox if item.label == "语言").set_value("en-US")

    app.run(timeout=15)

    assert not app.exception
    assert next(item for item in app.text_input if item.key == "llm_key_input").value == (
        "session-llm-key"
    )
    assert next(item for item in app.text_input if item.key == "pexels_key_input").value == (
        "session-pexels-key"
    )
    assert next(item for item in app.text_input if item.label == "Task ID").value == "a" * 32
    preserved_recording = app.session_state["topic_recording"]
    assert preserved_recording.name == "topic.wav"
    assert preserved_recording.getvalue() == b"RIFF-session-recording"


def test_locale_switch_normalizes_stale_translated_aspect_widget_value(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    app.session_state["locale"] = "en-US"
    app.session_state["aspect"] = "竖屏 9:16"

    app.run(timeout=15)

    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "Output Aspect").value == (
        "portrait"
    )


def test_provider_switch_preserves_each_session_credential(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    next(item for item in app.text_input if item.key == "pexels_key_input").set_value(
        "pexels-session-value"
    )
    next(item for item in app.selectbox if item.label == "图片素材服务商").set_value("pixabay")
    app.run(timeout=15)
    next(item for item in app.text_input if item.key == "pixabay_key_input").set_value(
        "pixabay-session-value"
    )
    next(item for item in app.selectbox if item.label == "图片素材服务商").set_value("pexels")

    app.run(timeout=15)

    assert next(item for item in app.text_input if item.key == "pexels_key_input").value == (
        "pexels-session-value"
    )
    next(item for item in app.selectbox if item.label == "图片素材服务商").set_value("pixabay")
    app.run(timeout=15)
    assert next(item for item in app.text_input if item.key == "pixabay_key_input").value == (
        "pixabay-session-value"
    )
    assert next(item for item in app.text_input if item.key == "pixabay_key_input").label == (
        "Pixabay API 密钥 (:red[必填]；[获取 Pixabay API 密钥](https://pixabay.com/api/docs/))"
    )


def test_llm_preset_switch_never_reuses_another_provider_credential(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    key_input = next(item for item in app.text_input if item.key == "llm_key_input")
    key_input.set_value("openai-session-value")
    next(item for item in app.selectbox if item.label == "大模型服务商").set_value("deepseek")

    app.run(timeout=15)

    assert next(item for item in app.text_input if item.key == "llm_key_input").value == ""
    assert next(item for item in app.text_input if item.key == "llm_key_input").label == (
        "API 密钥 (:red[必填]；[获取 DeepSeek API 密钥](https://platform.deepseek.com/api_keys))"
    )
    next(item for item in app.text_input if item.key == "llm_key_input").set_value(
        "deepseek-session-value"
    )
    next(item for item in app.selectbox if item.label == "大模型服务商").set_value("openai")

    app.run(timeout=15)

    assert next(item for item in app.text_input if item.key == "llm_key_input").value == (
        "openai-session-value"
    )


def test_llm_api_key_label_matches_selected_provider(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    provider = next(item for item in app.selectbox if item.key == "llm_preset")

    provider.set_value("moonshot")
    app.run(timeout=15)
    assert next(item for item in app.text_input if item.key == "llm_key_input").label == (
        "API 密钥 (:red[必填]；[获取 Moonshot API 密钥](https://platform.kimi.com/console/api-keys))"
    )
    assert next(item for item in app.text_input if item.key == "llm_base_url").value == (
        "https://api.moonshot.cn/v1"
    )
    assert next(item for item in app.text_input if item.key == "llm_model").value == "kimi-k2.6"

    provider = next(item for item in app.selectbox if item.key == "llm_preset")

    provider.set_value("qwen")
    app.run(timeout=15)
    assert next(item for item in app.text_input if item.key == "llm_key_input").label == (
        "API 密钥 (:red[必填]；[获取阿里云百炼 API Key]"
        "(https://help.aliyun.com/zh/model-studio/get-api-key))"
    )

    provider = next(item for item in app.selectbox if item.key == "llm_preset")
    provider.set_value("ollama")
    app.run(timeout=15)
    assert next(item for item in app.text_input if item.key == "llm_key_input").label == (
        "API 密钥 (本地 Ollama 可不填)"
    )

    provider = next(item for item in app.selectbox if item.key == "llm_preset")
    provider.set_value("custom")
    app.run(timeout=15)
    assert next(item for item in app.text_input if item.key == "llm_key_input").label == (
        "API 密钥 (:red[必填，请到所用服务商后台获取])"
    )


@pytest.mark.parametrize(
    ("preset", "expected_title", "expected_default", "expected_link"),
    [
        ("openai", "OpenAI 配置说明", "gpt-5.6-terra", "platform.openai.com/docs/models"),
        (
            "deepseek",
            "DeepSeek 配置说明",
            "deepseek-v4-flash",
            "api-docs.deepseek.com/api/list-models",
        ),
        (
            "moonshot",
            "Moonshot 配置说明",
            "kimi-k2.6",
            "platform.kimi.com/docs/models",
        ),
        (
            "qwen",
            "通义千问 Qwen 配置说明",
            "qwen3.7-flash",
            "help.aliyun.com/zh/model-studio/list-models",
        ),
        ("ollama", "Ollama 配置说明", "qwen3.5:9b", "ollama.com/library"),
        ("custom", "自定义服务商配置说明", "准确模型 ID", "本机回环地址"),
    ],
)
def test_llm_setup_card_follows_provider_without_replacing_fields(
    monkeypatch,
    preset: str,
    expected_title: str,
    expected_default: str,
    expected_link: str,
) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    next(item for item in app.selectbox if item.key == "llm_preset").set_value(preset)

    app.run(timeout=15)

    setup_card = next(item for item in app.info if expected_title in str(item.value))
    assert expected_default in str(setup_card.value)
    assert expected_link in str(setup_card.value)
    assert next(item for item in app.text_input if item.key == "llm_key_input")
    assert next(item for item in app.text_input if item.key == "llm_base_url")
    assert next(item for item in app.text_input if item.key == "llm_model")


def test_custom_llm_credentials_are_bound_to_https_origin() -> None:
    first = _llm_credential_slot(LLMPreset.CUSTOM, "https://API.example/v1")
    same_origin = _llm_credential_slot(LLMPreset.CUSTOM, "https://api.example/v2")
    another_origin = _llm_credential_slot(LLMPreset.CUSTOM, "https://other.example/v1")

    assert first == same_origin == "custom:https://api.example"
    assert another_origin != first


def test_ollama_preset_enables_reliable_structured_output(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class CapturingProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "ai_vocab_video_generator.webui.OpenAICompatibleVocabularyProvider",
        CapturingProvider,
    )
    form = {
        "llm_preset": LLMPreset.OLLAMA.value,
        "llm_key": "",
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "qwen3.5:9b",
    }

    provider = _llm_provider(form, AppSettings())

    assert provider is not None
    assert captured["strict_json_schema"] is True
    assert captured["reasoning_effort"] == "none"


def test_moonshot_preset_uses_its_independent_key_and_cost_effective_model(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class CapturingProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    _without_credentials(monkeypatch)
    monkeypatch.setenv("AIVVG_MOONSHOT_API_KEY", "moonshot-test-only")
    monkeypatch.setattr(
        "ai_vocab_video_generator.webui.OpenAICompatibleVocabularyProvider",
        CapturingProvider,
    )
    form = {
        "llm_preset": LLMPreset.MOONSHOT.value,
        "llm_key": "",
        "llm_base_url": "https://api.moonshot.cn/v1",
        "llm_model": "kimi-k2.6",
    }

    provider = _llm_provider(form, AppSettings())

    assert provider is not None
    assert captured["api_key"].get_secret_value() == "moonshot-test-only"
    assert captured["base_url"] == "https://api.moonshot.cn/v1"
    assert captured["model"] == "kimi-k2.6"
    assert captured["strict_json_schema"] is False
    assert captured["thinking_mode"] == "disabled"


def test_webui_restores_voice_and_material_defaults(monkeypatch) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    repeats = {item.key: item.value for item in app.number_input}
    assert repeats["chinese_narration_repeats"] == 0
    assert repeats["fast_narration_repeats"] == 1
    assert repeats["slow_narration_repeats"] == 0
    enabled = {item.key: item.value for item in app.checkbox}
    assert enabled["chinese_narration_enabled"] is False
    assert enabled["fast_narration_enabled"] is True
    assert enabled["slow_narration_enabled"] is False
    voices = {item.key: item for item in app.selectbox}
    assert voices["chinese_narration_voice"].value == "zh-CN-XiaoxiaoNeural"
    assert len(voices["chinese_narration_voice"].options) == 14
    assert voices["fast_narration_voice"].value == "en-US-JennyNeural"
    assert len(voices["fast_narration_voice"].options) == 47
    assert voices["slow_narration_voice"].value == "en-US-JennyNeural"
    assert len(voices["slow_narration_voice"].options) == 47
    assert voices["question_narration_voice"].value == "en-US-JennyNeural"
    assert len(voices["question_narration_voice"].options) == 47
    assert next(item for item in app.selectbox if item.label == "图片素材服务商").value == "pexels"
    assert next(item for item in app.selectbox if item.label == "素材来源").value == "remote"
    assert (
        next(item for item in app.selectbox if item.label == "素材分配方式").value == "sequential"
    )
    colors = {item.key: item.value for item in app.color_picker}
    for prefix in ("question_style", "english", "phonetic", "chinese"):
        assert colors[f"{prefix}_fill_color"] == "#000000"
        assert colors[f"{prefix}_stroke_color"] == "#FFFFFF"
    sliders = {item.key: item.value for item in app.slider}
    assert sliders["question_style_font_size"] == 80
    assert sliders["english_font_size"] == 100
    assert sliders["phonetic_font_size"] == 90
    assert sliders["chinese_font_size"] == 80
    for prefix in ("question_style", "english", "phonetic", "chinese"):
        assert sliders[f"{prefix}_weight"] == 1.0
        assert sliders[f"{prefix}_stroke_width"] == 1.5


@pytest.mark.parametrize(
    ("locale", "expected_emphasis"),
    [
        (
            "zh-CN",
            (
                ":red[AI 自动生成]",
                ":red[需要 API 密钥]",
                ":red[必须上传]",
                ":red[音色必须与文本语言一致]",
            ),
        ),
        (
            "en-US",
            (
                ":red[AI generate]",
                ":red[API key required]",
                ":red[must upload]",
                ":red[voice must match the text language]",
            ),
        ),
    ],
)
def test_webui_restores_red_emphasis_for_critical_guidance(
    monkeypatch,
    locale: str,
    expected_emphasis: tuple[str, ...],
) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["locale"] = locale

    app.run(timeout=15)

    guidance = "\n".join(
        [*(str(item.value) for item in app.caption)]
        + [
            str(value)
            for item in (*app.selectbox, *app.text_input, *app.number_input)
            for value in (item.label, getattr(item, "help", ""))
            if value
        ]
    )
    for expected in expected_emphasis:
        assert expected in guidance


def test_webui_shows_generation_blockers_without_credentials_or_network(monkeypatch) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    generate = next(button for button in app.button if button.label == "生成视频")
    assert generate.disabled
    messages = [element.value for element in app.info]
    assert "请输入视频主题，或直接填写有效的单词信息。" in messages
    assert any("背景" in message for message in messages)


def test_ai_vocabulary_button_reports_only_the_missing_topic(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    monkeypatch.setenv("AIVVG_OPENAI_API_KEY", "sk-test-only")
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    next(button for button in app.button if button.label == "使用 AI 根据主题生成单词信息").click()
    app.run(timeout=15)

    assert [element.value for element in app.error] == [
        "请先输入视频主题，再使用 AI 生成单词信息。"
    ]


def test_webui_shows_regenerate_control_for_a_completed_job(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test-video")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["last_video_path"] = str(video)
    app.session_state["last_job_id"] = "a" * 32

    app.run(timeout=15)

    assert not app.exception
    assert any(button.label == "替换素材并重新生成" for button in app.button)
    assert any(item.value == "a" * 32 for item in app.code)
    assert next(button for button in app.button if button.label == "替换素材并重新生成").disabled


def _saved_regeneration_job(storage: JobStorage, words: list[WordEntry]) -> str:
    paths = storage.create_job(GenerationRequest(entries=words))
    storage.update_manifest(paths.job_id, entries=[word.model_dump(mode="json") for word in words])
    return paths.job_id


def test_regeneration_loads_saved_words_without_changing_editor(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    job_id = _saved_regeneration_job(
        storage,
        [WordEntry(english="apple", chinese="苹果"), WordEntry(english="pear", chinese="梨")],
    )
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "香蕉\nbanana"
    app.session_state["task_id"] = job_id
    app.run(timeout=15)

    assert next(b for b in app.button if b.label == "替换素材并重新生成").disabled
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert not app.exception
    assert not next(b for b in app.button if b.label == "替换素材并重新生成").disabled
    assert "apple" in str(app.dataframe[0].value)
    assert "pear" in str(app.dataframe[0].value)
    assert "banana" not in str(app.dataframe[0].value)
    assert app.session_state["script_text"] == "香蕉\nbanana"


@pytest.mark.parametrize("bad_id", ["wrong-id", "f" * 32])
def test_changed_task_id_unbinds_replacements_without_losing_data(
    monkeypatch, tmp_path, bad_id
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    job_id = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    before = storage.paths(job_id).manifest.read_bytes()
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "香蕉\nbanana"
    app.session_state["task_id"] = job_id
    app.run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    app.text_input(key="task_id").set_value(bad_id).run(timeout=15)
    assert next(b for b in app.button if b.label == "替换素材并重新生成").disabled
    assert not app.dataframe
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert app.error
    assert not app.exception
    assert not app.dataframe
    assert app.session_state["script_text"] == "香蕉\nbanana"
    assert storage.paths(job_id).manifest.read_bytes() == before
    app.text_input(key="task_id").set_value(job_id).run(timeout=15)
    assert next(b for b in app.button if b.label == "替换素材并重新生成").disabled


def test_regenerated_video_is_displayed_on_the_same_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    job_id = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    original = storage.paths(job_id).artifacts / "original.mp4"
    original.write_bytes(b"original-video")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["task_id"] = job_id
    app.session_state["last_job_id"] = job_id
    app.session_state["last_video_path"] = str(original)
    app.run(timeout=15)
    original_url = app.get("video")[0].proto.url
    app.button(key="load_regeneration_task").click().run(timeout=15)
    next(b for b in app.button if b.label == "替换素材并重新生成").click().run(timeout=15)
    assert not app.exception
    assert len(app.get("video")) == 1
    assert app.get("video")[0].proto.url != original_url
    assert any(item.value == job_id for item in app.code)
    assert original.read_bytes() == b"original-video"
    assert _FakePipeline.regenerated_replacements == [{}]


def test_result_download_uses_result_identity_not_the_edited_task_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    downloads = []
    original_download = webui_module.st.download_button

    def capture_download(*args, **kwargs):
        downloads.append((kwargs["file_name"], kwargs["data"]))
        return original_download(*args, **kwargs)

    monkeypatch.setattr(webui_module.st, "download_button", capture_download)
    video = tmp_path / "video-0002.mp4"
    video.write_bytes(b"new-result")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["last_video_path"] = str(video)
    app.session_state["last_job_id"] = "a" * 32
    app.session_state["task_id"] = "b" * 32
    app.run(timeout=15)

    assert not app.exception
    assert downloads == [("a" * 32 + "-video-0002.mp4", b"new-result")]
    assert any(item.value == "a" * 32 for item in app.code)


def test_loading_another_task_drops_old_uploads_and_duplicate_word_mapping(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    first = _saved_regeneration_job(
        storage, [WordEntry(english="apple"), WordEntry(english="pear")]
    )
    second = _saved_regeneration_job(storage, [WordEntry(english="orange")])
    app = AppTest.from_file(str(WEBUI))
    app.session_state["task_id"] = first
    app.run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    uploader = next(item for item in app.file_uploader if item.label == "替换素材")
    contents = _image(tmp_path / "replacement.png").read_bytes()
    uploader.set_value([("one.png", contents, "image/png"), ("two.png", contents, "image/png")])
    app.run(timeout=15)
    next(item for item in app.selectbox if "two.png" in item.label).set_value(0).run(timeout=15)
    assert next(b for b in app.button if b.label == "替换素材并重新生成").disabled
    assert any("同一个单词" in item.value for item in app.error)
    app.text_input(key="task_id").set_value(second).run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert not app.exception
    assert "orange" in str(app.dataframe[0].value)
    assert "apple" not in str(app.dataframe[0].value)
    assert next(item for item in app.file_uploader if item.label == "替换素材").value == []
    assert not any("one.png" in item.label or "two.png" in item.label for item in app.selectbox)
    assert not next(b for b in app.button if b.label == "替换素材并重新生成").disabled


def test_task_without_saved_vocabulary_cannot_be_regenerated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    paths = storage.create_job(GenerationRequest(entries=[WordEntry(english="apple")]))
    app = AppTest.from_file(str(WEBUI))
    app.session_state["task_id"] = paths.job_id
    app.run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert not app.exception
    assert any("尚未保存单词" in item.value for item in app.error)
    assert next(b for b in app.button if b.label == "替换素材并重新生成").disabled


def test_saved_task_with_materials_disabled_has_no_ineffective_replacement_control(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    request = GenerationRequest(entries=[WordEntry(english="apple")])
    request.material.enabled = False
    paths = storage.create_job(request)
    storage.update_manifest(paths.job_id, entries=[{"english": "apple"}])
    app = AppTest.from_file(str(WEBUI))
    app.session_state["task_id"] = paths.job_id
    app.run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert not app.exception
    assert not any(item.label == "替换素材" for item in app.file_uploader)
    assert any("未启用图片素材" in item.value for item in app.info)
    assert not next(b for b in app.button if b.label == "替换素材并重新生成").disabled


@pytest.mark.parametrize(
    ("aspect", "expected_width"),
    [("portrait", 280), ("landscape", 480)],
)
def test_result_video_preview_is_bounded_by_aspect(
    tmp_path: Path,
    aspect: str,
    expected_width: int,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test-video")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["last_video_path"] = str(video)
    app.session_state["last_video_aspect"] = aspect

    app.run(timeout=15)

    assert not app.exception
    video_element = app.get("video")[0]
    assert video_element.proto.width_config.pixel_width == expected_width


def test_regeneration_uploader_preserves_a_real_video_replacement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    replacement = _write_color_video(tmp_path / "replacement.mp4")
    storage = JobStorage(tmp_path / "storage")
    job_id = _saved_regeneration_job(
        storage,
        [WordEntry(english="apple", chinese="苹果"), WordEntry(english="pear", chinese="梨")],
    )
    app = AppTest.from_file(str(WEBUI))
    app.session_state["task_id"] = job_id

    app.run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)

    uploader = next(item for item in app.file_uploader if item.label == "替换素材")
    assert {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".mp4",
        ".mov",
        ".m4v",
        ".webm",
    }.issubset(set(uploader.allowed_type))
    uploader.set_value([("replacement.mp4", replacement.read_bytes(), "video/mp4")])
    app.run(timeout=15)
    selector = next(item for item in app.selectbox if "replacement.mp4" in item.label)
    assert selector.options == ["第 1 个：apple · 苹果", "第 2 个：pear · 梨"]
    selector.set_value(1).run(timeout=15)
    next(button for button in app.button if button.label == "替换素材并重新生成").click().run(
        timeout=15
    )

    captured = _FakePipeline.regenerated_replacements[-1]
    assert set(captured) == {1}
    assert captured[1].suffix == ".mp4"
    assert captured[1].read_bytes() == replacement.read_bytes()
    assert images_module.probe_material(captured[1]).kind is MaterialKind.VIDEO


def _selected_materials(app):
    return {
        index: state.selection
        for index, state in app.session_state["word_material_states"].items()
        if state.selection is not None
    }


def _selected_gallery_app(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "gallery-test-only")
    _patch_generation_fakes(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/æpl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(_image(tmp_path / "background.png"))
    app.run(timeout=15)
    app.button(key="search_candidates_0").click().run(timeout=15)
    app.button(key="use_candidate_0_1").click().run(timeout=15)
    return app


def test_material_overview_jumps_to_word_without_changing_selection(monkeypatch, tmp_path):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    chosen = app.session_state["word_material_states"][0].selection
    app.session_state["script_text"] = "苹果\napple\n/æpl/\n梨\npear\n/peə/"
    app.run(timeout=15)
    assert "手动选图" in app.button(key="material_overview_0").label
    assert "待自动搜索" in app.button(key="material_overview_1").label
    app.button(key="material_overview_1").click().run(timeout=15)
    assert not app.exception
    assert app.selectbox(key="material_word_index").value == 1
    assert app.text_input(key="material_query_1").value == "pear"
    assert app.session_state["word_material_states"][0].selection == chosen


def test_recent_task_selection_loads_saved_words_without_replacing_editor(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage")
    job_id = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    before = storage.paths(job_id).manifest.read_bytes()
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "香蕉\nbanana"
    app.run(timeout=15)
    recent_key = f"recent_task_{Locale.ZH_CN.value}"
    app.selectbox(key=recent_key).select(job_id).run(timeout=15)
    assert not app.exception
    assert app.text_input(key="task_id").value == job_id
    assert app.session_state["regeneration_task"]["entries"][0].english == "apple"
    assert app.session_state["script_text"] == "香蕉\nbanana"
    assert storage.paths(job_id).manifest.read_bytes() == before
    second_id = _saved_regeneration_job(storage, [WordEntry(english="pear")])
    app.button(key="refresh_recent_tasks").click().run(timeout=15)
    app.selectbox(key=recent_key).select(second_id).run(timeout=15)
    assert app.session_state["regeneration_task"]["id"] == second_id
    assert app.session_state["script_text"] == "香蕉\nbanana"
    app.text_input(key="task_id").input("wrong-id").run(timeout=15)
    assert app.selectbox(key=recent_key).value is None
    assert app.selectbox(key=recent_key).proto.set_value
    assert "regeneration_task" not in app.session_state
    assert app.session_state["script_text"] == "香蕉\nbanana"


@pytest.mark.parametrize("depth", [600, 1500])
def test_deep_manifest_does_not_interrupt_task_browser(monkeypatch, tmp_path, depth):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    good = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    bad = _saved_regeneration_job(storage, [WordEntry(english="pear")])
    storage.paths(bad).manifest.write_text(
        '{"schema_version":3,"cache":' + "[" * depth + "0" + "]" * depth + "}"
    )
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    assert not app.exception
    recent = app.selectbox(key=f"recent_task_{Locale.ZH_CN.value}")
    assert len(recent.options) == 1
    recent.select(good).run(timeout=15)
    assert app.session_state["regeneration_task"]["id"] == good
    app.text_input(key="task_id").input(bad).run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert not app.exception
    assert app.error
    assert "regeneration_task" not in app.session_state


@pytest.mark.parametrize("locale", [Locale.ZH_CN, Locale.EN_US])
def test_new_generation_clears_previous_task_selection(monkeypatch, tmp_path, locale):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "generation-test-only")
    _patch_generation_fakes(monkeypatch)

    class SavedVocabularyPipeline(_FakePipeline):
        def run(self, request, on_progress=None):
            result = super().run(request, on_progress)
            self.storage.update_manifest(
                result.job_id,
                status="complete",
                entries=[entry.model_dump(mode="json") for entry in request.entries],
            )
            return result

    monkeypatch.setattr(pipeline_module, "GenerationPipeline", SavedVocabularyPipeline)
    storage = JobStorage(tmp_path / "storage")
    old_id = _saved_regeneration_job(storage, [WordEntry(english="pear")])
    before = storage.paths(old_id).manifest.read_bytes()
    app = AppTest.from_file(str(WEBUI))
    app.session_state["locale"] = locale.value
    app.session_state["script_text"] = "苹果\napple\n/æpl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(_image(tmp_path / "background.png"))
    app.run(timeout=15)
    recent_key = f"recent_task_{locale.value}"
    app.selectbox(key=recent_key).select(old_id).run(timeout=15)
    assert app.session_state["regeneration_task"]["id"] == old_id
    next(b for b in app.button if b.label == translate(locale, "generate_video")).click().run(
        timeout=15
    )

    assert not app.exception
    new_id = app.session_state["last_job_id"]
    assert new_id != old_id
    assert app.text_input(key="task_id").value == new_id
    assert app.selectbox(key=recent_key).value is None
    assert app.selectbox(key=recent_key).proto.set_value
    assert "regeneration_task" not in app.session_state
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert app.session_state["regeneration_task"]["id"] == new_id
    other_locale = Locale.EN_US if locale is Locale.ZH_CN else Locale.ZH_CN
    app.selectbox(key="locale").select(other_locale.value).run(timeout=15)
    recent_key = f"recent_task_{other_locale.value}"
    assert app.selectbox(key=recent_key).value is None
    app.selectbox(key=recent_key).select(old_id).run(timeout=15)
    assert not app.exception
    assert app.text_input(key="task_id").value == old_id
    assert app.session_state["regeneration_task"]["id"] == old_id
    assert app.session_state["last_job_id"] == new_id
    assert app.session_state["script_text"] == "苹果\napple\n/æpl/"
    assert storage.paths(old_id).manifest.read_bytes() == before


@pytest.mark.parametrize("locale", [Locale.ZH_CN, Locale.EN_US])
def test_invalid_manifest_structure_does_not_interrupt_ui(monkeypatch, tmp_path, locale):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    good = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    bad = _saved_regeneration_job(storage, [WordEntry(english="pear")])
    path = storage.paths(bad).manifest
    payload = json.loads(path.read_text())
    payload["cache"] = []
    path.write_text(json.dumps(payload))
    app = AppTest.from_file(str(WEBUI))
    app.session_state["locale"] = locale.value
    app.session_state["script_text"] = "香蕉\nbanana"
    app.run(timeout=15)
    assert not app.exception
    recent = app.selectbox(key=f"recent_task_{locale.value}")
    assert len(recent.options) == 1
    recent.select(good).run(timeout=15)
    assert app.session_state["regeneration_task"]["id"] == good
    app.text_input(key="task_id").set_value(bad).run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    assert not app.exception
    assert app.error
    assert app.session_state["script_text"] == "香蕉\nbanana"


@pytest.mark.parametrize("locale", [Locale.ZH_CN, Locale.EN_US])
def test_regeneration_sends_latest_history_selection_to_browser(monkeypatch, tmp_path, locale):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    downloads = []
    original_download = webui_module.st.download_button

    def capture_download(*args, **kwargs):
        downloads.append((kwargs["file_name"], kwargs["data"]))
        return original_download(*args, **kwargs)

    class HistoryPipeline(_FakePipeline):
        def regenerate(self, job_id, **kwargs):
            paths = self.storage.paths(job_id)
            videos = self.storage.load_manifest(job_id)["artifacts"]["videos"]
            reference = f"artifacts/videos/video-{len(videos) + 1:04d}.mp4"
            video = paths.root / reference
            video.write_bytes(b"new-video")
            self.storage.update_manifest(
                job_id, artifacts={"video": reference, "videos": [*videos, reference]}
            )
            return GenerationResult(
                job_id=job_id,
                status=JobStatus.COMPLETE,
                video_path=video,
                manifest_path=paths.manifest,
            )

    monkeypatch.setattr(webui_module.st, "download_button", capture_download)
    monkeypatch.setattr(pipeline_module, "GenerationPipeline", HistoryPipeline)
    storage = JobStorage(tmp_path / "storage", active_secrets=())
    job_id = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    paths = storage.paths(job_id)
    (paths.artifacts / "videos").mkdir()
    refs = [f"artifacts/videos/video-{i:04d}.mp4" for i in (1, 2, 3)]
    for index, ref in enumerate(refs[:2]):
        (paths.root / ref).write_bytes(f"old-video-{index}".encode())
    storage.update_manifest(job_id, artifacts={"video": refs[1], "videos": refs[:2]})
    app = AppTest.from_file(str(WEBUI))
    app.session_state["locale"] = locale.value
    app.session_state["script_text"] = "香蕉\nbanana"
    app.run(timeout=15)
    app.selectbox(key=f"recent_task_{locale.value}").select(job_id).run(timeout=15)
    key = f"history_version_{job_id}_{locale.value}"
    app.selectbox(key=key).select(refs[0]).run(timeout=15)
    next(b for b in app.button if b.label == translate(locale, "regenerate")).click().run(
        timeout=15
    )
    assert not app.exception
    history = app.selectbox(key=key)
    assert history.value == refs[2]
    # Removing session state alone does not update the browser's selected label.
    assert history.proto.set_value
    assert downloads[-2:] == [(f"{job_id}-video-0003.mp4", b"new-video")] * 2
    app.selectbox(key=key).select(refs[0]).run(timeout=15)
    assert downloads[-2:] == [
        (f"{job_id}-video-0001.mp4", b"old-video-0"),
        (f"{job_id}-video-0003.mp4", b"new-video"),
    ]
    other = Locale.EN_US if locale is Locale.ZH_CN else Locale.ZH_CN
    app.selectbox(key="locale").select(other.value).run(timeout=15)
    assert app.selectbox(key=f"history_version_{job_id}_{other.value}").value == refs[0]
    assert app.session_state["script_text"] == "香蕉\nbanana"
    assert app.session_state["last_video_path"] == str(paths.root / refs[2])
    assert (paths.root / refs[0]).read_bytes() == b"old-video-0"


def test_history_preview_changes_only_preview_and_download(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    downloads = []
    original_download = webui_module.st.download_button

    def capture_download(*args, **kwargs):
        downloads.append((kwargs["file_name"], kwargs["data"]))
        return original_download(*args, **kwargs)

    monkeypatch.setattr(webui_module.st, "download_button", capture_download)
    storage = JobStorage(tmp_path / "storage")
    job_id = _saved_regeneration_job(storage, [WordEntry(english="apple")])
    paths = storage.paths(job_id)
    directory = paths.artifacts / "videos"
    directory.mkdir(parents=True)
    refs = [f"artifacts/videos/video-{i:04d}.mp4" for i in (1, 2)]
    for index, reference in enumerate(refs):
        (paths.root / reference).write_bytes(f"version-{index}".encode())
    storage.update_manifest(job_id, artifacts={"video": refs[1], "videos": refs})
    before = paths.manifest.read_bytes()
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "香蕉\nbanana"
    app.session_state["task_id"] = job_id
    app.run(timeout=15)
    app.button(key="load_regeneration_task").click().run(timeout=15)
    version_key = f"history_version_{job_id}_{Locale.ZH_CN.value}"
    assert app.selectbox(key=version_key).value == refs[1]
    latest_url = app.get("video")[0].proto.url
    app.selectbox(key=version_key).select(refs[0]).run(timeout=15)
    assert not app.exception
    assert app.get("video")[0].proto.url != latest_url
    assert downloads[-1] == (f"{job_id}-video-0001.mp4", b"version-0")
    app.selectbox(key="locale").select(Locale.EN_US.value).run(timeout=15)
    version_key = f"history_version_{job_id}_{Locale.EN_US.value}"
    assert app.selectbox(key=version_key).value == refs[0]
    assert app.selectbox(key=version_key).label == "Version to preview"
    assert downloads[-1] == (f"{job_id}-video-0001.mp4", b"version-0")
    assert not app.exception
    assert app.session_state["script_text"] == "香蕉\nbanana"
    assert "last_job_id" not in app.session_state
    assert paths.manifest.read_bytes() == before
    (paths.root / refs[0]).rename(tmp_path / "moved-video.mp4")
    app.run(timeout=15)
    assert not app.exception
    assert not app.get("video")
    assert any("video is unavailable" in warning.value for warning in app.warning)


def test_custom_search_query_preserves_selected_image_and_vocabulary(monkeypatch, tmp_path):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    assert any(item.key == "material_query_0" for item in app.text_input)
    app.text_input(key="material_query_0").set_value("  apple fruit  ").run(timeout=15)
    assert _selected_materials(app)[0].asset.source_id == "apple-1"
    app.button(key="search_candidates_0").click().run(timeout=15)
    assert _FakeRemoteProvider.searches[-1] == ("apple fruit", 8)
    assert _selected_materials(app)[0].asset.source_id == "apple-1"
    next(b for b in app.button if b.label == "生成视频").click().run(timeout=15)
    request = _FakePipeline.run_requests[-1]
    assert request.entries[0].english == "apple"
    assert request.material_queries == {0: "apple fruit"}
    assert request.pinned_materials[0].asset.source_id == "apple-1"


def test_search_queries_survive_navigation_and_visibility_but_reset_for_changed_word(
    monkeypatch, tmp_path
):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    app.text_area(key="script_text").set_value("苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/").run()
    assert any(item.key == "material_query_0" for item in app.text_input)
    app.text_input(key="material_query_0").set_value("apple fruit").run()
    app.button(key="material_next_word").click().run()
    assert app.text_input(key="material_query_1").value == "banana"
    app.text_input(key="material_query_1").set_value("banana bunch").run()
    app.checkbox(key="material_enabled").uncheck().run()
    app.checkbox(key="material_enabled").check().run()
    app.selectbox(key="material_word_index").set_value(0).run()
    assert app.text_input(key="material_query_0").value == "apple fruit"
    app.text_input(key="material_query_0").set_value("  ").run()
    app.button(key="search_candidates_0").click().run()
    assert _FakeRemoteProvider.searches[-1] == ("apple", 8)
    app.text_area(key="script_text").set_value("梨\npear\n/peə/\n香蕉\nbanana\n/bənɑːnə/").run()
    assert app.text_input(key="material_query_0").value == "pear"
    app.button(key="material_next_word").click().run()
    assert app.text_input(key="material_query_1").value == "banana bunch"
    assert not app.exception


def _two_word_preview_app(monkeypatch, tmp_path, *, mode="sequential"):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    initial = {
        "script_text": "苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "loaded_background_path": str(_image(tmp_path / "background.png", "white")),
        "loaded_material_paths": [
            str(_image(tmp_path / "red.png", "red")),
            str(_image(tmp_path / "blue.png", "blue")),
        ],
        "material_source": "local",
        "selection_mode": mode,
        "loaded_job_seed": 0,
        "material_shape": "rectangle",
        "question_enabled": True,
    }
    for key, value in initial.items():
        app.session_state[key] = value
    return app.run(timeout=15)


@pytest.mark.parametrize("widget_key", ["preview_word_index", "material_word_index"])
@pytest.mark.parametrize(
    "incoming_label, expected",
    [("2. banana · 香蕉", 1), ("2. removed · 已删除", 0), ("示例单词", 0)],
)
def test_word_selectors_recover_serialized_labels(
    monkeypatch, tmp_path, widget_key, incoming_label, expected
):
    app = _two_word_preview_app(monkeypatch, tmp_path)
    app.selectbox(key="material_source_widget").set_value("remote").run(timeout=15)
    app.selectbox(key=widget_key).set_value(1).run(timeout=15)
    # Replay the string payload sent by the browser, including a label from
    # an older word list. AppTest.set_value(int) alone misses this boundary.
    widget_states = app._tree.get_widget_states()
    for state in widget_states.widgets:
        if state.id == app.selectbox(key=widget_key).id:
            state.string_value = incoming_label
    app._run(widget_states, timeout=15)

    assert not app.exception
    assert app.selectbox(key=widget_key).value == expected
    assert type(app.session_state[widget_key]) is int
    assert app.text_area(key="script_text").value == "苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/"
    app.run(timeout=15)
    assert not app.exception


@pytest.mark.parametrize("widget_key", ["preview_word_index", "material_word_index"])
@pytest.mark.parametrize(
    "value, expected", [(None, 0), (True, 0), (-1, 0), (99, 0), (1.5, 0), (1, 1)]
)
def test_word_selectors_validate_session_indices(
    monkeypatch, tmp_path, widget_key, value, expected
):
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n香蕉\nbanana"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = False
    app.session_state[widget_key] = value
    app.run(timeout=15)

    assert not app.exception
    assert app.selectbox(key=widget_key).value == expected
    assert type(app.session_state[widget_key]) is int


@pytest.mark.parametrize("widget_key", ["preview_word_index", "material_word_index"])
def test_word_selectors_send_updated_labels_after_editing_words(monkeypatch, tmp_path, widget_key):
    app = _two_word_preview_app(monkeypatch, tmp_path)
    app.selectbox(key="material_source_widget").set_value("remote").run(timeout=15)
    app.selectbox(key=widget_key).set_value(1).run(timeout=15)
    app.text_area(key="script_text").set_value("苹果\napple\n/æpl/\n梨\npear\n/peə/").run(
        timeout=15
    )

    assert not app.exception
    selector = app.selectbox(key=widget_key)
    assert selector.value == 1
    assert selector.proto.set_value
    assert selector.proto.raw_value == "2. pear · 梨"
    app.selectbox(key="locale").set_value("en-US").run(timeout=15)
    assert not app.exception
    assert app.selectbox(key=widget_key).value == 1


@pytest.mark.parametrize("mode, expected", [("sequential", (0, 0, 255)), ("random", (255, 0, 0))])
def test_preview_uses_selected_word_and_its_actual_local_assignment(
    monkeypatch, tmp_path, mode, expected
):
    from ai_vocab_video_generator.rendering.cards import CardRenderer

    rendered_words = []
    original = CardRenderer.render_answer

    def record(self, entry, *args, **kwargs):
        rendered_words.append(entry.english)
        return original(self, entry, *args, **kwargs)

    monkeypatch.setattr(CardRenderer, "render_answer", record)
    app = _two_word_preview_app(monkeypatch, tmp_path, mode=mode)
    assert any(item.key == "preview_word_index" for item in app.selectbox)
    app.selectbox(key="preview_word_index").set_value(1).run()
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    assert not app.exception
    assert rendered_words == ["banana"]
    with Image.open(app.session_state["last_preview_path"]) as rendered:
        assert rendered.convert("RGB").getpixel((540, 708)) == expected


def test_preview_question_type_and_style_changes_clear_old_card(monkeypatch, tmp_path):
    from ai_vocab_video_generator.rendering.cards import CardRenderer

    questions = []
    original = CardRenderer.render_question

    def record(self, question, *args, **kwargs):
        questions.append(question)
        return original(self, question, *args, **kwargs)

    monkeypatch.setattr(CardRenderer, "render_question", record)
    app = _two_word_preview_app(monkeypatch, tmp_path)
    assert any(item.label == "卡片类型" for item in app.radio)
    app.selectbox(key="preview_word_index").set_value(1).run()
    app.radio(key="preview_card_type_zh-CN").set_value("question").run()
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    assert questions == ["What is this?"]
    assert Path(app.session_state["last_preview_path"]).is_file()
    font_size = app.slider(key="question_style_font_size")
    font_size.set_value(font_size.value + 1).run()
    assert "last_preview_path" not in app.session_state
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    app.radio(key="preview_card_type_zh-CN").set_value("answer").run()
    assert "last_preview_path" not in app.session_state
    app.radio(key="preview_card_type_zh-CN").set_value("question").run()
    app.checkbox(key="question_enabled").uncheck().run()
    assert app.radio(key="preview_card_type_zh-CN").value == "answer"
    app.text_area(key="script_text").set_value("苹果\napple\n/æpl/").run()
    assert app.selectbox(key="preview_word_index").value == 0
    assert not app.exception


@pytest.mark.parametrize(
    "change", ["toggle", "source", "width", "height", "offset", "fit", "shape"]
)
def test_manual_selection_survives_material_visibility_and_layout_changes(
    monkeypatch, tmp_path, change
):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    if change == "toggle":
        app.checkbox(key="material_enabled").uncheck().run(timeout=15)
        app.checkbox(key="material_enabled").check().run(timeout=15)
    elif change == "source":
        app.selectbox(key="material_source_widget").set_value("local").run(timeout=15)
        app.selectbox(key="material_source_widget").set_value("remote").run(timeout=15)
    elif change in {"width", "height", "offset"}:
        key = {"width": "material_width", "height": "material_height", "offset": "material_top"}[
            change
        ]
        widget = app.number_input(key=key)
        widget.set_value(widget.value + 1).run(timeout=15)
    elif change == "fit":
        app.selectbox(key="material_fit_mode_widget").set_value("contain").run(timeout=15)
    else:
        app.selectbox(key="material_shape_widget").set_value("rectangle").run(timeout=15)
    assert not app.exception
    next(b for b in app.button if b.label == "生成视频").click().run(timeout=15)
    assert [pin.asset.source_id for pin in _FakePipeline.run_requests[-1].pinned_materials] == [
        "apple-1"
    ]
    assert _FakeRemoteProvider.searches == [("apple", 8)]


def test_local_upload_and_removal_invalidate_the_rendered_card(monkeypatch, tmp_path):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    assert "last_preview_path" in app.session_state
    material = _image(tmp_path / "new.png", "red")
    app.file_uploader(key="material_override_0").upload(
        "new.png", material.read_bytes(), "image/png"
    )
    app.run(timeout=15)
    assert "last_preview_path" not in app.session_state
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    assert "last_preview_path" in app.session_state
    app.button(key="remove_material_override_0").click().run(timeout=15)
    assert not app.exception
    assert "last_preview_path" not in app.session_state


def test_candidate_gallery_preserves_selection_across_words_refresh_and_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "gallery-private-key")
    _patch_generation_fakes(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(_image(tmp_path / "background.png"))
    app.session_state["draft_job_seed"] = 777_777
    app.run(timeout=15)

    assert not app.exception
    assert _FakeRemoteProvider.searches == []
    gallery = next(item for item in app.expander if "逐词选图" in item.label)
    assert not gallery.proto.expanded
    assert all(item is gallery for item in gallery.get("expander"))
    assert any(button.key == "search_candidates_0" for button in app.button)
    app.button(key="search_candidates_0").click().run(timeout=15)
    assert not app.exception
    assert _FakeRemoteProvider.searches == [("apple", 8)]
    assert len([item for item in app.button if str(item.key).startswith("use_candidate_")]) == 8
    assert [item for item in _FakeRemoteProvider.downloads if not item[1]] == [("apple-0", False)]
    app.button(key="use_candidate_0_1").click().run(timeout=15)
    first = _selected_materials(app)[0]
    assert first.asset.source_id == "apple-1"

    app.button(key="material_next_word").click().run(timeout=15)
    assert app.selectbox(key="material_word_index").value == 1
    assert _FakeRemoteProvider.searches == [("apple", 8)]
    app.button(key="search_candidates_1").click().run(timeout=15)
    second = _selected_materials(app)[1]
    assert second.asset.source_id == "banana-1"
    app.button(key="material_previous_word").click().run(timeout=15)
    assert _selected_materials(app)[0] == first
    app.button(key="search_candidates_0").click().run(timeout=15)
    assert not app.exception
    assert _selected_materials(app) == {0: first, 1: second}
    assert app.session_state["draft_job_seed"] == 777_777

    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    captured = _FakePipeline.run_requests[-1]
    assert captured.job_seed == 777_777
    assert [pin.asset.source_id for pin in captured.pinned_materials] == ["apple-1", "banana-1"]
    assert "gallery-private-key" not in "\n".join(str(item.value) for item in app.markdown)


def test_candidate_selection_can_replace_local_upload_and_survive_word_navigation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "gallery-test-key")
    _patch_generation_fakes(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(_image(tmp_path / "background.png"))
    app.run(timeout=15)
    assert any(button.key == "search_candidates_0" for button in app.button)
    app.button(key="search_candidates_0").click().run(timeout=15)
    local = _image(tmp_path / "local.png", "blue")
    app.file_uploader(key="material_override_0").upload(
        "local.png", local.read_bytes(), "image/png"
    )
    app.run(timeout=15)
    app.button(key="material_next_word").click().run(timeout=15)
    app.button(key="material_previous_word").click().run(timeout=15)
    assert app.button(key="use_candidate_0_0").disabled is False
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    assert (
        _FakePipeline.run_requests[-1].pinned_materials[0].asset.path.read_bytes()
        == local.read_bytes()
    )

    app.button(key="use_candidate_0_2").click().run(timeout=15)
    assert not app.exception
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    assert _FakePipeline.run_requests[-1].pinned_materials[0].asset.source_id == "apple-2"


def test_loading_another_task_clears_draft_gallery_choices(monkeypatch) -> None:
    state = {
        "word_material_states": {0: object()},
        "material_override_0": object(),
        "material_word_index": 2,
    }
    monkeypatch.setattr(webui_module.st, "session_state", state)
    webui_module._load_request_into_state(
        GenerationRequest(
            entries=[WordEntry(english="apple")], material_queries={0: "apple fruit"}
        ),
        "new-task",
    )
    assert "material_override_0" not in state
    assert state["word_material_states"][0].selection is None
    assert state["word_material_states"][0].upload is None
    assert state["word_material_states"][0].search_query == "apple fruit"
    assert "material_word_index" not in state


def test_preview_second_word_prefers_its_upload_then_its_remote_choice(monkeypatch, tmp_path):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    app.text_area(key="script_text").set_value("苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/").run()
    app.button(key="material_next_word").click().run()
    app.button(key="search_candidates_1").click().run()
    blue = _image(tmp_path / "blue-override.png", "blue")
    app.file_uploader(key="material_override_1").upload("blue.png", blue.read_bytes(), "image/png")
    app.run()
    app.text_input(key="material_query_1").set_value("banana bunch").run()
    app.button(key="search_candidates_1").click().run()
    assert app.session_state["word_material_states"][1].upload is not None
    app.selectbox(key="preview_word_index").set_value(1).run()
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    with Image.open(app.session_state["last_preview_path"]) as rendered:
        assert rendered.convert("RGB").getpixel((540, 708)) == (0, 0, 255)
    app.button(key="remove_material_override_1").click().run()
    assert "last_preview_path" not in app.session_state
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    with Image.open(app.session_state["last_preview_path"]) as rendered:
        r, g, b = rendered.convert("RGB").getpixel((540, 708))
        assert r > 240 and 140 < g < 190 and b < 20
    assert _selected_materials(app)[1].asset.source_id == "banana-1"
    assert not app.exception


def test_placeholder_preview_stays_labelled_after_rerun(monkeypatch, tmp_path):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    app.text_area(key="script_text").set_value("苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/").run()
    app.selectbox(key="preview_word_index").set_value(1).run()
    next(b for b in app.button if b.label == "预览").click().run(timeout=15)
    assert any("占位图" in item.value for item in app.info)
    app.run()
    assert Path(app.session_state["last_preview_path"]).is_file()
    assert any("占位图" in item.value for item in app.info)


def test_keyword_and_preview_controls_survive_language_change(monkeypatch, tmp_path):
    app = _selected_gallery_app(monkeypatch, tmp_path)
    app.text_area(key="script_text").set_value("苹果\napple\n/æpl/\n香蕉\nbanana\n/bənɑːnə/").run()
    app.text_input(key="material_query_0").set_value("apple fruit").run()
    app.checkbox(key="question_enabled").check().run()
    app.selectbox(key="preview_word_index").set_value(1).run()
    app.radio(key="preview_card_type_zh-CN").set_value("question").run()
    app.selectbox(key="locale").set_value("en-US").run()
    assert app.text_input(key="material_query_0").value == "apple fruit"
    assert app.text_input(key="material_query_0").label == "Image search keywords"
    assert app.selectbox(key="preview_word_index").value == 1
    assert app.radio(key="preview_card_type_en-US").value == "question"
    assert app.radio(key="preview_card_type_en-US").options == ["Answer card", "Question card"]
    # Locale changes rebuild rendered labels; the frontend's initial index must
    # agree with the preserved logical selection (otherwise neither radio is checked).
    assert app.radio(key="preview_card_type_en-US").proto.default == 1
    assert not app.exception


def test_failed_candidate_download_keeps_the_previous_selection(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "gallery-test-key")
    _patch_generation_fakes(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/æpl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.run(timeout=15)
    app.button(key="search_candidates_0").click().run(timeout=15)
    previous = _selected_materials(app)[0]

    def fail_download(*_args, **_kwargs):
        raise ProviderError("No Pexels image could be fetched.")

    monkeypatch.setattr(_FakeRemoteProvider, "download_candidate", fail_download)
    app.button(key="use_candidate_0_2").click().run(timeout=15)
    assert not app.exception
    assert app.error
    assert _selected_materials(app)[0] == previous
    app.button(key="search_candidates_0").click().run(timeout=15)
    assert len([item for item in app.button if str(item.key).startswith("use_candidate_")]) == 8
    assert _selected_materials(app)[0] == previous


def test_gallery_upload_rejects_invalid_image_without_losing_remote_selection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "gallery-test-key")
    _patch_generation_fakes(monkeypatch)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/æpl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.run(timeout=15)
    app.button(key="search_candidates_0").click().run(timeout=15)
    previous = _selected_materials(app)[0]
    app.file_uploader(key="material_override_0").upload("bad.png", b"not an image", "image/png")
    app.run(timeout=15)
    assert not app.exception
    assert app.error
    assert all(state.upload is None for state in app.session_state["word_material_states"].values())
    assert _selected_materials(app)[0] == previous


def test_local_material_override_replaces_only_its_remote_vocabulary_entry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "local-override-key")
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    banana = _image(tmp_path / "banana.png", "yellow")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/ˈæp.əl/\n香蕉\nbanana\n/bəˈnɑː.nə/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(background)

    app.run(timeout=15)
    app.button(key="material_next_word").click().run(timeout=15)
    next(
        uploader for uploader in app.file_uploader if uploader.key == "material_override_1"
    ).upload("banana.png", banana.read_bytes(), "image/png")
    app.run(timeout=15)
    assert app.image
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    captured = _FakePipeline.run_requests[-1]
    assert captured.material.source is MaterialSource.REMOTE
    assert captured.local_materials == []
    assert [pin.entry_index for pin in captured.pinned_materials] == [1]
    assert captured.pinned_materials[0].asset.kind is MaterialKind.IMAGE
    assert captured.pinned_materials[0].asset.path.read_bytes() == banana.read_bytes()


def test_card_preview_prefers_the_first_word_local_override_over_its_remote_pin(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "preview-override-key")
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    local_override = _image(tmp_path / "local-blue.png", "blue")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/ˈæp.əl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(background)
    app.session_state["material_shape"] = "rectangle"

    app.run(timeout=15)
    next(button for button in app.button if button.key == "search_candidates_0").click().run(
        timeout=15
    )
    next(
        uploader for uploader in app.file_uploader if uploader.key == "material_override_0"
    ).upload("local-blue.png", local_override.read_bytes(), "image/png")
    app.run(timeout=15)
    next(button for button in app.button if button.label == "预览").click().run(timeout=15)

    preview = Path(app.session_state["last_preview_path"])
    with Image.open(preview) as rendered:
        red, green, blue = rendered.convert("RGB").getpixel((540, 708))
    assert blue > 200
    assert red < 60
    assert green < 60


def test_changing_a_word_clears_its_previous_local_material_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "override-invalidation-key")
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    apple = _image(tmp_path / "apple.png", "red")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/ˈæp.əl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(background)

    app.run(timeout=15)
    next(
        uploader for uploader in app.file_uploader if uploader.key == "material_override_0"
    ).upload("apple.png", apple.read_bytes(), "image/png")
    app.run(timeout=15)
    next(item for item in app.text_area if item.key == "script_text").set_value(
        "香蕉\nbanana\n/bəˈnɑː.nə/"
    )
    app.run(timeout=15)
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    captured = _FakePipeline.run_requests[-1]
    assert captured.entries[0].english == "banana"
    assert captured.pinned_materials == []


def test_per_word_review_preserves_a_loaded_tasks_existing_pinned_material(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "loaded-pin-key")
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    pinned = _image(tmp_path / "saved-apple.png", "green")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/ˈæp.əl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(background)
    app.session_state["loaded_pinned_materials"] = [
        PinnedMaterial(
            entry_index=0,
            asset=MaterialAsset(path=pinned, kind=MaterialKind.IMAGE, source_id="saved"),
        ).model_dump(mode="json")
    ]

    app.run(timeout=15)
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    captured = _FakePipeline.run_requests[-1]
    assert [pin.entry_index for pin in captured.pinned_materials] == [0]
    assert captured.pinned_materials[0].asset.path.read_bytes() == pinned.read_bytes()


def test_missing_remote_result_prompts_for_the_existing_local_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class NoResultRemoteProvider(_FakeRemoteProvider):
        def search(self, query: str, _aspect: Any, *, pool_size: int) -> tuple:
            return ()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "no-result-key")
    _patch_generation_fakes(monkeypatch)
    monkeypatch.setattr(images_module, "PexelsImageProvider", NoResultRemoteProvider)
    background = _image(tmp_path / "background.png", "white")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "罕见词\nobscureword\n/əbˈskjʊə/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(background)

    app.run(timeout=15)
    next(button for button in app.button if button.key == "search_candidates_0").click().run(
        timeout=15
    )

    assert any("没有找到合适的远程图片" in item.value for item in app.warning)
    assert any(uploader.key == "material_override_0" for uploader in app.file_uploader)


@pytest.mark.parametrize(
    "change",
    ["word", "provider", "pool", "aspect", "fit", "credential"],
)
def test_remote_preview_card_is_cleared_when_any_preview_key_input_changes(
    monkeypatch,
    tmp_path: Path,
    change: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIVVG_PEXELS_API_KEY", "initial-pexels-key")
    monkeypatch.setenv("AIVVG_PIXABAY_API_KEY", "initial-pixabay-key")
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/ˈæp.əl/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["loaded_background_path"] = str(background)
    app.run(timeout=15)
    app.button(key="search_candidates_0").click().run(timeout=15)
    next(button for button in app.button if button.label == "预览").click().run(timeout=15)
    assert Path(app.session_state["last_preview_path"]).is_file()

    if change == "word":
        next(item for item in app.text_area if item.key == "script_text").set_value(
            "香蕉\nbanana\n/bəˈnɑː.nə/"
        )
    elif change == "provider":
        next(item for item in app.selectbox if item.label == "图片素材服务商").set_value("pixabay")
    elif change == "pool":
        next(item for item in app.number_input if item.label == "远程候选数量").set_value(9)
    elif change == "aspect":
        next(item for item in app.selectbox if item.label == "输出比例").set_value("landscape")
    elif change == "fit":
        next(item for item in app.selectbox if item.label == "素材填充方式").set_value("contain")
    else:
        next(item for item in app.text_input if item.key == "pexels_key_input").set_value(
            "changed-pexels-key"
        )
    app.run(timeout=15)

    if change == "fit":
        assert _selected_materials(app)[0].asset.source_id == "apple-0"
    else:
        assert _selected_materials(app) == {}
    assert "last_preview_path" not in app.session_state
    assert len(app.image) == (8 if change == "fit" else 0)


def test_remote_preview_controls_are_absent_for_local_materials(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    assert any(item.label == "远程候选数量" for item in app.number_input)
    next(item for item in app.selectbox if item.label == "素材来源").set_value("local")

    app.run(timeout=15)

    labels = [button.label for button in app.button]
    assert "搜索候选" not in labels
    assert "重新搜索" not in labels
    assert not any(item.label == "远程候选数量" for item in app.number_input)


def test_remote_pool_value_survives_local_source_round_trip(monkeypatch) -> None:
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    next(item for item in app.number_input if item.label == "远程候选数量").set_value(13)
    app.run(timeout=15)
    next(item for item in app.selectbox if item.label == "素材来源").set_value("local")
    app.run(timeout=15)
    assert not any(item.label == "远程候选数量" for item in app.number_input)

    next(item for item in app.selectbox if item.label == "素材来源").set_value("remote")
    app.run(timeout=15)

    assert next(item for item in app.number_input if item.label == "远程候选数量").value == 13


def test_music_controls_and_upload_survive_disable_enable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    material = _image(tmp_path / "material.png", "red")
    music = _write_sine_wav(tmp_path / "music.wav")
    app = AppTest.from_file(str(WEBUI))
    initial_state = {
        "script_text": "苹果\napple\n/ˈæp.əl/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "loaded_background_path": str(background),
        "loaded_material_paths": [str(material)],
        "material_source": "local",
    }
    for key, value in initial_state.items():
        app.session_state[key] = value
    app.run(timeout=15)
    next(item for item in app.checkbox if item.label == "启用背景音乐").set_value(True)
    app.run(timeout=15)
    next(item for item in app.slider if item.label == "音乐音量 (%)").set_value(27)
    next(item for item in app.slider if item.label == "朗读时压低比例 (%)").set_value(73)
    next(item for item in app.file_uploader if item.label == "本地音乐文件").upload(
        "music.wav", music.read_bytes(), "audio/wav"
    )
    app.run(timeout=15)
    next(item for item in app.checkbox if item.label == "启用背景音乐").set_value(False)
    app.run(timeout=15)
    next(item for item in app.checkbox if item.label == "启用背景音乐").set_value(True)
    app.run(timeout=15)

    assert next(item for item in app.slider if item.label == "音乐音量 (%)").value == 27
    assert next(item for item in app.slider if item.label == "朗读时压低比例 (%)").value == 73
    generate = next(button for button in app.button if button.label == "生成视频")
    assert generate.disabled is False
    generate.click().run(timeout=15)
    captured = _FakePipeline.run_requests[-1]
    assert captured.background_music.path is not None
    assert captured.background_music.path.read_bytes() == music.read_bytes()


def test_clearing_visible_music_upload_removes_effective_music(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    material = _image(tmp_path / "material.png", "red")
    music = _write_sine_wav(tmp_path / "music.wav")
    app = AppTest.from_file(str(WEBUI))
    for key, value in {
        "script_text": "苹果\napple\n/ˈæp.əl/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "loaded_background_path": str(background),
        "loaded_material_paths": [str(material)],
        "material_source": "local",
    }.items():
        app.session_state[key] = value
    app.run(timeout=15)
    next(item for item in app.checkbox if item.label == "启用背景音乐").set_value(True)
    app.run(timeout=15)
    next(item for item in app.file_uploader if item.label == "本地音乐文件").upload(
        "music.wav", music.read_bytes(), "audio/wav"
    )
    app.run(timeout=15)

    next(item for item in app.file_uploader if item.label == "本地音乐文件").clear()
    app.run(timeout=15)

    assert next(button for button in app.button if button.label == "生成视频").disabled
    assert any("音乐" in item.value for item in app.info)
    next(item for item in app.checkbox if item.label == "启用背景音乐").set_value(False)
    app.run(timeout=15)
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    captured = _FakePipeline.run_requests[-1]
    assert captured.background_music.enabled is False
    assert captured.background_music.path is None


def test_loading_request_discards_previous_music_widget_upload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    material = _image(tmp_path / "material.png", "red")
    old_music = _write_sine_wav(tmp_path / "old-music.wav", duration=0.25)
    imported_music = _write_sine_wav(tmp_path / "imported-music.wav", duration=0.5)
    app = AppTest.from_file(str(WEBUI))
    for key, value in {
        "script_text": "苹果\napple\n/ˈæp.əl/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "loaded_background_path": str(background),
        "loaded_material_paths": [str(material)],
        "material_source": "local",
    }.items():
        app.session_state[key] = value
    app.run(timeout=15)
    next(item for item in app.checkbox if item.label == "启用背景音乐").set_value(True)
    app.run(timeout=15)
    next(item for item in app.file_uploader if item.label == "本地音乐文件").upload(
        "old-music.wav", old_music.read_bytes(), "audio/wav"
    )
    app.run(timeout=15)
    app.session_state["pending_loaded_request"] = GenerationRequest(
        topic="Imported music",
        entries=[WordEntry(chinese="苹果", english="apple", phonetic="/apple/")],
        background_image=background,
        local_materials=[material],
        material={"source": MaterialSource.LOCAL},
        background_music={"enabled": True, "path": imported_music},
    )
    app.session_state["pending_loaded_job_id"] = "a" * 32

    app.run(timeout=15)

    uploader = next(item for item in app.file_uploader if item.label == "本地音乐文件")
    assert uploader.value is None
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    captured = _FakePipeline.run_requests[-1]
    assert captured.background_music.path is not None
    assert captured.background_music.path.read_bytes() == imported_music.read_bytes()
    assert captured.background_music.path.read_bytes() != old_music.read_bytes()


def test_fresh_local_video_preview_seed_matches_generated_manifest_offset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    real_pipeline = pipeline_module.GenerationPipeline
    real_extract = images_module.extract_seeded_video_frame
    preview_seeds: list[int] = []

    def record_preview_seed(
        source: Path,
        destination: Path,
        *,
        seed: int,
        entry_index: int,
    ) -> Path:
        preview_seeds.append(seed)
        return real_extract(
            source,
            destination,
            seed=seed,
            entry_index=entry_index,
        )

    monkeypatch.setattr(secrets, "randbits", lambda _bits: 123_456)
    monkeypatch.setattr(images_module, "extract_seeded_video_frame", record_preview_seed)
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    video = _write_color_video(tmp_path / "material.mp4")
    app = AppTest.from_file(str(WEBUI))
    initial_state = {
        "script_text": "苹果\napple\n/ˈæp.əl/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "loaded_background_path": str(background),
        "loaded_material_paths": [str(video)],
        "material_source": "local",
        "material_shape": "rectangle",
    }
    for key, value in initial_state.items():
        app.session_state[key] = value
    app.run(timeout=15)

    next(button for button in app.button if button.label == "预览").click().run(timeout=15)

    assert not app.exception
    preview = Path(app.session_state["last_preview_path"])
    assert preview.is_file()
    with Image.open(preview) as rendered:
        red, green, blue = rendered.convert("RGB").getpixel((540, 708))
    assert blue > 200
    assert red < 60
    assert green < 60
    assert preview_seeds == [123_456]

    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)
    captured = _FakePipeline.run_requests[-1]
    assert captured.job_seed == 123_456

    production_request = captured.model_copy(deep=True)
    production_request.question.enabled = False
    production_request.background_music.enabled = False
    for track in (
        production_request.narration.chinese,
        production_request.narration.slow_english,
        production_request.narration.question,
    ):
        track.enabled = False
    monkeypatch.setattr(pipeline_module, "GenerationPipeline", real_pipeline)
    storage = JobStorage(tmp_path / "manifest-jobs")
    result = real_pipeline(
        storage=storage,
        vocabulary_provider=None,
        image_provider=LocalImageProvider(production_request.local_materials),
        speech_provider=_ManifestSpeechProvider(),
        card_renderer=_ManifestCardRenderer(),
        video_composer=_ManifestComposer(),
    ).run(production_request)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    persisted = storage.load_request(result.job_id)

    assert persisted.job_seed == 123_456
    assert manifest["material_assignments"]["0"]["start_offset_seconds"] == pytest.approx(
        0.8056271362589
    )


def test_unsupported_task_import_controls_are_removed(monkeypatch) -> None:
    _without_credentials(monkeypatch)

    app = AppTest.from_file(str(WEBUI)).run(timeout=15)

    visible_labels = {
        str(item.label) for collection in (app.button, app.text_input) for item in collection
    }
    assert "分析 (不导入)" not in visible_labels
    assert "导入旧任务" not in visible_labels
    assert "旧任务目录" not in visible_labels


def test_safe_message_redacts_pydantic_font_paths_and_secret_like_application_text() -> None:
    private_font = PosixPath("/Users/alice/private/sk-font-secret.ttf")
    with pytest.raises(ValidationError) as caught:
        TextElementStyle(font_path=private_font)

    validation_message = _safe_message(caught.value)
    application_message = _safe_message(
        ApplicationError(
            "Provider rejected /Users/alice/private/config.json with api_key=sk-private-value"
        )
    )

    assert "font" in validation_message.casefold()
    assert str(private_font) not in validation_message
    assert "/Users/alice" not in application_message
    assert "sk-private-value" not in application_message
    assert "api_key" not in application_message


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ProviderError(
                "The vocabulary provider rejected the API key. Check the configured credential."
            ),
            "大模型服务商拒绝了 API 密钥，请检查当前服务商的密钥配置。",
        ),
        (
            ProviderError(
                "The vocabulary provider request timed out. "
                "Try again and check provider availability."
            ),
            "大模型请求超时，请稍后重试并检查网络或服务状态。",
        ),
        (
            ProviderError(
                "The vocabulary provider is unavailable. "
                "Check that the service is running and reachable."
            ),
            "无法连接大模型服务；使用 Ollama 时请确认服务已启动，否则请检查网络。",
        ),
        (
            ProviderError(
                "The configured vocabulary model or API endpoint was not found. "
                "Check the model name and provider URL."
            ),
            (
                "找不到配置的大模型或接口；请检查模型名称和服务地址。使用 Ollama 时，"
                "请先运行 ollama pull <模型名>。"
            ),
        ),
    ],
)
def test_safe_message_localizes_provider_failures(
    error: ProviderError,
    expected: str,
) -> None:
    assert _safe_message(error, Locale.ZH_CN) == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ValueError("The selected upload exceeds the size limit."),
            "上传文件超过大小限制，请选择更小的文件。",
        ),
        (
            ProviderError("The selected material image cannot be decoded."),
            "无法读取所选素材图片，请检查文件是否损坏或格式是否受支持。",
        ),
        (
            ProviderError("The selected material video cannot be decoded."),
            "无法读取所选素材视频，请检查文件是否损坏、过长或格式是否受支持。",
        ),
        (
            ProviderError("The selected material file type is not supported."),
            "不支持所选素材文件类型，请上传受支持的图片或视频。",
        ),
        (
            ApplicationError("Video generation was interrupted."),
            "视频生成已中断，未完成的视频文件已清理。",
        ),
    ],
)
def test_safe_message_localizes_invalid_uploads_and_generation_interruption(
    error: Exception,
    expected: str,
) -> None:
    assert _safe_message(error, Locale.ZH_CN) == expected


def test_safe_message_never_exposes_a_rejected_nested_url_credential() -> None:
    private_marker = "private-" + "public-error-marker"
    with pytest.raises(ValidationError) as caught:
        VocabularySettings(base_url=f"https://example.test/v1#redirect=api_key={private_marker}")

    message = _safe_message(caught.value)
    assert message == "Fix the invalid input values."
    assert private_marker not in message
    assert "api_key" not in message


def test_material_uploader_exists_only_for_enabled_local_materials(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    app = AppTest.from_file(str(WEBUI)).run(timeout=15)
    assert not [item for item in app.file_uploader if item.key == "material_uploads"]

    next(item for item in app.selectbox if item.label == "素材来源").set_value("local").run(
        timeout=15
    )
    uploader = next(item for item in app.file_uploader if item.key == "material_uploads")
    local = _image(tmp_path / "private-local.png", "purple")
    uploader.set_value([("private-local.png", local.read_bytes(), "image/png")])
    app.run(timeout=15)
    next(item for item in app.selectbox if item.label == "素材来源").set_value("remote").run(
        timeout=15
    )
    assert not [item for item in app.file_uploader if item.key == "material_uploads"]
    assert "material_uploads" not in app.session_state

    next(item for item in app.selectbox if item.label == "素材来源").set_value("local").run(
        timeout=15
    )
    next(item for item in app.checkbox if item.label == "启用图片素材").uncheck().run(timeout=15)
    assert not [item for item in app.file_uploader if item.key == "material_uploads"]
    assert "material_uploads" not in app.session_state


def test_generate_normalizes_upload_oserror_without_uncaught_private_details(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/apple/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["material_enabled"] = False
    app.run(timeout=15)
    next(item for item in app.file_uploader if item.key == "background_upload").upload(
        "background.png", background.read_bytes(), "image/png"
    )
    app.run(timeout=15)

    def fail_upload(*_args: object, **_kwargs: object) -> Path:
        raise OSError("/Users/alice/private/api_key=synthetic-private-value")

    monkeypatch.setattr("ai_vocab_video_generator.webui._save_upload", fail_upload)
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    assert not app.exception
    visible = "\n".join(item.value for item in app.error)
    assert "/Users/alice" not in visible
    assert "synthetic-private-value" not in visible


def test_preview_validates_a_fresh_background_before_rendering(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    background = _image(tmp_path / "background.png", "white")
    real_probe = images_module.probe_material

    def reject_uploaded_background(path: Path) -> MaterialAsset:
        if path.parent.name == "_session_uploads":
            raise ProviderError("The selected material image cannot be decoded.")
        return real_probe(path)

    monkeypatch.setattr(images_module, "probe_material", reject_uploaded_background)
    app = AppTest.from_file(str(WEBUI))
    app.session_state["script_text"] = "苹果\napple\n/apple/"
    app.session_state["auto_phonetic"] = False
    app.session_state["manual_phonetic"] = True
    app.session_state["material_enabled"] = False
    app.run(timeout=15)
    next(item for item in app.file_uploader if item.key == "background_upload").upload(
        "background.png", background.read_bytes(), "image/png"
    )
    app.run(timeout=15)

    next(button for button in app.button if button.label == "预览").click().run(timeout=15)

    assert not app.exception
    assert "last_preview_path" not in app.session_state
    assert any("无法读取所选素材图片" in item.value for item in app.error)


def test_every_complete_media_control_reaches_the_generation_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _without_credentials(monkeypatch)
    _patch_generation_fakes(monkeypatch)
    background = _image(tmp_path / "fixtures" / "background.webp", "white")
    still = _image(tmp_path / "fixtures" / "apple.png", "red")
    video = _write_color_video(tmp_path / "fixtures" / "banana.mp4")
    music = _write_sine_wav(tmp_path / "fixtures" / "music.wav")
    app = AppTest.from_file(str(WEBUI))
    initial_state = {
        "script_text": "苹果\napple\n/ˈæp.əl/",
        "auto_phonetic": False,
        "manual_phonetic": True,
        "material_source": "local",
        "selection_mode": "random",
        "aspect": "landscape",
        "fps": 48,
        "material_fit_mode": "stretch",
        "music_enabled": True,
        "music_volume": 27,
        "music_ducking": 73,
        "question_enabled": True,
        "question_text": "Which fruit is shown?",
        "question_narration_enabled": True,
        "question_narration_repeats": 3,
        "question_narration_voice": "en-GB-SoniaNeural",
        "question_narration_volume": -11,
        "question_narration_rate": 19,
        "llm_key_input": "sk-session-only-llm",
        "provider_credentials": {"pexels": "session-only-pexels"},
    }
    for key, value in initial_state.items():
        app.session_state[key] = value

    app.run(timeout=15)
    next(item for item in app.file_uploader if item.key == "background_upload").upload(
        "background.webp", background.read_bytes(), "image/webp"
    )
    next(item for item in app.file_uploader if item.key == "material_uploads").set_value(
        [
            ("apple.png", still.read_bytes(), "image/png"),
            ("banana.mp4", video.read_bytes(), "video/mp4"),
        ]
    )
    next(item for item in app.file_uploader if item.label == "本地音乐文件").upload(
        "music.wav", music.read_bytes(), "audio/wav"
    )
    app.run(timeout=15)
    next(button for button in app.button if button.label == "生成视频").click().run(timeout=15)

    assert not app.exception
    request = _FakePipeline.run_requests[-1]
    persisted_request = request.model_dump_json()
    assert "sk-session-only-llm" not in persisted_request
    assert "session-only-pexels" not in persisted_request
    persisted_job_json = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")
    )
    assert "sk-session-only-llm" not in persisted_job_json
    assert "session-only-pexels" not in persisted_job_json
    assert request.render.fps == 48
    assert request.canvas.aspect is VideoAspect.LANDSCAPE
    assert (request.canvas.width, request.canvas.height) == (1920, 1080)
    assert request.material.fit_mode is MaterialFitMode.STRETCH
    assert request.material.source is MaterialSource.LOCAL
    assert request.material.selection_mode.value == "random"
    assert [path.suffix for path in request.local_materials] == [".png", ".mp4"]
    assert [path.read_bytes() for path in request.local_materials] == [
        still.read_bytes(),
        video.read_bytes(),
    ]
    assert request.background_image is not None
    assert request.background_image.suffix == ".webp"
    assert request.background_image.read_bytes() == background.read_bytes()
    assert request.background_music.enabled is True
    assert request.background_music.path is not None
    assert request.background_music.path.suffix == ".wav"
    assert request.background_music.path.read_bytes() == music.read_bytes()
    assert request.background_music.volume_percent == 27
    assert request.background_music.ducking_percent == 73
    assert request.question.enabled is True
    assert request.question_text == "Which fruit is shown?"
    assert request.narration.question.enabled is True
    assert request.narration.question.repeats == 3
    assert request.narration.question.voice == "en-GB-SoniaNeural"
    assert request.narration.question.volume == -11
    assert request.narration.question.rate == 19
    assert request.pinned_materials == []

    provider = LocalImageProvider(request.local_materials)
    image_asset = provider.fetch(
        "apple",
        tmp_path / "provider" / "image",
        VideoAspect.PORTRAIT,
        ImageSelectionContext(
            entry_index=0,
            pool_size=1,
            mode=SelectionMode.SEQUENTIAL,
            seed=11,
        ),
    )
    video_asset = provider.fetch(
        "banana",
        tmp_path / "provider" / "video",
        VideoAspect.PORTRAIT,
        ImageSelectionContext(
            entry_index=1,
            pool_size=1,
            mode=SelectionMode.SEQUENTIAL,
            seed=11,
        ),
    )
    assert image_asset.kind is MaterialKind.IMAGE
    assert video_asset.kind is MaterialKind.VIDEO

    composed = tmp_path / "composed-with-uploaded-music.mp4"
    overlay_style = request.material.model_copy(
        update={
            "width": 20,
            "height": 20,
            "offsets": AnchorOffsets(top=2, left=2),
        }
    )
    VideoComposer().compose(
        [
            VideoSegment(
                image_path=request.background_image,
                duration=0.5,
                material_video=MaterialVideoOverlay(
                    asset=video_asset,
                    style=overlay_style,
                    start_offset_seconds=seeded_video_start_offset(1.0, seed=0, entry_index=0),
                ),
            )
        ],
        composed,
        render=request.render,
        music=request.background_music,
    )
    moviepy = pytest.importorskip("moviepy.editor")
    clip = moviepy.VideoFileClip(str(composed))
    try:
        assert clip.audio is not None
        red, green, blue = clip.get_frame(0.0)[12, 12]
        assert blue > 200
        assert red < 60
        assert green < 60
    finally:
        clip.close()
