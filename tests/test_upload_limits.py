import gc
import os
import wave
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from ai_vocab_video_generator.domain import (
    GenerationRequest,
    MaterialAsset,
    MaterialKind,
    PhoneticMode,
    PinnedMaterial,
    WordEntry,
)
from ai_vocab_video_generator.i18n import Locale
from ai_vocab_video_generator.pipeline import GenerationPipeline
from ai_vocab_video_generator.providers.images import LocalImageProvider, probe_material
from ai_vocab_video_generator.rendering.cards import CardRenderer
from ai_vocab_video_generator.rendering.video import VideoComposer
from ai_vocab_video_generator.storage import JobStorage
from ai_vocab_video_generator.webui import _safe_message, _save_upload

WEBUI = Path(__file__).parents[1] / "src" / "ai_vocab_video_generator" / "webui.py"


@pytest.fixture(scope="module")
def large_png() -> bytes:
    buffer = BytesIO()
    with Image.new("RGB", (2304, 2304), "#c1d5e8") as image:
        image.save(buffer, format="PNG", compress_level=0)
    contents = buffer.getvalue()
    assert 15 * 1024 * 1024 < len(contents) < 16 * 1024 * 1024
    return contents


@pytest.mark.parametrize("size", [15 * 1024 * 1024, 32 * 1024 * 1024])
def test_upload_accepts_local_image_bytes_up_to_32_mib(tmp_path, size):
    contents = b"x" * size
    upload = UploadedFile(UploadedFileRec("test", "image.png", "image/png", contents), FileURLs())
    saved = _save_upload(upload, tmp_path, ".png")
    assert saved.read_bytes() == contents


@pytest.mark.parametrize("trust_declared_size", [True, False])
@pytest.mark.parametrize("suffix, limit", [(".png", 32), (".mp4", 128), (".wav", 32)])
def test_oversized_upload_reports_actual_and_allowed_size(
    tmp_path, trust_declared_size, suffix, limit
):
    actual_size = (limit + 1) * 1024 * 1024

    class Upload:
        name = f"private-input{suffix}"
        size = actual_size if trust_declared_size else 1

        def getvalue(self):
            if trust_declared_size:
                raise AssertionError("Oversized uploads must be rejected before reading bytes")
            return b"x" * actual_size

    with pytest.raises(ValueError) as caught:
        _save_upload(Upload(), tmp_path, suffix)
    for locale in Locale:
        message = _safe_message(caught.value, locale)
        assert str(limit + 1) in message
        assert str(limit) in message
        assert "MiB" in message
        assert "private-input" not in message
    assert not (tmp_path / "_session_uploads").exists()


def test_large_local_image_survives_storage_and_replacement(tmp_path, large_png):
    source = tmp_path / "large.png"
    source.write_bytes(large_png)
    asset = probe_material(source)
    assert asset.kind is MaterialKind.IMAGE
    storage = JobStorage(tmp_path / "jobs")
    paths = storage.create_job(
        GenerationRequest(
            entries=[WordEntry(english="apple", chinese="苹果")],
            background_image=source,
            local_materials=[source],
            pinned_materials=[PinnedMaterial(entry_index=0, asset=asset)],
        )
    )
    restored = storage.load_request(paths.job_id)
    for saved in (
        restored.background_image,
        restored.local_materials[0],
        restored.pinned_materials[0].asset.path,
    ):
        assert saved.read_bytes() == large_png
        assert probe_material(saved).kind is MaterialKind.IMAGE
    replacements = storage.snapshot_replacements(paths.job_id, {0: source})
    assert replacements[0].read_bytes() == large_png


def test_image_byte_limit_accepts_exact_boundary_and_rejects_one_extra_byte(tmp_path, large_png):
    source = tmp_path / "boundary.png"
    source.write_bytes(large_png)
    with source.open("r+b") as handle:
        handle.truncate(32 * 1024 * 1024)
    assert probe_material(source).kind is MaterialKind.IMAGE
    with source.open("r+b") as handle:
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="size limit"):
        probe_material(source)


def test_large_images_generate_and_regenerate_real_videos(tmp_path, large_png):
    source = tmp_path / "large.png"
    source.write_bytes(large_png)

    class OfflineSpeech:
        def synthesize(self, text, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(destination), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 3200)
            return destination

    storage = JobStorage(tmp_path / "jobs")
    pipeline = GenerationPipeline(
        storage=storage,
        vocabulary_provider=None,
        image_provider=LocalImageProvider(source),
        speech_provider=OfflineSpeech(),
        card_renderer=CardRenderer(),
        video_composer=VideoComposer(),
    )
    request = GenerationRequest(
        entries=[WordEntry(english="apple", chinese="苹果")],
        phonetic_mode=PhoneticMode.DISABLED,
        background_image=source,
        local_materials=[source],
    )
    request.narration.fast_english.repeats = 1
    first = pipeline.run(request)
    first_bytes = first.video_path.read_bytes()
    second = pipeline.regenerate(first.job_id, replacements={0: source})
    third = pipeline.regenerate(first.job_id)
    assert first.video_path.read_bytes() == first_bytes
    assert len({first.video_path, second.video_path, third.video_path}) == 3
    assert all(result.video_path.stat().st_size > 0 for result in (first, second, third))
    assert (
        storage.load_request(first.job_id).pinned_materials[0].asset.path.read_bytes() == large_png
    )


@pytest.mark.parametrize("kind", ["background", "local", "pin"])
def test_storage_rejects_images_over_32_mib_with_size_details(tmp_path, kind):
    source = tmp_path / "large.png"
    with source.open("wb") as handle:
        handle.truncate(33 * 1024 * 1024)
    request = GenerationRequest(entries=[WordEntry(english="apple")])
    if kind == "background":
        request.background_image = source
    elif kind == "local":
        request.local_materials = [source]
    else:
        request.pinned_materials = [
            PinnedMaterial(entry_index=0, asset=MaterialAsset(path=source, kind=MaterialKind.IMAGE))
        ]
    storage = JobStorage(tmp_path / "jobs")
    with pytest.raises(ValueError) as caught:
        storage.create_job(request)
    message = _safe_message(caught.value, Locale.ZH_CN)
    assert "33" in message and "32" in message and "MiB" in message


@pytest.mark.parametrize("locale", ["zh-CN", "en-US"])
def test_upload_widgets_show_type_specific_limits(monkeypatch, tmp_path, locale):
    # Earlier rendering and AppTest cases can leave cyclic Streamlit message state alive until
    # the cyclic collector runs. Release it before starting this resource-heavy AppTest flow.
    gc.collect()
    monkeypatch.chdir(tmp_path)
    for name in tuple(os.environ):
        if name.startswith("AIVVG_"):
            monkeypatch.delenv(name)
    app = AppTest.from_file(str(WEBUI))
    for key, value in {
        "locale": locale,
        "script_text": "苹果\napple",
        "auto_phonetic": False,
        "manual_phonetic": False,
        "material_source": "local",
        "music_enabled": True,
    }.items():
        app.session_state[key] = value
    app.run(timeout=30)
    assert not app.exception
    assert app.file_uploader(key="background_upload").proto.max_upload_size_mb == 32
    assert app.file_uploader(key="music_upload_widget").proto.max_upload_size_mb == 32
    assert app.file_uploader(key="material_uploads").proto.max_upload_size_mb == 128
    captions = " ".join(item.value for item in app.get("caption"))
    assert "32 MiB" in captions and "128 MiB" in captions

    storage = JobStorage(tmp_path / "storage")
    paths = storage.create_job(GenerationRequest(entries=[WordEntry(english="apple")]))
    storage.update_manifest(paths.job_id, entries=[{"english": "apple", "chinese": "苹果"}])
    app.text_input(key="task_id").set_value(paths.job_id).run(timeout=30)
    label = "加载任务" if locale == "zh-CN" else "Load Task"
    next(button for button in app.button if button.label == label).click().run(timeout=30)
    assert not app.exception
    replacement = next(
        item for item in app.file_uploader if item.key.startswith("replacement_uploads_")
    )
    assert replacement.proto.max_upload_size_mb == 128
    captions = " ".join(item.value for item in app.get("caption"))
    assert captions.count("128 MiB") >= 2
    app.selectbox(key="material_source_widget").set_value("remote").run(timeout=30)
    assert app.file_uploader(key="material_override_0").proto.max_upload_size_mb == 128
    captions = " ".join(item.value for item in app.get("caption"))
    assert "32 MiB" in captions and "128 MiB" in captions


def test_large_background_upload_can_be_previewed(monkeypatch, tmp_path, large_png):
    monkeypatch.chdir(tmp_path)
    for name in tuple(os.environ):
        if name.startswith("AIVVG_"):
            monkeypatch.delenv(name)
    app = AppTest.from_file(str(WEBUI))
    for key, value in {
        "script_text": "苹果\napple",
        "auto_phonetic": False,
        "manual_phonetic": False,
        "material_enabled": False,
    }.items():
        app.session_state[key] = value
    app.run(timeout=15)
    app.file_uploader(key="background_upload").upload("large.png", large_png, "image/png")
    app.run(timeout=15)
    next(button for button in app.button if button.label == "预览").click().run(timeout=15)
    assert not app.exception
    assert not app.error
    preview = Path(app.session_state["last_preview_path"])
    with Image.open(preview) as image:
        assert image.size == (1080, 1920)
