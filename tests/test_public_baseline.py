import json
import math
import re
import shutil
import struct
import subprocess
import tarfile
import wave
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageFont

from ai_vocab_video_generator.domain import (
    AnchorOffsets,
    BackgroundMusicSettings,
    CanvasSettings,
    GenerationRequest,
    MaterialFitMode,
    MaterialShape,
    MaterialSource,
    MaterialStyle,
    PhoneticMode,
    ProgressBarStyle,
    RenderSettings,
    TextElementStyle,
    VideoAspect,
    WordEntry,
)
from ai_vocab_video_generator.pipeline import GenerationPipeline
from ai_vocab_video_generator.providers.images import LocalImageProvider
from ai_vocab_video_generator.rendering.cards import CardRenderer
from ai_vocab_video_generator.rendering.video import VideoComposer
from ai_vocab_video_generator.storage import JobStorage

ROOT = Path(__file__).resolve().parents[1]

_SECRET_PATTERNS = {
    "openai-classic": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "openai-scoped": re.compile(r"\bsk-(?:proj|svcacct|admin|live)-[A-Za-z0-9_-]{20,}\b"),
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "private-key": re.compile(r"-----BEGIN (?:ENCRYPTED |RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def _secret_findings(text: str) -> set[str]:
    return {name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(text) is not None}


def _private_identity_findings(text: str) -> set[str]:
    private_home = str(Path.home())
    return {"private-home"} if private_home in text else set()


class _LocalToneSpeechProvider:
    """Offline speech fixture that preserves the real pipeline/composer boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

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
        destination.parent.mkdir(parents=True, exist_ok=True)
        frequency = 440.0 if text == "What is this?" else 550.0
        _write_sine_wav(destination, duration=0.4, frequency=frequency, amplitude=0.15)
        return destination


def _public_candidate_files() -> list[Path]:
    checkout = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if checkout.returncode != 0 or Path(checkout.stdout.strip()).resolve() != ROOT:
        pytest.skip("requires the project Git checkout, not an unpacked source distribution")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode() for value in result.stdout.split(b"\0") if value]


def test_launchers_and_documentation_keep_the_local_webui_on_loopback() -> None:
    shell_launcher = (ROOT / "webui.sh").read_text(encoding="utf-8")
    windows_launcher = (ROOT / "webui.bat").read_text(encoding="utf-8")
    streamlit_config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    readmes = "\n".join(
        (ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "README.zh.md")
    )

    assert shell_launcher.index('"$@"') < shell_launcher.index("--server.address 127.0.0.1")
    assert windows_launcher.index("%*") < windows_launcher.index("--server.address 127.0.0.1")
    assert "address" not in streamlit_config
    assert "maxUploadSize = 128" in streamlit_config
    assert "--server.address 127.0.0.1" in readmes
    assert "authenticated" in readmes


def test_public_demo_has_a_cloud_entrypoint_and_system_dependencies() -> None:
    entrypoint = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    packages = (ROOT / "packages.txt").read_text(encoding="utf-8").splitlines()
    readmes = "\n".join(
        (ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "README.zh.md")
    )

    assert "main(public_demo=True)" in entrypoint
    assert packages == ["ffmpeg", "fonts-noto-cjk"]
    assert "streamlit_app.py" in readmes
    assert "Streamlit Community Cloud" in readmes


def _write_sine_wav(
    destination: Path,
    *,
    duration: float,
    frequency: float,
    amplitude: float,
    sample_rate: int = 44_100,
) -> Path:
    frames = b"".join(
        struct.pack(
            "<h",
            round(amplitude * 32_767 * math.sin(2.0 * math.pi * frequency * index / sample_rate)),
        )
        for index in range(round(duration * sample_rate))
    )
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return destination


def _write_moving_video(destination: Path) -> Path:
    moviepy = pytest.importorskip("moviepy.editor")
    np = pytest.importorskip("numpy")
    duration = 0.8

    def make_frame(time: float) -> Any:
        phase = (time % duration) / duration
        color = (round(255 * phase), 40, round(255 * (1.0 - phase)))
        return np.full((80, 160, 3), color, dtype=np.uint8)

    video = moviepy.VideoClip(make_frame=make_frame, duration=duration)
    source_audio = moviepy.AudioClip(
        lambda time: 0.35 * np.sin(2.0 * np.pi * 660.0 * time),
        duration=duration,
        fps=44_100,
    )
    clip = video.set_audio(source_audio)
    try:
        clip.write_videofile(
            str(destination),
            fps=20,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            threads=1,
        )
    finally:
        clip.close()
        source_audio.close()
        video.close()
    return destination


def _soundarray(audio: Any, *, sample_rate: int = 44_100) -> Any:
    np = pytest.importorskip("numpy")
    return np.vstack(list(audio.iter_chunks(fps=sample_rate, chunksize=4096, logger=None)))


def _tone_rms(samples: Any, *, frequency: float, sample_rate: int = 44_100) -> float:
    np = pytest.importorskip("numpy")
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    times = np.arange(len(mono)) / sample_rate
    sine = np.sin(2.0 * np.pi * frequency * times)
    cosine = np.cos(2.0 * np.pi * frequency * times)
    sine_amplitude = 2.0 * float(np.dot(mono, sine)) / len(mono)
    cosine_amplitude = 2.0 * float(np.dot(mono, cosine)) / len(mono)
    return math.sqrt((sine_amplitude**2 + cosine_amplitude**2) / 2.0)


def _text_style(*, top: int, size: int = 20) -> TextElementStyle:
    return TextElementStyle(
        font_size=size,
        stroke_width=0,
        weight=0,
        offsets=AnchorOffsets(top=top),
    )


def _small_system_font_path() -> Path:
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
    pytest.skip("No small system font is available for the real media acceptance test.")


def test_required_public_files_exist() -> None:
    required = {
        "LICENSE",
        "CHANGELOG.md",
        "NOTICE.md",
        "README.md",
        "README.zh.md",
        "pyproject.toml",
        ".gitignore",
        ".env.example",
        "config.example.toml",
        "src/ai_vocab_video_generator/assets/countdown1.wav",
        "src/ai_vocab_video_generator/providers/edge_voices.py",
    }

    missing = sorted(name for name in required if not (ROOT / name).is_file())

    assert missing == []


def test_fresh_release_archives_exclude_internal_and_ignored_runtime_files(
    tmp_path: Path,
) -> None:
    sentinel = ROOT / ".superpowers" / "archive-sentinel-private.txt"
    created_superpowers_directory = not sentinel.parent.exists()
    sentinel.parent.mkdir(exist_ok=True)
    runtime_dir = ROOT / "runtime"
    runtime_sentinel = runtime_dir / "archive-sentinel-private.txt"
    sentinel.write_text("ignored-private-sentinel", encoding="utf-8")
    runtime_dir.mkdir(exist_ok=False)
    runtime_sentinel.write_text("runtime-private-sentinel", encoding="utf-8")
    try:
        subprocess.run(
            ["uv", "build", "--out-dir", str(tmp_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        sentinel.unlink(missing_ok=True)
        if created_superpowers_directory:
            sentinel.parent.rmdir()
        runtime_sentinel.unlink(missing_ok=True)
        runtime_dir.rmdir()
    wheel = next(tmp_path.glob("*.whl"))
    sdist = next(tmp_path.glob("*.tar.gz"))
    try:
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = archive.namelist()
        with tarfile.open(sdist) as archive:
            sdist_names = archive.getnames()
    finally:
        wheel.unlink(missing_ok=True)
        sdist.unlink(missing_ok=True)
    names = wheel_names + sdist_names
    assert not any(".superpowers" in name for name in names)
    assert not any("archive-sentinel-private" in name for name in names)
    assert not any("/.worktrees/" in name for name in names)
    assert not any(
        "/storage/" in name or "/outputs/" in name or "/runtime/" in name for name in names
    )


def test_forbidden_private_artifacts_are_absent() -> None:
    forbidden = {
        "config.toml",
        "baidu_token.json",
        "resource/funasr_models",
        "resource/fonts",
        "resource/songs",
        "storage",
    }

    tracked = {str(path.relative_to(ROOT)) for path in _public_candidate_files()}
    present = sorted(
        name
        for name in forbidden
        if any(path == name or path.startswith(f"{name}/") for path in tracked)
    )

    assert present == []


def test_example_environment_contains_no_secret_values() -> None:
    env_file = ROOT / ".env.example"
    assignments = [
        line.strip()
        for line in env_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert assignments
    assert all(line.endswith("=") for line in assignments)


def test_public_package_contains_no_bundled_generated_outputs() -> None:
    forbidden_suffixes = {".log", ".mp3", ".mp4"}
    bundled = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    )

    assert bundled == []


def test_public_candidate_baseline_contains_no_private_inputs_or_generated_media() -> None:
    public_candidates = _public_candidate_files()
    relative = [path.relative_to(ROOT) for path in public_candidates]
    forbidden_media_suffixes = {
        ".aac",
        ".gif",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".m4v",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".png",
        ".wav",
        ".webm",
        ".webp",
    }
    forbidden_roots = {"storage", "model_cache", "models", "logs", "user_assets"}
    allowed_project_media = {
        Path("src/ai_vocab_video_generator/assets/countdown1.wav"),
    }

    assert [
        str(path)
        for path in relative
        if path.suffix.lower() in forbidden_media_suffixes and path not in allowed_project_media
    ] == []
    assert [str(path) for path in relative if path.parts[0] in forbidden_roots] == []
    assert Path(".env") not in relative
    assert Path("config.toml") not in relative

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in public_candidates
        if path.is_file() and b"\0" not in path.read_bytes()
    )
    assert _private_identity_findings(public_text) == set()
    assert _secret_findings(public_text) == set()


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("sk-" + "A" * 48, "openai-classic"),
        ("sk-" + "proj-" + "B" * 40, "openai-scoped"),
        ("sk-" + "svcacct-" + "C" * 40, "openai-scoped"),
        ("sk-" + "admin-" + "D" * 40, "openai-scoped"),
        ("AK" + "IA" + "E" * 16, "aws-access-key"),
        ("AI" + "za" + "F" * 35, "google-api-key"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private-key"),
        ("-----BEGIN " + "ENCRYPTED PRIVATE KEY-----", "private-key"),
        ("-----BEGIN " + "RSA PRIVATE KEY-----", "private-key"),
        ("-----BEGIN " + "DSA PRIVATE KEY-----", "private-key"),
        ("-----BEGIN " + "EC PRIVATE KEY-----", "private-key"),
        ("-----BEGIN " + "OPENSSH PRIVATE KEY-----", "private-key"),
    ],
    ids=[
        "openai-classic",
        "openai-project",
        "openai-service-account",
        "openai-admin",
        "aws",
        "google",
        "pkcs8",
        "encrypted-pkcs8",
        "rsa",
        "dsa",
        "ec",
        "openssh",
    ],
)
def test_secret_scanner_detects_seeded_nonlive_samples(sample: str, expected: str) -> None:
    assert expected in _secret_findings(sample)


def test_private_identity_scanner_detects_the_current_home_directory() -> None:
    assert _private_identity_findings(str(Path.home() / "private" / "file.txt")) == {"private-home"}


def test_public_package_and_git_metadata_use_project_identity() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'authors = [{ name = "SoloShow Labs" }]' in pyproject

    checkout = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if checkout.returncode != 0 or Path(checkout.stdout.strip()).resolve() != ROOT:
        pytest.skip("Git metadata is only available in the project checkout")
    result = subprocess.run(
        ["git", "log", "--format=%ae%n%ce"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    emails = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert emails
    allowed_automation_emails = {"noreply@github.com"}
    assert all(
        email.endswith("@users.noreply.github.com") or email in allowed_automation_emails
        for email in emails
    )


def test_public_docs_cover_the_complete_bilingual_media_workflow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_zh = (ROOT / "README.zh.md").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    required_readme_phrases = {
        ".mp4`, `.mov`, `.m4v`, and `.webm`",
        ".mp3`, `.wav`, `.m4a`, `.aac`, and `.ogg`",
        "12 through 60 FPS",
        "contain",
        "cover",
        "stretch",
        "50% ducking",
        "question narration",
        "pinned",
        "Only schema version 3 is supported",
        "saved job inputs",
        "composition is rendered again",
        "Quick reference",
    }

    assert sorted(phrase for phrase in required_readme_phrases if phrase not in readme) == []
    assert "## 中文快速说明" not in readme
    assert "## 常用规则速查" in readme_zh
    assert "local video" in notice.lower()
    assert "background music" in notice.lower()
    assert "Only schema version 3 is supported" in architecture
    assert "Dictionary-backed phonetics" in roadmap
    assert "LLM fallback" in roadmap


@pytest.mark.parametrize(
    ("filename", "local_heading", "first_video_heading", "demo_heading", "boundary_text"),
    [
        (
            "README.md",
            "## Local installation and use (recommended)",
            "### 4. Make a first video",
            "## Restricted public demo",
            "not a replacement for local installation",
        ),
        (
            "README.zh.md",
            "## 本地安装与使用\uff08推荐\uff09",
            "### 4. 生成第一个视频",
            "## 在线公开演示\uff08受限版本\uff09",
            "不能替代本地安装",
        ),
    ],
)
def test_readmes_keep_local_workflow_primary_and_public_demo_separate(
    filename: str,
    local_heading: str,
    first_video_heading: str,
    demo_heading: str,
    boundary_text: str,
) -> None:
    readme = (ROOT / filename).read_text(encoding="utf-8")

    assert boundary_text in readme
    local_position = readme.index(local_heading)
    first_video_position = readme.index(first_video_heading)
    demo_position = readme.index(demo_heading)
    assert local_position < first_video_position < demo_position


def test_real_local_media_job_regenerates_to_a_safe_schema_v3_mp4(tmp_path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None, "FFprobe is required for the documented local workflow."
    moviepy = pytest.importorskip("moviepy.editor")
    np = pytest.importorskip("numpy")

    fixtures = tmp_path / "acceptance-fixtures"
    fixtures.mkdir()
    background = fixtures / "background.png"
    material = fixtures / "moving-material.mp4"
    replacement = fixtures / "replacement-material.mp4"
    music = fixtures / "music.wav"
    Image.new("RGB", (320, 240), "#E8EEF6").save(background)
    _write_moving_video(material)
    _write_moving_video(replacement)
    _write_sine_wav(music, duration=2.0, frequency=220.0, amplitude=0.5)

    request = GenerationRequest(
        entries=[WordEntry(english="apple", phonetic="/ˈæp.əl/", chinese="苹果")],
        phonetic_mode=PhoneticMode.MANUAL,
        canvas=CanvasSettings(
            aspect=VideoAspect.LANDSCAPE,
            width=320,
            height=240,
        ),
        question_text="What is this?",
        question=_text_style(top=10),
        material=MaterialStyle(
            width=100,
            height=100,
            shape=MaterialShape.CIRCLE,
            fit_mode=MaterialFitMode.CONTAIN,
            source=MaterialSource.LOCAL,
            offsets=AnchorOffsets(top=50, left=110),
        ),
        progress=ProgressBarStyle(
            width=200,
            height=6,
            offsets=AnchorOffsets(top=225),
        ),
        english_text=_text_style(top=160),
        phonetic_text=_text_style(top=188, size=16),
        chinese_text=_text_style(top=210, size=16),
        render=RenderSettings(fps=15),
        background_music=BackgroundMusicSettings(
            enabled=True,
            path=music,
            volume_percent=30,
            ducking_percent=50,
        ),
        background_image=background,
        local_materials=[material],
        job_seed=0,
    )
    request.narration.question.repeats = 2
    request.narration.chinese.enabled = True
    request.narration.chinese.repeats = 1
    request.narration.fast_english.enabled = False
    request.narration.fast_english.repeats = 0
    request.narration.slow_english.enabled = False
    request.narration.slow_english.repeats = 0
    source_font = _small_system_font_path()
    request.english_text.font_path = source_font

    speech = _LocalToneSpeechProvider()
    storage = JobStorage(tmp_path / "jobs")
    pipeline = GenerationPipeline(
        storage=storage,
        vocabulary_provider=None,
        image_provider=LocalImageProvider([material]),
        speech_provider=speech,
        card_renderer=CardRenderer(),
        video_composer=VideoComposer(),
    )

    first = pipeline.run(request)
    second = pipeline.regenerate(first.job_id, replacements={0: replacement})
    assert first.video_path is not None and first.video_path.name == "video-0001.mp4"
    assert second.video_path is not None and second.video_path.name == "video-0002.mp4"
    assert first.video_path.is_file()
    assert second.video_path.is_file()
    assert [call[0] for call in speech.calls].count("What is this?") == 1

    manifest = storage.load_manifest(first.job_id)
    serialized_manifest = json.dumps(manifest, ensure_ascii=False)
    assert manifest["schema_version"] == 3
    assert manifest["request"]["narration"]["question"]["repeats"] == 2
    assert manifest["request"]["english_text"]["font_path"].startswith("inputs/fonts/")
    assert str(source_font) not in serialized_manifest
    assert manifest["material_assignments"]["0"]["kind"] == "video"
    assert manifest["material_assignments"]["0"]["source"] == "replacement"
    assert manifest["material_assignments"]["0"]["path"].startswith("inputs/replacements/")
    assert manifest["artifacts"]["video"] == "artifacts/videos/video-0002.mp4"
    assert manifest["artifacts"]["videos"] == [
        "artifacts/videos/video-0001.mp4",
        "artifacts/videos/video-0002.mp4",
    ]
    assert str(fixtures) not in serialized_manifest
    assert str(tmp_path) not in serialized_manifest
    loaded_after_regeneration = storage.load_request(first.job_id)
    assert loaded_after_regeneration.english_text.font_path is not None
    assert loaded_after_regeneration.english_text.font_path.is_relative_to(
        first.manifest_path.parent / "inputs" / "fonts"
    )

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(second.video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video_stream = next(stream for stream in streams if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in streams if stream["codec_type"] == "audio")
    numerator, denominator = map(int, video_stream["avg_frame_rate"].split("/"))
    assert video_stream["codec_name"] == "h264"
    assert (video_stream["width"], video_stream["height"]) == (320, 240)
    assert numerator / denominator == pytest.approx(15.0, abs=0.2)
    assert audio_stream["codec_name"] == "aac"

    clip = moviepy.VideoFileClip(str(second.video_path))
    try:
        assert clip.audio is not None
        assert clip.duration > 2.4
        early = clip.get_frame(2.35)
        late = clip.get_frame(2.55)
        samples = _soundarray(clip.audio)
    finally:
        clip.close()
    inside_difference = float(np.abs(early[100, 160].astype(float) - late[100, 160]).mean())
    outside_difference = float(np.abs(early[20, 20].astype(float) - late[20, 20]).mean())
    assert inside_difference > 20.0
    assert outside_difference < 3.0

    after_music_loop = samples[round(2.05 * 44_100) : round(2.25 * 44_100)]
    first_question_repeat = samples[round(1.55 * 44_100) : round(1.85 * 44_100)]
    second_question_repeat = samples[round(1.95 * 44_100) : round(2.25 * 44_100)]
    answer_narration = samples[round(2.35 * 44_100) : round(2.65 * 44_100)]
    music_rms = _tone_rms(after_music_loop, frequency=220.0)
    source_video_rms = _tone_rms(after_music_loop, frequency=660.0)
    assert _tone_rms(first_question_repeat, frequency=440.0) > 0.06
    assert _tone_rms(second_question_repeat, frequency=440.0) > 0.06
    assert _tone_rms(answer_narration, frequency=550.0) > 0.06
    assert source_video_rms < music_rms * 0.15

    control_request = request.model_copy(deep=True)
    control_request.background_music.ducking_percent = 0
    control_storage = JobStorage(tmp_path / "control-jobs")
    control = GenerationPipeline(
        storage=control_storage,
        vocabulary_provider=None,
        image_provider=LocalImageProvider([material]),
        speech_provider=_LocalToneSpeechProvider(),
        card_renderer=CardRenderer(),
        video_composer=VideoComposer(),
    ).run(control_request)
    assert control.video_path is not None
    control_clip = moviepy.VideoFileClip(str(control.video_path))
    try:
        assert control_clip.audio is not None
        control_samples = _soundarray(control_clip.audio)
    finally:
        control_clip.close()
    control_window = control_samples[round(2.05 * 44_100) : round(2.25 * 44_100)]
    unducked_music_rms = _tone_rms(control_window, frequency=220.0)
    assert 1.7 < unducked_music_rms / music_rms < 2.3
