"""Audio assets used by generated question segments."""

import shutil
from importlib.resources import files
from pathlib import Path

from ai_vocab_video_generator.private_fs import ensure_private_directory, mark_private_file

COUNTDOWN_CUE_FILENAME = "countdown-soft-chime-v1.wav"


def create_countdown_wav(destination: Path) -> Path:
    """Copy the bundled countdown cue into a private job directory."""
    ensure_private_directory(destination.parent)
    asset = files("ai_vocab_video_generator").joinpath("assets/countdown1.wav")
    with asset.open("rb") as source, destination.open("wb") as output:
        shutil.copyfileobj(source, output)
    return mark_private_file(destination)
