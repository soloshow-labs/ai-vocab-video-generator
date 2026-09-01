"""Owner-private filesystem helpers for runtime data."""

import os
import shutil
from pathlib import Path


def ensure_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)
    return path


def mark_private_file(path: Path) -> Path:
    if os.name == "posix":
        path.chmod(0o600)
    return path


def write_private_bytes(path: Path, contents: bytes) -> Path:
    ensure_private_directory(path.parent)
    path.write_bytes(contents)
    return mark_private_file(path)


def copy_private_file(source: Path, destination: Path) -> Path:
    ensure_private_directory(destination.parent)
    shutil.copyfile(source, destination)
    return mark_private_file(destination)
