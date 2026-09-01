"""Line-group vocabulary script parsing."""

from collections.abc import Sequence

from ai_vocab_video_generator.domain import (
    MAX_SCRIPT_LENGTH,
    MAX_VOCABULARY_ENTRIES,
    PhoneticMode,
    WordEntry,
)


class ScriptFormatError(ValueError):
    """A visible script group cannot be converted to a vocabulary entry."""


def parse_vocabulary_script(value: str, mode: PhoneticMode) -> list[WordEntry]:
    if len(value) > MAX_SCRIPT_LENGTH:
        raise ScriptFormatError(f"Script must not exceed {MAX_SCRIPT_LENGTH:,} characters.")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    group_size = 3 if mode is PhoneticMode.MANUAL else 2
    if len(lines) > MAX_VOCABULARY_ENTRIES * group_size:
        raise ScriptFormatError(
            f"Script must contain at most {MAX_VOCABULARY_ENTRIES} vocabulary entries."
        )
    if len(lines) % group_size:
        group = len(lines) // group_size + 1
        raise ScriptFormatError(
            f"Script group {group} needs {group_size} non-empty lines "
            "in Chinese, English, and optional phonetic order."
        )

    entries: list[WordEntry] = []
    for offset in range(0, len(lines), group_size):
        group = offset // group_size + 1
        chinese, english = lines[offset : offset + 2]
        phonetic = lines[offset + 2] if mode is PhoneticMode.MANUAL else ""
        try:
            entries.append(WordEntry(chinese=chinese, english=english, phonetic=phonetic))
        except ValueError as exc:
            raise ScriptFormatError(f"Script group {group} is invalid.") from exc
    return entries


def serialize_vocabulary_script(entries: Sequence[WordEntry], mode: PhoneticMode) -> str:
    lines: list[str] = []
    for entry in entries:
        lines.extend((entry.chinese, entry.english))
        if mode is PhoneticMode.MANUAL:
            lines.append(entry.phonetic)
    return "\n".join(lines)
