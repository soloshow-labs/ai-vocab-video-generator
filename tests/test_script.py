import pytest

from ai_vocab_video_generator.domain import (
    MAX_SCRIPT_LENGTH,
    MAX_VOCABULARY_ENTRIES,
    PhoneticMode,
    WordEntry,
)
from ai_vocab_video_generator.script import (
    ScriptFormatError,
    parse_vocabulary_script,
    serialize_vocabulary_script,
)


def test_manual_script_parses_chinese_english_phonetic_triples() -> None:
    assert parse_vocabulary_script("苹果\napple\n/ˈæp.əl/", PhoneticMode.MANUAL) == [
        WordEntry(chinese="苹果", english="apple", phonetic="/ˈæp.əl/")
    ]


def test_automatic_script_parses_pairs_and_ignores_blank_lines() -> None:
    parsed = parse_vocabulary_script(" 苹果 \n\n apple \n 香蕉\n banana ", PhoneticMode.AUTOMATIC)

    assert parsed == [
        WordEntry(chinese="苹果", english="apple"),
        WordEntry(chinese="香蕉", english="banana"),
    ]


def test_automatic_script_reports_the_invalid_group() -> None:
    with pytest.raises(ScriptFormatError, match="group 2"):
        parse_vocabulary_script("苹果\napple\n香蕉", PhoneticMode.AUTOMATIC)


def test_manual_script_reports_a_blank_required_value() -> None:
    with pytest.raises(ScriptFormatError, match="group 1"):
        parse_vocabulary_script("苹果\n   \n/ˈæp.əl/", PhoneticMode.MANUAL)


def test_script_rejects_oversized_text_before_parsing() -> None:
    with pytest.raises(ScriptFormatError, match="characters"):
        parse_vocabulary_script("x" * (MAX_SCRIPT_LENGTH + 1), PhoneticMode.AUTOMATIC)


def test_script_accepts_fifty_entries_and_rejects_fifty_one() -> None:
    fifty = "\n".join(
        value for index in range(MAX_VOCABULARY_ENTRIES) for value in (f"词{index}", f"word{index}")
    )
    fifty_one = f"{fifty}\n额外\nextra"

    assert len(parse_vocabulary_script(fifty, PhoneticMode.AUTOMATIC)) == MAX_VOCABULARY_ENTRIES
    with pytest.raises(ScriptFormatError, match="at most 50"):
        parse_vocabulary_script(fifty_one, PhoneticMode.AUTOMATIC)


@pytest.mark.parametrize("mode", list(PhoneticMode))
def test_script_round_trip_is_stable(mode: PhoneticMode) -> None:
    entries = [WordEntry(chinese="苹果", english="apple", phonetic="/ˈæp.əl/")]
    serialized = serialize_vocabulary_script(entries, mode)
    parsed = parse_vocabulary_script(serialized, mode)

    if mode is PhoneticMode.MANUAL:
        assert parsed == entries
    else:
        assert parsed == [WordEntry(chinese="苹果", english="apple")]
