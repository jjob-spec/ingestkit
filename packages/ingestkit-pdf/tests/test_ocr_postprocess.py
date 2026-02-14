"""Tests for ingestkit_pdf.utils.ocr_postprocess — OCR text postprocessing.

Covers all four operations from SPEC 11.2 step 6:
- Hyphenated line break merging
- Unicode normalization (NFC)
- Whitespace normalization
- OCR artifact stripping

All tests are unit tests with no external dependencies.
"""

from __future__ import annotations

import pytest

from ingestkit_pdf.utils.ocr_postprocess import (
    _merge_hyphenated_breaks,
    _normalize_unicode,
    _normalize_whitespace,
    _strip_ocr_artifacts,
    postprocess_ocr_text,
)


# ---------------------------------------------------------------------------
# T1-T5: Hyphenated line break merging
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t1_basic_hyphen_merge() -> None:
    """T1: Basic hyphenated word across line break."""
    assert _merge_hyphenated_breaks("docu-\nment") == "document"


@pytest.mark.unit
def test_t2_windows_line_ending_hyphen_merge() -> None:
    """T2: Hyphenated word with Windows (CRLF) line ending."""
    assert _merge_hyphenated_breaks("docu-\r\nment") == "document"


@pytest.mark.unit
def test_t3_multiple_hyphen_merges() -> None:
    """T3: Multiple hyphenated words in one string."""
    result = _merge_hyphenated_breaks("end-\npoint config-\nuration")
    assert result == "endpoint configuration"


@pytest.mark.unit
def test_t4_same_line_hyphen_preserved() -> None:
    """T4: Hyphen within the same line is preserved (e.g. compound words)."""
    assert _merge_hyphenated_breaks("self-service") == "self-service"


@pytest.mark.unit
def test_t5_digit_hyphen_not_merged() -> None:
    """T5: Digits separated by hyphen+newline are NOT merged."""
    assert _merge_hyphenated_breaks("123-\n456") == "123-\n456"


# ---------------------------------------------------------------------------
# T6-T7: Unicode normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t6_nfd_to_nfc() -> None:
    """T6: NFD decomposed character normalized to NFC composed form."""
    # "cafe\u0301" (e + combining acute) -> "caf\u00e9" (precomposed e-acute)
    nfd_input = "caf\u0065\u0301"
    expected = "caf\u00e9"
    assert _normalize_unicode(nfd_input) == expected


@pytest.mark.unit
def test_t7_ascii_unchanged() -> None:
    """T7: Pure ASCII text passes through unchanged."""
    assert _normalize_unicode("resume") == "resume"


# ---------------------------------------------------------------------------
# T8-T12: Whitespace normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t8_collapse_multiple_spaces() -> None:
    """T8: Multiple spaces collapsed to single space."""
    assert _normalize_whitespace("hello   world") == "hello world"


@pytest.mark.unit
def test_t9_collapse_tabs() -> None:
    """T9: Multiple tabs collapsed to single space."""
    assert _normalize_whitespace("a\t\tb") == "a b"


@pytest.mark.unit
def test_t10_normalize_line_endings() -> None:
    """T10: CRLF and CR normalized to LF."""
    result = _normalize_whitespace("line1\r\nline2\rline3")
    assert result == "line1\nline2\nline3"


@pytest.mark.unit
def test_t11_collapse_excessive_blank_lines() -> None:
    """T11: 3+ consecutive newlines collapsed to exactly 2."""
    assert _normalize_whitespace("a\n\n\n\nb") == "a\n\nb"


@pytest.mark.unit
def test_t12_strip_trailing_whitespace() -> None:
    """T12: Trailing whitespace on each line is removed."""
    result = _normalize_whitespace("  trailing  \n")
    assert result == " trailing\n"


# ---------------------------------------------------------------------------
# T13-T17: OCR artifact stripping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t13_isolated_single_char_removed() -> None:
    """T13: Isolated single character (not a common word) is removed."""
    result = _strip_ocr_artifacts("hello x world")
    assert result == "hello world"


@pytest.mark.unit
def test_t14_common_single_char_words_preserved() -> None:
    """T14: Common single-letter words I, a, A are preserved."""
    result = _strip_ocr_artifacts("I am a doctor")
    assert result == "I am a doctor"


@pytest.mark.unit
def test_t15_repeated_punctuation_collapsed() -> None:
    """T15: 4+ repeated punctuation collapsed to single character."""
    result = _strip_ocr_artifacts("price!!!!!")
    assert result == "price!"


@pytest.mark.unit
def test_t16_ellipsis_preserved() -> None:
    """T16: Ellipsis (exactly 3 dots) is preserved."""
    result = _strip_ocr_artifacts("wait...")
    assert result == "wait..."


@pytest.mark.unit
def test_t17_single_period_preserved() -> None:
    """T17: Single period in normal text is preserved."""
    result = _strip_ocr_artifacts("Mr. Smith")
    assert result == "Mr. Smith"


# ---------------------------------------------------------------------------
# T18-T21: Full pipeline integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_t18_combined_pipeline() -> None:
    """T18: All steps applied together."""
    result = postprocess_ocr_text("docu-\nment   with   extra  spaces")
    assert result == "document with extra spaces"


@pytest.mark.unit
def test_t19_empty_string() -> None:
    """T19: Empty string returns empty string."""
    assert postprocess_ocr_text("") == ""


@pytest.mark.unit
def test_t20_clean_text_preserved() -> None:
    """T20: Clean text passes through unchanged."""
    assert postprocess_ocr_text("Clean text preserved.") == "Clean text preserved."


@pytest.mark.unit
def test_t21_realistic_ocr_sample() -> None:
    """T21: Realistic multi-paragraph OCR output with multiple issues."""
    raw_ocr = (
        "The docu-\r\nment describes the com-\npany's poli-\ncy.\n"
        "\n"
        "\n"
        "\n"
        "Section 2:   Overview  \n"
        "\n"
        "This is a  x  normal para-\ngraph with   extra   spaces "
        "and caf\u0065\u0301 latt\u0065\u0301.!!!!\n"
    )
    result = postprocess_ocr_text(raw_ocr)

    # Hyphenated words should be merged
    assert "document" in result
    assert "company's" in result
    assert "policy" in result
    assert "paragraph" in result

    # Unicode should be NFC
    assert "caf\u00e9" in result
    assert "latt\u00e9" in result

    # Excessive blank lines collapsed
    assert "\n\n\n" not in result

    # Extra spaces collapsed
    assert "   " not in result

    # Trailing whitespace stripped
    for line in result.split("\n"):
        assert line == line.rstrip(), f"Trailing whitespace found: {line!r}"

    # Isolated 'x' artifact removed
    assert " x " not in result

    # Repeated punctuation collapsed
    assert "!!!!" not in result
