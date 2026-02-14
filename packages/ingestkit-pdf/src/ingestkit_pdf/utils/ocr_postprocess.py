"""OCR postprocessing utilities for cleaning raw OCR text output.

Implements the four postprocessing operations specified in SPEC 11.2 step 6:

1. Merge hyphenated line breaks (e.g. ``docu-\\nment`` -> ``document``)
2. Normalize Unicode to NFC form
3. Normalize whitespace (collapse spaces, normalize line endings)
4. Strip OCR artifacts (isolated characters, repeated punctuation)

This module is stateless and always applied after OCR extraction, before
optional LLM cleanup (step 7) and heading detection / chunking (step 9).
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger("ingestkit_pdf.utils.ocr_postprocess")

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Matches a letter followed by hyphen + newline + letter (word continuation).
# Uses [a-zA-Z] instead of \w to avoid merging digits (e.g. "123-\n456").
_HYPHEN_BREAK_RE = re.compile(r"([a-zA-Z])-\r?\n([a-zA-Z])")

# Matches runs of horizontal whitespace (spaces and tabs), not newlines.
_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")

# Matches trailing whitespace on each line (before newline or end of string).
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# Matches 3 or more consecutive newlines (to collapse to 2).
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

# Matches an isolated single character surrounded by whitespace boundaries,
# excluding common single-letter English words: I, a, A.
_ISOLATED_CHAR_RE = re.compile(
    r"(?<!\S)(?![IaA])([a-zA-Z])(?!\S)",
)

# Matches 4+ repetitions of the same punctuation character.
# Preserves sequences of exactly 3 (e.g. ellipsis "...").
_REPEATED_PUNCT_RE = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])\1{3,}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_hyphenated_breaks(text: str) -> str:
    """Merge hyphenated line breaks where a word is split across lines.

    Only merges when both sides of the hyphen are alphabetic characters,
    preserving hyphens in compounds (e.g. ``self-service``) and digit
    sequences (e.g. ``123-\\n456``).
    """
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode text to NFC (Canonical Decomposition + Composition).

    Ensures consistent representation of composed characters (e.g. ``e`` +
    combining acute accent becomes ``é``).
    """
    return unicodedata.normalize("NFC", text)


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse spaces, normalize line endings, strip trailing.

    Operations:
    - Normalize ``\\r\\n`` and ``\\r`` to ``\\n``
    - Collapse runs of spaces/tabs to a single space
    - Strip trailing whitespace on each line
    - Collapse 3+ consecutive newlines to exactly 2
    """
    # Normalize line endings first
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace runs to single space
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    # Strip trailing whitespace per line
    text = _TRAILING_WS_RE.sub("", text)
    # Collapse excessive blank lines (3+ newlines -> 2)
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text


def _strip_ocr_artifacts(text: str) -> str:
    """Remove common OCR artifacts from text.

    - Removes isolated single characters that are not common English words
      (``I``, ``a``, ``A``).
    - Collapses sequences of 4+ repeated punctuation to a single character.
      Preserves ellipsis (``...``, exactly 3 dots).
    """
    # Remove isolated single characters (except I, a, A)
    text = _ISOLATED_CHAR_RE.sub("", text)
    # Collapse repeated punctuation (4+ of same char -> single)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    # Clean up any double spaces left by artifact removal
    text = re.sub(r"  +", " ", text)
    # Strip leading/trailing whitespace on lines left by removal
    text = _TRAILING_WS_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def postprocess_ocr_text(text: str) -> str:
    """Apply all OCR postprocessing steps to raw OCR text.

    Steps (per SPEC 11.2 step 6):

    1. Merge hyphenated line breaks
    2. Normalize Unicode (NFC)
    3. Normalize whitespace
    4. Strip OCR artifacts

    Returns cleaned text. Returns empty string for empty/whitespace-only input.
    """
    if not text or not text.strip():
        logger.debug("postprocess_ocr_text: empty or whitespace-only input")
        return ""

    logger.debug("postprocess_ocr_text: processing %d chars", len(text))

    text = _merge_hyphenated_breaks(text)
    text = _normalize_unicode(text)
    text = _normalize_whitespace(text)
    text = _strip_ocr_artifacts(text)

    logger.debug("postprocess_ocr_text: result %d chars", len(text))
    return text
