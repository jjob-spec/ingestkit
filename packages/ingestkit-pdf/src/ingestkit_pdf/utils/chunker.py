"""Configurable text chunking with structural awareness.

Implements recursive character splitting with heading-aware boundaries
and table-aware chunking. Pure utility with no backend dependencies.

Spec references: §15.1 (default strategy), §15.2 (table-aware),
§15.3 (metadata attachment), §15.4 (public interface).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestkit_pdf.config import PDFProcessorConfig

logger = logging.getLogger("ingestkit_pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 heuristic."""
    return max(1, len(text) // 4)


def _compute_chunk_hash(text: str) -> str:
    """Return SHA-256 hex digest of the chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_content_type(text: str) -> str:
    """Classify chunk content as narrative, table, list, or form_field."""
    lines = text.strip().splitlines()
    if not lines:
        return "narrative"

    table_lines = sum(1 for line in lines if "|" in line)
    if table_lines > len(lines) / 2:
        return "table"

    list_pattern = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
    list_lines = sum(1 for line in lines if list_pattern.match(line))
    if list_lines > len(lines) / 2:
        return "list"

    form_pattern = re.compile(r"\[\s*[xX ]?\s*\]|_{3,}")
    form_lines = sum(1 for line in lines if form_pattern.search(line))
    if form_lines > len(lines) / 3:
        return "form_field"

    return "narrative"


def _find_page_for_position(position: int, page_boundaries: list[int]) -> int:
    """Return 1-based page number for a character position."""
    if not page_boundaries:
        return 1
    page = 1
    for i, boundary in enumerate(page_boundaries):
        if position >= boundary:
            page = i + 1
        else:
            break
    return page


def _get_page_numbers(start: int, end: int, page_boundaries: list[int]) -> list[int]:
    """Return list of page numbers a chunk spanning [start, end) covers."""
    start_page = _find_page_for_position(start, page_boundaries)
    end_page = _find_page_for_position(max(start, end - 1), page_boundaries)
    return list(range(start_page, end_page + 1))


def _get_heading_path(
    position: int, headings: list[tuple[int, str, int]]
) -> list[str]:
    """Return heading ancestry at a given character position.

    Parameters
    ----------
    position:
        Character offset in the document.
    headings:
        List of ``(level, title, char_offset)`` tuples sorted by offset.
    """
    stack: dict[int, str] = {}
    for level, title, offset in headings:
        if offset > position:
            break
        # Clear deeper levels when a higher-level heading appears
        keys_to_remove = [k for k in stack if k >= level]
        for k in keys_to_remove:
            del stack[k]
        stack[level] = title
    return [stack[k] for k in sorted(stack)]


def _extract_tables(text: str) -> list[tuple[int, int]]:
    """Find contiguous markdown table regions in *text*.

    Returns list of ``(start_offset, end_offset)`` for each table.
    A table is a run of consecutive lines where each line contains ``|``.
    """
    regions: list[tuple[int, int]] = []
    lines = text.split("\n")
    offset = 0
    in_table = False
    table_start = 0

    for line in lines:
        line_end = offset + len(line) + 1  # +1 for the newline
        if "|" in line:
            if not in_table:
                in_table = True
                table_start = offset
        else:
            if in_table:
                regions.append((table_start, offset))  # end before this line
                in_table = False
        offset = line_end

    if in_table:
        # Table goes to end of text
        regions.append((table_start, len(text)))

    return regions


def _split_recursive(
    text: str, separators: list[str], chunk_size_chars: int
) -> list[str]:
    """Recursively split *text* into pieces under *chunk_size_chars*.

    Tries separators in order.  Merges adjacent small pieces.
    Falls back to hard character split if no separators remain.
    """
    if len(text) <= chunk_size_chars:
        return [text]

    if not separators:
        # Hard split
        pieces = []
        for i in range(0, len(text), chunk_size_chars):
            pieces.append(text[i : i + chunk_size_chars])
        return pieces

    sep = separators[0]
    remaining_seps = separators[1:]
    parts = text.split(sep)

    # Merge adjacent parts to stay under chunk_size_chars
    merged: list[str] = []
    current = parts[0]
    for part in parts[1:]:
        candidate = current + sep + part
        if len(candidate) <= chunk_size_chars:
            current = candidate
        else:
            merged.append(current)
            current = part
    merged.append(current)

    # Recurse on oversized pieces
    result: list[str] = []
    for piece in merged:
        if len(piece) > chunk_size_chars:
            result.extend(_split_recursive(piece, remaining_seps, chunk_size_chars))
        else:
            result.append(piece)

    return result


# ---------------------------------------------------------------------------
# PDFChunker
# ---------------------------------------------------------------------------


class PDFChunker:
    """Configurable text chunker with heading and table awareness.

    Parameters
    ----------
    config:
        ``PDFProcessorConfig`` instance providing chunking parameters.
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self._config = config
        self._chunk_size_chars = config.chunk_size_tokens * 4
        self._overlap_chars = config.chunk_overlap_tokens * 4
        self._separators = ["\n## ", "\n### ", "\n\n", ". ", " "]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        headings: list[tuple[int, str, int]],
        page_boundaries: list[int],
    ) -> list[dict]:
        """Split *text* into metadata-enriched chunks.

        Parameters
        ----------
        text:
            Full document text to chunk.
        headings:
            List of ``(level, title, char_offset)`` for document headings.
        page_boundaries:
            Character offsets where each page starts.

        Returns
        -------
        list[dict]
            Each dict contains ``text``, ``page_numbers``, ``heading_path``,
            ``content_type``, ``chunk_index``, ``chunk_hash``.
        """
        if not text or not text.strip():
            return []

        raw_segments = self._split_into_segments(text)
        chunks_with_offsets = self._apply_overlap(raw_segments)
        return self._build_chunk_dicts(chunks_with_offsets, headings, page_boundaries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_into_segments(
        self, text: str
    ) -> list[tuple[str, int]]:
        """Split text into ``(chunk_text, start_offset)`` pairs."""
        if self._config.chunk_respect_tables:
            return self._split_with_tables(text)
        return self._split_plain(text, 0)

    def _split_plain(
        self, text: str, base_offset: int
    ) -> list[tuple[str, int]]:
        """Split text without table awareness."""
        if self._config.chunk_respect_headings:
            return self._split_heading_aware(text, base_offset)

        pieces = _split_recursive(text, self._separators, self._chunk_size_chars)
        segments: list[tuple[str, int]] = []
        offset = base_offset
        for piece in pieces:
            idx = text.find(piece, offset - base_offset)
            real_offset = base_offset + idx if idx >= 0 else offset
            segments.append((piece, real_offset))
            offset = real_offset + len(piece)
        return segments

    def _split_heading_aware(
        self, text: str, base_offset: int
    ) -> list[tuple[str, int]]:
        """Split text respecting heading boundaries as hard breaks."""
        # Split on heading markers first, then recurse within each section
        heading_seps = ["\n## ", "\n### "]
        sections: list[tuple[str, int]] = []

        # Use the first heading separator that appears
        split_sep = None
        for sep in heading_seps:
            if sep in text:
                split_sep = sep
                break

        if split_sep is None:
            # No heading separators — use remaining separators
            remaining_seps = self._separators[2:]  # "\n\n", ". ", " "
            pieces = _split_recursive(text, remaining_seps, self._chunk_size_chars)
            offset = base_offset
            for piece in pieces:
                idx = text.find(piece, offset - base_offset)
                real_offset = base_offset + idx if idx >= 0 else offset
                sections.append((piece, real_offset))
                offset = real_offset + len(piece)
            return sections

        parts = text.split(split_sep)
        offset = base_offset
        for i, part in enumerate(parts):
            if i > 0:
                part = split_sep.lstrip("\n") + part  # Re-attach heading marker
            if not part.strip():
                offset += len(parts[i]) + (len(split_sep) if i > 0 else 0)
                continue

            part_start = offset
            if len(part) > self._chunk_size_chars:
                # Recurse with deeper separators
                sep_idx = self._separators.index(split_sep) + 1
                deeper_seps = self._separators[sep_idx:]
                sub_pieces = _split_recursive(part, deeper_seps, self._chunk_size_chars)
                sub_offset = part_start
                for sp in sub_pieces:
                    idx = part.find(sp, sub_offset - part_start)
                    real_offset = part_start + idx if idx >= 0 else sub_offset
                    sections.append((sp, real_offset))
                    sub_offset = real_offset + len(sp)
            else:
                sections.append((part, part_start))

            # Advance offset past original part + separator
            offset += len(parts[i]) + (len(split_sep) if i > 0 else 0)

        return sections

    def _split_with_tables(
        self, text: str
    ) -> list[tuple[str, int]]:
        """Split text preserving tables as atomic chunks."""
        table_regions = _extract_tables(text)
        if not table_regions:
            return self._split_plain(text, 0)

        segments: list[tuple[str, int]] = []
        prev_end = 0

        for t_start, t_end in table_regions:
            # Process text before this table
            if t_start > prev_end:
                pre_text = text[prev_end:t_start]
                if pre_text.strip():
                    segments.extend(self._split_plain(pre_text, prev_end))

            # Table as atomic chunk
            table_text = text[t_start:t_end]
            if table_text.strip():
                segments.append((table_text, t_start))

            prev_end = t_end

        # Process text after last table
        if prev_end < len(text):
            post_text = text[prev_end:]
            if post_text.strip():
                segments.extend(self._split_plain(post_text, prev_end))

        return segments

    def _apply_overlap(
        self, segments: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        """Prepend overlap from previous chunk to each segment after the first."""
        if self._overlap_chars <= 0 or len(segments) <= 1:
            return segments

        result: list[tuple[str, int]] = [segments[0]]
        for i in range(1, len(segments)):
            prev_text = segments[i - 1][0]
            cur_text, cur_offset = segments[i]
            overlap = prev_text[-self._overlap_chars :]
            overlapped = overlap + cur_text
            # Offset stays at the overlap start position
            new_offset = max(0, cur_offset - len(overlap))
            result.append((overlapped, new_offset))

        return result

    def _build_chunk_dicts(
        self,
        chunks: list[tuple[str, int]],
        headings: list[tuple[int, str, int]],
        page_boundaries: list[int],
    ) -> list[dict]:
        """Attach metadata to each chunk and return list of dicts."""
        result: list[dict] = []
        for idx, (raw_text, start_offset) in enumerate(chunks):
            stripped = raw_text.strip()
            if not stripped:
                continue
            end_offset = start_offset + len(raw_text)
            result.append(
                {
                    "text": stripped,
                    "page_numbers": _get_page_numbers(
                        start_offset, end_offset, page_boundaries
                    ),
                    "heading_path": _get_heading_path(start_offset, headings),
                    "content_type": _detect_content_type(stripped),
                    "chunk_index": len(result),
                    "chunk_hash": _compute_chunk_hash(stripped),
                }
            )

        if logger.isEnabledFor(logging.DEBUG) and self._config.log_chunk_previews:
            for c in result:
                logger.debug(
                    "Chunk %d (%s): %s...",
                    c["chunk_index"],
                    c["content_type"],
                    c["text"][:80],
                )

        return result
