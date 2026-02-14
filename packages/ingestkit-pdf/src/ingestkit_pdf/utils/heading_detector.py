"""Heading hierarchy detection for PDF documents.

Provides ``HeadingDetector`` with three strategies for extracting heading
hierarchies from PDFs:

1. **PDF outline** — authoritative when present (``doc.get_toc()``).
2. **Font-based inference** — identifies bold spans larger than body text.
3. **Markdown parsing** — parses ``#``-style headers from pymupdf4llm output.

Strategies are tried in order; the first non-empty result wins.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fitz

from ingestkit_pdf.config import PDFProcessorConfig

logger = logging.getLogger("ingestkit_pdf")


class HeadingDetector:
    """Detect heading hierarchy in a PDF document.

    Parameters
    ----------
    config:
        Pipeline configuration — uses ``heading_min_font_size_ratio``.
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self._config = config
        self._headings: list[tuple[int, str, int, float]] = []
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, doc: fitz.Document) -> list[tuple[int, str, int]]:
        """Return headings as ``(level, title, page_number)`` tuples.

        Strategies are attempted in priority order: outline, font-based,
        markdown.  The first strategy that produces a non-empty result is
        used.
        """
        try:
            result = self._detect_from_outline(doc)
            if result:
                self._headings = result
                self._logger.debug("Heading detection: used outline strategy")
                return [(lvl, title, page) for lvl, title, page, _ in result]

            result = self._detect_from_fonts(doc)
            if result:
                self._headings = result
                self._logger.debug("Heading detection: used font-based strategy")
                return [(lvl, title, page) for lvl, title, page, _ in result]

            result = self._detect_from_markdown(doc)
            self._headings = result
            if result:
                self._logger.debug("Heading detection: used markdown strategy")
            else:
                self._logger.debug("Heading detection: no headings found by any strategy")
            return [(lvl, title, page) for lvl, title, page, _ in result]
        except Exception:
            self._logger.warning("Heading detection failed unexpectedly", exc_info=True)
            self._headings = []
            return []

    def get_heading_path(self, page_number: int, position_y: float) -> list[str]:
        """Return heading ancestry at the given position.

        Must be called after :meth:`detect`.  Returns a list from the
        outermost heading (H1) to the deepest heading that precedes the
        given ``(page_number, position_y)``.
        """
        if not self._headings:
            return []

        # Find the last heading that is "before" the query position.
        current_idx: int | None = None
        for i, (_, _, h_page, h_y) in enumerate(self._headings):
            if h_page < page_number:
                current_idx = i
            elif h_page == page_number and h_y <= position_y:
                current_idx = i
            elif h_page > page_number:
                break

        if current_idx is None:
            return []

        # Build ancestry by walking backward, collecting the most recent
        # heading at each shallower level.
        current_level, current_title, _, _ = self._headings[current_idx]
        path: list[str] = [current_title]

        needed_level = current_level - 1
        for j in range(current_idx - 1, -1, -1):
            lvl, title, _, _ = self._headings[j]
            if lvl == needed_level:
                path.append(title)
                needed_level -= 1
            if needed_level < 1:
                break

        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Strategy 1: PDF outline / bookmarks
    # ------------------------------------------------------------------

    def _detect_from_outline(
        self, doc: fitz.Document
    ) -> list[tuple[int, str, int, float]]:
        try:
            toc = doc.get_toc()
        except Exception:
            self._logger.warning("Failed to read PDF outline", exc_info=True)
            return []

        if not toc:
            return []

        headings: list[tuple[int, str, int, float]] = []
        for entry in toc:
            level = int(entry[0])
            title = str(entry[1]).strip()
            page = int(entry[2])
            if title:
                headings.append((level, title, page, 0.0))
        return headings

    # ------------------------------------------------------------------
    # Strategy 2: Font-size-based inference
    # ------------------------------------------------------------------

    def _detect_from_fonts(
        self, doc: fitz.Document
    ) -> list[tuple[int, str, int, float]]:
        try:
            return self._font_detection_impl(doc)
        except Exception:
            self._logger.warning("Font-based heading detection failed", exc_info=True)
            return []

    def _font_detection_impl(
        self, doc: fitz.Document
    ) -> list[tuple[int, str, int, float]]:
        # Phase 1: Collect font size data and heading candidates.
        size_weights: Counter[float] = Counter()
        raw_spans: list[tuple[float, str, int, float, bool]] = []  # (size, text, page, y, bold)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0.0)
                        text = span.get("text", "").strip()
                        flags = span.get("flags", 0)
                        y_pos = span.get("origin", [0, 0])[1] if "origin" in span else line.get("bbox", [0, 0, 0, 0])[1]
                        is_bold = bool(flags & (2**4))
                        if text:
                            size_weights[size] += len(text)
                            raw_spans.append((size, text, page_number, y_pos, is_bold))

        if not size_weights:
            return []

        # Phase 2: Compute body font size (most common by character count).
        body_size = size_weights.most_common(1)[0][0]

        # Phase 3: Identify heading spans.
        threshold = body_size * self._config.heading_min_font_size_ratio
        heading_spans: list[tuple[float, str, int, float]] = []
        for size, text, page_num, y_pos, is_bold in raw_spans:
            if size >= threshold and is_bold:
                heading_spans.append((size, text, page_num, y_pos))

        if not heading_spans:
            return []

        # Merge consecutive heading spans on the same line (same page,
        # y-position within 2pt tolerance).
        merged: list[tuple[float, str, int, float]] = [heading_spans[0]]
        for size, text, page_num, y_pos in heading_spans[1:]:
            prev_size, prev_text, prev_page, prev_y = merged[-1]
            if page_num == prev_page and abs(y_pos - prev_y) <= 2.0 and abs(size - prev_size) < 0.5:
                merged[-1] = (prev_size, f"{prev_text} {text}", prev_page, prev_y)
            else:
                merged.append((size, text, page_num, y_pos))

        # Phase 4: Map font sizes to levels (top 3 → H1, H2, H3).
        distinct_sizes = sorted({s for s, _, _, _ in merged}, reverse=True)
        size_to_level: dict[float, int] = {}
        for i, s in enumerate(distinct_sizes):
            size_to_level[s] = min(i + 1, 3)

        result: list[tuple[int, str, int, float]] = []
        for size, title, page_num, y_pos in merged:
            level = size_to_level[size]
            result.append((level, title, page_num, y_pos))

        return result

    # ------------------------------------------------------------------
    # Strategy 3: Markdown header parsing (pymupdf4llm)
    # ------------------------------------------------------------------

    def _detect_from_markdown(
        self, doc: fitz.Document
    ) -> list[tuple[int, str, int, float]]:
        try:
            import pymupdf4llm  # type: ignore[import-untyped]
        except ImportError:
            self._logger.warning("pymupdf4llm not installed; markdown heading strategy unavailable")
            return []

        try:
            page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        except Exception:
            self._logger.warning("pymupdf4llm.to_markdown failed", exc_info=True)
            return []

        header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        headings: list[tuple[int, str, int, float]] = []

        for chunk in page_chunks:
            metadata = chunk.get("metadata", {})
            page_number = metadata.get("page", 0)
            text = chunk.get("text", "")

            for match in header_re.finditer(text):
                level = len(match.group(1))
                title = match.group(2).strip()
                if title:
                    headings.append((level, title, page_number, 0.0))

        return headings
