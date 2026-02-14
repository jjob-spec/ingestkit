"""Header/footer detection and stripping for PDF documents.

Implements cross-page text similarity analysis to identify repeating
headers and footers, per SPEC sections 13.1-13.3.  Uses
``difflib.SequenceMatcher`` for fuzzy matching and operates on
``fitz.Document`` objects.

This module is a second-pass utility for edge cases not handled by
pymupdf4llm's built-in header/footer suppression (SPEC 13.2).
"""

from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING

from ingestkit_pdf.config import PDFProcessorConfig

if TYPE_CHECKING:
    import fitz  # type: ignore[import-untyped]

logger = logging.getLogger("ingestkit_pdf.utils.header_footer")


class HeaderFooterDetector:
    """Detect and strip repeating headers/footers from a PDF document.

    Parameters
    ----------
    config:
        Pipeline configuration providing ``header_footer_sample_pages``,
        ``header_footer_zone_ratio``, and ``header_footer_similarity_threshold``.
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self._config = config
        self._sample_pages = config.header_footer_sample_pages
        self._zone_ratio = config.header_footer_zone_ratio
        self._similarity_threshold = config.header_footer_similarity_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_sample_indices(self, page_count: int) -> list[int]:
        """Return evenly distributed page indices for sampling."""
        if page_count <= self._sample_pages:
            return list(range(page_count))
        return [
            round(i * (page_count - 1) / (self._sample_pages - 1))
            for i in range(self._sample_pages)
        ]

    def _extract_zone_text(
        self, page: fitz.Page, zone: str  # type: ignore[name-defined]
    ) -> list[str]:
        """Extract text blocks from the header or footer zone of *page*.

        Parameters
        ----------
        page:
            A ``fitz.Page`` object.
        zone:
            ``"header"`` for the top zone or ``"footer"`` for the bottom zone.

        Returns
        -------
        list[str]
            Stripped text strings from blocks falling within the zone.
        """
        page_height: float = page.rect.height
        zone_height = page_height * self._zone_ratio

        # get_text("blocks") returns (x0, y0, x1, y1, text, block_no, type)
        blocks = page.get_text("blocks")

        texts: list[str] = []
        for block in blocks:
            # type field: 0 = text, 1 = image
            if block[6] != 0:
                continue

            y0, y1 = block[1], block[3]
            text = block[4].strip()
            if not text:
                continue

            if zone == "header" and y0 < zone_height:
                texts.append(text)
            elif zone == "footer" and y1 > page_height - zone_height:
                texts.append(text)

        logger.debug(
            "Extracted %d %s zone block(s) from page",
            len(texts),
            zone,
        )
        return texts

    def _find_repeating_patterns(
        self, zone_texts: list[list[str]]
    ) -> list[str]:
        """Identify text that repeats across most sampled pages.

        A candidate is considered a repeating pattern when it appears on
        at least ``len(zone_texts) - 1`` pages (with similarity >=
        threshold).

        Parameters
        ----------
        zone_texts:
            One list of text strings per sampled page.

        Returns
        -------
        list[str]
            Deduplicated list of pattern strings.
        """
        if not zone_texts:
            return []

        min_occurrences = max(len(zone_texts) - 1, 1)

        # Gather all unique candidate texts across every sampled page.
        seen: set[str] = set()
        candidates: list[str] = []
        for page_texts in zone_texts:
            for text in page_texts:
                if text not in seen:
                    seen.add(text)
                    candidates.append(text)

        patterns: list[str] = []
        for candidate in candidates:
            match_count = 0
            for page_texts in zone_texts:
                for text in page_texts:
                    ratio = difflib.SequenceMatcher(
                        None, candidate, text
                    ).ratio()
                    if ratio >= self._similarity_threshold:
                        match_count += 1
                        break  # one match per page is enough
            if match_count >= min_occurrences:
                patterns.append(candidate)

        # Deduplicate patterns that are similar to each other.
        deduplicated: list[str] = []
        for pattern in patterns:
            is_duplicate = False
            for existing in deduplicated:
                if (
                    difflib.SequenceMatcher(None, pattern, existing).ratio()
                    >= self._similarity_threshold
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(pattern)

        return deduplicated

    # ------------------------------------------------------------------
    # Public interface (SPEC 13.3)
    # ------------------------------------------------------------------

    def detect(
        self, doc: fitz.Document  # type: ignore[name-defined]
    ) -> tuple[list[str], list[str]]:
        """Detect repeating header and footer patterns in *doc*.

        Parameters
        ----------
        doc:
            A ``fitz.Document`` (PyMuPDF) object.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(header_patterns, footer_patterns)``.  Empty lists when
            the document has fewer than 2 pages.
        """
        page_count = len(doc)
        if page_count < 2:
            logger.debug(
                "Document has %d page(s); skipping header/footer detection",
                page_count,
            )
            return ([], [])

        indices = self._select_sample_indices(page_count)

        header_zone_texts: list[list[str]] = []
        footer_zone_texts: list[list[str]] = []

        for idx in indices:
            page = doc[idx]
            header_zone_texts.append(self._extract_zone_text(page, "header"))
            footer_zone_texts.append(self._extract_zone_text(page, "footer"))

        header_patterns = self._find_repeating_patterns(header_zone_texts)
        footer_patterns = self._find_repeating_patterns(footer_zone_texts)

        logger.info(
            "Detected %d header pattern(s) and %d footer pattern(s)",
            len(header_patterns),
            len(footer_patterns),
        )
        return (header_patterns, footer_patterns)

    def strip(
        self,
        text: str,
        page_number: int,
        headers: list[str],
        footers: list[str],
    ) -> str:
        """Remove lines matching header/footer patterns from *text*.

        Parameters
        ----------
        text:
            The raw page text to clean.
        page_number:
            The page number (available for future positional logic).
        headers:
            Header patterns returned by :meth:`detect`.
        footers:
            Footer patterns returned by :meth:`detect`.

        Returns
        -------
        str
            Cleaned text with matching lines removed and leading/trailing
            blank lines stripped.
        """
        if not headers and not footers:
            return text

        all_patterns = list(headers) + list(footers)
        lines = text.splitlines()
        kept: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue

            is_match = False
            for pattern in all_patterns:
                ratio = difflib.SequenceMatcher(
                    None, stripped, pattern
                ).ratio()
                if ratio >= self._similarity_threshold:
                    is_match = True
                    break

            if not is_match:
                kept.append(line)

        result = "\n".join(kept).strip()
        return result
