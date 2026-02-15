"""Multi-column layout detection and reading order correction.

Detects multi-column layouts in PDF pages via text block x-coordinate
clustering and reorders text blocks into correct reading order, per
SPEC sections 9.2 (signal 5) and 11.3 step 4.

Uses gap-based 1D clustering on block x0 coordinates — no external
dependencies (stdlib only).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ingestkit_pdf.config import PDFProcessorConfig

if TYPE_CHECKING:
    import fitz  # type: ignore[import-untyped]

logger = logging.getLogger("ingestkit_pdf.utils.layout_analysis")

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_MIN_BLOCKS_FOR_CLUSTERING = 3
_GAP_RATIO = 0.10  # 10% of page width = significant column gap
_FULL_WIDTH_RATIO = 0.75  # Block spanning >75% of page width = full-width
_MAX_COLUMNS = 3
_MIN_BLOCKS_PER_CLUSTER = 2


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """A positioned text block extracted from a PDF page."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_number: int


class LayoutResult(BaseModel):
    """Result of layout analysis for a single page."""

    is_multi_column: bool
    column_count: int  # 1, 2, or 3
    column_boundaries: list[tuple[float, float]]  # (x_start, x_end) per column
    page_width: float  # Needed by reorder_blocks for full-width threshold


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_SINGLE_COLUMN_DEFAULT_WIDTH = 612.0


def _single_column_result(
    page_width: float = _SINGLE_COLUMN_DEFAULT_WIDTH,
) -> LayoutResult:
    """Return a safe single-column default."""
    return LayoutResult(
        is_multi_column=False,
        column_count=1,
        column_boundaries=[(0.0, page_width)],
        page_width=page_width,
    )


def extract_text_blocks(page: fitz.Page) -> list[TextBlock]:  # type: ignore[name-defined]
    """Extract text-only blocks with non-empty content from *page*.

    Parameters
    ----------
    page:
        A ``fitz.Page`` object.

    Returns
    -------
    list[TextBlock]
        Filtered list of text blocks (image blocks and whitespace-only
        blocks are excluded).
    """
    raw_blocks = page.get_text("blocks")
    result: list[TextBlock] = []
    for block in raw_blocks:
        # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        block_type = block[6]
        if block_type != 0:
            continue
        text = block[4].strip()
        if not text:
            continue
        result.append(
            TextBlock(
                x0=block[0],
                y0=block[1],
                x1=block[2],
                y1=block[3],
                text=block[4],
                block_number=block[5],
            )
        )
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LayoutAnalyzer:
    """Detect multi-column layouts and reorder text blocks for correct reading order.

    Parameters
    ----------
    config:
        Pipeline configuration (reserved for future layout thresholds).
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect_columns(self, page: fitz.Page) -> LayoutResult:  # type: ignore[name-defined]
        """Analyse *page* and return column layout information.

        Uses gap-based 1D clustering on the x0 coordinate of each text
        block.  Full-width blocks (>75% of page width) are excluded from
        clustering to avoid masking genuine column structure.

        Parameters
        ----------
        page:
            A ``fitz.Page`` object.

        Returns
        -------
        LayoutResult
            Detection result including column count and boundaries.
        """
        try:
            return self._detect_columns_impl(page)
        except Exception:
            logger.warning(
                "Unexpected error during column detection; "
                "returning single-column default",
                exc_info=True,
            )
            return _single_column_result()

    def reorder_blocks(
        self, blocks: list[TextBlock], layout: LayoutResult
    ) -> list[TextBlock]:
        """Reorder *blocks* into correct reading order given *layout*.

        For single-column layouts, blocks are returned sorted top-to-bottom.
        For multi-column layouts: full-width blocks first (by y0), then
        each column left-to-right with blocks sorted top-to-bottom within
        each column.

        Parameters
        ----------
        blocks:
            Text blocks to reorder.
        layout:
            The ``LayoutResult`` from :meth:`detect_columns`.

        Returns
        -------
        list[TextBlock]
            Blocks in reading order.
        """
        try:
            return self._reorder_blocks_impl(blocks, layout)
        except Exception:
            logger.warning(
                "Unexpected error during block reordering; "
                "returning original order",
                exc_info=True,
            )
            return list(blocks)

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _detect_columns_impl(self, page: fitz.Page) -> LayoutResult:  # type: ignore[name-defined]
        page_width: float = page.rect.width
        text_blocks = extract_text_blocks(page)

        if len(text_blocks) < _MIN_BLOCKS_FOR_CLUSTERING:
            logger.debug(
                "Only %d text block(s); returning single-column",
                len(text_blocks),
            )
            return _single_column_result(page_width)

        # Separate full-width blocks from narrow (columnar) blocks.
        full_width_threshold = page_width * _FULL_WIDTH_RATIO
        columnar_blocks: list[TextBlock] = []
        for blk in text_blocks:
            block_width = blk.x1 - blk.x0
            if block_width <= full_width_threshold:
                columnar_blocks.append(blk)

        if len(columnar_blocks) < _MIN_BLOCKS_FOR_CLUSTERING:
            logger.debug(
                "Only %d narrow block(s) after full-width filtering; "
                "returning single-column",
                len(columnar_blocks),
            )
            return _single_column_result(page_width)

        # Gap-based 1D clustering on x0 values.
        clusters = self._cluster_x0(columnar_blocks, page_width)

        # Validate: each cluster needs at least _MIN_BLOCKS_PER_CLUSTER members.
        valid_clusters: list[list[TextBlock]] = [
            c for c in clusters if len(c) >= _MIN_BLOCKS_PER_CLUSTER
        ]

        if len(valid_clusters) <= 1:
            logger.debug(
                "Only %d valid cluster(s); returning single-column",
                len(valid_clusters),
            )
            return _single_column_result(page_width)

        # Cap at _MAX_COLUMNS.
        if len(valid_clusters) > _MAX_COLUMNS:
            valid_clusters = valid_clusters[:_MAX_COLUMNS]

        # Compute column boundaries.
        boundaries: list[tuple[float, float]] = []
        for cluster in valid_clusters:
            x_start = min(blk.x0 for blk in cluster)
            x_end = max(blk.x1 for blk in cluster)
            boundaries.append((x_start, x_end))

        # Sort boundaries left-to-right.
        boundaries.sort(key=lambda b: b[0])

        column_count = len(boundaries)
        result = LayoutResult(
            is_multi_column=column_count > 1,
            column_count=column_count,
            column_boundaries=boundaries,
            page_width=page_width,
        )
        logger.debug(
            "Detected %d column(s) with boundaries %s",
            column_count,
            boundaries,
        )
        return result

    @staticmethod
    def _cluster_x0(
        blocks: list[TextBlock], page_width: float
    ) -> list[list[TextBlock]]:
        """Cluster *blocks* by x0 using gap-based splitting."""
        min_gap = page_width * _GAP_RATIO

        # Sort blocks by x0.
        sorted_blocks = sorted(blocks, key=lambda b: b.x0)

        clusters: list[list[TextBlock]] = [[sorted_blocks[0]]]
        for blk in sorted_blocks[1:]:
            prev_x0 = clusters[-1][-1].x0
            if blk.x0 - prev_x0 > min_gap:
                clusters.append([blk])
            else:
                clusters[-1].append(blk)

        return clusters

    def _reorder_blocks_impl(
        self, blocks: list[TextBlock], layout: LayoutResult
    ) -> list[TextBlock]:
        if layout.column_count == 1:
            return sorted(blocks, key=lambda b: b.y0)

        page_width = layout.page_width
        full_width_threshold = page_width * _FULL_WIDTH_RATIO

        full_width_blocks: list[TextBlock] = []
        columnar_blocks: list[TextBlock] = []

        for blk in blocks:
            block_width = blk.x1 - blk.x0
            if block_width > full_width_threshold:
                full_width_blocks.append(blk)
            else:
                columnar_blocks.append(blk)

        # Sort full-width blocks top-to-bottom.
        full_width_blocks.sort(key=lambda b: b.y0)

        # Assign columnar blocks to columns by midpoint.
        column_buckets: list[list[TextBlock]] = [
            [] for _ in layout.column_boundaries
        ]

        for blk in columnar_blocks:
            midpoint = (blk.x0 + blk.x1) / 2.0
            best_idx = 0
            best_dist = float("inf")
            for idx, (col_start, col_end) in enumerate(
                layout.column_boundaries
            ):
                col_mid = (col_start + col_end) / 2.0
                dist = abs(midpoint - col_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            column_buckets[best_idx].append(blk)

        # Sort each column top-to-bottom.
        for bucket in column_buckets:
            bucket.sort(key=lambda b: b.y0)

        # Concatenate: full-width first, then columns left-to-right.
        result: list[TextBlock] = list(full_width_blocks)
        for bucket in column_buckets:
            result.extend(bucket)

        return result
