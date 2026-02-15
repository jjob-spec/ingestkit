"""Form Matcher: layout fingerprinting and template matching.

Compares incoming documents against registered FormTemplates using
structural fingerprints to select the best-matching template
(spec section 6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PIL import Image, ImageFilter

from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    FormIngestRequest,
    FormTemplate,
    SourceFormat,
    TemplateMatch,
)

if TYPE_CHECKING:
    from ingestkit_forms.config import FormProcessorConfig
    from ingestkit_forms.protocols import FormTemplateStore, LayoutFingerprinter

logger = logging.getLogger("ingestkit_forms")

# Type alias for page rendering callables (file_path, dpi) -> list of page images
PageRenderer = Callable[[str, int], list[Image.Image]]

# Quantization thresholds (fill ratio -> level)
# Level 0 (empty):   fill_ratio < 0.05
# Level 1 (sparse):  0.05 <= fill_ratio < 0.25
# Level 2 (partial): 0.25 <= fill_ratio < 0.60
# Level 3 (dense):   fill_ratio >= 0.60
QUANT_THRESHOLDS: tuple[float, float, float] = (0.05, 0.25, 0.60)

# Supported image extensions for direct Pillow loading
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def _quantize_fill_ratio(fill_ratio: float) -> int:
    """Quantize a fill ratio to 4 levels.

    Returns:
        0 (empty), 1 (sparse), 2 (partial), or 3 (dense).
    """
    if fill_ratio < QUANT_THRESHOLDS[0]:
        return 0
    elif fill_ratio < QUANT_THRESHOLDS[1]:
        return 1
    elif fill_ratio < QUANT_THRESHOLDS[2]:
        return 2
    else:
        return 3


def _compute_otsu_threshold(image: Image.Image) -> int:
    """Compute Otsu's threshold for a grayscale PIL Image.

    Note: The spec calls for 'adaptive thresholding' (section 5.4 line 534).
    Classical adaptive thresholding uses local/block-based thresholds (e.g.,
    cv2.adaptiveThreshold). We use global Otsu thresholding here because
    Pillow lacks a built-in adaptive threshold and we avoid adding OpenCV
    as a dependency. For the structural grid fingerprint use case, global
    Otsu produces equivalent results.

    Returns:
        int: Threshold value (0-255).
    """
    histogram = image.histogram()  # 256 bins
    total_pixels = sum(histogram)
    if total_pixels == 0:
        return 128

    # Standard Otsu: maximize inter-class variance
    sum_total = sum(i * histogram[i] for i in range(256))
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = 128  # default midpoint for uniform images

    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    # If no meaningful threshold found (uniform image), use midpoint
    if max_variance == 0.0:
        return 128

    return best_threshold


def compute_layout_fingerprint(
    pages: list[Image.Image],
    config: FormProcessorConfig,
) -> bytes:
    """Compute a structural layout fingerprint from pre-rendered page images.

    Algorithm (spec section 5.4):
        1. Convert each page to grayscale.
        2. Apply adaptive thresholding to isolate structural elements.
        3. Divide page into grid_cols x grid_rows grid.
        4. For each cell: compute fill ratio (dark pixels / total pixels).
        5. Quantize to 4 levels: empty(0), sparse(1), partial(2), dense(3).
        6. Concatenate per-page fingerprints (1 byte per cell).

    Args:
        pages: List of PIL.Image objects (pre-rendered page images).
        config: FormProcessorConfig with fingerprint_grid_rows, fingerprint_grid_cols.

    Returns:
        bytes: Concatenated fingerprint. Length = len(pages) * grid_rows * grid_cols.

    Raises:
        ValueError: If pages list is empty.
    """
    if not pages:
        raise ValueError("pages list must not be empty")

    grid_rows = config.fingerprint_grid_rows
    grid_cols = config.fingerprint_grid_cols
    fingerprint = bytearray()

    for page in pages:
        # 1. Convert to grayscale
        gray = page.convert("L")

        # 2. Adaptive thresholding (Otsu approximation -- see _compute_otsu_threshold docstring)
        smoothed = gray.filter(ImageFilter.MedianFilter(size=3))
        threshold = _compute_otsu_threshold(smoothed)
        binary = smoothed.point(lambda p, t=threshold: 0 if p <= t else 255)

        # 3. Divide into grid
        width, height = binary.size
        cell_w = width / grid_cols
        cell_h = height / grid_rows

        # 4-5. Compute fill ratio per cell and quantize
        for r in range(grid_rows):
            for c in range(grid_cols):
                left = int(c * cell_w)
                top = int(r * cell_h)
                right = int((c + 1) * cell_w)
                bottom = int((r + 1) * cell_h)

                cell = binary.crop((left, top, right, bottom))
                cell_data = cell.get_flattened_data()
                total = cell.size[0] * cell.size[1]
                dark_count = sum(1 for p in cell_data if p == 0)

                fill_ratio = dark_count / total if total > 0 else 0.0
                level = _quantize_fill_ratio(fill_ratio)
                fingerprint.append(level)

    return bytes(fingerprint)


def compute_layout_similarity(
    fp_a: bytes,
    fp_b: bytes,
    grid_cols: int,
    grid_rows: int,
) -> float:
    """Compare two layout fingerprints and return similarity score.

    Algorithm (spec section 5.4):
        1. Deserialize fingerprints into per-page grids (grid_cols * grid_rows per page).
        2. If page counts differ -> 0.0.
        3. For each cell: exact match = 1.0, off-by-one = 0.5, off-by-two+ = 0.0.
        4. Similarity = sum(cell_scores) / total_cells.

    Note: The spec defines compute_layout_similarity(fp_a, fp_b) -> float with
    only two parameters. We add grid_cols and grid_rows to determine page boundaries
    without embedding dimensions in the fingerprint bytes. This is a pragmatic
    deviation documented in the plan-check.

    Args:
        fp_a: First fingerprint bytes.
        fp_b: Second fingerprint bytes.
        grid_cols: Number of columns in the grid.
        grid_rows: Number of rows in the grid.

    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    cells_per_page = grid_cols * grid_rows
    if cells_per_page == 0:
        return 0.0

    # Validate both fingerprints have length divisible by cells_per_page
    if len(fp_a) % cells_per_page != 0 or len(fp_b) % cells_per_page != 0:
        return 0.0

    pages_a = len(fp_a) // cells_per_page
    pages_b = len(fp_b) // cells_per_page

    if pages_a != pages_b:
        return 0.0

    total_cells = len(fp_a)
    if total_cells == 0:
        return 0.0

    score_sum = 0.0
    for i in range(total_cells):
        diff = abs(fp_a[i] - fp_b[i])
        if diff == 0:
            score_sum += 1.0
        elif diff == 1:
            score_sum += 0.5

    return score_sum / total_cells


def compute_layout_fingerprint_from_file(
    file_path: str,
    config: FormProcessorConfig,
    renderer: PageRenderer | None = None,
) -> bytes:
    """Compute fingerprint from a file path.

    For image files (.jpg, .jpeg, .png, .tiff, .tif): loads directly via Pillow.
    For other formats: delegates to the provided renderer callable.

    Args:
        file_path: Path to the document.
        config: FormProcessorConfig.
        renderer: Optional callable (file_path, dpi) -> list[Image.Image].
            Required for PDF and Excel formats.

    Returns:
        bytes: Layout fingerprint.

    Raises:
        FormIngestError: With E_FORM_FINGERPRINT_FAILED if rendering fails.
    """
    ext = Path(file_path).suffix.lower()

    try:
        if ext in IMAGE_EXTENSIONS:
            img = Image.open(file_path)
            pages = [img]
        elif renderer is not None:
            pages = renderer(file_path, config.fingerprint_dpi)
        else:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_FINGERPRINT_FAILED,
                message=(
                    f"No renderer available for format '{ext}'. "
                    "Provide a renderer callable for non-image formats."
                ),
                stage="fingerprint",
                recoverable=False,
            )
        return compute_layout_fingerprint(pages, config)
    except FormIngestException:
        raise
    except Exception as exc:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_FINGERPRINT_FAILED,
            message=f"Fingerprint computation failed for '{file_path}': {exc}",
            stage="fingerprint",
            recoverable=False,
        ) from exc


# ---------------------------------------------------------------------------
# Source format detection (spec 3.1)
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, SourceFormat] = {
    ".pdf": SourceFormat.PDF,
    ".xlsx": SourceFormat.XLSX,
    ".jpg": SourceFormat.IMAGE,
    ".jpeg": SourceFormat.IMAGE,
    ".png": SourceFormat.IMAGE,
    ".tiff": SourceFormat.IMAGE,
    ".tif": SourceFormat.IMAGE,
}


def detect_source_format(file_path: str) -> SourceFormat:
    """Map file extension to SourceFormat enum.

    Raises:
        FormIngestException with E_FORM_UNSUPPORTED_FORMAT if the
        extension is not recognized.
    """
    suffix = Path(file_path).suffix.lower()
    fmt = _EXTENSION_MAP.get(suffix)
    if fmt is None:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_UNSUPPORTED_FORMAT,
            message=f"Unsupported file extension '{suffix}'",
            stage="matching",
            recoverable=False,
        )
    return fmt


# ---------------------------------------------------------------------------
# Fingerprint deserialization and page-level comparison helpers
# ---------------------------------------------------------------------------


def _deserialize_fingerprint(
    fp: bytes,
    rows: int,
    cols: int,
) -> list[list[list[int]]]:
    """Deserialize concatenated fingerprint bytes into per-page NxM grids.

    Args:
        fp: Concatenated fingerprint bytes.  Length must be a positive
            multiple of ``rows * cols``.
        rows: Grid row count (default 20).
        cols: Grid column count (default 16).

    Returns:
        List of pages, each page is a list of rows, each row is a
        list of int quantization levels (0-3).

    Raises:
        ValueError: If byte length is zero or not a multiple of rows * cols.
    """
    page_size = rows * cols
    if len(fp) == 0 or len(fp) % page_size != 0:
        raise ValueError(
            f"Fingerprint length {len(fp)} is not a multiple of "
            f"page_size {page_size} (rows={rows}, cols={cols})"
        )
    pages: list[list[list[int]]] = []
    num_pages = len(fp) // page_size
    for p in range(num_pages):
        offset = p * page_size
        grid: list[list[int]] = []
        for r in range(rows):
            row_offset = offset + r * cols
            grid.append(list(fp[row_offset : row_offset + cols]))
        pages.append(grid)
    return pages


def _compute_page_similarity(
    page_a: list[list[int]],
    page_b: list[list[int]],
) -> float:
    """Compare two single-page fingerprint grids.

    Scoring per cell (spec 5.4):
        - Exact match: 1.0
        - Off by one quantization level: 0.5
        - Off by two or more: 0.0

    Returns:
        Similarity score 0.0-1.0.
    """
    total_cells = 0
    score_sum = 0.0
    for row_a, row_b in zip(page_a, page_b):
        for cell_a, cell_b in zip(row_a, row_b):
            diff = abs(cell_a - cell_b)
            if diff == 0:
                score_sum += 1.0
            elif diff == 1:
                score_sum += 0.5
            # else: 0.0
            total_cells += 1
    return score_sum / total_cells if total_cells > 0 else 0.0


def _windowed_match(
    doc_pages: list[list[list[int]]],
    tmpl_pages: list[list[list[int]]],
    per_page_minimum: float,
    extra_page_penalty: float,
) -> tuple[float, list[float], int] | None:
    """Sliding window match across document pages.

    Algorithm (spec 6.1 lines 601-611):
        Let T = len(tmpl_pages), D = len(doc_pages).
        - D < T: return None (no match possible).
        - D == T: compare all pages 1:1.
        - D > T: slide window of size T across D pages.

    For each window position i (0 to D-T):
        1. Compare template page j vs document page i+j for all j.
        2. If any per-page similarity < per_page_minimum: skip window.
        3. penalty = (D - T) * extra_page_penalty
        4. confidence = mean(per_page_similarities) - penalty

    Returns:
        (confidence, per_page_scores, best_window_start) for the best
        window, or None if no window passes per-page minimum.
    """
    t_count = len(tmpl_pages)
    d_count = len(doc_pages)

    if d_count < t_count:
        return None

    extra_pages = d_count - t_count
    penalty = extra_pages * extra_page_penalty

    best_confidence: float = -1.0
    best_scores: list[float] = []
    best_start: int = 0

    for i in range(d_count - t_count + 1):
        page_scores: list[float] = []
        window_valid = True

        for j in range(t_count):
            sim = _compute_page_similarity(doc_pages[i + j], tmpl_pages[j])
            if sim < per_page_minimum:
                window_valid = False
                break
            page_scores.append(sim)

        if not window_valid:
            continue

        confidence = sum(page_scores) / len(page_scores) - penalty
        if confidence > best_confidence:
            best_confidence = confidence
            best_scores = page_scores
            best_start = i

    if best_confidence < 0:
        return None

    return (best_confidence, best_scores, best_start)


# ---------------------------------------------------------------------------
# FormMatcher class (spec section 6)
# ---------------------------------------------------------------------------

# Warning floor: matches below 0.5 are excluded entirely.
# The spec does not make this configurable (spec 6.2 lines 641-647).
_WARNING_FLOOR = 0.5


class FormMatcher:
    """Match incoming documents against registered form templates.

    Two code paths:
        1. ``match_document()``: auto-detection via fingerprint comparison
           with windowed multi-page alignment.
        2. ``resolve_manual_override()``: admin-specified template with
           format compatibility validation.

    Args:
        template_store: Backend for template retrieval.
        fingerprinter: Computes per-page layout fingerprints.
        config: Form processor configuration.
    """

    def __init__(
        self,
        template_store: FormTemplateStore,
        fingerprinter: LayoutFingerprinter,
        config: FormProcessorConfig,
    ) -> None:
        self._store = template_store
        self._fingerprinter = fingerprinter
        self._config = config

    def match_document(self, file_path: str) -> list[TemplateMatch]:
        """Match an incoming document against all active templates.

        Algorithm (spec 6.1):
            1. Detect source format from file extension.
            2. Load all active template fingerprints for that format.
            3. Compute incoming document's fingerprint (once).
            4. For each candidate: windowed alignment comparison.
            5. Filter matches with confidence >= 0.5 (warning floor).
            6. Sort by confidence descending.

        Returns:
            List of TemplateMatch sorted by confidence descending.
            Includes all matches with confidence >= 0.5.
            Empty list if no matches found.
        """
        source_format = detect_source_format(file_path)
        candidates = self._store.get_all_fingerprints(
            tenant_id=self._config.tenant_id,
            source_format=source_format.value,
        )
        if not candidates:
            return []

        # Compute document fingerprint once (performance requirement)
        try:
            doc_page_bytes = self._fingerprinter.compute_fingerprint(file_path)
        except FormIngestException:
            raise
        except Exception as exc:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_FINGERPRINT_FAILED,
                message=f"Failed to compute fingerprint for '{file_path}': {exc}",
                stage="matching",
                recoverable=False,
            ) from exc

        doc_pages = self._deserialize_pages(doc_page_bytes)
        rows = self._config.fingerprint_grid_rows
        cols = self._config.fingerprint_grid_cols

        matches: list[TemplateMatch] = []
        for tmpl_id, tmpl_name, tmpl_version, tmpl_fp in candidates:
            try:
                tmpl_pages = _deserialize_fingerprint(tmpl_fp, rows, cols)
            except ValueError:
                logger.warning(
                    "Skipping template %s v%d: invalid fingerprint",
                    tmpl_id,
                    tmpl_version,
                )
                continue

            result = _windowed_match(
                doc_pages=doc_pages,
                tmpl_pages=tmpl_pages,
                per_page_minimum=self._config.form_match_per_page_minimum,
                extra_page_penalty=self._config.form_match_extra_page_penalty,
            )
            if result is None:
                continue

            confidence, per_page_scores, _window_start = result
            if confidence >= _WARNING_FLOOR:
                matches.append(
                    TemplateMatch(
                        template_id=tmpl_id,
                        template_name=tmpl_name,
                        template_version=tmpl_version,
                        confidence=confidence,
                        per_page_confidence=per_page_scores,
                        matched_features=["layout_grid"],
                    )
                )

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def resolve_manual_override(
        self,
        request: FormIngestRequest,
    ) -> FormTemplate:
        """Load and validate a manually specified template.

        Spec 6.3 algorithm:
            1. Load template by ID (+ version if given; latest otherwise).
            2. If not found: raise E_FORM_TEMPLATE_NOT_FOUND.
            3. If format mismatch: raise E_FORM_FORMAT_MISMATCH.
            4. Return template for extraction.

        Raises:
            FormIngestException with E_FORM_TEMPLATE_NOT_FOUND or
            E_FORM_FORMAT_MISMATCH.
        """
        template = self._store.get_template(
            request.template_id,  # type: ignore[arg-type]
            version=request.template_version,
        )
        if template is None:
            version_msg = (
                f" version {request.template_version}"
                if request.template_version is not None
                else ""
            )
            raise FormIngestException(
                code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                message=f"Template '{request.template_id}'{version_msg} not found",
                stage="matching",
                recoverable=False,
                template_id=request.template_id,
                template_version=request.template_version,
            )

        input_format = detect_source_format(request.file_path)
        if template.source_format != input_format:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_FORMAT_MISMATCH,
                message=(
                    f"Template source_format '{template.source_format.value}' "
                    f"incompatible with input format '{input_format.value}'"
                ),
                stage="matching",
                recoverable=False,
                template_id=request.template_id,
                template_version=request.template_version,
            )

        return template

    def _deserialize_pages(
        self,
        page_bytes: list[bytes],
    ) -> list[list[list[int]]]:
        """Convert per-page fingerprint bytes to grid matrices."""
        rows = self._config.fingerprint_grid_rows
        cols = self._config.fingerprint_grid_cols
        pages: list[list[list[int]]] = []
        for page_fp in page_bytes:
            grids = _deserialize_fingerprint(page_fp, rows, cols)
            if len(grids) != 1:
                raise FormIngestException(
                    code=FormErrorCode.E_FORM_FINGERPRINT_FAILED,
                    message=(
                        f"Expected 1 page per fingerprint element, got {len(grids)}"
                    ),
                    stage="matching",
                    recoverable=False,
                )
            pages.append(grids[0])
        return pages
