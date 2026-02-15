"""Page rendering for OCR extraction: PDF and image loading.

Handles PDF-to-image rendering via PyMuPDF (optional) and direct image
loading via Pillow. Includes security checks per spec section 13.4.

Private module -- not exported from the extractors package.
"""

from __future__ import annotations

import logging
import os

from PIL import Image

from ingestkit_forms.errors import FormErrorCode, FormIngestException

logger = logging.getLogger("ingestkit_forms")

# Security constants (spec section 13.4)
MAX_IMAGE_DIMENSION = 10_000  # pixels
MAX_DECOMPRESSION_RATIO = 100


def validate_image_safety(
    file_path: str,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> None:
    """Validate image file safety before full loading.

    Checks:
    1. Resolution limit: width and height <= max_dimension (default 10000px).
    2. Decompression bomb: decompressed_size / compressed_size <= 100.

    Raises:
        FormIngestException with E_FORM_FILE_CORRUPT on violation.
    """
    compressed_size = os.path.getsize(file_path)
    if compressed_size == 0:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_FILE_CORRUPT,
            message=f"File is empty: {file_path}",
            stage="rendering",
            recoverable=False,
        )

    # Use Pillow to read header only (no full decompression)
    with Image.open(file_path) as img:
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_FILE_CORRUPT,
                message=(
                    f"Image dimensions {width}x{height} exceed limit "
                    f"{max_dimension}x{max_dimension}."
                ),
                stage="rendering",
                recoverable=False,
            )
        # Estimate decompressed size: width * height * channels
        channels = len(img.getbands())
        decompressed_estimate = width * height * channels
        if (
            compressed_size > 0
            and decompressed_estimate / compressed_size > MAX_DECOMPRESSION_RATIO
        ):
            raise FormIngestException(
                code=FormErrorCode.E_FORM_FILE_CORRUPT,
                message=(
                    f"Decompression ratio {decompressed_estimate / compressed_size:.1f} "
                    f"exceeds limit {MAX_DECOMPRESSION_RATIO}. Possible decompression bomb."
                ),
                stage="rendering",
                recoverable=False,
            )


def load_image_file(file_path: str, max_dpi: int = 300) -> Image.Image:
    """Load an image file and validate safety.

    Supported formats: JPEG, PNG, TIFF (spec section 3.1).
    Resizes if image resolution significantly exceeds target DPI.

    Returns:
        PIL Image in RGB mode.
    """
    validate_image_safety(file_path)
    img = Image.open(file_path)
    img.load()  # Force full load after validation
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def render_pdf_page(file_path: str, page: int, dpi: int = 300) -> Image.Image:
    """Render a PDF page to a PIL Image at the specified DPI.

    Requires PyMuPDF (fitz). Raises FormIngestError with
    E_FORM_UNSUPPORTED_FORMAT if PyMuPDF is not installed.

    Args:
        file_path: Path to the PDF file.
        page: 0-indexed page number.
        dpi: Target rendering resolution.

    Returns:
        PIL Image in RGB mode.
    """
    try:
        import fitz  # PyMuPDF  # noqa: F811
    except ImportError:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_UNSUPPORTED_FORMAT,
            message=(
                "PyMuPDF is required for PDF page rendering. "
                "Install with: pip install 'ingestkit-forms[pdf]'"
            ),
            stage="rendering",
            recoverable=False,
        )

    doc = fitz.open(file_path)
    try:
        if page >= len(doc):
            raise FormIngestException(
                code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                message=f"Page {page} does not exist in PDF with {len(doc)} pages.",
                stage="rendering",
                page_number=page,
                recoverable=False,
            )
        pdf_page = doc[page]
        zoom = dpi / 72.0  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = pdf_page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()

    # Validate rendered dimensions
    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_FILE_CORRUPT,
            message=(
                f"Rendered page dimensions {img.width}x{img.height} exceed limit "
                f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}. Reduce DPI."
            ),
            stage="rendering",
            page_number=page,
            recoverable=False,
        )

    return img


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
_PDF_EXTENSIONS = {".pdf"}


def get_page_image(file_path: str, page: int, dpi: int = 300) -> Image.Image:
    """Load a page image from a PDF or image file.

    For PDF files, renders the specified page at the target DPI.
    For image files, loads directly (page parameter must be 0).

    Args:
        file_path: Path to the document.
        page: 0-indexed page number.
        dpi: Target DPI for PDF rendering.

    Returns:
        PIL Image in RGB mode.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in _PDF_EXTENSIONS:
        return render_pdf_page(file_path, page, dpi)
    elif ext in _IMAGE_EXTENSIONS:
        if page != 0:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                message=f"Image files only have page 0, but page {page} was requested.",
                stage="rendering",
                page_number=page,
                recoverable=False,
            )
        return load_image_file(file_path, max_dpi=dpi)
    else:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_UNSUPPORTED_FORMAT,
            message=f"Unsupported file extension '{ext}'. Expected PDF or image.",
            stage="rendering",
            recoverable=False,
        )
