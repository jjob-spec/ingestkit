"""Page rendering and OCR preprocessing for PDF documents.

Converts PyMuPDF ``fitz.Page`` objects to PIL Images at configurable DPI,
then applies an ordered sequence of OpenCV-based preprocessing steps
(deskew, denoise, binarize, contrast) to prepare images for OCR engines.

Implements SPEC 11.2 steps 1-2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from PIL import Image

from ingestkit_pdf.config import PDFProcessorConfig

if TYPE_CHECKING:
    import fitz  # type: ignore[import-untyped]

logger = logging.getLogger("ingestkit_pdf.utils.page_renderer")

_VALID_STEPS = {"deskew", "denoise", "binarize", "contrast"}

_LARGE_DIMENSION_THRESHOLD = 10000


def _require_cv2():
    """Import and return cv2, raising a clear error if unavailable."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for OCR preprocessing. "
            "Install it with: pip install ingestkit-pdf[opencv]"
        ) from exc
    return cv2


class PageRenderer:
    """Render PDF pages to images and apply OCR preprocessing.

    Parameters
    ----------
    config:
        Pipeline configuration providing ``ocr_dpi`` and
        ``ocr_preprocessing_steps``.
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self._dpi = config.ocr_dpi
        self._steps: list[str] = []

        for step in config.ocr_preprocessing_steps:
            if step in _VALID_STEPS:
                self._steps.append(step)
            else:
                logger.warning(
                    "Unknown preprocessing step '%s' — skipping", step
                )

        self._dispatch: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "deskew": self._deskew,
            "denoise": self._denoise,
            "binarize": self._binarize,
            "contrast": self._contrast,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render_page(self, page: fitz.Page) -> Image.Image:  # type: ignore[name-defined]
        """Render a PDF page to a PIL Image at the configured DPI.

        Parameters
        ----------
        page:
            A ``fitz.Page`` object from PyMuPDF.

        Returns
        -------
        PIL.Image.Image
            RGB image of the rendered page.
        """
        pix = page.get_pixmap(dpi=self._dpi)

        if pix.width > _LARGE_DIMENSION_THRESHOLD or pix.height > _LARGE_DIMENSION_THRESHOLD:
            logger.warning(
                "Large page dimensions: %dx%d at %d DPI",
                pix.width,
                pix.height,
                self._dpi,
            )

        logger.debug(
            "Rendered page to %dx%d image at %d DPI",
            pix.width,
            pix.height,
            self._dpi,
        )

        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return image

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Apply configured preprocessing steps to the image.

        Parameters
        ----------
        image:
            A PIL Image (RGB or L mode).

        Returns
        -------
        PIL.Image.Image
            Preprocessed RGB image.
        """
        if not self._steps:
            return image

        # Ensure RGB for consistent processing
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image)

        for step_name in self._steps:
            handler = self._dispatch.get(step_name)
            if handler is not None:
                logger.debug("Applying preprocessing step: %s", step_name)
                img_array = handler(img_array)

        return Image.fromarray(img_array)

    # ------------------------------------------------------------------
    # Private preprocessing methods
    # ------------------------------------------------------------------

    @staticmethod
    def _deskew(img_array: np.ndarray) -> np.ndarray:
        """Correct skew via Hough line transform."""
        cv2 = _require_cv2()

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10,
        )

        if lines is None:
            logger.debug("No lines detected for deskew — skipping rotation")
            return img_array

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.1:
            logger.debug("Skew angle %.2f° is negligible — skipping rotation", median_angle)
            return img_array

        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img_array, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        logger.debug("Deskewed by %.2f°", median_angle)
        return rotated

    @staticmethod
    def _denoise(img_array: np.ndarray) -> np.ndarray:
        """Reduce noise via fastNlMeansDenoisingColored."""
        cv2 = _require_cv2()

        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
        # Convert back to RGB
        result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

        logger.debug("Applied denoising")
        return result

    @staticmethod
    def _binarize(img_array: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding for binarization."""
        cv2 = _require_cv2()

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to 3-channel for pipeline consistency
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        logger.debug("Applied Otsu binarization")
        return result

    @staticmethod
    def _contrast(img_array: np.ndarray) -> np.ndarray:
        """Enhance contrast via CLAHE on the L channel in LAB space."""
        cv2 = _require_cv2()

        # Convert RGB to BGR then to LAB
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        result_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        logger.debug("Applied CLAHE contrast enhancement")
        return result
