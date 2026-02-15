"""Image preprocessing pipeline for OCR field extraction.

Implements the preprocessing steps from spec section 7.2:
deskew, CLAHE contrast enhancement, bilateral noise reduction,
adaptive thresholding, and field-type-specific pipelines.

Private module -- not exported from the extractors package.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ingestkit_forms.models import FieldType

logger = logging.getLogger("ingestkit_forms")


def deskew(image: Image.Image, max_angle: float = 15.0) -> Image.Image:
    """Correct rotation up to +/- max_angle degrees via projection profile analysis.

    Uses a binarized copy to compute horizontal projection profiles at candidate
    angles, selecting the angle whose profile has the highest variance (most
    aligned rows of text).

    If the image is smaller than 20x20 px or the best angle is < 0.5 degrees,
    the original image is returned unchanged.
    """
    if image.width < 20 or image.height < 20:
        return image

    # Binarize a grayscale copy for analysis
    gray = image.convert("L")
    binary = gray.point(lambda p: 0 if p < 128 else 255, "1")
    bin_img = binary.convert("L")

    best_angle = 0.0
    best_score = -1.0

    for angle_10x in range(int(-max_angle * 2), int(max_angle * 2) + 1):
        angle = angle_10x / 2.0
        rotated = bin_img.rotate(angle, expand=False, fillcolor=255)
        arr = np.asarray(rotated)
        # Projection profile: count dark pixels per row
        profile = np.sum(arr < 128, axis=1).astype(np.float64)
        score = float(np.var(profile))
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.5:
        return image

    return image.rotate(best_angle, expand=False, fillcolor=(255, 255, 255))


def enhance_contrast(
    image: Image.Image, clip_limit: float = 2.0, tile_size: int = 8
) -> Image.Image:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) via numpy.

    Divides the image into a grid of tiles, computes clipped histograms
    for each tile, and applies bilinear interpolation between tile CDFs.

    Falls back to ``ImageOps.autocontrast`` if numpy operations fail.
    """
    gray = image.convert("L")

    try:
        arr = np.asarray(gray, dtype=np.uint8)
        h, w = arr.shape

        # Minimum dimensions for tiled CLAHE
        if h < tile_size * 2 or w < tile_size * 2:
            return ImageOps.autocontrast(gray, cutoff=1)

        # Compute tile dimensions
        gh = max(1, h // tile_size)
        gw = max(1, w // tile_size)

        # Build CDF for each tile
        cdfs = np.zeros((tile_size, tile_size, 256), dtype=np.float64)
        for ty in range(tile_size):
            for tx in range(tile_size):
                y0 = ty * gh
                y1 = min((ty + 1) * gh, h)
                x0 = tx * gw
                x1 = min((tx + 1) * gw, w)
                tile = arr[y0:y1, x0:x1]
                tile_pixels = tile.size
                if tile_pixels == 0:
                    cdfs[ty, tx] = np.linspace(0, 255, 256)
                    continue

                hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))
                hist = hist.astype(np.float64)

                # Clip histogram
                clip_val = clip_limit * (tile_pixels / 256.0)
                excess = np.sum(np.maximum(hist - clip_val, 0))
                hist = np.minimum(hist, clip_val)
                hist += excess / 256.0

                # Compute CDF
                cdf = np.cumsum(hist)
                cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
                denominator = tile_pixels - cdf_min
                if denominator > 0:
                    cdfs[ty, tx] = ((cdf - cdf_min) / denominator * 255.0).clip(0, 255)
                else:
                    cdfs[ty, tx] = np.linspace(0, 255, 256)

        # Apply CLAHE using bilinear interpolation between tile CDFs
        result = np.zeros_like(arr, dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                # Find surrounding tile centers
                ty_f = (y / gh) - 0.5
                tx_f = (x / gw) - 0.5
                ty0 = max(0, min(int(math.floor(ty_f)), tile_size - 2))
                tx0 = max(0, min(int(math.floor(tx_f)), tile_size - 2))
                ty1 = ty0 + 1
                tx1 = tx0 + 1

                # Interpolation weights
                fy = max(0.0, min(1.0, ty_f - ty0))
                fx = max(0.0, min(1.0, tx_f - tx0))

                pixel_val = arr[y, x]
                # Bilinear interpolation of CDF values
                val = (
                    (1 - fy) * (1 - fx) * cdfs[ty0, tx0, pixel_val]
                    + (1 - fy) * fx * cdfs[ty0, tx1, pixel_val]
                    + fy * (1 - fx) * cdfs[ty1, tx0, pixel_val]
                    + fy * fx * cdfs[ty1, tx1, pixel_val]
                )
                result[y, x] = int(min(255, max(0, val)))

        return Image.fromarray(result, mode="L")

    except Exception:
        logger.warning("CLAHE failed, falling back to autocontrast")
        return ImageOps.autocontrast(gray, cutoff=1)


def reduce_noise(image: Image.Image) -> Image.Image:
    """Bilateral-like noise reduction that preserves edges.

    For small images (typical cropped field regions, < 4000x4000),
    applies a simplified bilateral filter via numpy. For larger images,
    falls back to a median filter.
    """
    gray = image.convert("L")

    if gray.width > 4000 or gray.height > 4000:
        logger.debug("Image too large for bilateral filter, using median filter")
        return gray.filter(ImageFilter.MedianFilter(size=3))

    try:
        arr = np.asarray(gray, dtype=np.float64)
        h, w = arr.shape
        result = np.copy(arr)

        # Bilateral filter parameters
        radius = 2  # 5x5 neighborhood
        sigma_s = 2.0  # spatial sigma
        sigma_r = 25.0  # intensity (range) sigma

        # Pre-compute spatial weights
        spatial_kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                spatial_kernel[dy + radius, dx + radius] = math.exp(
                    -(dy * dy + dx * dx) / (2.0 * sigma_s * sigma_s)
                )

        # Pad array
        padded = np.pad(arr, radius, mode="reflect")

        for y in range(h):
            for x in range(w):
                center_val = arr[y, x]
                neighborhood = padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1]

                # Intensity weights
                intensity_diff = neighborhood - center_val
                range_weights = np.exp(
                    -(intensity_diff * intensity_diff) / (2.0 * sigma_r * sigma_r)
                )

                weights = spatial_kernel * range_weights
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    result[y, x] = np.sum(weights * neighborhood) / weight_sum

        return Image.fromarray(result.astype(np.uint8), mode="L")

    except Exception:
        logger.debug("Bilateral filter failed, using median filter")
        return gray.filter(ImageFilter.MedianFilter(size=3))


def adaptive_threshold(
    image: Image.Image, block_size: int = 11, c_offset: int = 2
) -> Image.Image:
    """Binarize an image using adaptive thresholding.

    For each pixel, computes the mean of a ``block_size x block_size``
    neighborhood. The pixel is white (255) if its value exceeds
    ``local_mean - c_offset``, otherwise black (0).
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float64)
    h, w = arr.shape

    # Compute local means using a box filter (integral image approach)
    pad = block_size // 2
    padded = np.pad(arr, pad, mode="reflect")

    # Use cumulative sums for efficient box filter.
    # Prepend a row/col of zeros so integral[y2, x2] - integral[y1, x2] works
    # without going out of bounds.
    ph, pw = padded.shape
    integral = np.zeros((ph + 1, pw + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    # Compute local means via integral image
    y1 = np.arange(h)
    y2 = y1 + block_size
    x1_arr = np.arange(w)
    x2_arr = x1_arr + block_size

    # Vectorized local mean computation
    local_sums = (
        integral[np.ix_(y2, x2_arr)]
        - integral[np.ix_(y1, x2_arr)]
        - integral[np.ix_(y2, x1_arr)]
        + integral[np.ix_(y1, x1_arr)]
    )
    local_means = local_sums / (block_size * block_size)

    # Apply threshold
    binary = np.where(arr > local_means - c_offset, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def preprocess_for_ocr(image: Image.Image, field_type: FieldType) -> Image.Image:
    """Apply field-type-specific preprocessing pipeline.

    TEXT/NUMBER/DATE: deskew -> CLAHE contrast -> bilateral noise reduction -> binarize
    CHECKBOX/RADIO/SIGNATURE: adaptive_threshold only (binary analysis)

    Note: The full 4-step pipeline for text fields follows the preprocessing
    pipeline diagram in spec section 7.2 (lines 806-827). The step 2c bullet
    points abbreviate this as "deskew, contrast" but the diagram is authoritative.
    """
    if field_type in (FieldType.TEXT, FieldType.NUMBER, FieldType.DATE):
        image = deskew(image)
        image = enhance_contrast(image)
        image = reduce_noise(image)
        image = adaptive_threshold(image)
    elif field_type in (FieldType.CHECKBOX, FieldType.RADIO, FieldType.SIGNATURE):
        image = adaptive_threshold(image)
    # DROPDOWN: no preprocessing (should not reach OCR overlay)
    return image


def compute_fill_ratio(image: Image.Image) -> float:
    """Compute ratio of dark pixels to total pixels.

    Used for CHECKBOX and RADIO field type detection.
    Returns float in [0.0, 1.0].
    """
    gray = image.convert("L")
    arr = np.asarray(gray)
    # Threshold at 128: below = dark
    dark_pixels = int(np.sum(arr < 128))
    total_pixels = arr.size
    if total_pixels == 0:
        return 0.0
    return float(dark_pixels / total_pixels)


def compute_ink_ratio(image: Image.Image) -> float:
    """Compute ratio of ink (dark) pixels to total pixels.

    Used for SIGNATURE field detection. Currently identical to
    compute_fill_ratio; separated for semantic clarity and future
    connected-component enhancements.
    """
    return compute_fill_ratio(image)
