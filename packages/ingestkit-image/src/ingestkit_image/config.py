"""Configuration model for the ingestkit-image pipeline.

Provides ``ImageProcessorConfig`` with all tunable parameters and sensible
defaults.  Supports loading overrides from YAML or JSON files via the
``from_file()`` classmethod.
"""

from __future__ import annotations

import json
import pathlib

from pydantic import BaseModel


class ImageProcessorConfig(BaseModel):
    """All tunable parameters with sensible defaults.

    Override individual values via constructor kwargs or load a complete
    config from a file with ``ImageProcessorConfig.from_file(path)``.
    """

    # --- Identity ---
    parser_version: str = "ingestkit_image:1.0.0"
    tenant_id: str | None = None

    # --- Security / Resource Limits ---
    max_file_size_mb: int = 50
    max_image_width: int = 10000
    max_image_height: int = 10000
    supported_formats: list[str] = ["jpeg", "png", "tiff", "webp", "bmp", "gif"]

    # --- VLM Settings ---
    vision_model: str = "llama3.2-vision:11b"
    caption_prompt: str = (
        "Describe this image in detail for search indexing. "
        "Include key objects, text, colors, layout, and context."
    )
    vlm_temperature: float = 0.3
    vlm_timeout_seconds: int = 30
    vlm_max_retries: int = 1
    min_caption_length: int = 10

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # --- Vector Store ---
    default_collection: str = "helpdesk"

    # --- Backend Resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2
    backend_backoff_base: float = 1.0

    # --- OCR Settings ---
    enable_ocr: bool = False
    ocr_language: str = "eng"  # Tesseract language code
    ocr_config: str | None = None  # e.g., "--psm 1 --oem 3"
    ocr_timeout_seconds: float = 60.0  # OCR can be slow on large images
    ocr_max_dimension: int = 4096  # Resize if any side exceeds this
    ocr_megapixel_threshold: float = 20.0  # Resize trigger (megapixels)
    ocr_min_text_length: int = 3  # Below this, warn low confidence
    ocr_max_retries: int = 1

    # --- Logging / PII Safety ---
    log_sample_data: bool = False
    log_captions: bool = False

    @classmethod
    def from_file(cls, path: str) -> ImageProcessorConfig:
        """Load configuration from a YAML or JSON file.

        File format is detected by extension: ``.yaml`` / ``.yml`` for YAML,
        ``.json`` for JSON.  Any keys present in the file override the
        corresponding defaults; keys not present retain their defaults.
        """
        file_path = pathlib.Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = file_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "pyyaml is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                ) from exc
            with open(file_path) as fh:
                data = yaml.safe_load(fh)
        elif suffix == ".json":
            with open(file_path) as fh:
                data = json.load(fh)
        else:
            raise ValueError(
                f"Unsupported config file extension '{suffix}'. "
                "Use .yaml, .yml, or .json."
            )

        if data is None:
            data = {}

        return cls(**data)
