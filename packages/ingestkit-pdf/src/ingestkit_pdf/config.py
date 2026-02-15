"""Configuration model for the ingestkit-pdf pipeline.

Provides ``PDFProcessorConfig`` with all tunable parameters and sensible
defaults drawn from the specification.  Supports loading overrides from YAML
or JSON files via the ``from_file()`` classmethod.

Security override governance (§7.5): disabling any security default
requires an explicit reason string in the corresponding
``*_override_reason`` field, enforced via Pydantic model validators.
"""

from __future__ import annotations

import json
import pathlib

from pydantic import BaseModel, model_validator

from ingestkit_pdf.models import OCREngine


class PDFProcessorConfig(BaseModel):
    """All tunable parameters with sensible defaults.

    Every field matches the default value specified in SPEC.md §6.
    Override individual values via constructor kwargs or load a complete
    config from a file with ``PDFProcessorConfig.from_file(path)``.
    """

    # --- Identity ---
    parser_version: str = "ingestkit_pdf:1.0.0"
    tenant_id: str | None = None

    # --- Security / Resource Limits ---
    max_file_size_mb: int = 500
    max_page_count: int = 5000
    per_document_timeout_seconds: int = 300
    max_decompression_ratio: int = 100
    reject_javascript: bool = True

    # --- Security Override Governance ---
    reject_javascript_override_reason: str | None = None
    max_file_size_override_reason: str | None = None
    max_page_count_override_reason: str | None = None

    # --- Tier 1 Thresholds ---
    min_chars_per_page: int = 200
    max_image_coverage_for_text: float = 0.3
    min_table_count_for_complex: int = 1
    min_font_count_for_digital: int = 1
    tier1_high_confidence_signals: int = 4
    tier1_medium_confidence_signals: int = 3

    # --- Tier 2/3 LLM Settings ---
    classification_model: str = "qwen2.5:7b"
    reasoning_model: str = "deepseek-r1:14b"
    tier2_confidence_threshold: float = 0.6
    llm_temperature: float = 0.1
    enable_tier3: bool = True

    # --- OCR Settings ---
    ocr_engine: OCREngine = OCREngine.TESSERACT
    ocr_dpi: int = 300
    ocr_language: str = "en"
    ocr_confidence_threshold: float = 0.7
    ocr_preprocessing_steps: list[str] = ["deskew"]
    ocr_max_workers: int = 4
    ocr_per_page_timeout_seconds: int = 60
    enable_ocr_cleanup: bool = False
    ocr_cleanup_model: str = "qwen2.5:7b"

    # --- Extraction Quality ---
    quality_min_printable_ratio: float = 0.85
    quality_min_words_per_page: int = 10
    auto_ocr_fallback: bool = True

    # --- Header/Footer Detection ---
    header_footer_sample_pages: int = 5
    header_footer_zone_ratio: float = 0.10
    header_footer_similarity_threshold: float = 0.7

    # --- Heading Detection ---
    heading_min_font_size_ratio: float = 1.2

    # --- Table Extraction ---
    table_max_rows_for_serialization: int = 20
    table_min_rows_for_db: int = 20
    table_continuation_column_match_threshold: float = 0.8

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    chunk_respect_headings: bool = True
    chunk_respect_tables: bool = True

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector Store ---
    default_collection: str = "helpdesk"

    # --- Language Detection ---
    enable_language_detection: bool = True
    default_language: str = "en"

    # --- Deduplication ---
    enable_content_dedup: bool = True

    # --- Backend Resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2
    backend_backoff_base: float = 1.0

    # --- Execution Backend ---
    execution_backend: str = "local"                # "local" or "distributed"
    execution_max_workers: int = 4                  # max ProcessPoolExecutor workers
    execution_queue_url: str | None = None          # Redis/RabbitMQ URL for distributed

    # --- Logging / PII Safety ---
    log_sample_text: bool = False
    log_llm_prompts: bool = False
    log_chunk_previews: bool = False
    log_ocr_output: bool = False
    redact_patterns: list[str] = []

    @model_validator(mode="after")
    def _validate_security_overrides(self) -> PDFProcessorConfig:
        """Enforce security override governance (§7.5).

        Disabling a security default without providing an explicit reason
        string raises a ValidationError.
        """
        if not self.reject_javascript and self.reject_javascript_override_reason is None:
            raise ValueError(
                "reject_javascript=False requires reject_javascript_override_reason "
                "to be set (see SPEC §7.5 Security Override Governance)"
            )
        if self.max_file_size_mb > 500 and self.max_file_size_override_reason is None:
            raise ValueError(
                "max_file_size_mb > 500 requires max_file_size_override_reason "
                "to be set (see SPEC §7.5 Security Override Governance)"
            )
        if self.max_page_count > 5000 and self.max_page_count_override_reason is None:
            raise ValueError(
                "max_page_count > 5000 requires max_page_count_override_reason "
                "to be set (see SPEC §7.5 Security Override Governance)"
            )
        return self

    @classmethod
    def from_file(cls, path: str) -> PDFProcessorConfig:
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
