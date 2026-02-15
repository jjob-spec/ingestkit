"""FormProcessorConfig and configuration defaults.

Provides ``FormProcessorConfig`` with all tunable parameters and sensible
defaults drawn from the specification (section 11).  Supports loading
overrides from YAML or JSON files via the ``from_file()`` classmethod.

Also exports ``RedactTarget`` enum for scoping redaction behaviour.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class RedactTarget(str, Enum):
    """Where redaction applies."""

    BOTH = "both"
    CHUNKS_ONLY = "chunks_only"
    DB_ONLY = "db_only"


class FormProcessorConfig(BaseModel):
    """All tunable parameters for the form ingestor plugin.

    Override individual values via constructor kwargs or load a complete
    config from a file with ``FormProcessorConfig.from_file(path)``.
    """

    # --- Identity ---
    parser_version: str = "ingestkit_forms:1.0.0"
    tenant_id: str | None = None

    # --- Form Matching ---
    form_match_enabled: bool = True
    form_match_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum layout similarity for auto-template assignment.",
    )
    form_match_per_page_minimum: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum per-page similarity for multi-page matching.",
    )
    form_match_extra_page_penalty: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Confidence penalty per unmatched extra page in document.",
    )
    page_match_strategy: str = Field(
        default="windowed",
        description="Multi-page matching strategy: 'windowed' (v1 only).",
    )

    # --- Fingerprinting ---
    fingerprint_dpi: int = Field(
        default=150,
        description="DPI for rendering documents during fingerprint computation.",
    )
    fingerprint_grid_rows: int = Field(
        default=20,
        description="Number of rows in the fingerprint grid.",
    )
    fingerprint_grid_cols: int = Field(
        default=16,
        description="Number of columns in the fingerprint grid.",
    )

    # --- OCR Settings ---
    form_ocr_dpi: int = Field(
        default=300,
        description="DPI for rendering pages during OCR extraction.",
    )
    form_ocr_engine: str = Field(
        default="paddleocr",
        description="OCR engine: 'paddleocr' (primary) or 'tesseract' (lightweight fallback).",
    )
    form_ocr_language: str = Field(
        default="en",
        description="OCR language code.",
    )
    form_ocr_per_field_timeout_seconds: int = Field(
        default=10,
        description="Timeout per field OCR operation.",
    )

    # --- Native PDF Backend ---
    pdf_widget_backend: str = Field(
        default="pymupdf",
        description=(
            "PDF widget extraction backend: 'pymupdf' (AGPL, highest accuracy) "
            "or 'pdfplumber' (MIT, licensing-safe alternative). See \u00a77.1.1."
        ),
    )

    # --- VLM Fallback (Optional) ---
    form_vlm_enabled: bool = Field(
        default=False,
        description="Enable VLM fallback for low-confidence OCR fields. Requires VLMBackend.",
    )
    form_vlm_model: str = Field(
        default="qwen2.5-vl:7b",
        description="VLM model identifier for Ollama (or equivalent backend).",
    )
    form_vlm_fallback_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="OCR confidence below this triggers VLM fallback (when enabled).",
    )
    form_vlm_timeout_seconds: int = Field(
        default=15,
        description="Timeout per VLM field extraction call.",
    )
    form_vlm_max_fields_per_document: int = Field(
        default=10,
        ge=1,
        description="Maximum fields per document sent to VLM (cost/latency guard).",
    )

    # --- Field Extraction ---
    form_extraction_min_field_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Below this confidence, field value is set to None.",
    )
    form_extraction_min_overall_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Below this overall confidence, extraction is considered failed. "
            "Result is returned with E_FORM_EXTRACTION_LOW_CONFIDENCE."
        ),
    )
    checkbox_fill_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fill ratio above which a checkbox is considered checked.",
    )
    signature_ink_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Ink ratio above which a signature field is considered signed.",
    )

    # --- Native PDF Field Matching ---
    native_pdf_iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for matching PDF widgets to template fields.",
    )

    # --- Output: Structured DB ---
    form_db_table_prefix: str = Field(
        default="form_",
        description="Prefix for structured DB table names.",
    )

    # --- Output: Chunking ---
    chunk_max_fields: int = Field(
        default=20,
        description="Maximum fields per chunk. Multi-page forms exceeding this are split.",
    )

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector Store ---
    default_collection: str = "helpdesk"

    # --- Template Storage ---
    form_template_storage_path: str = Field(
        default="./form_templates",
        description="Directory or connection string for template persistence.",
    )

    # --- Resource Limits ---
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum file size for form documents.",
    )
    per_document_timeout_seconds: int = Field(
        default=120,
        description="Maximum processing time per form document.",
    )

    # --- Backend Resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2
    backend_backoff_base: float = 1.0

    # --- Dual-Write ---
    dual_write_mode: str = Field(
        default="best_effort",
        description="Dual-write failure semantics: 'best_effort' or 'strict_atomic'.",
    )

    # --- Logging / PII Safety ---
    log_sample_data: bool = Field(
        default=False,
        description="If True, extracted field values may appear in logs. Default is PII-safe.",
    )
    log_ocr_output: bool = False
    log_extraction_details: bool = False
    redact_patterns: list[str] = []
    redact_target: str = Field(
        default="both",
        description="Where redaction applies: 'both', 'chunks_only', 'db_only'.",
    )

    @model_validator(mode="after")
    def _validate_enum_fields(self) -> FormProcessorConfig:
        allowed_dual_write = {"best_effort", "strict_atomic"}
        if self.dual_write_mode not in allowed_dual_write:
            raise ValueError(f"dual_write_mode must be one of {allowed_dual_write}")
        allowed_redact = {"both", "chunks_only", "db_only"}
        if self.redact_target not in allowed_redact:
            raise ValueError(f"redact_target must be one of {allowed_redact}")
        allowed_page_match = {"windowed"}
        if self.page_match_strategy not in allowed_page_match:
            raise ValueError(f"page_match_strategy must be one of {allowed_page_match}")
        allowed_ocr_engines = {"paddleocr", "tesseract"}
        if self.form_ocr_engine not in allowed_ocr_engines:
            raise ValueError(f"form_ocr_engine must be one of {allowed_ocr_engines}")
        allowed_pdf_backends = {"pymupdf", "pdfplumber"}
        if self.pdf_widget_backend not in allowed_pdf_backends:
            raise ValueError(f"pdf_widget_backend must be one of {allowed_pdf_backends}")
        if self.form_vlm_fallback_threshold >= self.form_extraction_min_field_confidence:
            raise ValueError(
                "form_vlm_fallback_threshold must be less than "
                "form_extraction_min_field_confidence"
            )
        return self

    @classmethod
    def from_file(cls, path: str) -> FormProcessorConfig:
        """Load configuration from a YAML or JSON file.

        File format is detected by extension: ``.yaml`` / ``.yml`` for YAML,
        ``.json`` for JSON.  Any keys present in the file override the
        corresponding defaults; keys not present retain their defaults.
        """
        import json as json_mod
        import pathlib

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
                data = json_mod.load(fh)
        else:
            raise ValueError(
                f"Unsupported config file extension '{suffix}'. "
                "Use .yaml, .yml, or .json."
            )

        if data is None:
            data = {}

        return cls(**data)
