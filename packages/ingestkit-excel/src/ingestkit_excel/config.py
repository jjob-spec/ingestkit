"""Configuration model for the ingestkit-excel pipeline.

Provides ``ExcelProcessorConfig`` with all tunable parameters and sensible
defaults drawn from the specification.  Supports loading overrides from YAML
or JSON files via the ``from_file()`` classmethod.
"""

from __future__ import annotations

import json
import pathlib

from pydantic import BaseModel


class ExcelProcessorConfig(BaseModel):
    """All tunable parameters with sensible defaults.

    Every field matches the default value specified in the project
    specification (SPEC.md, section 5).  Override individual values via
    constructor kwargs or load a complete config from a file with
    ``ExcelProcessorConfig.from_file(path)``.
    """

    # --- Identity ---
    parser_version: str = "ingestkit_excel:1.0.0"
    tenant_id: str | None = None

    # --- Tier 1 thresholds ---
    tier1_high_confidence_signals: int = 4
    tier1_medium_confidence_signals: int = 3
    merged_cell_ratio_threshold: float = 0.05
    numeric_ratio_threshold: float = 0.3
    column_consistency_threshold: float = 0.7
    min_row_count_for_tabular: int = 5

    # --- Tier 2/3 LLM settings ---
    classification_model: str = "qwen2.5:7b"
    reasoning_model: str = "deepseek-r1:14b"
    tier2_confidence_threshold: float = 0.6
    llm_temperature: float = 0.1

    # --- Path A settings ---
    row_serialization_limit: int = 5000
    clean_column_names: bool = True

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector store ---
    default_collection: str = "helpdesk"

    # --- General ---
    max_sample_rows: int = 3
    enable_tier3: bool = True
    max_rows_in_memory: int = 100_000

    # --- Backend resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2
    backend_backoff_base: float = 1.0

    # --- Logging / PII safety ---
    log_sample_data: bool = False
    log_llm_prompts: bool = False
    log_chunk_previews: bool = False
    redact_patterns: list[str] = []

    @classmethod
    def from_file(cls, path: str) -> ExcelProcessorConfig:
        """Load configuration from a YAML or JSON file.

        File format is detected by extension: ``.yaml`` / ``.yml`` for YAML,
        ``.json`` for JSON.  Any keys present in the file override the
        corresponding defaults; keys not present retain their defaults.

        Args:
            path: Filesystem path to the configuration file.

        Returns:
            A fully-populated ``ExcelProcessorConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file extension is not recognized.
            ImportError: If a YAML file is provided but ``pyyaml`` is not
                installed.
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
