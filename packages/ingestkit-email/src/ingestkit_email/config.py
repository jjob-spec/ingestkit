"""Configuration model for the ingestkit-email pipeline.

Provides ``EmailProcessorConfig`` with all tunable parameters and sensible
defaults.  Supports loading overrides from YAML or JSON files via the
``from_file()`` classmethod.
"""

from __future__ import annotations

import json
import pathlib

from pydantic import BaseModel


class EmailProcessorConfig(BaseModel):
    """All tunable parameters with sensible defaults."""

    # --- Identity ---
    parser_version: str = "ingestkit_email:1.0.0"
    tenant_id: str | None = None

    # --- Email Parsing ---
    prefer_plain_text: bool = True
    include_headers: bool = True
    header_format: str = "key_value"
    skip_attachments: bool = True

    # --- Security / Resource Limits ---
    max_file_size_mb: int = 50

    # --- Embedding ---
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64

    # --- Vector Store ---
    default_collection: str = "helpdesk"

    # --- Backend Resilience ---
    backend_timeout_seconds: float = 30.0
    backend_max_retries: int = 2

    # --- Logging / PII Safety ---
    log_sample_data: bool = False
    redact_patterns: list[str] = []

    @classmethod
    def from_file(cls, path: str) -> EmailProcessorConfig:
        """Load configuration from a YAML or JSON file.

        File format is detected by extension: ``.yaml`` / ``.yml`` for YAML,
        ``.json`` for JSON.  Keys present in the file override the
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
