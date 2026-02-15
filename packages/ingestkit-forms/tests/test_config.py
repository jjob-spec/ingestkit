"""Tests for FormProcessorConfig."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ingestkit_forms.config import FormProcessorConfig, RedactTarget


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


class TestDefaultConstruction:
    """Default construction should succeed and produce correct values."""

    def test_default_construction_succeeds(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg is not None

    def test_field_count(self) -> None:
        assert len(FormProcessorConfig.model_fields) == 43

    def test_default_identity_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.parser_version == "ingestkit_forms:1.0.0"
        assert cfg.tenant_id is None

    def test_default_form_matching_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.form_match_enabled is True
        assert cfg.form_match_confidence_threshold == 0.8
        assert cfg.form_match_per_page_minimum == 0.6
        assert cfg.form_match_extra_page_penalty == 0.02
        assert cfg.page_match_strategy == "windowed"

    def test_default_fingerprinting_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.fingerprint_dpi == 150
        assert cfg.fingerprint_grid_rows == 20
        assert cfg.fingerprint_grid_cols == 16

    def test_default_ocr_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.form_ocr_dpi == 300
        assert cfg.form_ocr_engine == "paddleocr"
        assert cfg.form_ocr_language == "en"
        assert cfg.form_ocr_per_field_timeout_seconds == 10

    def test_default_vlm_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.form_vlm_enabled is False
        assert cfg.form_vlm_model == "qwen2.5-vl:7b"
        assert cfg.form_vlm_fallback_threshold == 0.4
        assert cfg.form_vlm_timeout_seconds == 15
        assert cfg.form_vlm_max_fields_per_document == 10

    def test_default_extraction_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.form_extraction_min_field_confidence == 0.5
        assert cfg.form_extraction_min_overall_confidence == 0.3
        assert cfg.checkbox_fill_threshold == 0.3
        assert cfg.signature_ink_threshold == 0.05
        assert cfg.native_pdf_iou_threshold == 0.5

    def test_default_embedding_and_storage_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.embedding_model == "nomic-embed-text"
        assert cfg.embedding_dimension == 768
        assert cfg.embedding_batch_size == 64
        assert cfg.default_collection == "helpdesk"
        assert cfg.form_template_storage_path == "./form_templates"
        assert cfg.form_db_table_prefix == "form_"
        assert cfg.chunk_max_fields == 20

    def test_default_resilience_and_limits(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.max_file_size_mb == 100
        assert cfg.per_document_timeout_seconds == 120
        assert cfg.backend_timeout_seconds == 30.0
        assert cfg.backend_max_retries == 2
        assert cfg.backend_backoff_base == 1.0
        assert cfg.dual_write_mode == "best_effort"

    def test_default_logging_pii_fields(self) -> None:
        cfg = FormProcessorConfig()
        assert cfg.log_sample_data is False
        assert cfg.log_ocr_output is False
        assert cfg.log_extraction_details is False
        assert cfg.redact_patterns == []
        assert cfg.redact_target == "both"


# ---------------------------------------------------------------------------
# Enum-like string field validators
# ---------------------------------------------------------------------------


class TestEnumFieldValidators:
    """model_validator should reject invalid enum-like string values."""

    def test_invalid_dual_write_mode(self) -> None:
        with pytest.raises(ValueError, match="dual_write_mode"):
            FormProcessorConfig(dual_write_mode="invalid")

    def test_valid_dual_write_mode_strict_atomic(self) -> None:
        cfg = FormProcessorConfig(dual_write_mode="strict_atomic")
        assert cfg.dual_write_mode == "strict_atomic"

    def test_invalid_redact_target(self) -> None:
        with pytest.raises(ValueError, match="redact_target"):
            FormProcessorConfig(redact_target="invalid")

    def test_valid_redact_targets(self) -> None:
        for val in ("both", "chunks_only", "db_only"):
            cfg = FormProcessorConfig(redact_target=val)
            assert cfg.redact_target == val

    def test_invalid_page_match_strategy(self) -> None:
        with pytest.raises(ValueError, match="page_match_strategy"):
            FormProcessorConfig(page_match_strategy="sliding")

    def test_invalid_form_ocr_engine(self) -> None:
        with pytest.raises(ValueError, match="form_ocr_engine"):
            FormProcessorConfig(form_ocr_engine="easyocr")

    def test_valid_tesseract_engine(self) -> None:
        cfg = FormProcessorConfig(form_ocr_engine="tesseract")
        assert cfg.form_ocr_engine == "tesseract"

    def test_invalid_pdf_widget_backend(self) -> None:
        with pytest.raises(ValueError, match="pdf_widget_backend"):
            FormProcessorConfig(pdf_widget_backend="camelot")

    def test_valid_pdfplumber_backend(self) -> None:
        cfg = FormProcessorConfig(pdf_widget_backend="pdfplumber")
        assert cfg.pdf_widget_backend == "pdfplumber"


# ---------------------------------------------------------------------------
# Cross-field threshold validation
# ---------------------------------------------------------------------------


class TestCrossFieldThreshold:
    """VLM fallback threshold must be less than field confidence threshold."""

    def test_vlm_threshold_equal_to_field_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="form_vlm_fallback_threshold"):
            FormProcessorConfig(
                form_vlm_fallback_threshold=0.5,
                form_extraction_min_field_confidence=0.5,
            )

    def test_vlm_threshold_greater_than_field_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="form_vlm_fallback_threshold"):
            FormProcessorConfig(
                form_vlm_fallback_threshold=0.6,
                form_extraction_min_field_confidence=0.5,
            )

    def test_valid_threshold_ordering(self) -> None:
        cfg = FormProcessorConfig(
            form_vlm_fallback_threshold=0.3,
            form_extraction_min_field_confidence=0.5,
        )
        assert cfg.form_vlm_fallback_threshold < cfg.form_extraction_min_field_confidence


# ---------------------------------------------------------------------------
# Field constraints (ge/le)
# ---------------------------------------------------------------------------


class TestFieldConstraints:
    """Field() ge/le constraints should reject out-of-range values."""

    def test_confidence_threshold_too_high(self) -> None:
        with pytest.raises(ValueError):
            FormProcessorConfig(form_match_confidence_threshold=1.5)

    def test_confidence_threshold_too_low(self) -> None:
        with pytest.raises(ValueError):
            FormProcessorConfig(form_match_confidence_threshold=-0.1)

    def test_extra_page_penalty_too_high(self) -> None:
        with pytest.raises(ValueError):
            FormProcessorConfig(form_match_extra_page_penalty=0.6)

    def test_vlm_max_fields_below_minimum(self) -> None:
        with pytest.raises(ValueError):
            FormProcessorConfig(form_vlm_max_fields_per_document=0)


# ---------------------------------------------------------------------------
# RedactTarget enum
# ---------------------------------------------------------------------------


class TestRedactTarget:
    """RedactTarget enum should have correct members and values."""

    def test_member_count(self) -> None:
        assert len(RedactTarget) == 3

    def test_member_values(self) -> None:
        assert RedactTarget.BOTH.value == "both"
        assert RedactTarget.CHUNKS_ONLY.value == "chunks_only"
        assert RedactTarget.DB_ONLY.value == "db_only"

    def test_string_comparison(self) -> None:
        assert RedactTarget.BOTH == "both"
        assert RedactTarget.CHUNKS_ONLY == "chunks_only"
        assert RedactTarget.DB_ONLY == "db_only"


# ---------------------------------------------------------------------------
# from_file()
# ---------------------------------------------------------------------------


class TestFromFile:
    """from_file() should load JSON and YAML configs."""

    def test_load_json(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"tenant_id": "acme", "fingerprint_dpi": 200}))
        cfg = FormProcessorConfig.from_file(str(cfg_path))
        assert cfg.tenant_id == "acme"
        assert cfg.fingerprint_dpi == 200
        # Non-overridden defaults remain
        assert cfg.form_match_enabled is True

    def test_load_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            textwrap.dedent("""\
                tenant_id: acme
                form_ocr_engine: tesseract
            """)
        )
        cfg = FormProcessorConfig.from_file(str(cfg_path))
        assert cfg.tenant_id == "acme"
        assert cfg.form_ocr_engine == "tesseract"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            FormProcessorConfig.from_file("/nonexistent/path/config.json")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text("")
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            FormProcessorConfig.from_file(str(cfg_path))

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "empty.yml"
        cfg_path.write_text("")
        cfg = FormProcessorConfig.from_file(str(cfg_path))
        assert cfg.parser_version == "ingestkit_forms:1.0.0"


# ---------------------------------------------------------------------------
# Serialization round-trip and overrides
# ---------------------------------------------------------------------------


class TestSerializationAndOverrides:
    """Pydantic serialization and constructor overrides should work."""

    def test_model_dump_and_validate_round_trip(self) -> None:
        original = FormProcessorConfig(tenant_id="test-tenant")
        data = original.model_dump()
        restored = FormProcessorConfig.model_validate(data)
        assert restored.tenant_id == "test-tenant"
        assert restored == original

    def test_override_individual_fields(self) -> None:
        cfg = FormProcessorConfig(
            tenant_id="corp-a",
            embedding_dimension=384,
            dual_write_mode="strict_atomic",
        )
        assert cfg.tenant_id == "corp-a"
        assert cfg.embedding_dimension == 384
        assert cfg.dual_write_mode == "strict_atomic"
        # Other defaults unchanged
        assert cfg.parser_version == "ingestkit_forms:1.0.0"
