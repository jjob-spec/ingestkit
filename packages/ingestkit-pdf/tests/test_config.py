"""Tests for ingestkit_pdf.config — PDFProcessorConfig defaults, validators, and file loading."""

from __future__ import annotations

import json
import textwrap

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import OCREngine


# ---------------------------------------------------------------------------
# Default Values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_identity(self):
        c = PDFProcessorConfig()
        assert c.parser_version == "ingestkit_pdf:1.0.0"
        assert c.tenant_id is None

    def test_security_defaults(self):
        c = PDFProcessorConfig()
        assert c.max_file_size_mb == 500
        assert c.max_page_count == 5000
        assert c.per_document_timeout_seconds == 300
        assert c.max_decompression_ratio == 100
        assert c.reject_javascript is True

    def test_security_override_defaults_none(self):
        c = PDFProcessorConfig()
        assert c.reject_javascript_override_reason is None
        assert c.max_file_size_override_reason is None
        assert c.max_page_count_override_reason is None

    def test_tier1_defaults(self):
        c = PDFProcessorConfig()
        assert c.min_chars_per_page == 200
        assert c.max_image_coverage_for_text == 0.3
        assert c.min_table_count_for_complex == 1
        assert c.min_font_count_for_digital == 1
        assert c.tier1_high_confidence_signals == 4
        assert c.tier1_medium_confidence_signals == 3

    def test_tier2_defaults(self):
        c = PDFProcessorConfig()
        assert c.classification_model == "qwen2.5:7b"
        assert c.reasoning_model == "deepseek-r1:14b"
        assert c.tier2_confidence_threshold == 0.6
        assert c.llm_temperature == 0.1
        assert c.enable_tier3 is True

    def test_ocr_defaults(self):
        c = PDFProcessorConfig()
        assert c.ocr_engine == OCREngine.TESSERACT
        assert c.ocr_dpi == 300
        assert c.ocr_language == "en"
        assert c.ocr_confidence_threshold == 0.7
        assert c.ocr_preprocessing_steps == ["deskew"]
        assert c.ocr_max_workers == 4
        assert c.ocr_per_page_timeout_seconds == 60
        assert c.enable_ocr_cleanup is False
        assert c.ocr_cleanup_model == "qwen2.5:7b"

    def test_quality_defaults(self):
        c = PDFProcessorConfig()
        assert c.quality_min_printable_ratio == 0.85
        assert c.quality_min_words_per_page == 10
        assert c.auto_ocr_fallback is True

    def test_header_footer_defaults(self):
        c = PDFProcessorConfig()
        assert c.header_footer_sample_pages == 5
        assert c.header_footer_zone_ratio == 0.10
        assert c.header_footer_similarity_threshold == 0.7

    def test_heading_defaults(self):
        c = PDFProcessorConfig()
        assert c.heading_min_font_size_ratio == 1.2

    def test_table_defaults(self):
        c = PDFProcessorConfig()
        assert c.table_max_rows_for_serialization == 20
        assert c.table_min_rows_for_db == 20
        assert c.table_continuation_column_match_threshold == 0.8

    def test_chunking_defaults(self):
        c = PDFProcessorConfig()
        assert c.chunk_size_tokens == 512
        assert c.chunk_overlap_tokens == 50
        assert c.chunk_respect_headings is True
        assert c.chunk_respect_tables is True

    def test_embedding_defaults(self):
        c = PDFProcessorConfig()
        assert c.embedding_model == "nomic-embed-text"
        assert c.embedding_dimension == 768
        assert c.embedding_batch_size == 64

    def test_vector_store_defaults(self):
        c = PDFProcessorConfig()
        assert c.default_collection == "helpdesk"

    def test_language_defaults(self):
        c = PDFProcessorConfig()
        assert c.enable_language_detection is True
        assert c.default_language == "en"

    def test_dedup_defaults(self):
        c = PDFProcessorConfig()
        assert c.enable_content_dedup is True

    def test_backend_resilience_defaults(self):
        c = PDFProcessorConfig()
        assert c.backend_timeout_seconds == 30.0
        assert c.backend_max_retries == 2
        assert c.backend_backoff_base == 1.0

    def test_logging_defaults(self):
        c = PDFProcessorConfig()
        assert c.log_sample_text is False
        assert c.log_llm_prompts is False
        assert c.log_chunk_previews is False
        assert c.log_ocr_output is False
        assert c.redact_patterns == []


# ---------------------------------------------------------------------------
# Security Override Governance
# ---------------------------------------------------------------------------


class TestSecurityOverrides:
    def test_reject_javascript_false_without_reason_raises(self):
        with pytest.raises(ValueError, match="reject_javascript_override_reason"):
            PDFProcessorConfig(reject_javascript=False)

    def test_reject_javascript_false_with_reason_succeeds(self):
        c = PDFProcessorConfig(
            reject_javascript=False,
            reject_javascript_override_reason="TICKET-4521: Legacy HR forms",
        )
        assert c.reject_javascript is False
        assert c.reject_javascript_override_reason == "TICKET-4521: Legacy HR forms"

    def test_reject_javascript_true_no_reason_needed(self):
        c = PDFProcessorConfig(reject_javascript=True)
        assert c.reject_javascript is True

    def test_max_file_size_override_without_reason_raises(self):
        with pytest.raises(ValueError, match="max_file_size_override_reason"):
            PDFProcessorConfig(max_file_size_mb=1000)

    def test_max_file_size_override_with_reason_succeeds(self):
        c = PDFProcessorConfig(
            max_file_size_mb=1000,
            max_file_size_override_reason="TICKET-4530: Annual compliance bundle",
        )
        assert c.max_file_size_mb == 1000

    def test_max_file_size_at_default_no_reason_needed(self):
        c = PDFProcessorConfig(max_file_size_mb=500)
        assert c.max_file_size_mb == 500

    def test_max_file_size_below_default_no_reason_needed(self):
        c = PDFProcessorConfig(max_file_size_mb=100)
        assert c.max_file_size_mb == 100

    def test_max_page_count_override_without_reason_raises(self):
        with pytest.raises(ValueError, match="max_page_count_override_reason"):
            PDFProcessorConfig(max_page_count=10000)

    def test_max_page_count_override_with_reason_succeeds(self):
        c = PDFProcessorConfig(
            max_page_count=10000,
            max_page_count_override_reason="TICKET-4540: Large document set",
        )
        assert c.max_page_count == 10000

    def test_max_page_count_at_default_no_reason_needed(self):
        c = PDFProcessorConfig(max_page_count=5000)
        assert c.max_page_count == 5000

    def test_multiple_overrides(self):
        c = PDFProcessorConfig(
            reject_javascript=False,
            reject_javascript_override_reason="TICKET-1",
            max_file_size_mb=2000,
            max_file_size_override_reason="TICKET-2",
            max_page_count=20000,
            max_page_count_override_reason="TICKET-3",
        )
        assert c.reject_javascript is False
        assert c.max_file_size_mb == 2000
        assert c.max_page_count == 20000


# ---------------------------------------------------------------------------
# Override via Constructor
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_override_single_field(self):
        c = PDFProcessorConfig(ocr_dpi=600)
        assert c.ocr_dpi == 600
        assert c.ocr_engine == OCREngine.TESSERACT  # others unchanged

    def test_override_ocr_engine(self):
        c = PDFProcessorConfig(ocr_engine=OCREngine.PADDLEOCR)
        assert c.ocr_engine == OCREngine.PADDLEOCR

    def test_override_chunking(self):
        c = PDFProcessorConfig(chunk_size_tokens=1024, chunk_overlap_tokens=100)
        assert c.chunk_size_tokens == 1024
        assert c.chunk_overlap_tokens == 100


# ---------------------------------------------------------------------------
# File Loading
# ---------------------------------------------------------------------------


class TestFromFile:
    def test_load_json(self, tmp_path):
        config_data = {"ocr_dpi": 600, "chunk_size_tokens": 1024}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        c = PDFProcessorConfig.from_file(str(config_file))
        assert c.ocr_dpi == 600
        assert c.chunk_size_tokens == 1024
        assert c.max_file_size_mb == 500  # default preserved

    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            ocr_dpi: 600
            chunk_size_tokens: 1024
            """)
        )
        c = PDFProcessorConfig.from_file(str(config_file))
        assert c.ocr_dpi == 600
        assert c.chunk_size_tokens == 1024

    def test_load_yml_extension(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("ocr_dpi: 400\n")
        c = PDFProcessorConfig.from_file(str(config_file))
        assert c.ocr_dpi == 400

    def test_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        c = PDFProcessorConfig.from_file(str(config_file))
        assert c.max_file_size_mb == 500  # all defaults

    def test_empty_json(self, tmp_path):
        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")
        c = PDFProcessorConfig.from_file(str(config_file))
        assert c.max_file_size_mb == 500

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PDFProcessorConfig.from_file("/nonexistent/config.yaml")

    def test_unsupported_extension(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        with pytest.raises(ValueError, match="Unsupported"):
            PDFProcessorConfig.from_file(str(config_file))

    def test_security_override_via_file(self, tmp_path):
        config_data = {
            "reject_javascript": False,
            "reject_javascript_override_reason": "TICKET-99",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        c = PDFProcessorConfig.from_file(str(config_file))
        assert c.reject_javascript is False

    def test_security_override_missing_reason_via_file(self, tmp_path):
        config_data = {"reject_javascript": False}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        with pytest.raises(ValueError, match="reject_javascript_override_reason"):
            PDFProcessorConfig.from_file(str(config_file))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self):
        c = PDFProcessorConfig(ocr_dpi=600, tenant_id="t1")
        data = c.model_dump()
        c2 = PDFProcessorConfig.model_validate(data)
        assert c2.ocr_dpi == 600
        assert c2.tenant_id == "t1"

    def test_ocr_engine_serializes_as_string(self):
        c = PDFProcessorConfig()
        data = c.model_dump()
        assert data["ocr_engine"] == "tesseract"
