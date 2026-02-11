"""Tests for ExcelProcessorConfig defaults, from_file(), and ErrorCode completeness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode


# ---------------------------------------------------------------------------
# Default value tests -- every field must match SPEC.md section 5 exactly
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Assert that ExcelProcessorConfig() with no args produces spec-compliant defaults."""

    def test_parser_version(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.parser_version == "ingestkit_excel:1.0.0"

    def test_tenant_id(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.tenant_id is None

    def test_tier1_high_confidence_signals(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.tier1_high_confidence_signals == 4

    def test_tier1_medium_confidence_signals(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.tier1_medium_confidence_signals == 3

    def test_merged_cell_ratio_threshold(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.merged_cell_ratio_threshold == 0.05

    def test_numeric_ratio_threshold(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.numeric_ratio_threshold == 0.3

    def test_column_consistency_threshold(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.column_consistency_threshold == 0.7

    def test_min_row_count_for_tabular(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.min_row_count_for_tabular == 5

    def test_classification_model(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.classification_model == "qwen2.5:7b"

    def test_reasoning_model(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.reasoning_model == "deepseek-r1:14b"

    def test_tier2_confidence_threshold(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.tier2_confidence_threshold == 0.6

    def test_llm_temperature(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.llm_temperature == 0.1

    def test_row_serialization_limit(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.row_serialization_limit == 5000

    def test_clean_column_names(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.clean_column_names is True

    def test_embedding_model(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.embedding_model == "nomic-embed-text"

    def test_embedding_dimension(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.embedding_dimension == 768

    def test_embedding_batch_size(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.embedding_batch_size == 64

    def test_default_collection(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.default_collection == "helpdesk"

    def test_max_sample_rows(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.max_sample_rows == 3

    def test_enable_tier3(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.enable_tier3 is True

    def test_max_rows_in_memory(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.max_rows_in_memory == 100_000

    def test_backend_timeout_seconds(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.backend_timeout_seconds == 30.0

    def test_backend_max_retries(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.backend_max_retries == 2

    def test_backend_backoff_base(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.backend_backoff_base == 1.0

    def test_log_sample_data(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.log_sample_data is False

    def test_log_llm_prompts(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.log_llm_prompts is False

    def test_log_chunk_previews(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.log_chunk_previews is False

    def test_redact_patterns(self, sample_config: ExcelProcessorConfig) -> None:
        assert sample_config.redact_patterns == []

    def test_total_field_count(self) -> None:
        """Ensure we have at least 27 fields (the spec count)."""
        assert len(ExcelProcessorConfig.model_fields) >= 27


# ---------------------------------------------------------------------------
# Custom value override
# ---------------------------------------------------------------------------


class TestConfigCustom:
    """Verify that custom values override defaults."""

    def test_override_classification_model(self) -> None:
        cfg = ExcelProcessorConfig(classification_model="llama3:8b")
        assert cfg.classification_model == "llama3:8b"

    def test_override_tenant_id(self) -> None:
        cfg = ExcelProcessorConfig(tenant_id="acme")
        assert cfg.tenant_id == "acme"

    def test_override_embedding_dimension(self) -> None:
        cfg = ExcelProcessorConfig(embedding_dimension=1024)
        assert cfg.embedding_dimension == 1024


# ---------------------------------------------------------------------------
# from_file() tests
# ---------------------------------------------------------------------------


class TestConfigFromFileJSON:
    """Test loading config from a JSON file."""

    def test_loads_json(self) -> None:
        data = {"classification_model": "custom:3b", "embedding_dimension": 512}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            fh.flush()
            cfg = ExcelProcessorConfig.from_file(fh.name)
        assert cfg.classification_model == "custom:3b"
        assert cfg.embedding_dimension == 512
        assert cfg.parser_version == "ingestkit_excel:1.0.0"  # default preserved

    def test_empty_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump({}, fh)
            fh.flush()
            cfg = ExcelProcessorConfig.from_file(fh.name)
        assert cfg.parser_version == "ingestkit_excel:1.0.0"


class TestConfigFromFileYAML:
    """Test loading config from a YAML file."""

    def test_loads_yaml(self) -> None:
        data = {"classification_model": "yaml_model:7b", "max_sample_rows": 10}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
            yaml.dump(data, fh)
            fh.flush()
            cfg = ExcelProcessorConfig.from_file(fh.name)
        assert cfg.classification_model == "yaml_model:7b"
        assert cfg.max_sample_rows == 10
        assert cfg.enable_tier3 is True  # default preserved

    def test_loads_yml(self) -> None:
        data = {"tenant_id": "test_yml"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as fh:
            yaml.dump(data, fh)
            fh.flush()
            cfg = ExcelProcessorConfig.from_file(fh.name)
        assert cfg.tenant_id == "test_yml"

    def test_empty_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
            fh.write("")
            fh.flush()
            cfg = ExcelProcessorConfig.from_file(fh.name)
        assert cfg.parser_version == "ingestkit_excel:1.0.0"


class TestConfigFromFileErrors:
    """Test error handling in from_file()."""

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ExcelProcessorConfig.from_file("/nonexistent/path/config.json")

    def test_unsupported_extension(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as fh:
            fh.write("")
            fh.flush()
            with pytest.raises(ValueError, match="Unsupported config file extension"):
                ExcelProcessorConfig.from_file(fh.name)


# ---------------------------------------------------------------------------
# ErrorCode completeness tests
# ---------------------------------------------------------------------------


class TestErrorCodeCompleteness:
    """Verify ErrorCode enum has all codes from spec section 4.1."""

    def test_total_count(self) -> None:
        assert len(ErrorCode) == 26

    @pytest.mark.parametrize(
        "code_value",
        [
            "E_PARSE_CORRUPT",
            "E_PARSE_OPENPYXL_FAIL",
            "E_PARSE_PANDAS_FAIL",
            "E_PARSE_PASSWORD",
            "E_PARSE_EMPTY",
            "E_PARSE_TOO_LARGE",
            "E_CLASSIFY_INCONCLUSIVE",
            "E_LLM_TIMEOUT",
            "E_LLM_MALFORMED_JSON",
            "E_LLM_SCHEMA_INVALID",
            "E_LLM_CONFIDENCE_OOB",
            "E_BACKEND_VECTOR_TIMEOUT",
            "E_BACKEND_VECTOR_CONNECT",
            "E_BACKEND_DB_TIMEOUT",
            "E_BACKEND_DB_CONNECT",
            "E_BACKEND_EMBED_TIMEOUT",
            "E_BACKEND_EMBED_CONNECT",
            "E_PROCESS_REGION_DETECT",
            "E_PROCESS_SERIALIZE",
            "E_PROCESS_SCHEMA_GEN",
            "W_SHEET_SKIPPED_CHART",
            "W_SHEET_SKIPPED_HIDDEN",
            "W_SHEET_SKIPPED_PASSWORD",
            "W_PARSER_FALLBACK",
            "W_LLM_RETRY",
            "W_ROWS_TRUNCATED",
        ],
    )
    def test_code_exists_by_value(self, code_value: str) -> None:
        assert ErrorCode(code_value).value == code_value

    def test_error_codes_are_str_enum(self) -> None:
        for member in ErrorCode:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_error_code_count_by_prefix(self) -> None:
        errors = [c for c in ErrorCode if c.value.startswith("E_")]
        warnings = [c for c in ErrorCode if c.value.startswith("W_")]
        assert len(errors) == 20
        assert len(warnings) == 6
