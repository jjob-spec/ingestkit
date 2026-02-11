"""Tests for the LLMClassifier Tier 2/3 LLM-based classifier.

Covers schema validation, malformed JSON retry, confidence clamping,
tier escalation, fail-closed behavior, structural summary generation,
prompt correctness, and the mock LLM backend.
"""

from __future__ import annotations

import json

import pytest

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.llm_classifier import (
    LLMClassificationResponse,
    LLMClassifier,
    _infer_cell_type,
)
from ingestkit_excel.models import (
    ClassificationResult,
    ClassificationTier,
    FileProfile,
    FileType,
    ParserUsed,
    SheetProfile,
)


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------


class MockLLMBackend:
    """Mock LLM backend for testing LLMClassifier.

    Supports configurable responses via a list of return values or
    exceptions. Each call to ``classify()`` pops the next item from
    the response queue, allowing tests to simulate retry sequences.
    """

    def __init__(
        self,
        responses: list[dict | Exception] | None = None,
    ) -> None:
        self._responses: list[dict | Exception] = list(responses or [])
        self.calls: list[dict] = []  # records all calls for assertion

    def classify(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout: float | None = None,
    ) -> dict:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise RuntimeError("MockLLMBackend: no more responses configured")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        raise NotImplementedError("generate() not used by LLMClassifier")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sheet_profile(**overrides: object) -> SheetProfile:
    """Build a SheetProfile with sensible tabular defaults."""
    defaults: dict = dict(
        name="Sheet1",
        row_count=100,
        col_count=5,
        merged_cell_count=0,
        merged_cell_ratio=0.0,
        header_row_detected=True,
        header_values=["A", "B", "C", "D", "E"],
        column_type_consistency=0.9,
        numeric_ratio=0.4,
        text_ratio=0.5,
        empty_ratio=0.1,
        sample_rows=[["1", "hello", "3.14", "", "world"]],
        has_formulas=False,
        is_hidden=False,
        parser_used=ParserUsed.OPENPYXL,
    )
    defaults.update(overrides)
    return SheetProfile(**defaults)


def _make_file_profile(
    sheets: list[SheetProfile] | None = None, **overrides: object
) -> FileProfile:
    """Build a FileProfile from a list of SheetProfiles."""
    if sheets is None:
        sheets = [_make_sheet_profile()]
    defaults: dict = dict(
        file_path="/tmp/test.xlsx",
        file_size_bytes=1024,
        sheet_count=len(sheets),
        sheet_names=[s.name for s in sheets],
        sheets=sheets,
        has_password_protected_sheets=False,
        has_chart_only_sheets=False,
        total_merged_cells=sum(s.merged_cell_count for s in sheets),
        total_rows=sum(s.row_count for s in sheets),
        content_hash="a" * 64,
    )
    defaults.update(overrides)
    return FileProfile(**defaults)


def _valid_response(
    type_: str = "tabular_data",
    confidence: float = 0.85,
    reasoning: str = "Consistent column types and header row detected.",
    sheet_types: dict[str, str] | None = None,
) -> dict:
    """Build a valid LLM response dict."""
    d: dict[str, object] = {
        "type": type_,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    if sheet_types is not None:
        d["sheet_types"] = sheet_types
    return d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> ExcelProcessorConfig:
    return ExcelProcessorConfig()


@pytest.fixture()
def profile() -> FileProfile:
    return _make_file_profile()


# ---------------------------------------------------------------------------
# TestValidResponseParsing
# ---------------------------------------------------------------------------


class TestValidResponseParsing:
    """Tests for valid LLM responses producing correct ClassificationResults."""

    def test_valid_tabular_response_returns_correct_result(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.85
        assert result.reasoning == "Consistent column types and header row detected."

    def test_valid_formatted_document_response(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="formatted_document",
                    reasoning="Merged cells and irregular structure detected.",
                )
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.FORMATTED_DOCUMENT

    def test_valid_hybrid_response_with_sheet_types(
        self, config: ExcelProcessorConfig
    ) -> None:
        sheets = [
            _make_sheet_profile(name="Data"),
            _make_sheet_profile(name="Cover"),
        ]
        fp = _make_file_profile(sheets=sheets)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="hybrid",
                    reasoning="Mixed content.",
                    sheet_types={
                        "Data": "tabular_data",
                        "Cover": "formatted_document",
                    },
                )
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(fp, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.HYBRID
        assert result.per_sheet_types is not None
        assert result.per_sheet_types["Data"] == FileType.TABULAR_DATA
        assert result.per_sheet_types["Cover"] == FileType.FORMATTED_DOCUMENT

    def test_hybrid_response_per_sheet_types_are_filetype_enums(
        self, config: ExcelProcessorConfig
    ) -> None:
        sheets = [
            _make_sheet_profile(name="S1"),
            _make_sheet_profile(name="S2"),
        ]
        fp = _make_file_profile(sheets=sheets)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="hybrid",
                    reasoning="Mixed.",
                    sheet_types={
                        "S1": "tabular_data",
                        "S2": "formatted_document",
                    },
                )
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(fp, ClassificationTier.LLM_BASIC)

        assert result.per_sheet_types is not None
        for val in result.per_sheet_types.values():
            assert isinstance(val, FileType)

    def test_tier_used_reflects_llm_basic(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.tier_used == ClassificationTier.LLM_BASIC

    def test_tier_used_reflects_llm_reasoning(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_REASONING)

        assert result.tier_used == ClassificationTier.LLM_REASONING

    def test_signals_is_none_for_llm_tiers(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.signals is None


# ---------------------------------------------------------------------------
# TestMalformedJsonRetry
# ---------------------------------------------------------------------------


class TestMalformedJsonRetry:
    """Tests for malformed JSON handling and retry behavior."""

    def test_json_decode_error_triggers_retry(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("Expecting value", "", 0),
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.85
        assert len(backend.calls) == 2

    def test_generic_exception_triggers_retry(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                RuntimeError("Connection refused"),
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert len(backend.calls) == 2

    def test_two_json_failures_returns_fail_closed(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("Expecting value", "", 0),
                json.JSONDecodeError("Expecting value", "", 0),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0
        assert len(backend.calls) == 2

    def test_correction_hint_appended_to_prompt_on_retry(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("Expecting value", "", 0),
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        # Second call should have correction hint in prompt
        assert "IMPORTANT" in backend.calls[1]["prompt"]
        assert "not valid JSON" in backend.calls[1]["prompt"]


# ---------------------------------------------------------------------------
# TestSchemaValidation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for Pydantic schema validation of LLM responses."""

    def test_missing_type_field_triggers_schema_error(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        # First response has no 'type', second is valid
        backend = MockLLMBackend(
            responses=[
                {"confidence": 0.8, "reasoning": "No type field."},
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert len(backend.calls) == 2

    def test_invalid_type_value_triggers_schema_error(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        # Uses Python name "TABULAR_DATA" instead of value "tabular_data"
        backend = MockLLMBackend(
            responses=[
                {
                    "type": "TABULAR_DATA",
                    "confidence": 0.8,
                    "reasoning": "Wrong type string.",
                },
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert len(backend.calls) == 2

    def test_missing_reasoning_triggers_schema_error(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "tabular_data", "confidence": 0.8},
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert len(backend.calls) == 2

    def test_empty_reasoning_triggers_schema_error(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "tabular_data", "confidence": 0.8, "reasoning": ""},
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert len(backend.calls) == 2

    def test_schema_error_retry_then_success(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "invalid_type", "confidence": 0.8, "reasoning": "bad"},
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.85

    def test_two_schema_failures_returns_fail_closed(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "bad1", "confidence": 0.8, "reasoning": "nope"},
                {"type": "bad2", "confidence": 0.8, "reasoning": "nope"},
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# TestConfidenceBounds
# ---------------------------------------------------------------------------


class TestConfidenceBounds:
    """Tests for confidence out-of-bounds clamping behavior."""

    def test_confidence_above_1_is_clamped(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=1.5)]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 1.0

    def test_confidence_below_0_is_clamped(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=-0.3)]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0

    def test_confidence_exactly_0_is_valid(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=0.0)]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0

    def test_confidence_exactly_1_is_valid(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=1.0)]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 1.0

    def test_confidence_oob_does_not_trigger_retry(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=1.5)]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        # Only 1 LLM call -- OOB is clamped, not retried
        assert len(backend.calls) == 1


# ---------------------------------------------------------------------------
# TestTimeoutHandling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Tests for LLM timeout handling."""

    def test_timeout_triggers_retry(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                TimeoutError("Backend timed out"),
                _valid_response(),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.file_type == FileType.TABULAR_DATA
        assert len(backend.calls) == 2

    def test_two_timeouts_returns_fail_closed(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                TimeoutError("Backend timed out"),
                TimeoutError("Backend timed out again"),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# TestFailClosed
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Tests for fail-closed behavior after all retries exhausted."""

    def test_fail_closed_result_has_zero_confidence(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0

    def test_fail_closed_result_has_correct_tier_used(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.tier_used == ClassificationTier.LLM_BASIC

    def test_fail_closed_reasoning_mentions_failure(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = LLMClassifier(llm=backend, config=config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert "fail" in result.reasoning.lower() or "failed" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# TestTierModelSelection
# ---------------------------------------------------------------------------


class TestTierModelSelection:
    """Tests for model selection based on tier."""

    def test_tier2_uses_classification_model(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["model"] == "qwen2.5:7b"

    def test_tier3_uses_reasoning_model(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_REASONING)

        assert backend.calls[0]["model"] == "deepseek-r1:14b"

    def test_rule_based_tier_raises_value_error(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        with pytest.raises(ValueError, match="rule_based"):
            classifier.classify(profile, ClassificationTier.RULE_BASED)

    def test_custom_model_names_respected(
        self, profile: FileProfile
    ) -> None:
        custom_config = ExcelProcessorConfig(
            classification_model="custom-small:3b",
            reasoning_model="custom-large:70b",
        )
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=custom_config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["model"] == "custom-small:3b"

    def test_temperature_from_config(
        self, profile: FileProfile
    ) -> None:
        custom_config = ExcelProcessorConfig(llm_temperature=0.5)
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=custom_config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["temperature"] == 0.5

    def test_timeout_from_config(
        self, profile: FileProfile
    ) -> None:
        custom_config = ExcelProcessorConfig(backend_timeout_seconds=60.0)
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=custom_config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["timeout"] == 60.0


# ---------------------------------------------------------------------------
# TestStructuralSummary
# ---------------------------------------------------------------------------


class TestStructuralSummary:
    """Tests for structural summary generation."""

    def test_summary_contains_no_raw_values_by_default(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        # Access private method for testing
        summary = classifier._generate_structural_summary(profile)

        # Should contain type labels
        assert "int" in summary
        assert "str" in summary
        assert "float" in summary
        assert "empty" in summary
        # Should NOT contain actual values from sample_rows
        assert "hello" not in summary
        assert "world" not in summary

    def test_summary_contains_values_when_log_sample_data_true(
        self, profile: FileProfile
    ) -> None:
        custom_config = ExcelProcessorConfig(log_sample_data=True)
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=custom_config
        )

        summary = classifier._generate_structural_summary(profile)

        assert "hello" in summary
        assert "world" in summary

    def test_summary_contains_filename_not_full_path(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(profile)

        assert "test.xlsx" in summary
        assert "/tmp/test.xlsx" not in summary

    def test_summary_contains_sheet_names(
        self, config: ExcelProcessorConfig
    ) -> None:
        sheets = [
            _make_sheet_profile(name="Revenue"),
            _make_sheet_profile(name="Expenses"),
        ]
        fp = _make_file_profile(sheets=sheets)
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(fp)

        assert "Revenue" in summary
        assert "Expenses" in summary

    def test_summary_contains_header_values(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(profile)

        assert "A" in summary
        assert "B" in summary
        assert "C" in summary
        assert "D" in summary
        assert "E" in summary

    def test_summary_contains_merged_cell_info(
        self, config: ExcelProcessorConfig
    ) -> None:
        sheet = _make_sheet_profile(merged_cell_count=10, merged_cell_ratio=0.15)
        fp = _make_file_profile(sheets=[sheet])
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(fp)

        assert "10" in summary
        assert "0.150" in summary

    def test_summary_mentions_hidden_sheets(
        self, config: ExcelProcessorConfig
    ) -> None:
        sheet = _make_sheet_profile(is_hidden=True)
        fp = _make_file_profile(sheets=[sheet])
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(fp)

        assert "Hidden sheet: yes" in summary

    def test_summary_mentions_formulas(
        self, config: ExcelProcessorConfig
    ) -> None:
        sheet = _make_sheet_profile(has_formulas=True)
        fp = _make_file_profile(sheets=[sheet])
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(fp)

        assert "Contains formulas: yes" in summary

    def test_summary_mentions_password_protected(
        self, config: ExcelProcessorConfig
    ) -> None:
        fp = _make_file_profile(has_password_protected_sheets=True)
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=config
        )

        summary = classifier._generate_structural_summary(fp)

        assert "password-protected" in summary

    def test_summary_respects_max_sample_rows(self) -> None:
        sheet = _make_sheet_profile(
            sample_rows=[
                ["1", "a", "2.0", "", "x"],
                ["2", "b", "3.0", "", "y"],
                ["3", "c", "4.0", "", "z"],
            ]
        )
        fp = _make_file_profile(sheets=[sheet])
        custom_config = ExcelProcessorConfig(max_sample_rows=1)
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=custom_config
        )

        summary = classifier._generate_structural_summary(fp)

        # Only Row 1 should appear
        assert "Row 1:" in summary
        assert "Row 2:" not in summary
        assert "Row 3:" not in summary

    def test_summary_redacts_when_log_sample_data_with_patterns(self) -> None:
        sheet = _make_sheet_profile(
            sample_rows=[["John", "555-1234", "test@example.com"]]
        )
        fp = _make_file_profile(sheets=[sheet])
        custom_config = ExcelProcessorConfig(
            log_sample_data=True,
            redact_patterns=[r"\d{3}-\d{4}"],
        )
        classifier = LLMClassifier(
            llm=MockLLMBackend(), config=custom_config
        )

        summary = classifier._generate_structural_summary(fp)

        assert "[REDACTED]" in summary
        assert "555-1234" not in summary
        # Non-matching values should still be present
        assert "John" in summary


# ---------------------------------------------------------------------------
# TestPromptContent
# ---------------------------------------------------------------------------


class TestPromptContent:
    """Tests for classification prompt content."""

    def test_prompt_contains_tabular_data_enum_value(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert '"tabular_data"' in prompt

    def test_prompt_contains_formatted_document_enum_value(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert '"formatted_document"' in prompt

    def test_prompt_contains_hybrid_enum_value(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert '"hybrid"' in prompt

    def test_prompt_contains_structural_summary(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        # Summary should contain the filename and sheet name
        assert "test.xlsx" in prompt
        assert "Sheet1" in prompt

    def test_prompt_requests_json_only(
        self, config: ExcelProcessorConfig, profile: FileProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = LLMClassifier(llm=backend, config=config)

        classifier.classify(profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert "Respond with JSON only" in prompt


# ---------------------------------------------------------------------------
# TestCellTypeInference
# ---------------------------------------------------------------------------


class TestCellTypeInference:
    """Tests for the _infer_cell_type helper."""

    def test_infer_cell_type_empty_string(self) -> None:
        assert _infer_cell_type("") == "empty"

    def test_infer_cell_type_whitespace(self) -> None:
        assert _infer_cell_type("  ") == "empty"

    def test_infer_cell_type_integer(self) -> None:
        assert _infer_cell_type("42") == "int"

    def test_infer_cell_type_float(self) -> None:
        assert _infer_cell_type("3.14") == "float"

    def test_infer_cell_type_string(self) -> None:
        assert _infer_cell_type("hello") == "str"

    def test_infer_cell_type_negative_int(self) -> None:
        assert _infer_cell_type("-5") == "int"

    def test_infer_cell_type_negative_float(self) -> None:
        assert _infer_cell_type("-3.14") == "float"


# ---------------------------------------------------------------------------
# TestLLMClassificationResponseModel
# ---------------------------------------------------------------------------


class TestLLMClassificationResponseModel:
    """Tests for the LLMClassificationResponse Pydantic model itself."""

    def test_valid_model_creation(self) -> None:
        resp = LLMClassificationResponse(
            type="tabular_data",
            confidence=0.85,
            reasoning="Valid reasoning.",
        )
        assert resp.type == "tabular_data"
        assert resp.confidence == 0.85

    def test_invalid_type_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMClassificationResponse(
                type="invalid",
                confidence=0.5,
                reasoning="test",
            )

    def test_empty_reasoning_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMClassificationResponse(
                type="tabular_data",
                confidence=0.5,
                reasoning="",
            )

    def test_sheet_types_with_hybrid_value_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMClassificationResponse(
                type="hybrid",
                confidence=0.5,
                reasoning="test",
                sheet_types={"S1": "hybrid"},
            )

    def test_confidence_not_bounded_by_field(self) -> None:
        # Should NOT raise -- confidence has no ge/le in Field
        resp = LLMClassificationResponse(
            type="tabular_data",
            confidence=5.0,
            reasoning="High confidence test.",
        )
        assert resp.confidence == 5.0
