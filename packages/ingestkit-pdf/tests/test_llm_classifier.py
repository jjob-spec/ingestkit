"""Tests for the PDFLLMClassifier Tier 2/3 LLM-based classifier.

Covers schema validation, malformed JSON retry, confidence clamping,
tier model selection, fail-closed behavior, structural summary generation,
prompt correctness, page_types conversion, connection error propagation,
degraded flag, and the LLMClassificationResponse Pydantic model.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.llm_classifier import (
    LLMClassificationResponse,
    PageTypeEntry,
    PDFLLMClassifier,
)
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationTier,
    DocumentProfile,
    PageType,
    PDFType,
)

from tests.conftest import (
    MockLLMBackend,
    _make_document_profile,
    _make_page_profile,
    _valid_response,
)


# ---------------------------------------------------------------------------
# TestValidResponseParsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidResponseParsing:
    """Tests for valid LLM responses producing correct ClassificationResults."""

    def test_valid_text_native_response_returns_correct_result(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert result.confidence == 0.85
        assert result.reasoning == "Digital PDF with extractable text throughout."

    def test_valid_scanned_response(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="scanned",
                    reasoning="Pages are primarily images requiring OCR.",
                )
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.SCANNED

    def test_valid_complex_response_with_page_types(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        pages = [
            _make_page_profile(page_number=1, page_type=PageType.TEXT),
            _make_page_profile(page_number=5, page_type=PageType.SCANNED),
        ]
        profile = _make_document_profile(pages=pages)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="complex",
                    reasoning="Mixed content.",
                    page_types=[
                        {"page": 1, "type": "text"},
                        {"page": 5, "type": "scanned"},
                    ],
                )
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.COMPLEX
        assert result.per_page_types == {
            1: PageType.TEXT,
            5: PageType.SCANNED,
        }

    def test_complex_response_per_page_types_are_pagetype_enums(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        pages = [
            _make_page_profile(page_number=1),
            _make_page_profile(page_number=2),
        ]
        profile = _make_document_profile(pages=pages)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="complex",
                    reasoning="Mixed.",
                    page_types=[
                        {"page": 1, "type": "text"},
                        {"page": 2, "type": "table_heavy"},
                    ],
                )
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        for val in result.per_page_types.values():
            assert isinstance(val, PageType)

    def test_tier_used_reflects_llm_basic(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.tier_used == ClassificationTier.LLM_BASIC

    def test_tier_used_reflects_llm_reasoning(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_REASONING)

        assert result.tier_used == ClassificationTier.LLM_REASONING

    def test_signals_is_none_for_llm_tiers(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.signals is None


# ---------------------------------------------------------------------------
# TestDegradedFlag
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDegradedFlag:
    """Tests for the degraded flag behavior."""

    def test_successful_classification_has_degraded_false(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.degraded is False

    def test_fail_closed_result_has_degraded_false(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.degraded is False


# ---------------------------------------------------------------------------
# TestMalformedJsonRetry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMalformedJsonRetry:
    """Tests for malformed JSON handling and retry behavior."""

    def test_json_decode_error_triggers_retry(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("Expecting value", "", 0),
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert result.confidence == 0.85
        assert len(backend.calls) == 2

    def test_generic_exception_triggers_retry(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                RuntimeError("Something went wrong"),
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert len(backend.calls) == 2

    def test_two_json_failures_returns_fail_closed(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("Expecting value", "", 0),
                json.JSONDecodeError("Expecting value", "", 0),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0
        assert len(backend.calls) == 2

    def test_correction_hint_appended_to_prompt_on_retry(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("Expecting value", "", 0),
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert "IMPORTANT" in backend.calls[1]["prompt"]
        assert "not valid JSON" in backend.calls[1]["prompt"]


# ---------------------------------------------------------------------------
# TestSchemaValidation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchemaValidation:
    """Tests for Pydantic schema validation of LLM responses."""

    def test_missing_type_field_triggers_schema_error(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"confidence": 0.8, "reasoning": "No type field."},
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert len(backend.calls) == 2

    def test_invalid_type_value_triggers_schema_error(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {
                    "type": "TEXT_NATIVE",
                    "confidence": 0.8,
                    "reasoning": "Wrong type string.",
                },
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert len(backend.calls) == 2

    def test_missing_reasoning_triggers_schema_error(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "text_native", "confidence": 0.8},
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert len(backend.calls) == 2

    def test_empty_reasoning_triggers_schema_error(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "text_native", "confidence": 0.8, "reasoning": ""},
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert len(backend.calls) == 2

    def test_schema_error_retry_then_success(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "invalid_type", "confidence": 0.8, "reasoning": "bad"},
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert result.confidence == 0.85

    def test_two_schema_failures_returns_fail_closed(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                {"type": "bad1", "confidence": 0.8, "reasoning": "nope"},
                {"type": "bad2", "confidence": 0.8, "reasoning": "nope"},
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# TestConfidenceBounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfidenceBounds:
    """Tests for confidence out-of-bounds clamping behavior."""

    def test_confidence_above_1_is_clamped(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=1.5)]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 1.0

    def test_confidence_below_0_is_clamped(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=-0.3)]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0

    def test_confidence_exactly_0_is_valid(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=0.0)]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0

    def test_confidence_exactly_1_is_valid(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=1.0)]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 1.0

    def test_confidence_oob_does_not_trigger_retry(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[_valid_response(confidence=1.5)]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert len(backend.calls) == 1


# ---------------------------------------------------------------------------
# TestTimeoutHandling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeoutHandling:
    """Tests for LLM timeout handling."""

    def test_timeout_triggers_retry(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                TimeoutError("Backend timed out"),
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.pdf_type == PDFType.TEXT_NATIVE
        assert len(backend.calls) == 2

    def test_two_timeouts_returns_fail_closed(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                TimeoutError("Backend timed out"),
                TimeoutError("Backend timed out again"),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# TestConnectionErrorPropagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConnectionErrorPropagation:
    """Tests for ConnectionError propagation to caller."""

    def test_connection_error_propagates_to_caller(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[ConnectionError("Connection refused")]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        with pytest.raises(ConnectionError):
            classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

    def test_connection_error_not_retried(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                ConnectionError("Connection refused"),
                _valid_response(),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        with pytest.raises(ConnectionError):
            classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert len(backend.calls) == 1


# ---------------------------------------------------------------------------
# TestFailClosed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFailClosed:
    """Tests for fail-closed behavior after all retries exhausted."""

    def test_fail_closed_result_has_zero_confidence(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.confidence == 0.0

    def test_fail_closed_result_has_correct_tier_used(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.tier_used == ClassificationTier.LLM_BASIC

    def test_fail_closed_reasoning_mentions_failure(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert "fail" in result.reasoning.lower()

    def test_fail_closed_per_page_types_is_empty_dict(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(
            responses=[
                json.JSONDecodeError("bad", "", 0),
                json.JSONDecodeError("bad", "", 0),
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.per_page_types == {}


# ---------------------------------------------------------------------------
# TestTierModelSelection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTierModelSelection:
    """Tests for model selection based on tier."""

    def test_tier2_uses_classification_model(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["model"] == "qwen2.5:7b"

    def test_tier3_uses_reasoning_model(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_REASONING)

        assert backend.calls[0]["model"] == "deepseek-r1:14b"

    def test_rule_based_tier_raises_value_error(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        with pytest.raises(ValueError, match="rule_based"):
            classifier.classify(document_profile, ClassificationTier.RULE_BASED)

    def test_custom_model_names_respected(
        self, document_profile: DocumentProfile
    ) -> None:
        custom_config = PDFProcessorConfig(
            classification_model="custom-small:3b",
            reasoning_model="custom-large:70b",
        )
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=custom_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["model"] == "custom-small:3b"

    def test_temperature_from_config(
        self, document_profile: DocumentProfile
    ) -> None:
        custom_config = PDFProcessorConfig(llm_temperature=0.5)
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=custom_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["temperature"] == 0.5

    def test_timeout_from_config(
        self, document_profile: DocumentProfile
    ) -> None:
        custom_config = PDFProcessorConfig(backend_timeout_seconds=60.0)
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=custom_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert backend.calls[0]["timeout"] == 60.0


# ---------------------------------------------------------------------------
# TestStructuralSummary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStructuralSummary:
    """Tests for structural summary generation."""

    def test_summary_contains_page_count(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "Pages: 1" in summary

    def test_summary_contains_file_size(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "102400" in summary

    def test_summary_contains_creator(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "TestApp" in summary

    def test_summary_contains_pdf_version(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "1.7" in summary

    def test_summary_contains_page_type_distribution(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        pages = [
            _make_page_profile(page_number=1, page_type=PageType.TEXT),
            _make_page_profile(page_number=2, page_type=PageType.TEXT),
            _make_page_profile(page_number=3, page_type=PageType.SCANNED),
        ]
        profile = _make_document_profile(pages=pages)
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(profile)

        assert "text: 2" in summary
        assert "scanned: 1" in summary

    def test_summary_contains_sample_page_profiles(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "Words: 300" in summary
        assert "Text length: 1500" in summary
        assert "Images: 0" in summary
        assert "Tables: 0" in summary

    def test_summary_contains_detected_languages(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "en" in summary

    def test_summary_contains_toc_info_when_present(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        profile = _make_document_profile(
            has_toc=True,
            toc_entries=[(1, "Chapter 1", 1), (2, "Chapter 2", 5)],
        )
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(profile)

        assert "present" in summary
        assert "2 entries" in summary

    def test_summary_contains_form_fields_info(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        from ingestkit_pdf.models import DocumentMetadata

        profile = _make_document_profile(
            metadata=DocumentMetadata(
                creator="TestApp",
                pdf_version="1.7",
                page_count=1,
                file_size_bytes=102400,
                has_form_fields=True,
            ),
        )
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(profile)

        assert "Form fields: present" in summary

    def test_summary_contains_no_raw_text(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        # The summary should only contain structural metadata, never raw text.
        # Since our test profile has no raw text content field, we verify
        # the summary does not contain unexpected content markers.
        assert "raw text" not in summary.lower()

    def test_summary_contains_filename_not_full_path(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "test.pdf" in summary
        assert "/tmp/test.pdf" not in summary

    def test_summary_font_names_included(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=pdf_config)

        summary = classifier._generate_structural_summary(document_profile)

        assert "Arial" in summary
        assert "Times" in summary
        assert "Courier" in summary


# ---------------------------------------------------------------------------
# TestPromptContent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptContent:
    """Tests for classification prompt content."""

    def test_prompt_contains_text_native_enum_value(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert '"text_native"' in prompt

    def test_prompt_contains_scanned_enum_value(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert '"scanned"' in prompt

    def test_prompt_contains_complex_enum_value(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert '"complex"' in prompt

    def test_prompt_contains_structural_summary(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert "test.pdf" in prompt
        assert "Pages:" in prompt

    def test_prompt_requests_json_only(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        prompt = backend.calls[0]["prompt"]
        assert "Respond with JSON only" in prompt


# ---------------------------------------------------------------------------
# TestPageTypesValidation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPageTypesValidation:
    """Tests for page_types list-to-dict conversion and validation."""

    def test_valid_page_types_converted_to_dict(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        pages = [
            _make_page_profile(page_number=1, page_type=PageType.TEXT),
            _make_page_profile(page_number=5, page_type=PageType.SCANNED),
        ]
        profile = _make_document_profile(pages=pages)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="complex",
                    reasoning="Mixed content.",
                    page_types=[
                        {"page": 1, "type": "text"},
                        {"page": 5, "type": "scanned"},
                    ],
                )
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert result.per_page_types == {
            1: PageType.TEXT,
            5: PageType.SCANNED,
        }

    def test_unknown_page_numbers_logged_as_warning(
        self, pdf_config: PDFProcessorConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        pages = [_make_page_profile(page_number=1)]
        profile = _make_document_profile(pages=pages)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="complex",
                    reasoning="Mixed content.",
                    page_types=[
                        {"page": 1, "type": "text"},
                        {"page": 999, "type": "scanned"},
                    ],
                )
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        with caplog.at_level(logging.WARNING, logger="ingestkit_pdf"):
            result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        # Result should still succeed (warning only, not rejection)
        assert result.pdf_type == PDFType.COMPLEX
        assert 999 in result.per_page_types
        assert "999" in caplog.text

    def test_no_page_types_returns_empty_dict(
        self, pdf_config: PDFProcessorConfig, document_profile: DocumentProfile
    ) -> None:
        backend = MockLLMBackend(responses=[_valid_response()])
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(document_profile, ClassificationTier.LLM_BASIC)

        assert result.per_page_types == {}

    def test_all_eight_page_types_accepted(
        self, pdf_config: PDFProcessorConfig
    ) -> None:
        all_types = [
            "text", "scanned", "table_heavy", "form",
            "mixed", "blank", "vector_only", "toc",
        ]
        pages = [
            _make_page_profile(page_number=i + 1)
            for i in range(len(all_types))
        ]
        profile = _make_document_profile(pages=pages)
        backend = MockLLMBackend(
            responses=[
                _valid_response(
                    type_="complex",
                    reasoning="Many page types.",
                    page_types=[
                        {"page": i + 1, "type": t}
                        for i, t in enumerate(all_types)
                    ],
                )
            ]
        )
        classifier = PDFLLMClassifier(llm=backend, config=pdf_config)

        result = classifier.classify(profile, ClassificationTier.LLM_BASIC)

        assert len(result.per_page_types) == 8
        for i, t in enumerate(all_types):
            assert result.per_page_types[i + 1] == PageType(t)


# ---------------------------------------------------------------------------
# TestLLMClassificationResponseModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMClassificationResponseModel:
    """Tests for the LLMClassificationResponse Pydantic model itself."""

    def test_valid_model_creation(self) -> None:
        resp = LLMClassificationResponse(
            type="text_native",
            confidence=0.85,
            reasoning="Valid reasoning.",
        )
        assert resp.type == "text_native"
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
                type="text_native",
                confidence=0.5,
                reasoning="",
            )

    def test_page_types_with_invalid_page_type_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMClassificationResponse(
                type="complex",
                confidence=0.5,
                reasoning="test",
                page_types=[{"page": 1, "type": "invalid_page"}],
            )

    def test_confidence_not_bounded_by_field(self) -> None:
        resp = LLMClassificationResponse(
            type="text_native",
            confidence=5.0,
            reasoning="High confidence test.",
        )
        assert resp.confidence == 5.0


# ---------------------------------------------------------------------------
# TestRedaction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRedaction:
    """Tests for the _redact() helper method."""

    def test_redact_applies_patterns(self) -> None:
        config = PDFProcessorConfig(redact_patterns=[r"\d{3}-\d{4}"])
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=config)

        result = classifier._redact("Call 555-1234 for info")

        assert "[REDACTED]" in result
        assert "555-1234" not in result

    def test_redact_with_no_patterns_is_identity(self) -> None:
        config = PDFProcessorConfig(redact_patterns=[])
        classifier = PDFLLMClassifier(llm=MockLLMBackend(), config=config)

        text = "Some text with 555-1234"
        result = classifier._redact(text)

        assert result == text
