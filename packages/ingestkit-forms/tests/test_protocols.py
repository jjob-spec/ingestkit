"""Unit tests for ingestkit_forms.protocols.

Covers all 4 form-specific protocols (isinstance checks with conforming and
non-conforming mock classes), 3 result models (field access, validation,
bounds), and re-export identity checks for shared protocols.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestkit_forms.models import BoundingBox
from ingestkit_forms.protocols import (
    EmbeddingBackend,
    FormTemplateStore,
    OCRBackend,
    OCRRegionResult,
    PDFWidgetBackend,
    StructuredDBBackend,
    VectorStoreBackend,
    VLMBackend,
    VLMFieldResult,
    WidgetField,
)


# ---------------------------------------------------------------------------
# Minimal Stub Classes for isinstance Checks
# ---------------------------------------------------------------------------


class _StubFormTemplateStore:
    def save_template(self, template):
        ...

    def get_template(self, template_id, version=None):
        ...

    def list_templates(self, tenant_id=None, source_format=None, active_only=True):
        return []

    def list_versions(self, template_id):
        return []

    def delete_template(self, template_id, version=None):
        ...

    def get_all_fingerprints(self, tenant_id=None, source_format=None):
        return []


class _StubOCRBackend:
    def ocr_region(self, image_bytes, language="en", config=None, timeout=None):
        return OCRRegionResult(text="hello", confidence=0.9, engine="stub")

    def engine_name(self):
        return "stub"


class _StubPDFWidgetBackend:
    def extract_widgets(self, file_path, page):
        return []

    def has_form_fields(self, file_path):
        return False

    def engine_name(self):
        return "stub"


class _StubVLMBackend:
    def extract_field(
        self, image_bytes, field_type, field_name, extraction_hint=None, timeout=None
    ):
        return VLMFieldResult(value="test", confidence=0.8, model="stub")

    def model_name(self):
        return "stub"

    def is_available(self):
        return True


class _NonConformingBackend:
    """Deliberately missing required methods."""

    def some_unrelated_method(self):
        ...


# ---------------------------------------------------------------------------
# Protocol isinstance Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormTemplateStoreProtocol:
    def test_conforming_class_passes_isinstance(self):
        store = _StubFormTemplateStore()
        assert isinstance(store, FormTemplateStore)

    def test_non_conforming_class_fails_isinstance(self):
        obj = _NonConformingBackend()
        assert not isinstance(obj, FormTemplateStore)


@pytest.mark.unit
class TestOCRBackendProtocol:
    def test_conforming_class_passes_isinstance(self):
        backend = _StubOCRBackend()
        assert isinstance(backend, OCRBackend)


@pytest.mark.unit
class TestPDFWidgetBackendProtocol:
    def test_conforming_class_passes_isinstance(self):
        backend = _StubPDFWidgetBackend()
        assert isinstance(backend, PDFWidgetBackend)


@pytest.mark.unit
class TestVLMBackendProtocol:
    def test_conforming_class_passes_isinstance(self):
        backend = _StubVLMBackend()
        assert isinstance(backend, VLMBackend)


# ---------------------------------------------------------------------------
# Result Model Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCRRegionResult:
    def test_basic_construction(self):
        result = OCRRegionResult(text="hello", confidence=0.95, engine="tesseract")
        assert result.text == "hello"
        assert result.confidence == 0.95
        assert result.char_confidences is None
        assert result.engine == "tesseract"

    def test_with_char_confidences(self):
        result = OCRRegionResult(
            text="hi",
            confidence=0.9,
            char_confidences=[0.95, 0.85],
            engine="paddleocr",
        )
        assert result.char_confidences == [0.95, 0.85]

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            OCRRegionResult(text="x", confidence=-0.1, engine="t")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            OCRRegionResult(text="x", confidence=1.1, engine="t")


@pytest.mark.unit
class TestWidgetField:
    def test_basic_construction(self):
        bbox = BoundingBox(x=0.1, y=0.2, width=0.5, height=0.3)
        widget = WidgetField(
            field_name="name",
            field_value="John",
            field_type="text",
            bbox=bbox,
            page=0,
        )
        assert widget.field_name == "name"
        assert widget.field_value == "John"
        assert widget.field_type == "text"
        assert widget.bbox.x == 0.1
        assert widget.page == 0

    def test_none_field_value(self):
        bbox = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        widget = WidgetField(
            field_name="sig",
            field_value=None,
            field_type="signature",
            bbox=bbox,
            page=1,
        )
        assert widget.field_value is None


@pytest.mark.unit
class TestVLMFieldResult:
    def test_basic_construction(self):
        result = VLMFieldResult(value="2024-01-15", confidence=0.85, model="qwen2.5-vl")
        assert result.value == "2024-01-15"
        assert result.confidence == 0.85
        assert result.model == "qwen2.5-vl"
        assert result.prompt_tokens is None
        assert result.completion_tokens is None

    def test_bool_value(self):
        result = VLMFieldResult(value=True, confidence=0.99, model="qwen2.5-vl")
        assert result.value is True

    def test_none_value(self):
        result = VLMFieldResult(value=None, confidence=0.1, model="qwen2.5-vl")
        assert result.value is None

    def test_with_token_counts(self):
        result = VLMFieldResult(
            value="test",
            confidence=0.8,
            model="m",
            prompt_tokens=100,
            completion_tokens=20,
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            VLMFieldResult(value="x", confidence=-0.01, model="m")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            VLMFieldResult(value="x", confidence=1.001, model="m")


# ---------------------------------------------------------------------------
# Re-export Identity Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReexportedProtocols:
    def test_vector_store_is_same_object(self):
        from ingestkit_core.protocols import (
            VectorStoreBackend as CoreVSB,
        )

        assert VectorStoreBackend is CoreVSB

    def test_structured_db_is_same_object(self):
        from ingestkit_core.protocols import (
            StructuredDBBackend as CoreSDB,
        )

        assert StructuredDBBackend is CoreSDB

    def test_embedding_is_same_object(self):
        from ingestkit_core.protocols import (
            EmbeddingBackend as CoreEB,
        )

        assert EmbeddingBackend is CoreEB
