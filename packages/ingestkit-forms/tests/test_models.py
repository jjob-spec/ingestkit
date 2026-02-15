"""Unit tests for ingestkit_forms.models.

Covers all 19 new models/enums (3 enums + 16 models): enum values,
field constraints, validators, serialization, inheritance, and round-trips.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from ingestkit_core.models import BaseChunkMetadata, WrittenArtifacts
from ingestkit_forms.models import (
    BoundingBox,
    CellAddress,
    DualWriteMode,
    ExtractionPreview,
    ExtractedField,
    FieldMapping,
    FieldType,
    FormChunkMetadata,
    FormChunkPayload,
    FormExtractionResult,
    FormIngestRequest,
    FormProcessingResult,
    FormTemplate,
    FormTemplateCreateRequest,
    FormTemplateUpdateRequest,
    FormWrittenArtifacts,
    RollbackResult,
    SourceFormat,
    TemplateMatch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_region(**overrides: object) -> BoundingBox:
    defaults = {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.3}
    defaults.update(overrides)
    return BoundingBox(**defaults)


def _make_cell_address(**overrides: object) -> CellAddress:
    defaults = {"cell": "B2"}
    defaults.update(overrides)
    return CellAddress(**defaults)


def _make_field_mapping(*, use_region: bool = True, **overrides: object) -> FieldMapping:
    defaults: dict = {
        "field_name": "employee_name",
        "field_label": "Employee Name",
        "field_type": FieldType.TEXT,
        "page_number": 0,
    }
    if use_region:
        defaults["region"] = _make_region()
    else:
        defaults["cell_address"] = _make_cell_address()
    defaults.update(overrides)
    return FieldMapping(**defaults)


def _make_extracted_field(**overrides: object) -> ExtractedField:
    defaults: dict = {
        "field_id": str(uuid.uuid4()),
        "field_name": "employee_name",
        "field_label": "Employee Name",
        "field_type": FieldType.TEXT,
        "value": "John Smith",
        "confidence": 0.95,
        "extraction_method": "native_fields",
    }
    defaults.update(overrides)
    return ExtractedField(**defaults)


def _make_form_extraction_result(**overrides: object) -> FormExtractionResult:
    defaults: dict = {
        "template_id": str(uuid.uuid4()),
        "template_name": "Leave Request",
        "template_version": 1,
        "source_uri": "/tmp/form.pdf",
        "source_format": "pdf",
        "fields": [_make_extracted_field()],
        "overall_confidence": 0.92,
        "extraction_method": "native_fields",
        "match_method": "auto_detect",
        "pages_processed": 1,
        "extraction_duration_seconds": 0.5,
    }
    defaults.update(overrides)
    return FormExtractionResult(**defaults)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSourceFormat:
    def test_values_are_lowercase(self) -> None:
        assert SourceFormat.PDF.value == "pdf"
        assert SourceFormat.XLSX.value == "xlsx"
        assert SourceFormat.IMAGE.value == "image"

    def test_member_count(self) -> None:
        assert len(SourceFormat) == 3


@pytest.mark.unit
class TestFieldType:
    def test_values_are_lowercase(self) -> None:
        assert FieldType.TEXT.value == "text"
        assert FieldType.NUMBER.value == "number"
        assert FieldType.DATE.value == "date"
        assert FieldType.CHECKBOX.value == "checkbox"
        assert FieldType.RADIO.value == "radio"
        assert FieldType.SIGNATURE.value == "signature"
        assert FieldType.DROPDOWN.value == "dropdown"

    def test_member_count(self) -> None:
        assert len(FieldType) == 7


@pytest.mark.unit
class TestDualWriteMode:
    def test_values(self) -> None:
        assert DualWriteMode.BEST_EFFORT.value == "best_effort"
        assert DualWriteMode.STRICT_ATOMIC.value == "strict_atomic"

    def test_member_count(self) -> None:
        assert len(DualWriteMode) == 2


# ---------------------------------------------------------------------------
# BoundingBox Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoundingBox:
    def test_valid(self) -> None:
        bb = BoundingBox(x=0.1, y=0.2, width=0.5, height=0.3)
        assert bb.x == 0.1
        assert bb.width == 0.5

    def test_x_out_of_range(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            BoundingBox(x=1.5, y=0.2, width=0.5, height=0.3)

    def test_x_negative(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            BoundingBox(x=-0.1, y=0.2, width=0.5, height=0.3)

    def test_width_zero(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            BoundingBox(x=0.1, y=0.2, width=0.0, height=0.3)


# ---------------------------------------------------------------------------
# FieldMapping Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFieldMapping:
    def test_with_region(self) -> None:
        fm = _make_field_mapping(use_region=True)
        assert fm.region is not None
        assert fm.cell_address is None

    def test_with_cell_address(self) -> None:
        fm = _make_field_mapping(use_region=False)
        assert fm.cell_address is not None
        assert fm.region is None

    def test_both_addresses_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot have both"):
            FieldMapping(
                field_name="test",
                field_label="Test",
                field_type=FieldType.TEXT,
                page_number=0,
                region=_make_region(),
                cell_address=_make_cell_address(),
            )

    def test_no_address_raises(self) -> None:
        with pytest.raises(ValueError, match="must have either"):
            FieldMapping(
                field_name="test",
                field_label="Test",
                field_type=FieldType.TEXT,
                page_number=0,
            )

    def test_auto_generated_field_id(self) -> None:
        fm = _make_field_mapping()
        uuid.UUID(fm.field_id)  # Should not raise


# ---------------------------------------------------------------------------
# FormTemplate Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormTemplate:
    def test_defaults(self) -> None:
        field = _make_field_mapping()
        t = FormTemplate(
            name="Leave Request",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[field],
        )
        uuid.UUID(t.template_id)  # Valid UUID
        assert t.version == 1
        assert t.created_by == "system"
        assert t.description == ""
        assert isinstance(t.created_at, datetime)
        assert isinstance(t.updated_at, datetime)

    def test_fields_min_length(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            FormTemplate(
                name="Empty",
                source_format=SourceFormat.PDF,
                page_count=1,
                fields=[],
            )

    def test_bytes_serialization(self) -> None:
        field = _make_field_mapping()
        t = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[field],
            layout_fingerprint=b"\xde\xad\xbe\xef",
            thumbnail=b"\xca\xfe",
        )
        data = t.model_dump(mode="json")
        assert data["layout_fingerprint"] == "deadbeef"
        assert data["thumbnail"] == "cafe"

    def test_bytes_none_serialization(self) -> None:
        field = _make_field_mapping()
        t = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[field],
        )
        data = t.model_dump(mode="json")
        assert data["layout_fingerprint"] is None
        assert data["thumbnail"] is None

    def test_datetime_is_utc(self) -> None:
        field = _make_field_mapping()
        t = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[field],
        )
        assert t.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Matching Model Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemplateMatch:
    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TemplateMatch(
                template_id="t1",
                template_name="Test",
                template_version=1,
                confidence=1.5,
                per_page_confidence=[0.9],
                matched_features=["layout_grid"],
            )

    def test_valid(self) -> None:
        m = TemplateMatch(
            template_id="t1",
            template_name="Test",
            template_version=1,
            confidence=0.85,
            per_page_confidence=[0.9, 0.8],
            matched_features=["layout_grid"],
        )
        assert m.confidence == 0.85


@pytest.mark.unit
class TestFormIngestRequest:
    def test_optional_fields_default_none(self) -> None:
        r = FormIngestRequest(file_path="/tmp/form.pdf")
        assert r.template_id is None
        assert r.template_version is None
        assert r.tenant_id is None
        assert r.source_uri is None
        assert r.metadata is None


# ---------------------------------------------------------------------------
# Extraction Result Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractedField:
    def test_confidence_too_high(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _make_extracted_field(confidence=1.5)

    def test_confidence_too_low(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            _make_extracted_field(confidence=-0.1)

    def test_union_value_types(self) -> None:
        """value field accepts str, bool, float, and None."""
        for val in ["hello", True, 3.14, None]:
            ef = _make_extracted_field(value=val)
            assert ef.value == val

    def test_warnings_default_empty(self) -> None:
        ef = _make_extracted_field()
        assert ef.warnings == []


@pytest.mark.unit
class TestFormExtractionResult:
    def test_defaults(self) -> None:
        r = _make_form_extraction_result()
        uuid.UUID(r.form_id)  # Auto-generated UUID
        assert r.warnings == []

    def test_required_fields(self) -> None:
        r = _make_form_extraction_result()
        assert r.template_name == "Leave Request"
        assert r.extraction_method == "native_fields"


# ---------------------------------------------------------------------------
# Chunk Model Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormChunkMetadata:
    def test_extends_base(self) -> None:
        assert issubclass(FormChunkMetadata, BaseChunkMetadata)

    def test_ingestion_method_default(self) -> None:
        m = FormChunkMetadata(
            source_uri="/tmp/form.pdf",
            source_format="pdf",
            parser_version="1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key1",
            template_id="t1",
            template_name="Leave Request",
            template_version=1,
            form_id="f1",
            field_names=["name"],
            extraction_method="native_fields",
            overall_confidence=0.9,
            per_field_confidence={"name": 0.9},
            page_numbers=[0],
            match_method="auto_detect",
        )
        assert m.ingestion_method == "form_extraction"

    def test_form_chunk_payload_structure(self) -> None:
        meta = FormChunkMetadata(
            source_uri="/tmp/form.pdf",
            source_format="pdf",
            parser_version="1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key1",
            template_id="t1",
            template_name="Leave Request",
            template_version=1,
            form_id="f1",
            field_names=["name"],
            extraction_method="native_fields",
            overall_confidence=0.9,
            per_field_confidence={"name": 0.9},
            page_numbers=[0],
            match_method="auto_detect",
        )
        payload = FormChunkPayload(
            id="chunk-1",
            text="Employee Name: John Smith",
            vector=[0.1, 0.2, 0.3],
            metadata=meta,
        )
        assert payload.metadata.template_name == "Leave Request"


# ---------------------------------------------------------------------------
# Result Model Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormWrittenArtifacts:
    def test_extends_core(self) -> None:
        assert issubclass(FormWrittenArtifacts, WrittenArtifacts)

    def test_db_row_ids_default(self) -> None:
        w = FormWrittenArtifacts()
        assert w.db_row_ids == []
        assert w.vector_point_ids == []
        assert w.db_table_names == []

    def test_with_values(self) -> None:
        w = FormWrittenArtifacts(
            vector_point_ids=["v1"],
            db_table_names=["form_leave"],
            db_row_ids=["r1", "r2"],
        )
        assert w.db_row_ids == ["r1", "r2"]


@pytest.mark.unit
class TestFormProcessingResult:
    def test_structure(self) -> None:
        extraction = _make_form_extraction_result()
        result = FormProcessingResult(
            file_path="/tmp/form.pdf",
            ingest_key="key1",
            ingest_run_id="run1",
            extraction_result=extraction,
            chunks_created=1,
            tables_created=1,
            tables=["form_leave_request"],
            written=FormWrittenArtifacts(
                vector_point_ids=["v1"],
                db_table_names=["form_leave_request"],
                db_row_ids=["r1"],
            ),
            errors=[],
            warnings=[],
            processing_time_seconds=1.5,
        )
        assert result.chunks_created == 1
        assert isinstance(result.written, FormWrittenArtifacts)
        assert result.error_details == []


@pytest.mark.unit
class TestRollbackResult:
    def test_defaults(self) -> None:
        r = RollbackResult()
        assert r.vector_points_deleted == 0
        assert r.db_rows_deleted == 0
        assert r.errors == []
        assert r.fully_rolled_back is True


# ---------------------------------------------------------------------------
# Request/Response Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormTemplateCreateRequest:
    def test_required_fields(self) -> None:
        field = _make_field_mapping()
        req = FormTemplateCreateRequest(
            name="Leave Request",
            source_format=SourceFormat.PDF,
            sample_file_path="/tmp/sample.pdf",
            page_count=1,
            fields=[field],
        )
        assert req.created_by == "system"
        assert req.tenant_id is None

    def test_fields_min_length(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            FormTemplateCreateRequest(
                name="Empty",
                source_format=SourceFormat.PDF,
                sample_file_path="/tmp/sample.pdf",
                page_count=1,
                fields=[],
            )


@pytest.mark.unit
class TestFormTemplateUpdateRequest:
    def test_all_optional(self) -> None:
        req = FormTemplateUpdateRequest()
        assert req.name is None
        assert req.description is None
        assert req.sample_file_path is None
        assert req.page_count is None
        assert req.fields is None


@pytest.mark.unit
class TestExtractionPreview:
    def test_structure(self) -> None:
        ef = _make_extracted_field()
        p = ExtractionPreview(
            template_id="t1",
            template_name="Test",
            template_version=1,
            fields=[ef],
            overall_confidence=0.9,
            extraction_method="native_fields",
            warnings=[],
        )
        assert p.template_name == "Test"


# ---------------------------------------------------------------------------
# Serialization Round-Trip Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSerializationRoundTrips:
    def test_form_template_json_round_trip(self) -> None:
        field = _make_field_mapping()
        original = FormTemplate(
            name="Leave Request",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[field],
            layout_fingerprint=b"\xde\xad",
        )
        data = original.model_dump(mode="json")
        restored = FormTemplate.model_validate(data)
        assert restored.name == original.name
        assert restored.source_format == original.source_format
        assert restored.version == original.version
        assert len(restored.fields) == len(original.fields)
        # Note: bytes round-trip produces hex string, not bytes
        assert data["layout_fingerprint"] == "dead"

    def test_form_extraction_result_json_round_trip(self) -> None:
        original = _make_form_extraction_result()
        data = original.model_dump(mode="json")
        restored = FormExtractionResult.model_validate(data)
        assert restored.form_id == original.form_id
        assert restored.template_name == original.template_name
        assert restored.overall_confidence == original.overall_confidence
        assert len(restored.fields) == len(original.fields)
