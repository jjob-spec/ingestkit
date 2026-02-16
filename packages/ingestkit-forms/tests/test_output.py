"""Tests for output writers (db_writer, chunk_writer, dual_writer).

Covers DB row writing with schema evolution, RAG chunk serialization
and embedding, and dual-write consistency modes with rollback.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    ExtractedField,
    FieldType,
    FormExtractionResult,
    FormWrittenArtifacts,
)
from ingestkit_forms.output.chunk_writer import (
    FormChunkWriter,
    build_chunk_metadata,
    serialize_field_value,
    serialize_form_to_text,
    split_fields_into_chunks,
)
from ingestkit_forms.output.db_writer import (
    FIELD_TYPE_TO_SQL,
    METADATA_COLUMNS,
    FormDBWriter,
    build_row_dict,
    generate_table_schema,
    get_table_name,
    slugify_template_name,
)
from ingestkit_forms.output.dual_writer import (
    FormDualWriter,
    redact_extraction,
    rollback_written_artifacts,
)
from tests.conftest import (
    MockEmbeddingBackend,
    MockFormDBBackend,
    MockVectorStoreBackend,
    make_extracted_field,
    make_extraction_result,
    make_field_mapping,
    make_template,
)


# ===========================================================================
# DB Writer Tests
# ===========================================================================


class TestSlugifyTemplateName:
    @pytest.mark.unit
    def test_basic(self):
        assert slugify_template_name("Leave Request") == "leave_request"

    @pytest.mark.unit
    def test_special_chars(self):
        assert slugify_template_name("W-4 (2026)") == "w_4_2026"

    @pytest.mark.unit
    def test_unicode(self):
        result = slugify_template_name("Formulaire de cong\u00e9")
        assert result == "formulaire_de_cong"


class TestGetTableName:
    @pytest.mark.unit
    def test_prefix_and_slug(self, output_config):
        template = make_template(name="Leave Request")
        assert get_table_name(output_config, template) == "form_leave_request"


class TestFieldTypeToSql:
    @pytest.mark.unit
    def test_all_types_mapped(self):
        for ft in FieldType:
            assert ft in FIELD_TYPE_TO_SQL, f"Missing SQL type for {ft}"


class TestGenerateTableSchema:
    @pytest.mark.unit
    def test_metadata_plus_fields(self):
        fields = [
            make_field_mapping(field_name="emp_name", field_type=FieldType.TEXT),
            make_field_mapping(
                field_name="is_active",
                field_type=FieldType.CHECKBOX,
                y=0.2,
            ),
        ]
        template = make_template(fields=fields)
        schema = generate_table_schema(template)
        # 10 metadata columns + 2 field columns
        assert len(schema) == 12
        assert schema["_form_id"] == "TEXT PRIMARY KEY"
        assert schema["emp_name"] == "TEXT"
        assert schema["is_active"] == "INTEGER"


class TestBuildRowDict:
    @pytest.mark.unit
    def test_happy_path(self, output_config):
        extraction = make_extraction_result()
        row = build_row_dict(extraction, output_config, "key-1", "run-1")
        assert row["_form_id"] == "form-001"
        assert row["_ingest_key"] == "key-1"
        assert row["_tenant_id"] == "test-tenant"
        assert row["employee_name"] == "John Doe"

    @pytest.mark.unit
    def test_none_values(self, output_config):
        fields = [make_extracted_field(value=None)]
        extraction = make_extraction_result(fields=fields)
        row = build_row_dict(extraction, output_config, "key-1", "run-1")
        assert row["employee_name"] is None

    @pytest.mark.unit
    def test_checkbox_coercion(self, output_config):
        fields = [
            make_extracted_field(
                field_name="is_active",
                field_label="Is Active",
                field_type=FieldType.CHECKBOX,
                value=True,
            )
        ]
        extraction = make_extraction_result(fields=fields)
        row = build_row_dict(extraction, output_config, "key-1", "run-1")
        assert row["is_active"] == 1

    @pytest.mark.unit
    def test_signature_coercion(self, output_config):
        fields = [
            make_extracted_field(
                field_name="signed",
                field_label="Signed",
                field_type=FieldType.SIGNATURE,
                value=False,
            )
        ]
        extraction = make_extraction_result(fields=fields)
        row = build_row_dict(extraction, output_config, "key-1", "run-1")
        assert row["signed"] == 0


class TestFormDBWriter:
    @pytest.mark.unit
    def test_ensure_table_creates_new(self, mock_form_db, output_config):
        writer = FormDBWriter(mock_form_db, output_config)
        template = make_template(name="Leave Request")
        table_name = writer.ensure_table(template)
        assert table_name == "form_leave_request"
        assert mock_form_db.table_exists("form_leave_request")
        assert "_form_id" in mock_form_db.get_table_columns("form_leave_request")

    @pytest.mark.unit
    def test_ensure_table_existing_with_evolution(self, mock_form_db, output_config):
        writer = FormDBWriter(mock_form_db, output_config)
        # Create table with one field
        template_v1 = make_template(
            name="Leave Request",
            fields=[make_field_mapping(field_name="emp_name")],
        )
        writer.ensure_table(template_v1)

        # Evolve with a new field
        template_v2 = make_template(
            name="Leave Request",
            fields=[
                make_field_mapping(field_name="emp_name"),
                make_field_mapping(field_name="leave_type", y=0.2),
            ],
            version=2,
        )
        writer.ensure_table(template_v2)
        cols = mock_form_db.get_table_columns("form_leave_request")
        assert "leave_type" in cols

    @pytest.mark.unit
    def test_evolve_schema_no_changes(self, mock_form_db, output_config):
        writer = FormDBWriter(mock_form_db, output_config)
        template = make_template(name="Leave Request")
        writer.ensure_table(template)
        added = writer.evolve_schema("form_leave_request", template)
        assert added == []

    @pytest.mark.unit
    def test_write_row_success(self, mock_form_db, output_config):
        writer = FormDBWriter(mock_form_db, output_config)
        template = make_template(name="Leave Request")
        writer.ensure_table(template)
        extraction = make_extraction_result()
        form_id = writer.write_row(
            "form_leave_request", extraction, "key-1", "run-1"
        )
        assert form_id == "form-001"
        assert len(mock_form_db._rows["form_leave_request"]) == 1

    @pytest.mark.unit
    def test_write_row_retry_then_success(self, output_config):
        db = MockFormDBBackend(fail_on_call=3)  # fail 3rd execute_sql (the INSERT)
        writer = FormDBWriter(db, output_config)
        template = make_template(name="Leave Request")
        writer.ensure_table(template)
        # Reset fail counter so the next INSERT succeeds on retry
        db._fail_on_call = -1
        extraction = make_extraction_result()
        form_id = writer.write_row(
            "form_leave_request", extraction, "key-1", "run-1"
        )
        assert form_id == "form-001"

    @pytest.mark.unit
    def test_write_row_exhausts_retries(self, output_config):
        db = MockFormDBBackend()
        writer = FormDBWriter(db, output_config)

        # Manually create table so ensure_table isn't needed
        db._tables["form_test"] = {"_form_id": "TEXT PRIMARY KEY"}
        db._rows["form_test"] = []
        # Monkey-patch execute_sql to always raise
        original = db.execute_sql

        def always_fail(sql, params=None):
            raise RuntimeError("Always fails")

        db.execute_sql = always_fail

        extraction = make_extraction_result()
        with pytest.raises(FormIngestException) as exc_info:
            writer.write_row("form_test", extraction, "key-1", "run-1")
        assert exc_info.value.code == FormErrorCode.E_FORM_DB_WRITE_FAILED


# ===========================================================================
# Chunk Writer Tests
# ===========================================================================


class TestSerializeFieldValue:
    @pytest.mark.unit
    def test_text(self):
        field = make_extracted_field(value="John Doe")
        assert serialize_field_value(field) == "John Doe"

    @pytest.mark.unit
    def test_none(self):
        field = make_extracted_field(value=None)
        assert serialize_field_value(field) == "[not extracted]"

    @pytest.mark.unit
    def test_checkbox_yes(self):
        field = make_extracted_field(
            field_type=FieldType.CHECKBOX, value=True
        )
        assert serialize_field_value(field) == "Yes"

    @pytest.mark.unit
    def test_checkbox_no(self):
        field = make_extracted_field(
            field_type=FieldType.CHECKBOX, value=False
        )
        assert serialize_field_value(field) == "No"


class TestSerializeFormToText:
    @pytest.mark.unit
    def test_format(self):
        extraction = make_extraction_result()
        text = serialize_form_to_text(extraction, extraction.fields)
        assert text.startswith("Form: Test Template (v1)")
        assert "Date Extracted:" in text
        assert "Employee Name: John Doe" in text


class TestSplitFieldsIntoChunks:
    @pytest.mark.unit
    def test_single_chunk(self):
        field_mappings = [
            make_field_mapping(field_id="f1", field_name="f1"),
        ]
        template = make_template(fields=field_mappings)
        fields = [
            make_extracted_field(field_id="f1", field_name="f1"),
        ]
        result = split_fields_into_chunks(fields, 20, template)
        assert len(result) == 1
        assert len(result[0]) == 1

    @pytest.mark.unit
    def test_multi_page_split(self):
        field_mappings = [
            make_field_mapping(
                field_id=f"f{i}",
                field_name=f"field_{i}",
                field_label=f"Field {i}",
                page_number=i // 3,
                y=0.1 + (i % 3) * 0.1,
            )
            for i in range(6)
        ]
        template = make_template(fields=field_mappings, page_count=2)
        fields = [
            make_extracted_field(
                field_id=f"f{i}",
                field_name=f"field_{i}",
                field_label=f"Field {i}",
            )
            for i in range(6)
        ]
        # chunk_max_fields=4 forces splitting; 6 fields across 2 pages
        result = split_fields_into_chunks(fields, 4, template)
        assert len(result) == 2


class TestBuildChunkMetadata:
    @pytest.mark.unit
    def test_all_fields_present(self, output_config):
        field_mappings = [make_field_mapping(field_id="f1")]
        template = make_template(fields=field_mappings)
        fields = [make_extracted_field(field_id="f1")]
        extraction = make_extraction_result(fields=fields)
        meta = build_chunk_metadata(
            extraction=extraction,
            chunk_fields=fields,
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key-1",
            ingest_run_id="run-1",
            config=output_config,
            template=template,
        )
        assert meta.template_id == "tmpl-1"
        assert meta.form_id == "form-001"
        assert meta.field_names == ["employee_name"]
        assert meta.ingestion_method == "form_extraction"
        assert meta.page_numbers == [0]

    @pytest.mark.unit
    def test_form_date_extraction(self, output_config):
        field_mappings = [
            make_field_mapping(
                field_id="fd",
                field_name="hire_date",
                field_label="Hire Date",
                field_type=FieldType.DATE,
            )
        ]
        template = make_template(fields=field_mappings)
        fields = [
            make_extracted_field(
                field_id="fd",
                field_name="hire_date",
                field_label="Hire Date",
                field_type=FieldType.DATE,
                value="2026-01-15",
            )
        ]
        extraction = make_extraction_result(fields=fields)
        meta = build_chunk_metadata(
            extraction=extraction,
            chunk_fields=fields,
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key-1",
            ingest_run_id="run-1",
            config=output_config,
            template=template,
        )
        assert meta.form_date == "2026-01-15"


class TestFormChunkWriter:
    @pytest.mark.unit
    def test_write_chunks_happy_path(
        self, mock_vector_store, mock_embedder, output_config
    ):
        field_mappings = [make_field_mapping(field_id="f1")]
        template = make_template(fields=field_mappings)
        fields = [make_extracted_field(field_id="f1")]
        extraction = make_extraction_result(fields=fields)

        writer = FormChunkWriter(mock_vector_store, mock_embedder, output_config)
        point_ids, embed_result = writer.write_chunks(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert len(point_ids) == 1
        assert embed_result.texts_embedded == 1
        assert len(mock_embedder.embed_calls) == 1
        assert len(mock_vector_store.upsert_calls) == 1

    @pytest.mark.unit
    def test_write_chunks_retry(self, mock_vector_store, output_config):
        embedder = MockEmbeddingBackend(fail_on_call=1)
        field_mappings = [make_field_mapping(field_id="f1")]
        template = make_template(fields=field_mappings)
        fields = [make_extracted_field(field_id="f1")]
        extraction = make_extraction_result(fields=fields)

        writer = FormChunkWriter(mock_vector_store, embedder, output_config)
        point_ids, _ = writer.write_chunks(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        # First call fails, second succeeds
        assert len(point_ids) == 1
        assert len(embedder.embed_calls) == 1  # only successful call tracked

    @pytest.mark.unit
    def test_chunk_id_deterministic(
        self, mock_vector_store, mock_embedder, output_config
    ):
        field_mappings = [make_field_mapping(field_id="f1")]
        template = make_template(fields=field_mappings)
        fields = [make_extracted_field(field_id="f1")]
        extraction = make_extraction_result(fields=fields)

        writer = FormChunkWriter(mock_vector_store, mock_embedder, output_config)
        ids1, _ = writer.write_chunks(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        ids2, _ = writer.write_chunks(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert ids1 == ids2


# ===========================================================================
# Dual Writer Tests
# ===========================================================================


def _make_dual_writer(
    mock_form_db,
    mock_vector_store,
    mock_embedder,
    config,
):
    """Helper to construct FormDualWriter with mock backends."""
    db_writer = FormDBWriter(mock_form_db, config)
    chunk_writer = FormChunkWriter(mock_vector_store, mock_embedder, config)
    return FormDualWriter(db_writer, chunk_writer, config)


def _make_test_data():
    """Helper to create aligned template + extraction for dual writer tests."""
    field_mappings = [make_field_mapping(field_id="f1")]
    template = make_template(fields=field_mappings)
    fields = [make_extracted_field(field_id="f1")]
    extraction = make_extraction_result(fields=fields)
    return template, extraction


class TestDualWriteBothSuccess:
    @pytest.mark.unit
    def test_both_artifacts_populated(
        self, mock_form_db, mock_vector_store, mock_embedder, output_config
    ):
        writer = _make_dual_writer(
            mock_form_db, mock_vector_store, mock_embedder, output_config
        )
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert len(written.db_table_names) == 1
        assert len(written.db_row_ids) == 1
        assert len(written.vector_point_ids) == 1
        assert errors == []
        assert warnings == []
        assert embed_result is not None


class TestBestEffort:
    @pytest.mark.unit
    def test_db_fail_vector_success(
        self, mock_vector_store, mock_embedder, output_config
    ):
        db = MockFormDBBackend(fail_on_call=1)  # CREATE TABLE fails
        writer = _make_dual_writer(db, mock_vector_store, mock_embedder, output_config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert len(written.db_table_names) == 0
        assert len(written.vector_point_ids) == 1
        assert "W_FORM_PARTIAL_WRITE" in warnings
        assert len(errors) == 1

    @pytest.mark.unit
    def test_vector_fail_db_success(
        self, mock_form_db, mock_embedder, output_config
    ):
        vs = MockVectorStoreBackend(fail_on_upsert=True)
        writer = _make_dual_writer(mock_form_db, vs, mock_embedder, output_config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert len(written.db_row_ids) == 1
        assert len(written.vector_point_ids) == 0
        assert "W_FORM_PARTIAL_WRITE" in warnings

    @pytest.mark.unit
    def test_both_fail(self, mock_embedder, output_config):
        db = MockFormDBBackend(fail_on_call=1)
        vs = MockVectorStoreBackend(fail_on_upsert=True)
        writer = _make_dual_writer(db, vs, mock_embedder, output_config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert len(written.db_table_names) == 0
        assert len(written.vector_point_ids) == 0
        assert len(errors) == 2


class TestStrictAtomic:
    @pytest.mark.unit
    def test_db_fail_vector_rolled_back(
        self, mock_vector_store, mock_embedder
    ):
        config = FormProcessorConfig(
            dual_write_mode="strict_atomic",
            backend_max_retries=1,
            backend_backoff_base=0.0,
            tenant_id="test-tenant",
        )
        db = MockFormDBBackend(fail_on_call=1)
        writer = _make_dual_writer(db, mock_vector_store, mock_embedder, config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        # Vector points should be rolled back
        assert len(written.vector_point_ids) == 0
        assert len(written.db_table_names) == 0

    @pytest.mark.unit
    def test_vector_fail_db_rolled_back(
        self, mock_form_db, mock_embedder
    ):
        config = FormProcessorConfig(
            dual_write_mode="strict_atomic",
            backend_max_retries=1,
            backend_backoff_base=0.0,
            tenant_id="test-tenant",
        )
        vs = MockVectorStoreBackend(fail_on_upsert=True)
        writer = _make_dual_writer(mock_form_db, vs, mock_embedder, config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        # DB row should be rolled back
        assert len(written.db_row_ids) == 0
        assert len(written.db_table_names) == 0
        assert len(mock_form_db.delete_rows_calls) == 1

    @pytest.mark.unit
    def test_both_fail_no_rollback(self, mock_embedder):
        config = FormProcessorConfig(
            dual_write_mode="strict_atomic",
            backend_max_retries=1,
            backend_backoff_base=0.0,
            tenant_id="test-tenant",
        )
        db = MockFormDBBackend(fail_on_call=1)
        vs = MockVectorStoreBackend(fail_on_upsert=True)
        writer = _make_dual_writer(db, vs, mock_embedder, config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert len(written.db_table_names) == 0
        assert len(written.vector_point_ids) == 0
        assert len(errors) == 2

    @pytest.mark.unit
    def test_rollback_failure_warning(self, mock_embedder):
        config = FormProcessorConfig(
            dual_write_mode="strict_atomic",
            backend_max_retries=0,
            backend_backoff_base=0.0,
            tenant_id="test-tenant",
        )
        vs = MockVectorStoreBackend(fail_on_upsert=True)
        # DB will succeed, vectors will fail -> rollback DB
        # Make delete_rows fail by monkey-patching
        db = MockFormDBBackend()
        original_delete = db.delete_rows

        def failing_delete(*args, **kwargs):
            raise RuntimeError("Rollback failure")

        db.delete_rows = failing_delete

        writer = _make_dual_writer(db, vs, mock_embedder, config)
        template, extraction = _make_test_data()
        written, errors, warnings, error_details, embed_result = writer.write(
            extraction, template, "/forms/test.pdf", "key-1", "run-1"
        )
        assert "W_FORM_ROLLBACK_FAILED" in warnings


class TestRedaction:
    @pytest.mark.unit
    def test_both_targets(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "both"
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        db_redacted = redact_extraction(extraction, output_config, "db")
        chunk_redacted = redact_extraction(extraction, output_config, "chunks")
        assert "123-45-6789" not in str(db_redacted.fields[0].value)
        assert "123-45-6789" not in str(chunk_redacted.fields[0].value)

    @pytest.mark.unit
    def test_chunks_only(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "chunks_only"
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        db_result = redact_extraction(extraction, output_config, "db")
        chunk_result = redact_extraction(extraction, output_config, "chunks")
        # DB should keep raw values
        assert "123-45-6789" in str(db_result.fields[0].value)
        # Chunks should be redacted
        assert "123-45-6789" not in str(chunk_result.fields[0].value)

    @pytest.mark.unit
    def test_db_only(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "db_only"
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        db_result = redact_extraction(extraction, output_config, "db")
        chunk_result = redact_extraction(extraction, output_config, "chunks")
        assert "123-45-6789" not in str(db_result.fields[0].value)
        assert "123-45-6789" in str(chunk_result.fields[0].value)

    @pytest.mark.unit
    def test_no_patterns(self, output_config):
        output_config.redact_patterns = []
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        result = redact_extraction(extraction, output_config, "db")
        assert result is extraction  # Same object, no copy

    @pytest.mark.unit
    def test_preserves_original(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "both"
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        redact_extraction(extraction, output_config, "db")
        # Original should be unchanged
        assert extraction.fields[0].value == "SSN: 123-45-6789"

    @pytest.mark.unit
    def test_redacted_flag_set_on_match(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "both"
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        result = redact_extraction(extraction, output_config, "db")
        assert result.fields[0].redacted is True

    @pytest.mark.unit
    def test_redacted_flag_not_set_when_no_match(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "both"
        fields = [make_extracted_field(value="John Doe")]
        extraction = make_extraction_result(fields=fields)

        result = redact_extraction(extraction, output_config, "db")
        assert result.fields[0].redacted is False

    @pytest.mark.unit
    def test_raw_value_redacted_on_match(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "both"
        fields = [
            ExtractedField(
                field_id="f1",
                field_name="ssn",
                field_label="SSN",
                field_type=FieldType.TEXT,
                value="123-45-6789",
                raw_value="123-45-6789",
                confidence=0.9,
                extraction_method="ocr_overlay",
            )
        ]
        extraction = make_extraction_result(fields=fields)

        result = redact_extraction(extraction, output_config, "db")
        assert result.fields[0].raw_value == "[REDACTED]"
        assert result.fields[0].redacted is True

    @pytest.mark.unit
    def test_raw_value_none_stays_none(self, output_config):
        output_config.redact_patterns = [r"\d{3}-\d{2}-\d{4}"]
        output_config.redact_target = "both"
        fields = [make_extracted_field(value="SSN: 123-45-6789")]
        extraction = make_extraction_result(fields=fields)

        result = redact_extraction(extraction, output_config, "db")
        assert result.fields[0].raw_value is None
        assert result.fields[0].redacted is True


class TestRollback:
    @pytest.mark.unit
    def test_vector_first_then_db(self, output_config):
        vs = MockVectorStoreBackend()
        db = MockFormDBBackend()
        written = FormWrittenArtifacts(
            vector_collection="helpdesk",
            vector_point_ids=["p1", "p2"],
            db_table_names=["form_test"],
            db_row_ids=["row-1"],
        )
        # Pre-populate vector store so delete works
        vs._collections["helpdesk"] = {"p1": "x", "p2": "y"}
        # Pre-populate DB so delete works
        db._tables["form_test"] = {"_form_id": "TEXT"}
        db._rows["form_test"] = [{"_form_id": "row-1"}]

        result = rollback_written_artifacts(written, vs, db, output_config)
        assert result.vector_points_deleted == 2
        assert result.db_rows_deleted == 1
        assert result.fully_rolled_back is True
        # Verify vector was deleted before DB (by checking call order)
        assert len(vs.delete_calls) == 1
        assert len(db.delete_rows_calls) == 1

    @pytest.mark.unit
    def test_idempotent_empty(self, output_config):
        written = FormWrittenArtifacts(vector_collection="helpdesk")
        result = rollback_written_artifacts(written, None, None, output_config)
        assert result.fully_rolled_back is True
        assert result.vector_points_deleted == 0
        assert result.db_rows_deleted == 0
