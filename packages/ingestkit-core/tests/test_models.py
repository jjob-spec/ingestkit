"""Tests for ingestkit_core.models -- IngestKey, WrittenArtifacts, EmbedStageResult,
BaseChunkMetadata, ChunkPayload, ClassificationTier."""

from __future__ import annotations

import hashlib
import re

import pytest
from pydantic import ValidationError

from ingestkit_core.models import (
    BaseChunkMetadata,
    ChunkPayload,
    ClassificationTier,
    EmbedStageResult,
    IngestKey,
    WrittenArtifacts,
)


class TestIngestKey:
    """Deterministic key computation tests."""

    def test_key_is_deterministic(self):
        ik = IngestKey(content_hash="abc", source_uri="file:///test", parser_version="1.0.0")
        assert ik.key == ik.key

    def test_key_is_64_char_hex(self):
        ik = IngestKey(content_hash="abc", source_uri="u", parser_version="v")
        assert len(ik.key) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", ik.key)

    def test_different_content_hash_different_key(self):
        ik1 = IngestKey(content_hash="aaa", source_uri="u", parser_version="v")
        ik2 = IngestKey(content_hash="bbb", source_uri="u", parser_version="v")
        assert ik1.key != ik2.key

    def test_different_source_uri_different_key(self):
        ik1 = IngestKey(content_hash="h", source_uri="a", parser_version="v")
        ik2 = IngestKey(content_hash="h", source_uri="b", parser_version="v")
        assert ik1.key != ik2.key

    def test_different_parser_version_different_key(self):
        ik1 = IngestKey(content_hash="h", source_uri="u", parser_version="v1")
        ik2 = IngestKey(content_hash="h", source_uri="u", parser_version="v2")
        assert ik1.key != ik2.key

    def test_tenant_none_vs_set(self):
        ik1 = IngestKey(content_hash="h", source_uri="u", parser_version="v")
        ik2 = IngestKey(content_hash="h", source_uri="u", parser_version="v", tenant_id="t")
        assert ik1.key != ik2.key

    def test_same_fields_same_key(self):
        kwargs = dict(content_hash="h", source_uri="u", parser_version="v", tenant_id="t")
        assert IngestKey(**kwargs).key == IngestKey(**kwargs).key

    def test_known_value(self):
        """Verify against hand-computed SHA-256."""
        ik = IngestKey(content_hash="abc", source_uri="file:///test.pdf", parser_version="1.0.0")
        expected = hashlib.sha256("abc|file:///test.pdf|1.0.0".encode()).hexdigest()
        assert ik.key == expected

    def test_known_value_with_tenant(self):
        ik = IngestKey(content_hash="abc", source_uri="uri", parser_version="v", tenant_id="t1")
        expected = hashlib.sha256("abc|uri|v|t1".encode()).hexdigest()
        assert ik.key == expected

    def test_serialization_round_trip(self):
        ik = IngestKey(content_hash="abc", source_uri="u", parser_version="v", tenant_id="t")
        data = ik.model_dump()
        ik2 = IngestKey.model_validate(data)
        assert ik2.key == ik.key

    def test_json_round_trip(self):
        ik = IngestKey(content_hash="abc", source_uri="u", parser_version="v")
        json_str = ik.model_dump_json()
        ik2 = IngestKey.model_validate_json(json_str)
        assert ik2.key == ik.key

    def test_empty_tenant_id_is_falsy(self):
        """Empty string tenant_id behaves same as None (not appended to hash)."""
        ik_none = IngestKey(content_hash="h", source_uri="u", parser_version="v", tenant_id=None)
        ik_empty = IngestKey(content_hash="h", source_uri="u", parser_version="v", tenant_id="")
        assert ik_none.key == ik_empty.key

    def test_unicode_source_uri(self):
        ik = IngestKey(content_hash="h", source_uri="file:///tmp/cafe\u0301.pdf", parser_version="v")
        assert len(ik.key) == 64


class TestWrittenArtifacts:
    """WrittenArtifacts construction and serialization."""

    def test_defaults(self):
        wa = WrittenArtifacts()
        assert wa.vector_point_ids == []
        assert wa.vector_collection is None
        assert wa.db_table_names == []

    def test_populated(self):
        wa = WrittenArtifacts(
            vector_point_ids=["p1", "p2"],
            vector_collection="helpdesk",
            db_table_names=["employees"],
        )
        assert len(wa.vector_point_ids) == 2
        assert wa.vector_collection == "helpdesk"

    def test_explicit_empty_lists(self):
        wa = WrittenArtifacts(vector_point_ids=[], db_table_names=[])
        assert wa.vector_point_ids == []
        assert wa.db_table_names == []

    def test_serialization_round_trip(self):
        wa = WrittenArtifacts(
            vector_point_ids=["p1"],
            vector_collection="col",
            db_table_names=["t1"],
        )
        data = wa.model_dump()
        wa2 = WrittenArtifacts.model_validate(data)
        assert wa2.vector_point_ids == wa.vector_point_ids
        assert wa2.vector_collection == wa.vector_collection
        assert wa2.db_table_names == wa.db_table_names

    def test_large_id_list(self):
        ids = [f"point-{i}" for i in range(10_000)]
        wa = WrittenArtifacts(vector_point_ids=ids)
        assert len(wa.vector_point_ids) == 10_000
        data = wa.model_dump()
        wa2 = WrittenArtifacts.model_validate(data)
        assert len(wa2.vector_point_ids) == 10_000


class TestEmbedStageResult:
    """EmbedStageResult construction and validation."""

    def test_valid(self):
        r = EmbedStageResult(texts_embedded=10, embedding_dimension=768, embed_duration_seconds=2.0)
        assert r.texts_embedded == 10

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            EmbedStageResult()  # type: ignore[call-arg]

    def test_serialization_round_trip(self):
        r = EmbedStageResult(texts_embedded=5, embedding_dimension=384, embed_duration_seconds=1.0)
        data = r.model_dump()
        r2 = EmbedStageResult.model_validate(data)
        assert r2.texts_embedded == r.texts_embedded
        assert r2.embedding_dimension == r.embedding_dimension

    def test_zero_texts_embedded(self):
        r = EmbedStageResult(texts_embedded=0, embedding_dimension=768, embed_duration_seconds=0.0)
        assert r.texts_embedded == 0


class TestBaseChunkMetadata:
    """BaseChunkMetadata construction and subclassing."""

    def _make_base(self, **overrides):
        defaults = dict(
            source_uri="file:///test.xlsx",
            source_format="xlsx",
            ingestion_method="sql_agent",
            parser_version="1.0.0",
            chunk_index=0,
            chunk_hash="abc123",
            ingest_key="key123",
        )
        defaults.update(overrides)
        return BaseChunkMetadata(**defaults)

    def test_construction_all_required(self):
        m = self._make_base()
        assert m.source_uri == "file:///test.xlsx"
        assert m.source_format == "xlsx"
        assert m.chunk_index == 0

    def test_optional_fields_default_none(self):
        m = self._make_base()
        assert m.ingest_run_id is None
        assert m.tenant_id is None
        assert m.table_name is None
        assert m.row_count is None
        assert m.columns is None
        assert m.section_title is None

    def test_subclassing_adds_fields(self):
        """Excel-style subclass can add sheet_name."""

        class ExcelChunk(BaseChunkMetadata):
            sheet_name: str
            source_format: str = "xlsx"

        m = ExcelChunk(
            source_uri="u", ingestion_method="sql_agent",
            parser_version="v", chunk_index=0, chunk_hash="h",
            ingest_key="k", sheet_name="Sheet1",
        )
        assert m.sheet_name == "Sheet1"
        assert m.source_format == "xlsx"

    def test_chunk_payload_accepts_subclass(self):
        """ChunkPayload works with subclassed metadata."""

        class PDFChunk(BaseChunkMetadata):
            page_numbers: list[int]
            source_format: str = "pdf"

        meta = PDFChunk(
            source_uri="u", ingestion_method="ocr",
            parser_version="v", chunk_index=0, chunk_hash="h",
            ingest_key="k", page_numbers=[1, 2],
        )
        payload = ChunkPayload(id="c1", text="hello", vector=[0.1], metadata=meta)
        assert payload.metadata.source_format == "pdf"


class TestClassificationTier:
    """ClassificationTier enum checks."""

    def test_members(self):
        assert ClassificationTier.RULE_BASED.value == "rule_based"
        assert ClassificationTier.LLM_BASIC.value == "llm_basic"
        assert ClassificationTier.LLM_REASONING.value == "llm_reasoning"

    def test_is_str_enum(self):
        for member in ClassificationTier:
            assert isinstance(member, str)

    def test_member_count(self):
        assert len(ClassificationTier) == 3
