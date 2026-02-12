"""Tests for ingestkit_core.protocols -- runtime-checkable protocol definitions."""

from __future__ import annotations

import pytest

from ingestkit_core.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)


# -- Runtime checkable decorator verification --

class TestRuntimeCheckable:
    """All four protocols must be @runtime_checkable."""

    @pytest.mark.parametrize(
        "protocol",
        [VectorStoreBackend, StructuredDBBackend, LLMBackend, EmbeddingBackend],
        ids=["VectorStore", "StructuredDB", "LLM", "Embedding"],
    )
    def test_is_runtime_checkable(self, protocol):
        """Protocol has _is_runtime_protocol attribute set by decorator."""
        assert getattr(protocol, "_is_runtime_protocol", False) is True


# -- Conforming mock classes (structural subtyping, NO inheritance) --

class _FakeVectorStore:
    def upsert_chunks(self, collection, chunks): return 0
    def ensure_collection(self, collection, vector_size): pass
    def create_payload_index(self, collection, field, field_type): pass
    def delete_by_ids(self, collection, ids): return 0

class _FakeStructuredDB:
    def create_table_from_dataframe(self, table_name, df): pass
    def drop_table(self, table_name): pass
    def table_exists(self, table_name): return False
    def get_table_schema(self, table_name): return {}
    def get_connection_uri(self): return "sqlite:///:memory:"

class _FakeLLM:
    def classify(self, prompt, model, temperature=0.1, timeout=None): return {}
    def generate(self, prompt, model, temperature=0.7, timeout=None): return ""

class _FakeEmbedding:
    def embed(self, texts, timeout=None): return [[0.0] * 768 for _ in texts]
    def dimension(self): return 768


class TestStructuralSubtyping:
    """Classes implementing protocol methods WITHOUT inheriting pass isinstance."""

    def test_vector_store_isinstance(self):
        assert isinstance(_FakeVectorStore(), VectorStoreBackend)

    def test_structured_db_isinstance(self):
        assert isinstance(_FakeStructuredDB(), StructuredDBBackend)

    def test_llm_isinstance(self):
        assert isinstance(_FakeLLM(), LLMBackend)

    def test_embedding_isinstance(self):
        assert isinstance(_FakeEmbedding(), EmbeddingBackend)


class TestNonConforming:
    """Empty or partial implementations must fail isinstance."""

    def test_empty_class_fails_all(self):
        class Empty: pass
        assert not isinstance(Empty(), VectorStoreBackend)
        assert not isinstance(Empty(), StructuredDBBackend)
        assert not isinstance(Empty(), LLMBackend)
        assert not isinstance(Empty(), EmbeddingBackend)

    def test_partial_vector_store_fails(self):
        class Partial:
            def upsert_chunks(self, collection, chunks): return 0
            # Missing: ensure_collection, create_payload_index, delete_by_ids
        assert not isinstance(Partial(), VectorStoreBackend)


class TestExtraMethodsOk:
    """A class with extra methods beyond the protocol should still conform."""

    def test_extra_methods_on_vector_store(self):
        class Extended(_FakeVectorStore):
            def extra_method(self): return "extra"
        assert isinstance(Extended(), VectorStoreBackend)
