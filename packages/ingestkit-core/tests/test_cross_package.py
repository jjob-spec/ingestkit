"""Integration tests: verify shared core primitives work across packages."""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestSharedMockBackend:
    """A single mock backend implementing core protocols works in both packages."""

    def test_mock_vector_store_passes_both_protocols(self):
        from ingestkit_core.protocols import VectorStoreBackend as CoreVSB

        class SharedMock:
            def upsert_chunks(self, collection, chunks): return 0
            def ensure_collection(self, collection, vector_size): pass
            def create_payload_index(self, collection, field, field_type): pass
            def delete_by_ids(self, collection, ids): return 0

        mock = SharedMock()
        assert isinstance(mock, CoreVSB)

        # Also passes via package-level re-exports
        from ingestkit_excel.protocols import VectorStoreBackend as ExcelVSB
        from ingestkit_pdf.protocols import VectorStoreBackend as PdfVSB
        assert isinstance(mock, ExcelVSB)
        assert isinstance(mock, PdfVSB)

    def test_mock_embedding_backend_passes_both(self):
        from ingestkit_core.protocols import EmbeddingBackend as CoreEB

        class SharedEmbedder:
            def embed(self, texts, timeout=None): return [[0.0] * 768 for _ in texts]
            def dimension(self): return 768

        mock = SharedEmbedder()
        assert isinstance(mock, CoreEB)

        from ingestkit_excel.protocols import EmbeddingBackend as ExcelEB
        from ingestkit_pdf.protocols import EmbeddingBackend as PdfEB
        assert isinstance(mock, ExcelEB)
        assert isinstance(mock, PdfEB)


@pytest.mark.integration
class TestIngestKeyInterop:
    """IngestKey from core produces identical keys whether used via excel or pdf."""

    def test_same_computation(self):
        from ingestkit_core.models import IngestKey as CoreIK
        from ingestkit_excel.models import IngestKey as ExcelIK
        from ingestkit_pdf.models import IngestKey as PdfIK

        kwargs = dict(content_hash="abc", source_uri="file:///test", parser_version="1.0.0")
        assert CoreIK(**kwargs).key == ExcelIK(**kwargs).key == PdfIK(**kwargs).key


@pytest.mark.integration
class TestWrittenArtifactsInterop:
    """WrittenArtifacts from core usable in both ProcessingResults."""

    def test_core_artifacts_type_compatible(self):
        from ingestkit_core.models import WrittenArtifacts
        wa = WrittenArtifacts(vector_point_ids=["p1"], vector_collection="col", db_table_names=["t1"])

        # Verify it can be used in model_dump / model_validate context
        data = wa.model_dump()
        wa2 = WrittenArtifacts.model_validate(data)
        assert wa2.vector_point_ids == ["p1"]
