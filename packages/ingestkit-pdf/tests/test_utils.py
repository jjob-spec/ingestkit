"""Tests for test infrastructure: mock backends and PDF fixtures.

Validates that mock backends support error injection and assertion helpers,
and that all programmatic PDF fixtures produce valid, well-formed PDFs.
"""

from __future__ import annotations

import string

import fitz  # PyMuPDF
import pdfplumber
import pytest

from tests.conftest import (
    ENCRYPTED_PDF_PASSWORD,
    MockEmbeddingBackend,
    MockLLMBackend,
    MockStructuredDBBackend,
    MockVectorStoreBackend,
    _SENTINEL_CONNECTION_ERROR,
    _SENTINEL_MALFORMED_JSON,
    _SENTINEL_TIMEOUT,
)


# ---------------------------------------------------------------------------
# MockLLMBackend Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockLLMBackend:
    def test_classify_returns_enqueued_response(self) -> None:
        llm = MockLLMBackend(responses=[{"type": "text_native", "confidence": 0.9}])
        result = llm.classify("prompt", "model-a")
        assert result == {"type": "text_native", "confidence": 0.9}

    def test_generate_returns_enqueued_response(self) -> None:
        llm = MockLLMBackend(responses=["generated text"])
        result = llm.generate("prompt", "model-b")
        assert result == "generated text"

    def test_classify_raises_on_empty_queue(self) -> None:
        llm = MockLLMBackend()
        with pytest.raises(RuntimeError, match="no more responses"):
            llm.classify("prompt", "model-a")

    def test_timeout_sentinel_raises_timeout_error(self) -> None:
        llm = MockLLMBackend(responses=[_SENTINEL_TIMEOUT])
        with pytest.raises(TimeoutError, match="simulated timeout"):
            llm.classify("prompt", "model-a")

    def test_connection_error_sentinel_raises(self) -> None:
        llm = MockLLMBackend(responses=[_SENTINEL_CONNECTION_ERROR])
        with pytest.raises(ConnectionError, match="simulated connection error"):
            llm.classify("prompt", "model-a")

    def test_malformed_json_sentinel_returns_garbled_dict(self) -> None:
        llm = MockLLMBackend(responses=[_SENTINEL_MALFORMED_JSON])
        result = llm.classify("prompt", "model-a")
        assert result == {"raw": "<<<not json>>>"}

    def test_enqueue_timeout_convenience(self) -> None:
        llm = MockLLMBackend()
        llm.enqueue_timeout()
        with pytest.raises(TimeoutError):
            llm.classify("prompt", "model-a")

    def test_enqueue_connection_error_convenience(self) -> None:
        llm = MockLLMBackend()
        llm.enqueue_connection_error()
        with pytest.raises(ConnectionError):
            llm.generate("prompt", "model-a")

    def test_calls_tracked(self) -> None:
        llm = MockLLMBackend(responses=[{"ok": True}, "text"])
        llm.classify("p1", "m1")
        llm.generate("p2", "m2")
        assert llm.call_count == 2
        assert llm.calls[0]["model"] == "m1"
        assert llm.calls[1]["model"] == "m2"

    def test_assert_called_with_model(self) -> None:
        llm = MockLLMBackend(responses=[{"ok": True}])
        llm.classify("prompt", "qwen2.5:7b")
        llm.assert_called_with_model("qwen2.5:7b")

    def test_assert_called_with_model_fails(self) -> None:
        llm = MockLLMBackend(responses=[{"ok": True}])
        llm.classify("prompt", "qwen2.5:7b")
        with pytest.raises(AssertionError, match="Expected call with model"):
            llm.assert_called_with_model("nonexistent-model")

    def test_generate_timeout_sentinel(self) -> None:
        """Verify generate() also handles timeout sentinel."""
        llm = MockLLMBackend(responses=[_SENTINEL_TIMEOUT])
        with pytest.raises(TimeoutError):
            llm.generate("prompt", "model-a")

    def test_generate_connection_error_sentinel(self) -> None:
        """Verify generate() also handles connection error sentinel."""
        llm = MockLLMBackend(responses=[_SENTINEL_CONNECTION_ERROR])
        with pytest.raises(ConnectionError):
            llm.generate("prompt", "model-a")


# ---------------------------------------------------------------------------
# MockVectorStoreBackend Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockVectorStoreBackend:
    def test_upsert_and_tracking(self) -> None:
        vs = MockVectorStoreBackend()
        result = vs.upsert_chunks("col1", ["chunk_a", "chunk_b"])
        assert result == 2
        assert len(vs.upserted) == 1
        assert vs.upserted[0] == ("col1", ["chunk_a", "chunk_b"])

    def test_ensure_collection_tracked(self) -> None:
        vs = MockVectorStoreBackend()
        vs.ensure_collection("my_col", 768)
        assert vs.collections_ensured == [("my_col", 768)]

    def test_delete_by_ids_tracked(self) -> None:
        vs = MockVectorStoreBackend()
        result = vs.delete_by_ids("col1", ["id1", "id2"])
        assert result == 2
        assert vs.deleted == [("col1", ["id1", "id2"])]

    def test_fail_next_upsert(self) -> None:
        vs = MockVectorStoreBackend()
        vs.fail_next_upsert()
        with pytest.raises(ConnectionError, match="simulated error"):
            vs.upsert_chunks("col1", ["chunk"])

    def test_error_clears_after_raise(self) -> None:
        vs = MockVectorStoreBackend()
        vs.fail_next_upsert()
        with pytest.raises(ConnectionError):
            vs.upsert_chunks("col1", ["chunk"])
        # Second call should succeed
        result = vs.upsert_chunks("col1", ["chunk"])
        assert result == 1

    def test_fail_next_upsert_custom_error(self) -> None:
        vs = MockVectorStoreBackend()
        vs.fail_next_upsert(ValueError("custom"))
        with pytest.raises(ValueError, match="custom"):
            vs.upsert_chunks("col1", ["chunk"])

    def test_fail_next_ensure(self) -> None:
        vs = MockVectorStoreBackend()
        vs.fail_next_ensure()
        with pytest.raises(ConnectionError, match="simulated error"):
            vs.ensure_collection("col1", 768)

    def test_total_chunks_upserted_property(self) -> None:
        vs = MockVectorStoreBackend()
        vs.upsert_chunks("col1", ["a", "b"])
        vs.upsert_chunks("col2", ["c"])
        assert vs.total_chunks_upserted == 3


# ---------------------------------------------------------------------------
# MockEmbeddingBackend Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockEmbeddingBackend:
    def test_returns_correct_dimension_vectors(self) -> None:
        emb = MockEmbeddingBackend(dim=384)
        result = emb.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 384
        assert emb.dimension() == 384

    def test_calls_tracked(self) -> None:
        emb = MockEmbeddingBackend()
        emb.embed(["a", "b"])
        emb.embed(["c"])
        assert len(emb.calls) == 2
        assert emb.calls[0] == ["a", "b"]

    def test_fail_next_embed(self) -> None:
        emb = MockEmbeddingBackend()
        emb.fail_next_embed()
        with pytest.raises(TimeoutError, match="simulated timeout"):
            emb.embed(["hello"])

    def test_fail_next_embed_custom_error(self) -> None:
        emb = MockEmbeddingBackend()
        emb.fail_next_embed(RuntimeError("custom"))
        with pytest.raises(RuntimeError, match="custom"):
            emb.embed(["hello"])

    def test_error_clears_after_raise(self) -> None:
        emb = MockEmbeddingBackend()
        emb.fail_next_embed()
        with pytest.raises(TimeoutError):
            emb.embed(["hello"])
        # Second call should succeed
        result = emb.embed(["hello"])
        assert len(result) == 1

    def test_total_texts_embedded_property(self) -> None:
        emb = MockEmbeddingBackend()
        emb.embed(["a", "b", "c"])
        emb.embed(["d"])
        assert emb.total_texts_embedded == 4


# ---------------------------------------------------------------------------
# MockStructuredDBBackend Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockStructuredDBBackend:
    def test_create_and_exists(self) -> None:
        import pandas as pd

        db = MockStructuredDBBackend()
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        db.create_table_from_dataframe("test_table", df)
        assert db.table_exists("test_table") is True
        assert db.table_exists("nonexistent") is False

    def test_drop_table(self) -> None:
        import pandas as pd

        db = MockStructuredDBBackend()
        df = pd.DataFrame({"col_a": [1]})
        db.create_table_from_dataframe("t1", df)
        db.drop_table("t1")
        assert db.table_exists("t1") is False
        assert db.dropped == ["t1"]

    def test_get_table_schema(self) -> None:
        import pandas as pd

        db = MockStructuredDBBackend()
        df = pd.DataFrame({"name": ["Alice"], "age": [30]})
        db.create_table_from_dataframe("people", df)
        schema = db.get_table_schema("people")
        assert "name" in schema
        assert "age" in schema

    def test_get_table_schema_missing(self) -> None:
        db = MockStructuredDBBackend()
        assert db.get_table_schema("missing") == {}

    def test_get_connection_uri(self) -> None:
        db = MockStructuredDBBackend()
        assert db.get_connection_uri() == "sqlite:///:memory:"

    def test_fail_next_create(self) -> None:
        import pandas as pd

        db = MockStructuredDBBackend()
        db.fail_next_create()
        with pytest.raises(ConnectionError, match="simulated error"):
            db.create_table_from_dataframe("t1", pd.DataFrame({"a": [1]}))

    def test_fail_next_create_clears(self) -> None:
        import pandas as pd

        db = MockStructuredDBBackend()
        db.fail_next_create()
        with pytest.raises(ConnectionError):
            db.create_table_from_dataframe("t1", pd.DataFrame({"a": [1]}))
        # Second call should succeed
        db.create_table_from_dataframe("t2", pd.DataFrame({"a": [2]}))
        assert db.table_exists("t2") is True


# ---------------------------------------------------------------------------
# PDF Fixture Tests
# ---------------------------------------------------------------------------

_PRINTABLE = set(string.printable)


@pytest.mark.unit
class TestPDFFixtures:
    def test_text_native_pdf_is_valid(self, text_native_pdf) -> None:
        doc = fitz.open(str(text_native_pdf))
        assert doc.page_count == 3
        # Verify text is extractable
        text = doc[0].get_text()
        assert len(text.strip()) > 0
        doc.close()

    def test_text_native_pdf_has_headings(self, text_native_pdf) -> None:
        doc = fitz.open(str(text_native_pdf))
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        doc.close()
        assert "Chapter 1: Introduction" in all_text
        assert "Chapter 2: Methods" in all_text
        assert "Chapter 3: Results" in all_text

    def test_text_native_pdf_has_page_numbers(self, text_native_pdf) -> None:
        doc = fitz.open(str(text_native_pdf))
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        doc.close()
        assert "Page 1 of 3" in all_text
        assert "Page 3 of 3" in all_text

    def test_scanned_pdf_has_no_text(self, scanned_pdf) -> None:
        doc = fitz.open(str(scanned_pdf))
        # Pages should have images but minimal/no extractable text
        for page in doc:
            text = page.get_text().strip()
            images = page.get_images()
            assert len(images) > 0, "Scanned page should contain images"
            # Allow some minimal metadata text but not full page content
            assert len(text) < 50, (
                f"Scanned page should have minimal text, got {len(text)} chars"
            )
        doc.close()

    def test_scanned_pdf_page_count(self, scanned_pdf) -> None:
        doc = fitz.open(str(scanned_pdf))
        assert doc.page_count >= 2
        doc.close()

    def test_complex_pdf_has_tables(self, complex_pdf) -> None:
        with pdfplumber.open(str(complex_pdf)) as pdf:
            # Page 1 should have a table
            tables = pdf.pages[0].extract_tables()
            assert len(tables) >= 1, "Complex PDF page 1 should contain a table"
            # Verify table has header row
            header = tables[0][0]
            assert "ID" in header or "Name" in header

    def test_complex_pdf_is_multi_page(self, complex_pdf) -> None:
        doc = fitz.open(str(complex_pdf))
        assert doc.page_count >= 2
        doc.close()

    def test_encrypted_pdf_requires_password(self, encrypted_pdf) -> None:
        doc = fitz.open(str(encrypted_pdf))
        assert doc.needs_pass, "Encrypted PDF should require a password"
        # Verify we can authenticate with the known password
        assert doc.authenticate(ENCRYPTED_PDF_PASSWORD) > 0
        doc.close()

    def test_garbled_pdf_low_printable_ratio(self, garbled_pdf) -> None:
        doc = fitz.open(str(garbled_pdf))
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        doc.close()

        if len(all_text.strip()) == 0:
            # If fitz stripped all garbled chars, printable ratio is
            # effectively 0 (no useful text extracted) which is acceptable
            # for exercising the E_PARSE_GARBLED detection path.
            return

        printable_count = sum(1 for ch in all_text if ch in _PRINTABLE)
        total = len(all_text)
        ratio = printable_count / total if total > 0 else 0.0
        # The garbled text should have a low printable ratio
        assert ratio < 0.8, (
            f"Expected low printable ratio, got {ratio:.2f} "
            f"(printable={printable_count}, total={total})"
        )
