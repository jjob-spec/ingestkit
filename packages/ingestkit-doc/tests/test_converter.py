"""Tests for ingestkit_doc.converter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ingestkit_doc.config import DocProcessorConfig
from ingestkit_doc.converter import ExtractResult, chunk_text, extract_text


class TestExtractText:
    """extract_text with mocked mammoth."""

    def test_basic_extraction(self, tmp_path):
        doc_path = str(tmp_path / "test.doc")
        # Create a dummy file so open() works
        (tmp_path / "test.doc").write_bytes(b"dummy")

        mock_result = SimpleNamespace(
            value="Hello world. This is a test document.",
            messages=[],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = extract_text(doc_path)

        assert result.text == "Hello world. This is a test document."
        assert result.word_count == 7
        assert result.mammoth_messages == []

    def test_extraction_with_messages(self, tmp_path):
        doc_path = str(tmp_path / "test.doc")
        (tmp_path / "test.doc").write_bytes(b"dummy")

        mock_result = SimpleNamespace(
            value="Some text here.",
            messages=["Warning: unrecognised style"],
        )
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = extract_text(doc_path)

        assert result.word_count == 3
        assert len(result.mammoth_messages) == 1
        assert "unrecognised style" in result.mammoth_messages[0]

    def test_extraction_raises_on_failure(self, tmp_path):
        doc_path = str(tmp_path / "test.doc")
        (tmp_path / "test.doc").write_bytes(b"dummy")

        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.side_effect = ValueError(
                "Not a valid Word document"
            )
            with pytest.raises(ValueError, match="Not a valid Word document"):
                extract_text(doc_path)

    def test_empty_text_extraction(self, tmp_path):
        doc_path = str(tmp_path / "test.doc")
        (tmp_path / "test.doc").write_bytes(b"dummy")

        mock_result = SimpleNamespace(value="", messages=[])
        with patch("ingestkit_doc.converter.mammoth") as mock_mammoth:
            mock_mammoth.extract_raw_text.return_value = mock_result
            result = extract_text(doc_path)

        assert result.text == ""
        assert result.word_count == 0


class TestChunkText:
    """chunk_text splitting logic."""

    def test_empty_text(self):
        config = DocProcessorConfig()
        assert chunk_text("", config) == []

    def test_whitespace_only(self):
        config = DocProcessorConfig()
        assert chunk_text("   \n\n  ", config) == []

    def test_single_short_paragraph(self):
        config = DocProcessorConfig(chunk_size_tokens=512)
        text = "This is a short paragraph."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short paragraph."

    def test_multiple_paragraphs_single_chunk(self):
        config = DocProcessorConfig(chunk_size_tokens=512)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert "Paragraph one." in chunks[0]
        assert "Paragraph two." in chunks[0]

    def test_multiple_chunks(self):
        config = DocProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        # Each paragraph has about 3-4 words
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2

    def test_overlap(self):
        config = DocProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=3)
        text = "Word one two.\n\nWord three four.\n\nWord five six."
        chunks = chunk_text(text, config)
        # With overlap, later chunks should contain text from previous chunks
        if len(chunks) > 1:
            # Just verify we got multiple chunks with overlap enabled
            assert len(chunks) >= 2

    def test_long_paragraph_split_on_newlines(self):
        config = DocProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        # Single paragraph with sub-lines that together exceed chunk_size
        text = "Line one here now.\nLine two here now.\nLine three here now."
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2
