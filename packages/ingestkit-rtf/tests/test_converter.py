"""Tests for ingestkit_rtf.converter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ingestkit_rtf.config import RTFProcessorConfig
from ingestkit_rtf.converter import ExtractResult, chunk_text, extract_text


class TestExtractText:
    """extract_text with mocked striprtf."""

    def test_basic_extraction(self, tmp_path):
        rtf_path = str(tmp_path / "test.rtf")
        (tmp_path / "test.rtf").write_text("{\\rtf1\\ansi Hello world.}")

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "Hello world."
            result = extract_text(rtf_path)

        assert result.text == "Hello world."
        assert result.word_count == 2
        mock_rtf_to_text.assert_called_once()

    def test_multiline_extraction(self, tmp_path):
        rtf_path = str(tmp_path / "test.rtf")
        (tmp_path / "test.rtf").write_text("{\\rtf1\\ansi Some text here.}")

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "First paragraph.\n\nSecond paragraph."
            result = extract_text(rtf_path)

        assert result.word_count == 4

    def test_extraction_raises_on_failure(self, tmp_path):
        rtf_path = str(tmp_path / "test.rtf")
        (tmp_path / "test.rtf").write_text("not rtf content")

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.side_effect = ValueError("Not a valid RTF document")
            with pytest.raises(ValueError, match="Not a valid RTF document"):
                extract_text(rtf_path)

    def test_empty_text_extraction(self, tmp_path):
        rtf_path = str(tmp_path / "test.rtf")
        (tmp_path / "test.rtf").write_text("{\\rtf1\\ansi }")

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = ""
            result = extract_text(rtf_path)

        assert result.text == ""
        assert result.word_count == 0

    def test_whitespace_only_extraction(self, tmp_path):
        rtf_path = str(tmp_path / "test.rtf")
        (tmp_path / "test.rtf").write_text("{\\rtf1\\ansi  }")

        with patch("ingestkit_rtf.converter.rtf_to_text") as mock_rtf_to_text:
            mock_rtf_to_text.return_value = "   \n\n  "
            result = extract_text(rtf_path)

        assert result.word_count == 0


class TestChunkText:
    """chunk_text splitting logic."""

    def test_empty_text(self):
        config = RTFProcessorConfig()
        assert chunk_text("", config) == []

    def test_whitespace_only(self):
        config = RTFProcessorConfig()
        assert chunk_text("   \n\n  ", config) == []

    def test_single_short_paragraph(self):
        config = RTFProcessorConfig(chunk_size_tokens=512)
        text = "This is a short paragraph."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short paragraph."

    def test_multiple_paragraphs_single_chunk(self):
        config = RTFProcessorConfig(chunk_size_tokens=512)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert "Paragraph one." in chunks[0]
        assert "Paragraph two." in chunks[0]

    def test_multiple_chunks(self):
        config = RTFProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2

    def test_overlap(self):
        config = RTFProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=3)
        text = "Word one two.\n\nWord three four.\n\nWord five six."
        chunks = chunk_text(text, config)
        if len(chunks) > 1:
            assert len(chunks) >= 2

    def test_long_paragraph_split_on_newlines(self):
        config = RTFProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        text = "Line one here now.\nLine two here now.\nLine three here now."
        chunks = chunk_text(text, config)
        assert len(chunks) >= 2
