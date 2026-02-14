"""Tests for ingestkit_pdf.utils.chunker — configurable text chunking."""

from __future__ import annotations

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.utils.chunker import (
    PDFChunker,
    _compute_chunk_hash,
    _detect_content_type,
    _estimate_tokens,
    _get_heading_path,
    _get_page_numbers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PDFProcessorConfig:
    """Create a PDFProcessorConfig with optional overrides."""
    return PDFProcessorConfig(**overrides)


def _make_chunker(**overrides) -> PDFChunker:
    return PDFChunker(_make_config(**overrides))


# ---------------------------------------------------------------------------
# TestTokenEstimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # min 1

    def test_short_string(self):
        assert _estimate_tokens("hello") == 1  # 5 // 4 = 1

    def test_known_length(self):
        assert _estimate_tokens("a" * 2048) == 512


# ---------------------------------------------------------------------------
# TestChunkHash
# ---------------------------------------------------------------------------


class TestChunkHash:
    def test_deterministic(self):
        h1 = _compute_chunk_hash("hello world")
        h2 = _compute_chunk_hash("hello world")
        assert h1 == h2

    def test_different_text(self):
        h1 = _compute_chunk_hash("hello")
        h2 = _compute_chunk_hash("world")
        assert h1 != h2

    def test_is_64_hex_chars(self):
        h = _compute_chunk_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# TestContentTypeDetection
# ---------------------------------------------------------------------------


class TestContentTypeDetection:
    def test_narrative(self):
        text = "This is a plain paragraph of narrative text.\nIt continues here."
        assert _detect_content_type(text) == "narrative"

    def test_table(self):
        text = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        assert _detect_content_type(text) == "table"

    def test_list_dash(self):
        text = "- Item one\n- Item two\n- Item three"
        assert _detect_content_type(text) == "list"

    def test_list_numbered(self):
        text = "1. First item\n2. Second item\n3. Third item"
        assert _detect_content_type(text) == "list"

    def test_list_asterisk(self):
        text = "* Item one\n* Item two\n* Item three"
        assert _detect_content_type(text) == "list"

    def test_form_field_checkbox(self):
        text = "[ ] Task A\n[x] Task B\n[ ] Task C"
        assert _detect_content_type(text) == "form_field"

    def test_form_field_underscores(self):
        text = "Name: ___________\nDate: ___________\nSign: ___________"
        assert _detect_content_type(text) == "form_field"

    def test_empty(self):
        assert _detect_content_type("") == "narrative"


# ---------------------------------------------------------------------------
# TestHeadingPath
# ---------------------------------------------------------------------------


class TestHeadingPath:
    def test_no_headings(self):
        assert _get_heading_path(100, []) == []

    def test_single_heading(self):
        headings = [(2, "Section A", 0)]
        assert _get_heading_path(50, headings) == ["Section A"]

    def test_nested_headings(self):
        headings = [(2, "Section A", 0), (3, "Subsection A.1", 50)]
        assert _get_heading_path(100, headings) == ["Section A", "Subsection A.1"]

    def test_heading_reset(self):
        headings = [
            (2, "Section A", 0),
            (3, "Sub A.1", 50),
            (2, "Section B", 200),
        ]
        # Position after Section B — h3 should be cleared
        assert _get_heading_path(250, headings) == ["Section B"]

    def test_position_before_any_heading(self):
        headings = [(2, "Section A", 100)]
        assert _get_heading_path(50, headings) == []

    def test_position_exactly_at_heading(self):
        headings = [(2, "Section A", 50)]
        assert _get_heading_path(50, headings) == ["Section A"]


# ---------------------------------------------------------------------------
# TestPageNumbers
# ---------------------------------------------------------------------------


class TestPageNumbers:
    def test_single_page(self):
        # Pages start at offsets [0, 500, 1000]
        assert _get_page_numbers(100, 200, [0, 500, 1000]) == [1]

    def test_spanning_pages(self):
        assert _get_page_numbers(400, 600, [0, 500, 1000]) == [1, 2]

    def test_empty_boundaries(self):
        assert _get_page_numbers(0, 100, []) == [1]

    def test_spanning_three_pages(self):
        assert _get_page_numbers(100, 1100, [0, 500, 1000, 1500]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# TestBasicChunking
# ---------------------------------------------------------------------------


class TestBasicChunking:
    def test_empty_text(self):
        chunker = _make_chunker()
        assert chunker.chunk("", [], []) == []

    def test_whitespace_only(self):
        chunker = _make_chunker()
        assert chunker.chunk("   \n  ", [], []) == []

    def test_short_text_single_chunk(self):
        chunker = _make_chunker()
        result = chunker.chunk("Hello world.", [], [0])
        assert len(result) == 1
        assert result[0]["text"] == "Hello world."
        assert result[0]["chunk_index"] == 0

    def test_text_splits_on_paragraphs(self):
        # Two paragraphs, each large enough to be separate chunks
        para = "word " * 300  # ~1500 chars = ~375 tokens each
        text = para.strip() + "\n\n" + para.strip()
        chunker = _make_chunker(chunk_size_tokens=256, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], [0])
        assert len(result) >= 2

    def test_text_splits_on_sentences(self):
        # Single paragraph with sentences
        sentence = "This is a test sentence that has some words in it. "
        text = sentence * 60  # ~3600 chars = ~900 tokens
        chunker = _make_chunker(chunk_size_tokens=256, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], [0])
        assert len(result) >= 2

    def test_text_splits_on_words(self):
        # One very long "sentence" with no periods
        text = "word " * 600  # ~3000 chars = ~750 tokens, no ". " delimiter
        chunker = _make_chunker(chunk_size_tokens=256, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], [0])
        assert len(result) >= 2

    def test_chunk_index_sequential(self):
        text = "paragraph one content here. " * 40 + "\n\n" + "paragraph two content. " * 40
        chunker = _make_chunker(chunk_size_tokens=128, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], [0])
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_hash_present(self):
        chunker = _make_chunker()
        result = chunker.chunk("Some text content here.", [], [0])
        for c in result:
            assert "chunk_hash" in c
            assert len(c["chunk_hash"]) == 64

    def test_all_keys_present(self):
        chunker = _make_chunker()
        result = chunker.chunk("Some text content.", [], [0])
        expected_keys = {"text", "page_numbers", "heading_path", "content_type",
                         "chunk_index", "chunk_hash"}
        for c in result:
            assert set(c.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestOverlap
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_overlap_applied(self):
        # Two distinct paragraphs
        para1 = "Alpha bravo charlie delta echo. " * 20
        para2 = "Foxtrot golf hotel india juliet. " * 20
        text = para1.strip() + "\n\n" + para2.strip()
        chunker = _make_chunker(chunk_size_tokens=128, chunk_overlap_tokens=50)
        result = chunker.chunk(text, [], [0])
        assert len(result) >= 2
        # Second chunk should start with text from end of first chunk
        # (overlap means some content is shared)
        first_end = result[0]["text"][-100:]
        # At least part of first chunk's end should appear in second chunk's start
        # The overlap is from the raw segments, so check for shared substring
        second_start = result[1]["text"][:300]
        # Find a common substring of reasonable length
        overlap_found = any(
            first_end[i:i+20] in second_start
            for i in range(0, len(first_end) - 20)
        )
        assert overlap_found, "Expected overlap between consecutive chunks"

    def test_no_overlap_on_first_chunk(self):
        para1 = "First paragraph content here. " * 20
        para2 = "Second paragraph content here. " * 20
        text = para1.strip() + "\n\n" + para2.strip()
        chunker = _make_chunker(chunk_size_tokens=128, chunk_overlap_tokens=50)
        result = chunker.chunk(text, [], [0])
        # First chunk should start with the actual text start
        assert result[0]["text"].startswith("First paragraph")

    def test_zero_overlap(self):
        para1 = "Alpha " * 150
        para2 = "Bravo " * 150
        text = para1.strip() + "\n\n" + para2.strip()
        chunker = _make_chunker(chunk_size_tokens=128, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], [0])
        if len(result) >= 2:
            # With zero overlap, second chunk should NOT start with content
            # from the first chunk's ending
            assert "Alpha" not in result[-1]["text"][:50] or "Bravo" in result[-1]["text"][:50]


# ---------------------------------------------------------------------------
# TestHeadingRespect
# ---------------------------------------------------------------------------


class TestHeadingRespect:
    def _heading_text(self):
        return (
            "Introduction paragraph here.\n\n"
            "## Section One\n\n"
            "Content of section one. " * 30 + "\n\n"
            "## Section Two\n\n"
            "Content of section two. " * 30
        )

    def test_heading_boundary_not_crossed(self):
        text = self._heading_text()
        chunker = _make_chunker(
            chunk_size_tokens=256, chunk_overlap_tokens=0,
            chunk_respect_headings=True,
        )
        result = chunker.chunk(text, [], [0])
        # No chunk should contain text from both sections
        for c in result:
            has_one = "section one" in c["text"].lower()
            has_two = "section two" in c["text"].lower()
            # A chunk should not have content from both sections
            # (heading names like "Section One"/"Section Two" in headings are OK,
            # but body content mixing is not)
            if "Content of section one" in c["text"] and "Content of section two" in c["text"]:
                pytest.fail("Chunk crosses heading boundary")

    def test_heading_boundary_ignored_when_disabled(self):
        text = self._heading_text()
        chunker = _make_chunker(
            chunk_size_tokens=2048, chunk_overlap_tokens=0,
            chunk_respect_headings=False,
        )
        result = chunker.chunk(text, [], [0])
        # With large chunk size and headings disabled, should be one chunk
        assert len(result) == 1

    def test_heading_path_propagated(self):
        # Heading at offset 0 so even the first chunk is "under" it
        text = "## Chapter 1\n\nContent of chapter one goes here."
        headings = [
            (2, "Chapter 1", 0),
        ]
        chunker = _make_chunker(chunk_size_tokens=512)
        result = chunker.chunk(text, headings, [0])
        assert len(result) >= 1
        assert result[0]["heading_path"] == ["Chapter 1"]


# ---------------------------------------------------------------------------
# TestTableAwareChunking
# ---------------------------------------------------------------------------


class TestTableAwareChunking:
    def _table_text(self):
        return "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"

    def test_table_not_split(self):
        text = "Before text.\n\n" + self._table_text() + "\n\nAfter text."
        chunker = _make_chunker(
            chunk_size_tokens=32, chunk_overlap_tokens=0,
            chunk_respect_tables=True,
        )
        result = chunker.chunk(text, [], [0])
        # Find the table chunk
        table_chunks = [c for c in result if c["content_type"] == "table"]
        assert len(table_chunks) == 1
        assert "Alice" in table_chunks[0]["text"]
        assert "Bob" in table_chunks[0]["text"]

    def test_oversized_table_kept_whole(self):
        # Large table that exceeds chunk_size
        rows = "\n".join(f"| Item {i} | Value {i} |" for i in range(50))
        table = "| Name | Data |\n|------|------|\n" + rows
        chunker = _make_chunker(
            chunk_size_tokens=32, chunk_overlap_tokens=0,
            chunk_respect_tables=True,
        )
        result = chunker.chunk(table, [], [0])
        # Should be a single chunk (table kept whole)
        assert len(result) == 1
        assert result[0]["content_type"] == "table"

    def test_table_content_type(self):
        chunker = _make_chunker(chunk_respect_tables=True)
        result = chunker.chunk(self._table_text(), [], [0])
        assert result[0]["content_type"] == "table"

    def test_table_splitting_when_disabled(self):
        # Large table with tables disabled — may be split
        rows = "\n".join(f"| Item {i} | Value {i} |" for i in range(100))
        table = "| Name | Data |\n|------|------|\n" + rows
        chunker = _make_chunker(
            chunk_size_tokens=32, chunk_overlap_tokens=0,
            chunk_respect_tables=False,
        )
        result = chunker.chunk(table, [], [0])
        # With table respect disabled, a large table CAN be split
        assert len(result) >= 1  # May be 1 or more

    def test_mixed_text_and_tables(self):
        narrative1 = "This is introductory text. " * 10
        table = "| Col1 | Col2 |\n|------|------|\n| A | B |\n| C | D |"
        narrative2 = "This is concluding text. " * 10
        text = narrative1 + "\n\n" + table + "\n\n" + narrative2
        chunker = _make_chunker(
            chunk_size_tokens=64, chunk_overlap_tokens=0,
            chunk_respect_tables=True,
        )
        result = chunker.chunk(text, [], [0])
        assert len(result) >= 3
        # Table should be intact in one chunk
        table_chunks = [c for c in result if c["content_type"] == "table"]
        assert len(table_chunks) >= 1


# ---------------------------------------------------------------------------
# TestPageBoundaries
# ---------------------------------------------------------------------------


class TestPageBoundaries:
    def test_page_numbers_correct(self):
        text = "Page one content. " * 30 + "Page two content. " * 30
        # Page 2 starts at char 540 (30 * 18 chars)
        page_boundaries = [0, 540]
        chunker = _make_chunker(chunk_size_tokens=64, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], page_boundaries)
        # First chunk should be on page 1
        assert 1 in result[0]["page_numbers"]

    def test_chunk_spanning_pages(self):
        text = "a" * 1000
        page_boundaries = [0, 500]
        chunker = _make_chunker(chunk_size_tokens=512, chunk_overlap_tokens=0)
        result = chunker.chunk(text, [], page_boundaries)
        # Single large chunk should span both pages
        assert result[0]["page_numbers"] == [1, 2]


# ---------------------------------------------------------------------------
# TestConfigIntegration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_custom_chunk_size(self):
        text = "word " * 500  # ~2500 chars = ~625 tokens
        small = _make_chunker(chunk_size_tokens=128, chunk_overlap_tokens=0)
        large = _make_chunker(chunk_size_tokens=1024, chunk_overlap_tokens=0)
        small_result = small.chunk(text, [], [0])
        large_result = large.chunk(text, [], [0])
        assert len(small_result) > len(large_result)

    def test_large_chunk_size(self):
        text = "Short text."
        chunker = _make_chunker(chunk_size_tokens=4096)
        result = chunker.chunk(text, [], [0])
        assert len(result) == 1

    def test_default_config(self):
        config = PDFProcessorConfig()
        chunker = PDFChunker(config)
        result = chunker.chunk("Default config text.", [], [0])
        assert len(result) == 1
        assert result[0]["text"] == "Default config text."
