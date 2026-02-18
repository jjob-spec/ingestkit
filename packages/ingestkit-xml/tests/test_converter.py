"""Unit tests for ingestkit_xml.converter -- extraction and chunking."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.converter import chunk_text, extract_xml


# ======================================================================
# extract_xml tests
# ======================================================================


class TestExtractSimple:
    """Tests for simple element extraction."""

    def test_simple_text_content(self, default_config):
        root = ET.fromstring("<root><title>Hello</title></root>")
        result = extract_xml(root, default_config)
        assert any("Hello" in line for line in result.lines)

    def test_nested_elements(self, default_config):
        root = ET.fromstring(
            "<root><parent><child>Content</child></parent></root>"
        )
        result = extract_xml(root, default_config)
        assert any("Content" in line for line in result.lines)
        assert result.total_elements == 3  # root, parent, child

    def test_empty_elements_produce_no_lines(self, default_config):
        root = ET.fromstring("<root><empty/></root>")
        result = extract_xml(root, default_config)
        # Neither root nor empty have text, so no lines
        assert len(result.lines) == 0

    def test_root_tag_captured(self, default_config):
        root = ET.fromstring("<document><item>x</item></document>")
        result = extract_xml(root, default_config)
        assert result.root_tag == "document"

    def test_depth_tracking(self, default_config):
        root = ET.fromstring(
            "<a><b><c><d>deep</d></c></b></a>"
        )
        result = extract_xml(root, default_config)
        assert result.max_depth == 3  # a=0, b=1, c=2, d=3


class TestExtractNamespaces:
    """Tests for namespace handling."""

    def test_namespace_stripping(self, default_config):
        root = ET.fromstring(
            '<root xmlns:ns="http://example.com">'
            '<ns:item>Namespaced</ns:item>'
            '</root>'
        )
        # ElementTree resolves ns:item to {http://example.com}item
        result = extract_xml(root, default_config)
        # With strip_namespaces=True, tag should be "item" not "{uri}item"
        assert any("item" in line and "Namespaced" in line for line in result.lines)
        assert not any("{http://example.com}" in line for line in result.lines)

    def test_namespace_stripping_disabled(self):
        config = XMLProcessorConfig(strip_namespaces=False)
        root = ET.fromstring(
            '<root xmlns="http://example.com"><item>test</item></root>'
        )
        result = extract_xml(root, config)
        assert any("{http://example.com}" in line for line in result.lines)

    def test_namespaces_collected(self, default_config):
        root = ET.fromstring(
            '<root xmlns:ns="http://example.com"><ns:item>x</ns:item></root>'
        )
        result = extract_xml(root, default_config)
        assert "http://example.com" in result.namespaces


class TestExtractAttributes:
    """Tests for attribute handling."""

    def test_attributes_included(self, default_config):
        root = ET.fromstring('<root><item id="1" type="widget">text</item></root>')
        result = extract_xml(root, default_config)
        assert any("id=1" in line for line in result.lines)
        assert any("type=widget" in line for line in result.lines)

    def test_xmlns_attributes_skipped(self, default_config):
        # xmlns:* declarations are handled by ElementTree as namespace bindings
        # and don't appear as regular attributes. Test with explicit xmlns-prefixed
        # attributes that would appear in the attrib dict.
        root = ET.fromstring(
            '<root>'
            '<item id="1">data</item>'
            '</root>'
        )
        # Manually add an attribute with xmlns prefix to test skip logic
        root[0].set("xmlns:foo", "http://example.com")
        result = extract_xml(root, default_config)
        # The item line should contain id but not xmlns:foo
        for line in result.lines:
            if "data" in line:
                assert "id=1" in line
                assert "xmlns" not in line

    def test_attributes_disabled(self):
        config = XMLProcessorConfig(include_attributes=False)
        root = ET.fromstring('<root><item id="1">text</item></root>')
        result = extract_xml(root, config)
        assert not any("id=1" in line for line in result.lines)


class TestExtractTailText:
    """Tests for tail text handling."""

    def test_tail_text_included(self, default_config):
        root = ET.fromstring("<root><b>bold</b> and normal</root>")
        result = extract_xml(root, default_config)
        assert any("and normal" in line for line in result.lines)

    def test_tail_text_excluded(self):
        config = XMLProcessorConfig(include_tail_text=False)
        root = ET.fromstring("<root><b>bold</b> and normal</root>")
        result = extract_xml(root, config)
        assert not any("and normal" in line for line in result.lines)


class TestExtractMixedContent:
    """Tests for mixed content (text + children + tail)."""

    def test_mixed_content(self, default_config):
        root = ET.fromstring(
            "<root>Preamble <child>inner</child> postlude</root>"
        )
        result = extract_xml(root, default_config)
        assert any("Preamble" in line for line in result.lines)
        assert any("inner" in line for line in result.lines)
        assert any("postlude" in line for line in result.lines)


class TestExtractElementLimit:
    """Tests for element count limit."""

    def test_element_count_limit_triggers_truncation(self):
        config = XMLProcessorConfig(max_elements=2)
        root = ET.fromstring("<root><a>1</a><b>2</b><c>3</c></root>")
        result = extract_xml(root, config)
        assert result.truncated is True
        # Should have processed some but not all
        assert result.total_elements <= 3


class TestExtractIndentation:
    """Tests for indentation option."""

    def test_indent_text_adds_spaces(self):
        config = XMLProcessorConfig(indent_text=True)
        root = ET.fromstring("<root><child>text</child></root>")
        result = extract_xml(root, config)
        # child is at depth 1, so should have 2 spaces indent
        assert any(line.startswith("  ") for line in result.lines)


# ======================================================================
# chunk_text tests
# ======================================================================


class TestChunkText:
    """Tests for chunk_text function."""

    def test_empty_lines(self, default_config):
        assert chunk_text([], default_config) == []

    def test_single_chunk_when_all_fit(self, default_config):
        lines = ["line one", "line two", "line three"]
        chunks = chunk_text(lines, default_config)
        assert len(chunks) == 1
        assert "line one" in chunks[0]

    def test_multiple_chunks(self):
        config = XMLProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        # Each line has ~3 tokens
        lines = ["word one two", "three four five", "six seven eight", "nine ten eleven"]
        chunks = chunk_text(lines, config)
        assert len(chunks) > 1

    def test_overlap_between_chunks(self):
        config = XMLProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=3)
        lines = ["word one two", "three four five", "six seven eight"]
        chunks = chunk_text(lines, config)
        if len(chunks) > 1:
            # Last line of first chunk should appear in second chunk
            first_lines = chunks[0].split("\n")
            second_chunk = chunks[1]
            # At least one line from end of first chunk should be in second
            assert any(line in second_chunk for line in first_lines[-2:])

    def test_single_oversized_line(self):
        config = XMLProcessorConfig(chunk_size_tokens=3, chunk_overlap_tokens=0)
        lines = ["this is a very long line with many words exceeding the limit"]
        chunks = chunk_text(lines, config)
        assert len(chunks) == 1  # Emitted as a single chunk

    def test_lines_never_split(self):
        config = XMLProcessorConfig(chunk_size_tokens=10, chunk_overlap_tokens=0)
        lines = ["line one two three", "line four five six"]
        chunks = chunk_text(lines, config)
        for chunk in chunks:
            for original_line in lines:
                # If the original line appears partially, it should appear fully
                words = original_line.split()
                if words[0] in chunk:
                    assert original_line in chunk or original_line not in "\n".join(lines)
