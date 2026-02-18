"""Unit tests for ingestkit_json.converter -- flattening and chunking."""

from __future__ import annotations

import pytest

from ingestkit_json.config import JSONProcessorConfig
from ingestkit_json.converter import chunk_text, flatten_json


# ---------------------------------------------------------------------------
# Flatten tests
# ---------------------------------------------------------------------------


class TestFlattenFlat:
    """Tests for flat object flattening."""

    def test_flat_object(self, default_config):
        result = flatten_json({"a": 1, "b": "hello"}, default_config)
        assert "a: 1" in result.lines
        assert "b: hello" in result.lines
        assert result.total_keys == 2
        assert not result.truncated

    def test_flat_booleans(self, default_config):
        result = flatten_json({"x": True, "y": False}, default_config)
        assert "x: true" in result.lines
        assert "y: false" in result.lines

    def test_flat_float(self, default_config):
        result = flatten_json({"pi": 3.14}, default_config)
        assert "pi: 3.14" in result.lines


class TestFlattenNested:
    """Tests for nested object flattening."""

    def test_nested_object(self, sample_json_nested, default_config):
        result = flatten_json(sample_json_nested, default_config)
        assert "user.name: Alice" in result.lines
        assert "user.address.city: Springfield" in result.lines
        assert "user.address.state: IL" in result.lines
        assert "active: true" in result.lines
        assert "score: 42.5" in result.lines

    def test_depth_tracked(self, sample_json_nested, default_config):
        result = flatten_json(sample_json_nested, default_config)
        assert result.max_depth >= 2  # user -> address -> city


class TestFlattenArrays:
    """Tests for array flattening."""

    def test_array_of_objects(self, default_config):
        data = {"items": [{"n": "A"}, {"n": "B"}]}
        result = flatten_json(data, default_config)
        assert "items[0].n: A" in result.lines
        assert "items[1].n: B" in result.lines

    def test_root_array(self, sample_json_array, default_config):
        result = flatten_json(sample_json_array, default_config)
        assert "[0].id: 1" in result.lines
        assert "[0].name: Widget" in result.lines
        assert "[1].id: 2" in result.lines
        assert "[1].name: Gadget" in result.lines

    def test_array_of_scalars(self, default_config):
        data = {"tags": ["a", "b", "c"]}
        result = flatten_json(data, default_config)
        assert "tags[0]: a" in result.lines
        assert "tags[1]: b" in result.lines
        assert "tags[2]: c" in result.lines


class TestFlattenScalar:
    """Tests for root scalar values."""

    def test_root_string(self, default_config):
        result = flatten_json("hello", default_config)
        assert result.lines == ["hello"]
        assert result.total_keys == 1

    def test_root_number(self, default_config):
        result = flatten_json(42, default_config)
        assert result.lines == ["42"]

    def test_root_bool(self, default_config):
        result = flatten_json(True, default_config)
        assert result.lines == ["true"]

    def test_root_null(self, default_config):
        result = flatten_json(None, default_config)
        assert result.lines == ["null"]


class TestFlattenNulls:
    """Tests for null value handling."""

    def test_null_excluded_by_default(self, default_config):
        result = flatten_json({"a": 1, "b": None}, default_config)
        assert len(result.lines) == 1
        assert "a: 1" in result.lines

    def test_null_included_when_configured(self):
        config = JSONProcessorConfig(include_null_values=True)
        result = flatten_json({"a": 1, "b": None}, config)
        assert "a: 1" in result.lines
        assert "b: null" in result.lines
        assert result.total_keys == 2


class TestFlattenEmpty:
    """Tests for empty structures."""

    def test_empty_object(self, default_config):
        result = flatten_json({}, default_config)
        assert result.lines == []
        assert result.total_keys == 0

    def test_empty_array(self, default_config):
        result = flatten_json([], default_config)
        assert result.lines == []
        assert result.total_keys == 0


class TestFlattenLimits:
    """Tests for depth and key limits."""

    def test_nesting_depth_limit(self):
        config = JSONProcessorConfig(max_nesting_depth=3)
        # Build deeply nested structure
        data: dict = {"a": 1}
        current = data
        for i in range(10):
            current[f"level{i}"] = {}
            current = current[f"level{i}"]
        current["deep"] = "value"

        result = flatten_json(data, config)
        assert result.truncated

    def test_key_count_limit(self):
        config = JSONProcessorConfig(max_keys=5)
        data = {f"key_{i}": i for i in range(20)}
        result = flatten_json(data, config)
        assert result.truncated
        assert len(result.lines) == 5


class TestFlattenCustomSeparator:
    """Tests for custom path separator."""

    def test_slash_separator(self):
        config = JSONProcessorConfig(path_separator="/")
        data = {"user": {"name": "Jo"}}
        result = flatten_json(data, config)
        assert "user/name: Jo" in result.lines


class TestFlattenMixedTypes:
    """Tests for mixed value types."""

    def test_mixed(self, default_config):
        data = {"s": "text", "i": 42, "f": 3.14, "b": True, "n": None}
        result = flatten_json(data, default_config)
        assert "s: text" in result.lines
        assert "i: 42" in result.lines
        assert "f: 3.14" in result.lines
        assert "b: true" in result.lines
        # null excluded by default
        assert result.total_keys == 4


# ---------------------------------------------------------------------------
# Chunk tests
# ---------------------------------------------------------------------------


class TestChunkText:
    """Tests for chunk_text function."""

    def test_empty_lines(self, default_config):
        assert chunk_text([], default_config) == []

    def test_single_chunk(self, default_config):
        lines = ["a: 1", "b: 2"]
        chunks = chunk_text(lines, default_config)
        assert len(chunks) == 1
        assert "a: 1" in chunks[0]
        assert "b: 2" in chunks[0]

    def test_multiple_chunks(self):
        config = JSONProcessorConfig(chunk_size_tokens=3, chunk_overlap_tokens=0)
        # Each line is ~2 tokens: "key: value"
        lines = [f"key{i}: value{i}" for i in range(10)]
        chunks = chunk_text(lines, config)
        assert len(chunks) > 1

    def test_overlap(self):
        config = JSONProcessorConfig(chunk_size_tokens=4, chunk_overlap_tokens=2)
        lines = ["a: 1", "b: 2", "c: 3", "d: 4", "e: 5", "f: 6"]
        chunks = chunk_text(lines, config)
        # Chunks should have overlapping lines
        assert len(chunks) > 1
        # Check that some content from end of first chunk appears in start of second
        if len(chunks) >= 2:
            first_lines = chunks[0].split("\n")
            second_lines = chunks[1].split("\n")
            # At least one line from end of first chunk should be in start of second
            overlap_found = any(
                line in second_lines for line in first_lines[-2:]
            )
            assert overlap_found

    def test_single_large_line(self):
        config = JSONProcessorConfig(chunk_size_tokens=3, chunk_overlap_tokens=0)
        lines = ["this is a very long line with many words that exceeds the chunk size limit"]
        chunks = chunk_text(lines, config)
        assert len(chunks) == 1
        assert lines[0] in chunks[0]

    def test_preserves_complete_lines(self):
        config = JSONProcessorConfig(chunk_size_tokens=5, chunk_overlap_tokens=0)
        lines = ["key1: val1", "key2: val2", "key3: val3"]
        chunks = chunk_text(lines, config)
        # Each chunk should contain complete lines only
        for chunk in chunks:
            for line in chunk.split("\n"):
                assert line in lines
