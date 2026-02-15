"""Tests for idempotency key computation (spec section 4.3)."""

from __future__ import annotations

import hashlib
import re
import uuid

import pytest

from ingestkit_forms.idempotency import (
    IngestKey,
    compute_form_extraction_key,
    compute_ingest_key,
    compute_vector_point_id,
)

PARSER_VERSION = "ingestkit_forms:1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(tmp_path, name: str, content: bytes) -> str:
    """Write *content* to a file under *tmp_path* and return its string path."""
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


# ---------------------------------------------------------------------------
# TestComputeIngestKey -- delegates to ingestkit_core
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeIngestKey:
    """Global ingest key must be deterministic and delegate to core."""

    def test_same_file_produces_same_key(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"hello world")
        key1 = compute_ingest_key(path, PARSER_VERSION)
        key2 = compute_ingest_key(path, PARSER_VERSION)
        assert key1.key == key2.key

    def test_different_content_produces_different_key(self, tmp_path):
        path_a = _write_file(tmp_path, "a.pdf", b"content-v1")
        path_b = _write_file(tmp_path, "b.pdf", b"content-v2")
        key_a = compute_ingest_key(path_a, PARSER_VERSION)
        key_b = compute_ingest_key(path_b, PARSER_VERSION)
        assert key_a.content_hash != key_b.content_hash

    def test_different_parser_version_produces_different_key(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"same content")
        key_v1 = compute_ingest_key(path, "ingestkit_forms:1.0.0")
        key_v2 = compute_ingest_key(path, "ingestkit_forms:2.0.0")
        assert key_v1.content_hash == key_v2.content_hash
        assert key_v1.key != key_v2.key

    def test_tenant_id_affects_key(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"tenant test")
        key_no = compute_ingest_key(path, PARSER_VERSION)
        key_yes = compute_ingest_key(path, PARSER_VERSION, tenant_id="acme")
        assert key_no.key != key_yes.key

    def test_source_uri_override(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"uri test")
        uri = "s3://bucket/form.pdf"
        key = compute_ingest_key(path, PARSER_VERSION, source_uri=uri)
        assert key.source_uri == uri

    def test_returns_ingest_key_instance(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"type test")
        result = compute_ingest_key(path, PARSER_VERSION)
        assert isinstance(result, IngestKey)

    def test_key_is_64_char_hex(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"hex test")
        key = compute_ingest_key(path, PARSER_VERSION)
        assert len(key.key) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", key.key)

    def test_file_not_found_raises(self, tmp_path):
        bogus = str(tmp_path / "does_not_exist.pdf")
        with pytest.raises(FileNotFoundError):
            compute_ingest_key(bogus, PARSER_VERSION)


# ---------------------------------------------------------------------------
# TestComputeFormExtractionKey
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeFormExtractionKey:
    """Form extraction key = sha256(global_key | template_id | version)."""

    GLOBAL_KEY = "a" * 64  # Fake 64-char hex global key
    TEMPLATE_ID = "template-safety-checklist"
    TEMPLATE_VERSION = 1

    def test_deterministic_same_inputs(self):
        k1 = compute_form_extraction_key(
            self.GLOBAL_KEY, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        k2 = compute_form_extraction_key(
            self.GLOBAL_KEY, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        assert k1 == k2

    def test_different_ingest_key_produces_different_result(self):
        k1 = compute_form_extraction_key(
            "a" * 64, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        k2 = compute_form_extraction_key(
            "b" * 64, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        assert k1 != k2

    def test_different_template_id_produces_different_result(self):
        k1 = compute_form_extraction_key(
            self.GLOBAL_KEY, "template-alpha", self.TEMPLATE_VERSION
        )
        k2 = compute_form_extraction_key(
            self.GLOBAL_KEY, "template-beta", self.TEMPLATE_VERSION
        )
        assert k1 != k2

    def test_different_template_version_produces_different_result(self):
        k1 = compute_form_extraction_key(
            self.GLOBAL_KEY, self.TEMPLATE_ID, 1
        )
        k2 = compute_form_extraction_key(
            self.GLOBAL_KEY, self.TEMPLATE_ID, 2
        )
        assert k1 != k2

    def test_result_is_64_char_hex(self):
        result = compute_form_extraction_key(
            self.GLOBAL_KEY, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        assert len(result) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", result)

    def test_matches_expected_formula(self):
        """Verify the output matches manual sha256 computation."""
        payload = f"{self.GLOBAL_KEY}|{self.TEMPLATE_ID}|{self.TEMPLATE_VERSION}"
        expected = hashlib.sha256(payload.encode()).hexdigest()
        result = compute_form_extraction_key(
            self.GLOBAL_KEY, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        assert result == expected

    def test_re_extraction_new_version_new_key(self):
        """Same document + same template but bumped version -> new key."""
        k_v1 = compute_form_extraction_key(self.GLOBAL_KEY, self.TEMPLATE_ID, 1)
        k_v2 = compute_form_extraction_key(self.GLOBAL_KEY, self.TEMPLATE_ID, 2)
        assert k_v1 != k_v2


# ---------------------------------------------------------------------------
# TestComputeVectorPointId
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeVectorPointId:
    """Vector point ID = uuid5(NAMESPACE_URL, extraction_key : chunk_index)."""

    EXTRACTION_KEY = "c" * 64

    def test_deterministic_same_inputs(self):
        id1 = compute_vector_point_id(self.EXTRACTION_KEY, 0)
        id2 = compute_vector_point_id(self.EXTRACTION_KEY, 0)
        assert id1 == id2

    def test_different_form_extraction_key_produces_different_id(self):
        id1 = compute_vector_point_id("a" * 64, 0)
        id2 = compute_vector_point_id("b" * 64, 0)
        assert id1 != id2

    def test_different_chunk_index_produces_different_id(self):
        id1 = compute_vector_point_id(self.EXTRACTION_KEY, 0)
        id2 = compute_vector_point_id(self.EXTRACTION_KEY, 1)
        assert id1 != id2

    def test_result_is_valid_uuid(self):
        result = compute_vector_point_id(self.EXTRACTION_KEY, 0)
        parsed = uuid.UUID(result)
        assert parsed.version == 5

    def test_sequential_chunks_produce_unique_ids(self):
        ids = [compute_vector_point_id(self.EXTRACTION_KEY, i) for i in range(100)]
        assert len(set(ids)) == 100

    def test_matches_expected_formula(self):
        """Verify the output matches manual uuid5 computation."""
        name = f"{self.EXTRACTION_KEY}:7"
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
        result = compute_vector_point_id(self.EXTRACTION_KEY, 7)
        assert result == expected


# ---------------------------------------------------------------------------
# TestEndToEndIdempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEndToEndIdempotency:
    """Full pipeline: file -> global key -> extraction key -> point IDs."""

    TEMPLATE_ID = "template-end-to-end"
    TEMPLATE_VERSION = 3

    def test_full_pipeline_determinism(self, tmp_path):
        path = _write_file(tmp_path, "form.pdf", b"end-to-end content")

        # Run the full chain twice
        for _ in range(2):
            ik = compute_ingest_key(path, PARSER_VERSION)
            fek = compute_form_extraction_key(
                ik.key, self.TEMPLATE_ID, self.TEMPLATE_VERSION
            )
            point_ids = [compute_vector_point_id(fek, i) for i in range(5)]

        # Re-run and compare
        ik2 = compute_ingest_key(path, PARSER_VERSION)
        fek2 = compute_form_extraction_key(
            ik2.key, self.TEMPLATE_ID, self.TEMPLATE_VERSION
        )
        point_ids2 = [compute_vector_point_id(fek2, i) for i in range(5)]

        assert ik.key == ik2.key
        assert fek == fek2
        assert point_ids == point_ids2

    def test_template_version_change_changes_downstream_keys(self, tmp_path):
        """Same file, different template version -> different extraction key
        and point IDs but same global ingest key."""
        path = _write_file(tmp_path, "form.pdf", b"version change test")

        ik = compute_ingest_key(path, PARSER_VERSION)

        fek_v1 = compute_form_extraction_key(ik.key, self.TEMPLATE_ID, 1)
        fek_v2 = compute_form_extraction_key(ik.key, self.TEMPLATE_ID, 2)

        # Global key unchanged
        assert fek_v1 != fek_v2

        # Downstream point IDs also differ
        pid_v1 = compute_vector_point_id(fek_v1, 0)
        pid_v2 = compute_vector_point_id(fek_v2, 0)
        assert pid_v1 != pid_v2
