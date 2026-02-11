"""Tests for deterministic ingest-key computation (idempotency module)."""

from __future__ import annotations

import hashlib
import re

import pytest

from ingestkit_excel.idempotency import compute_ingest_key
from ingestkit_excel.models import IngestKey

PARSER_VERSION = "ingestkit_excel:1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(tmp_path, name: str, content: bytes) -> str:
    """Write *content* to a file under *tmp_path* and return its string path."""
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


# ---------------------------------------------------------------------------
# Identical files -> identical keys
# ---------------------------------------------------------------------------


class TestIdenticalFilesProduceIdenticalKeys:
    """Same content + same parser_version must always yield the same key."""

    def test_same_file_read_twice(self, tmp_path):
        path = _write_file(tmp_path, "a.xlsx", b"hello world")
        key1 = compute_ingest_key(path, PARSER_VERSION)
        key2 = compute_ingest_key(path, PARSER_VERSION)
        assert key1.key == key2.key

    def test_two_files_with_identical_bytes(self, tmp_path):
        content = b"\x00\x01\x02\x03"
        path_a = _write_file(tmp_path, "a.bin", content)
        path_b = _write_file(tmp_path, "b.bin", content)
        key_a = compute_ingest_key(path_a, PARSER_VERSION)
        key_b = compute_ingest_key(path_b, PARSER_VERSION)
        # content_hash must match
        assert key_a.content_hash == key_b.content_hash
        # composite keys differ because source_uri differs
        assert key_a.key != key_b.key

    def test_content_hash_is_sha256_of_bytes(self, tmp_path):
        content = b"deterministic content"
        path = _write_file(tmp_path, "file.bin", content)
        key = compute_ingest_key(path, PARSER_VERSION)
        expected = hashlib.sha256(content).hexdigest()
        assert key.content_hash == expected


# ---------------------------------------------------------------------------
# Modified files -> different keys
# ---------------------------------------------------------------------------


class TestModifiedFilesProduceDifferentKeys:
    """Any change to file content must change the key."""

    def test_one_byte_difference(self, tmp_path):
        path_a = _write_file(tmp_path, "a.bin", b"content-v1")
        path_b = _write_file(tmp_path, "b.bin", b"content-v2")
        key_a = compute_ingest_key(path_a, PARSER_VERSION)
        key_b = compute_ingest_key(path_b, PARSER_VERSION)
        assert key_a.content_hash != key_b.content_hash
        assert key_a.key != key_b.key

    def test_empty_vs_nonempty(self, tmp_path):
        path_empty = _write_file(tmp_path, "empty.bin", b"")
        path_full = _write_file(tmp_path, "full.bin", b"data")
        key_empty = compute_ingest_key(path_empty, PARSER_VERSION)
        key_full = compute_ingest_key(path_full, PARSER_VERSION)
        assert key_empty.content_hash != key_full.content_hash


# ---------------------------------------------------------------------------
# Different parser_version -> different keys
# ---------------------------------------------------------------------------


class TestDifferentParserVersionProducesDifferentKeys:
    """Same file but different parser_version must produce a different key."""

    def test_version_bump(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"same content")
        key_v1 = compute_ingest_key(path, "ingestkit_excel:1.0.0")
        key_v2 = compute_ingest_key(path, "ingestkit_excel:2.0.0")
        assert key_v1.content_hash == key_v2.content_hash
        assert key_v1.key != key_v2.key


# ---------------------------------------------------------------------------
# tenant_id present vs absent -> different keys
# ---------------------------------------------------------------------------


class TestTenantIdAffectsKey:
    """Presence or absence of tenant_id must change the composite key."""

    def test_with_and_without_tenant(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"tenant test")
        key_no_tenant = compute_ingest_key(path, PARSER_VERSION)
        key_with_tenant = compute_ingest_key(
            path, PARSER_VERSION, tenant_id="acme-corp"
        )
        assert key_no_tenant.key != key_with_tenant.key

    def test_different_tenants(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"tenant test")
        key_a = compute_ingest_key(path, PARSER_VERSION, tenant_id="tenant-a")
        key_b = compute_ingest_key(path, PARSER_VERSION, tenant_id="tenant-b")
        assert key_a.key != key_b.key


# ---------------------------------------------------------------------------
# source_uri override
# ---------------------------------------------------------------------------


class TestSourceUriOverride:
    """When source_uri is provided, it must be used verbatim."""

    def test_override_uses_given_value(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"uri test")
        custom_uri = "s3://my-bucket/data/file.xlsx"
        key = compute_ingest_key(
            path, PARSER_VERSION, source_uri=custom_uri
        )
        assert key.source_uri == custom_uri

    def test_default_source_uri_is_absolute_posix(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"posix test")
        key = compute_ingest_key(path, PARSER_VERSION)
        # Must be an absolute POSIX path (starts with /)
        assert key.source_uri.startswith("/")
        # Must contain the filename
        assert "file.xlsx" in key.source_uri

    def test_override_changes_key(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"uri key test")
        key_default = compute_ingest_key(path, PARSER_VERSION)
        key_custom = compute_ingest_key(
            path, PARSER_VERSION, source_uri="gs://bucket/file.xlsx"
        )
        assert key_default.key != key_custom.key


# ---------------------------------------------------------------------------
# IngestKey.key is a hex string
# ---------------------------------------------------------------------------


class TestIngestKeyIsHexString:
    """The composite key must be a 64-character lowercase hex string."""

    def test_key_is_64_char_hex(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"hex test")
        key = compute_ingest_key(path, PARSER_VERSION)
        assert len(key.key) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", key.key)

    def test_content_hash_is_64_char_hex(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"hash test")
        key = compute_ingest_key(path, PARSER_VERSION)
        assert len(key.content_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", key.content_hash)


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


class TestFileNotFoundRaisesError:
    """Missing file must raise FileNotFoundError."""

    def test_nonexistent_file(self, tmp_path):
        bogus = str(tmp_path / "does_not_exist.xlsx")
        with pytest.raises(FileNotFoundError):
            compute_ingest_key(bogus, PARSER_VERSION)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    """compute_ingest_key must return an IngestKey instance."""

    def test_returns_ingest_key_instance(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"type test")
        result = compute_ingest_key(path, PARSER_VERSION)
        assert isinstance(result, IngestKey)

    def test_fields_populated(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"fields test")
        result = compute_ingest_key(
            path,
            PARSER_VERSION,
            tenant_id="t1",
            source_uri="file:///custom",
        )
        assert result.content_hash
        assert result.source_uri == "file:///custom"
        assert result.parser_version == PARSER_VERSION
        assert result.tenant_id == "t1"

    def test_tenant_id_none_by_default(self, tmp_path):
        path = _write_file(tmp_path, "file.xlsx", b"default test")
        result = compute_ingest_key(path, PARSER_VERSION)
        assert result.tenant_id is None
