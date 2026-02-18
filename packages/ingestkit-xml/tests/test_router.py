"""Unit tests for ingestkit_xml.router -- orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.errors import ErrorCode
from ingestkit_xml.router import XMLRouter


class TestCanHandle:
    """Tests for can_handle method."""

    def test_xml_extension(self, mock_vector_store, mock_embedder):
        router = XMLRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data.xml") is True

    def test_xml_uppercase(self, mock_vector_store, mock_embedder):
        router = XMLRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data.XML") is True

    def test_json_extension(self, mock_vector_store, mock_embedder):
        router = XMLRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data.json") is False

    def test_no_extension(self, mock_vector_store, mock_embedder):
        router = XMLRouter(mock_vector_store, mock_embedder)
        assert router.can_handle("data") is False


class TestProcessHappyPath:
    """Tests for process() on valid XML files."""

    def test_simple_xml(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><title>Test</title><body>Content here</body></root>")

        result = router.process(fp)

        assert result.chunks_created > 0
        assert result.ingest_key != ""
        assert len(result.errors) == 0
        mock_vector_store.upsert_chunks.assert_called_once()
        mock_embedder.embed.assert_called_once()

    def test_ingest_key_is_64_char_hex(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp)

        assert len(result.ingest_key) == 64

    def test_tenant_id_propagates(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        config = XMLProcessorConfig(tenant_id="tenant-42")
        router = XMLRouter(mock_vector_store, mock_embedder, config=config)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp)

        assert result.tenant_id == "tenant-42"

    def test_source_uri_override(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp, source_uri="s3://bucket/data.xml")

        assert result.ingest_key != ""

    def test_embed_result_populated(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp)

        assert result.embed_result is not None
        assert result.embed_result.texts_embedded > 0

    def test_written_artifacts(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp)

        assert len(result.written.vector_point_ids) > 0
        assert result.written.vector_collection == "helpdesk"


class TestProcessSecurityFailure:
    """Tests for process() when security scan fails."""

    def test_invalid_xml_file(self, mock_vector_store, mock_embedder, tmp_path):
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_path / "bad.xml"
        fp.write_text("<root><unclosed>")

        result = router.process(str(fp))

        assert result.chunks_created == 0
        assert len(result.errors) > 0
        assert any("INVALID_XML" in e for e in result.errors)

    def test_empty_file(self, mock_vector_store, mock_embedder, tmp_path):
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_path / "empty.xml"
        fp.write_text("")

        result = router.process(str(fp))

        assert result.chunks_created == 0
        assert any("EMPTY" in e for e in result.errors)

    def test_entity_declaration(self, mock_vector_store, mock_embedder, tmp_path):
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_path / "entity.xml"
        fp.write_text(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE foo [\n'
            '  <!ENTITY xxe "bad">\n'
            ']>\n'
            '<root>&xxe;</root>'
        )

        result = router.process(str(fp))

        assert result.chunks_created == 0
        assert any("ENTITY" in e for e in result.errors)


class TestProcessMalformedFallback:
    """Tests for plain-text fallback on malformed XML."""

    def test_malformed_fallback(self, mock_vector_store, mock_embedder, tmp_xml_file):
        """XML that passes security scan but fails ET.parse should fall back to text.

        We mock the security scanner to return no errors, then provide content
        that will fail ET.parse, triggering the plain-text fallback in the router.
        """
        from unittest.mock import patch

        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("This is not XML at all\nJust plain text\nWith multiple lines")

        # Bypass security scanner so we reach the parse step in the router
        with patch.object(router._security_scanner, "scan", return_value=[]):
            result = router.process(fp)

        assert result.chunks_created > 0
        assert ErrorCode.W_MALFORMED_FALLBACK.value in result.warnings


class TestProcessBackendFailure:
    """Tests for process() when backend fails."""

    def test_embed_timeout(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.side_effect = TimeoutError("timeout")
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp)

        assert result.chunks_created == 0
        assert any("EMBED" in e for e in result.errors)

    def test_vector_store_failure(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        mock_vector_store.ensure_collection.side_effect = ConnectionError("refused")
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = router.process(fp)

        assert result.chunks_created == 0
        assert any("VECTOR" in e for e in result.errors)


class TestAprocess:
    """Tests for async aprocess() method."""

    @pytest.mark.asyncio
    async def test_aprocess_returns_result(self, mock_vector_store, mock_embedder, tmp_xml_file):
        mock_embedder.embed.return_value = [[0.1] * 768]
        router = XMLRouter(mock_vector_store, mock_embedder)
        fp = tmp_xml_file("<root><item>data</item></root>")

        result = await router.aprocess(fp)

        assert result.chunks_created > 0
        assert result.ingest_key != ""
