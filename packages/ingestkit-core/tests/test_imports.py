"""Tests for import paths, re-exports, and circular import prevention."""

from __future__ import annotations

import sys


class TestCoreImportable:
    """Core package is importable and exports expected symbols."""

    def test_import_package(self):
        import ingestkit_core
        assert ingestkit_core is not None

    def test_import_protocols(self):
        from ingestkit_core.protocols import (
            VectorStoreBackend,
            StructuredDBBackend,
            LLMBackend,
            EmbeddingBackend,
        )
        assert VectorStoreBackend is not None

    def test_import_models(self):
        from ingestkit_core.models import (
            IngestKey,
            WrittenArtifacts,
            EmbedStageResult,
        )
        assert IngestKey is not None

    def test_import_errors(self):
        from ingestkit_core.errors import CoreErrorCode, BaseIngestError
        assert CoreErrorCode is not None

    def test_all_exports(self):
        import ingestkit_core
        expected = {
            "VectorStoreBackend",
            "StructuredDBBackend",
            "LLMBackend",
            "EmbeddingBackend",
            "IngestKey",
            "WrittenArtifacts",
            "EmbedStageResult",
            "CoreErrorCode",
            "BaseIngestError",
            "ClassificationTier",
            "BaseChunkMetadata",
            "ChunkPayload",
            "compute_ingest_key",
        }
        actual = set(ingestkit_core.__all__) if hasattr(ingestkit_core, "__all__") else set()
        assert expected.issubset(actual), f"Missing exports: {expected - actual}"


class TestReExportsExcel:
    """After migration, old import paths in ingestkit_excel still work."""

    def test_protocols_re_exported(self):
        from ingestkit_excel.protocols import VectorStoreBackend
        from ingestkit_core.protocols import VectorStoreBackend as CoreVSB
        assert VectorStoreBackend is CoreVSB

    def test_ingest_key_re_exported(self):
        from ingestkit_excel.models import IngestKey
        from ingestkit_core.models import IngestKey as CoreIK
        assert IngestKey is CoreIK

    def test_written_artifacts_re_exported(self):
        from ingestkit_excel.models import WrittenArtifacts
        from ingestkit_core.models import WrittenArtifacts as CoreWA
        assert WrittenArtifacts is CoreWA

    def test_embed_stage_result_re_exported(self):
        from ingestkit_excel.models import EmbedStageResult
        from ingestkit_core.models import EmbedStageResult as CoreESR
        assert EmbedStageResult is CoreESR


class TestReExportsPDF:
    """After migration, old import paths in ingestkit_pdf still work."""

    def test_protocols_re_exported(self):
        from ingestkit_pdf.protocols import VectorStoreBackend
        from ingestkit_core.protocols import VectorStoreBackend as CoreVSB
        assert VectorStoreBackend is CoreVSB

    def test_ingest_key_re_exported(self):
        from ingestkit_pdf.models import IngestKey
        from ingestkit_core.models import IngestKey as CoreIK
        assert IngestKey is CoreIK

    def test_written_artifacts_re_exported(self):
        from ingestkit_pdf.models import WrittenArtifacts
        from ingestkit_core.models import WrittenArtifacts as CoreWA
        assert WrittenArtifacts is CoreWA

    def test_embed_stage_result_re_exported(self):
        from ingestkit_pdf.models import EmbedStageResult
        from ingestkit_core.models import EmbedStageResult as CoreESR
        assert EmbedStageResult is CoreESR


class TestNoCircularImports:
    """Importing core must never transitively import excel or pdf."""

    def test_core_does_not_import_excel(self):
        # Clear caches to get clean state
        for mod_name in list(sys.modules):
            if mod_name.startswith("ingestkit_"):
                del sys.modules[mod_name]

        import ingestkit_core  # noqa: F811
        loaded = [m for m in sys.modules if m.startswith("ingestkit_excel")]
        assert len(loaded) == 0, f"Core imported excel modules: {loaded}"

    def test_core_does_not_import_pdf(self):
        for mod_name in list(sys.modules):
            if mod_name.startswith("ingestkit_"):
                del sys.modules[mod_name]

        import ingestkit_core  # noqa: F811
        loaded = [m for m in sys.modules if m.startswith("ingestkit_pdf")]
        assert len(loaded) == 0, f"Core imported pdf modules: {loaded}"
