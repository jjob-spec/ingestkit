"""Shared fixtures for ingestkit-pdf tests.

Provides MockLLMBackend, factory helpers for DocumentProfile / PageProfile,
and common pytest fixtures used across test modules.
"""

from __future__ import annotations

from typing import Any

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.inspector import PDFInspector
from ingestkit_pdf.models import (
    DocumentMetadata,
    DocumentProfile,
    ExtractionQuality,
    PageProfile,
    PageType,
)


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------


class MockLLMBackend:
    """Mock LLM backend for testing PDFLLMClassifier.

    Supports configurable responses via a list of return values or
    exceptions.  Each call to ``classify()`` pops the next item from
    the response queue, allowing tests to simulate retry sequences.
    """

    def __init__(
        self,
        responses: list[dict | Exception] | None = None,
    ) -> None:
        self._responses: list[dict | Exception] = list(responses or [])
        self.calls: list[dict] = []  # records all calls for assertion

    def classify(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout: float | None = None,
    ) -> dict:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise RuntimeError("MockLLMBackend: no more responses configured")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        raise NotImplementedError("generate() not used by PDFLLMClassifier")


# ---------------------------------------------------------------------------
# Factory Helpers
# ---------------------------------------------------------------------------


def _make_extraction_quality(**overrides: Any) -> ExtractionQuality:
    """Build an ExtractionQuality with sensible defaults."""
    defaults: dict[str, Any] = dict(
        printable_ratio=0.95,
        avg_words_per_page=300.0,
        pages_with_text=1,
        total_pages=1,
        extraction_method="pdfminer",
    )
    defaults.update(overrides)
    return ExtractionQuality(**defaults)


def _make_page_profile(**overrides: Any) -> PageProfile:
    """Build a PageProfile with sensible text-page defaults."""
    defaults: dict[str, Any] = dict(
        page_number=1,
        text_length=1500,
        word_count=300,
        image_count=0,
        image_coverage_ratio=0.0,
        table_count=0,
        font_count=3,
        font_names=["Arial", "Times", "Courier"],
        has_form_fields=False,
        is_multi_column=False,
        page_type=PageType.TEXT,
        extraction_quality=_make_extraction_quality(),
    )
    defaults.update(overrides)
    return PageProfile(**defaults)


def _make_document_profile(
    pages: list[PageProfile] | None = None,
    **overrides: Any,
) -> DocumentProfile:
    """Build a DocumentProfile from a list of PageProfiles."""
    if pages is None:
        pages = [_make_page_profile()]

    # Auto-compute page_type_distribution from pages
    distribution: dict[str, int] = {}
    for p in pages:
        key = p.page_type.value
        distribution[key] = distribution.get(key, 0) + 1

    defaults: dict[str, Any] = dict(
        file_path="/tmp/test.pdf",
        file_size_bytes=102400,
        page_count=len(pages),
        content_hash="a" * 64,
        metadata=DocumentMetadata(
            creator="TestApp",
            pdf_version="1.7",
            page_count=len(pages),
            file_size_bytes=102400,
        ),
        pages=pages,
        page_type_distribution=distribution,
        detected_languages=["en"],
        has_toc=False,
        toc_entries=None,
        overall_quality=_make_extraction_quality(
            pages_with_text=len(pages),
            total_pages=len(pages),
        ),
        security_warnings=[],
    )
    defaults.update(overrides)
    return DocumentProfile(**defaults)


def _valid_response(
    type_: str = "text_native",
    confidence: float = 0.85,
    reasoning: str = "Digital PDF with extractable text throughout.",
    page_types: list[dict[str, Any]] | None = None,
) -> dict:
    """Build a valid LLM response dict."""
    d: dict[str, Any] = {
        "type": type_,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    if page_types is not None:
        d["page_types"] = page_types
    return d


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pdf_config() -> PDFProcessorConfig:
    return PDFProcessorConfig()


@pytest.fixture()
def pdf_inspector(pdf_config: PDFProcessorConfig) -> PDFInspector:
    return PDFInspector(pdf_config)


@pytest.fixture()
def document_profile() -> DocumentProfile:
    return _make_document_profile()
