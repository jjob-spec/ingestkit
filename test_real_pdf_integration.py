"""Integration test: run all ingestkit-pdf modules against a real PDF."""
from __future__ import annotations

import sys
import traceback

import fitz  # PyMuPDF

PDF_PATH = "/home/jjob/project_data_files/sample-docs-for-rag/output/Marketing_Sales/Product_Catalog_2025.pdf"

# Add package to path
sys.path.insert(0, "packages/ingestkit-pdf/src")

from ingestkit_pdf.config import PDFProcessorConfig

config = PDFProcessorConfig()
doc = fitz.open(PDF_PATH)
print(f"PDF: {PDF_PATH}")
print(f"Pages: {len(doc)}")
print("=" * 70)

results = {}


def run_module(name: str, fn):
    print(f"\n{'─' * 70}")
    print(f"Module: {name}")
    print(f"{'─' * 70}")
    try:
        fn()
        results[name] = "PASS"
        print(f"  Result: PASS")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"  Result: FAIL")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 1. Language Detection
# ──────────────────────────────────────────────────────────────────────
def test_language():
    from ingestkit_pdf.utils.language import detect_language, map_language_to_ocr

    page = doc[0]
    text = page.get_text()
    lang, confidence = detect_language(text)
    print(f"  Language: {lang} (confidence: {confidence:.2f})")
    ocr_lang = map_language_to_ocr(lang, "tesseract")
    print(f"  OCR mapping (tesseract): {ocr_lang}")


run_module("Language Detection", test_language)


# ──────────────────────────────────────────────────────────────────────
# 2. Heading Detection
# ──────────────────────────────────────────────────────────────────────
def test_headings():
    from ingestkit_pdf.utils.heading_detector import HeadingDetector

    detector = HeadingDetector(config)
    headings = detector.detect(doc)
    print(f"  Headings found: {len(headings)}")
    for level, title, page_num in headings[:10]:
        print(f"    H{level}: '{title}' (page {page_num})")
    if len(headings) > 10:
        print(f"    ... and {len(headings) - 10} more")


run_module("Heading Detection", test_headings)


# ──────────────────────────────────────────────────────────────────────
# 3. Header/Footer Detection
# ──────────────────────────────────────────────────────────────────────
def test_header_footer():
    from ingestkit_pdf.utils.header_footer import HeaderFooterDetector

    detector = HeaderFooterDetector(config)
    headers, footers = detector.detect(doc)
    print(f"  Headers: {headers}")
    print(f"  Footers: {footers}")


run_module("Header/Footer Detection", test_header_footer)


# ──────────────────────────────────────────────────────────────────────
# 4. Layout Analysis
# ──────────────────────────────────────────────────────────────────────
def test_layout():
    from ingestkit_pdf.utils.layout_analysis import (
        LayoutAnalyzer,
        extract_text_blocks,
    )

    analyzer = LayoutAnalyzer(config)
    for i in range(min(5, len(doc))):
        page = doc[i]
        blocks = extract_text_blocks(page)
        layout = analyzer.detect_columns(page)
        print(
            f"  Page {i + 1}: {layout.column_count} column(s), "
            f"{len(blocks)} text blocks, "
            f"multi-column={layout.is_multi_column}"
        )
        if layout.column_boundaries:
            for j, (x0, x1) in enumerate(layout.column_boundaries):
                print(f"    Column {j + 1}: x={x0:.0f}..{x1:.0f}")


run_module("Layout Analysis", test_layout)


# ──────────────────────────────────────────────────────────────────────
# 5. Text Chunker
# ──────────────────────────────────────────────────────────────────────
def test_chunker():
    from ingestkit_pdf.utils.chunker import PDFChunker
    from ingestkit_pdf.utils.heading_detector import HeadingDetector

    # Get text from all pages
    full_text = ""
    page_boundaries = []
    for i in range(len(doc)):
        page_text = doc[i].get_text()
        page_boundaries.append(len(full_text))
        full_text += page_text

    # Get headings
    detector = HeadingDetector(config)
    headings = detector.detect(doc)

    chunker = PDFChunker(config)
    chunks = chunker.chunk(full_text, headings, page_boundaries)
    print(f"  Total text length: {len(full_text)} chars")
    print(f"  Chunks produced: {len(chunks)}")
    if chunks:
        for i, c in enumerate(chunks[:3]):
            preview = c.get("text", "")[:80].replace("\n", " ")
            print(f"    Chunk {i + 1}: {len(c.get('text', ''))} chars - '{preview}...'")
        if len(chunks) > 3:
            print(f"    ... and {len(chunks) - 3} more")


run_module("Text Chunker", test_chunker)


# ──────────────────────────────────────────────────────────────────────
# 6. OCR Postprocessing
# ──────────────────────────────────────────────────────────────────────
def test_ocr_postprocess():
    from ingestkit_pdf.utils.ocr_postprocess import postprocess_ocr_text

    # Use text from first page as sample input
    sample = doc[0].get_text()[:500]
    cleaned = postprocess_ocr_text(sample)
    print(f"  Input length: {len(sample)} chars")
    print(f"  Output length: {len(cleaned)} chars")
    print(f"  Sample: '{cleaned[:100]}...'")


run_module("OCR Postprocessing", test_ocr_postprocess)


# ──────────────────────────────────────────────────────────────────────
# 7. Page Renderer
# ──────────────────────────────────────────────────────────────────────
def test_page_renderer():
    from ingestkit_pdf.utils.page_renderer import PageRenderer

    renderer = PageRenderer(config)
    page = doc[0]
    img = renderer.render_page(page)
    print(f"  Rendered page 1: {img.size} ({img.mode})")
    preprocessed = renderer.preprocess(img)
    print(f"  Preprocessed: {preprocessed.size} ({preprocessed.mode})")


run_module("Page Renderer", test_page_renderer)


# ──────────────────────────────────────────────────────────────────────
# 8. OCR Engine
# ──────────────────────────────────────────────────────────────────────
def test_ocr_engine():
    from ingestkit_pdf.utils.ocr_engines import create_ocr_engine
    from ingestkit_pdf.utils.page_renderer import PageRenderer

    from ingestkit_pdf.utils.ocr_engines import EngineUnavailableError
    try:
        engine = create_ocr_engine(config)
        print(f"  Engine: {type(engine).__name__}")
        renderer = PageRenderer(config)
        img = renderer.render_page(doc[0])
        result = engine.recognise(img)
        print(f"  OCR text length: {len(result.text)} chars")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Preview: '{result.text[:100]}...'")
    except EngineUnavailableError as e:
        print(f"  Skipped (engine not installed): {e}")
        results["OCR Engine"] = "SKIP (not installed)"


run_module("OCR Engine", test_ocr_engine)


# ──────────────────────────────────────────────────────────────────────
# 9. Table Extractor
# ──────────────────────────────────────────────────────────────────────
def test_table_extractor():
    from ingestkit_pdf.processors.table_extractor import TableExtractor

    extractor = TableExtractor(config=config)
    page_numbers = list(range(1, len(doc) + 1))  # 1-indexed
    result = extractor.extract_tables(
        file_path=PDF_PATH,
        page_numbers=page_numbers,
        ingest_key="test-key-123",
        ingest_run_id="test-run-001",
    )
    print(f"  Tables found: {len(result.tables)}")
    print(f"  Chunks produced: {len(result.chunks)}")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"  Errors: {len(result.errors)}")
    for i, table in enumerate(result.tables):
        print(
            f"    Table {i + 1}: {table.row_count} rows x {table.col_count} cols "
            f"(page {table.page_number})"
        )


run_module("Table Extractor", test_table_extractor)


# ──────────────────────────────────────────────────────────────────────
# 10. Inspector (classification)
# ──────────────────────────────────────────────────────────────────────
def test_inspector():
    from ingestkit_pdf.inspector import PDFInspector
    from ingestkit_pdf.models import (
        DocumentMetadata,
        DocumentProfile,
        ExtractionQuality,
        PageProfile,
        PageType,
    )
    import hashlib

    # Build a DocumentProfile from the raw PDF
    pages = []
    pages_with_text = sum(1 for i in range(len(doc)) if len(doc[i].get_text().strip()) > 50)
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        blocks = page.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b.get("type") == 1]
        total_area = page.rect.width * page.rect.height
        img_area = sum(
            (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
            for b in img_blocks
        ) if img_blocks else 0
        fonts = set()
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    fonts.add(span.get("font", "unknown"))

        page_quality = ExtractionQuality(
            printable_ratio=0.95,
            avg_words_per_page=len(text.split()),
            pages_with_text=pages_with_text,
            total_pages=len(doc),
            extraction_method="pymupdf",
        )

        pages.append(
            PageProfile(
                page_number=i + 1,
                text_length=len(text),
                word_count=len(text.split()),
                image_count=len(img_blocks),
                image_coverage_ratio=img_area / total_area if total_area > 0 else 0.0,
                table_count=0,
                font_count=len(fonts),
                font_names=sorted(fonts),
                has_form_fields=False,
                is_multi_column=False,
                page_type=PageType.TEXT if len(text.strip()) > 50 else PageType.BLANK,
                extraction_quality=page_quality,
            )
        )

    # Page type distribution
    dist: dict[str, int] = {}
    for p in pages:
        key = p.page_type.value
        dist[key] = dist.get(key, 0) + 1

    pdf_bytes = open(PDF_PATH, "rb").read()
    profile = DocumentProfile(
        file_path=PDF_PATH,
        file_size_bytes=len(pdf_bytes),
        page_count=len(doc),
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        metadata=DocumentMetadata(
            title=doc.metadata.get("title", ""),
            author=doc.metadata.get("author", ""),
            creation_date=doc.metadata.get("creationDate", ""),
            modification_date=doc.metadata.get("modDate", ""),
            producer=doc.metadata.get("producer", ""),
            page_count=len(doc),
            file_size_bytes=len(pdf_bytes),
        ),
        pages=pages,
        page_type_distribution=dist,
        detected_languages=[],
        has_toc=len(doc.get_toc()) > 0,
        toc_entries=[(level, title, page) for level, title, page in doc.get_toc()],
        overall_quality=ExtractionQuality(
            printable_ratio=0.95,
            avg_words_per_page=500,
            pages_with_text=pages_with_text,
            total_pages=len(doc),
            extraction_method="pymupdf",
        ),
        security_warnings=[],
    )

    inspector = PDFInspector(config)
    classification = inspector.classify(profile)
    print(f"  PDF Type: {classification.pdf_type}")
    print(f"  Confidence: {classification.confidence:.2f}")
    print(f"  Tier: {classification.tier_used}")
    print(f"  Reasoning: {classification.reasoning}")
    print(f"  Page types: {classification.per_page_types}")


run_module("Inspector (Classification)", test_inspector)


# ──────────────────────────────────────────────────────────────────────
# 11. PDFRouter — Full End-to-End Pipeline
# ──────────────────────────────────────────────────────────────────────
def test_pdf_router():
    # Import mock backends from conftest
    sys.path.insert(0, "packages/ingestkit-pdf/tests")
    from conftest import (
        MockEmbeddingBackend,
        MockLLMBackend,
        MockStructuredDBBackend,
        MockVectorStoreBackend,
    )

    from ingestkit_pdf.router import PDFRouter

    # Set up mock backends
    # LLM needs at least one response for potential Tier 2 classification
    mock_llm = MockLLMBackend(responses=[
        {"pdf_type": "text_native", "confidence": 0.9, "reasoning": "All pages text"},
    ])
    mock_vector = MockVectorStoreBackend()
    mock_db = MockStructuredDBBackend()
    mock_embedder = MockEmbeddingBackend()

    router = PDFRouter(
        vector_store=mock_vector,
        structured_db=mock_db,
        llm=mock_llm,
        embedder=mock_embedder,
        config=config,
    )

    # Test can_handle
    assert router.can_handle(PDF_PATH), "can_handle should return True for .pdf"
    assert not router.can_handle("file.xlsx"), "can_handle should return False for .xlsx"
    print(f"  can_handle: OK")

    # Process the real PDF
    result = router.process(PDF_PATH)
    print(f"  PDF Type: {result.classification_result.pdf_type}")
    print(f"  Tier: {result.classification_result.tier_used}")
    print(f"  Ingestion method: {result.ingestion_method}")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Tables created: {result.tables_created}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Degraded: {result.classification_result.degraded}")
    print(f"  Ingest key: {result.ingest_key[:16]}...")
    print(f"  Processing time: {result.processing_time_seconds:.2f}s")

    # Check vector store got chunks
    if hasattr(mock_vector, 'total_chunks_upserted'):
        print(f"  Vector store chunks upserted: {mock_vector.total_chunks_upserted}")
    if hasattr(mock_embedder, 'total_texts_embedded'):
        print(f"  Texts embedded: {mock_embedder.total_texts_embedded}")


run_module("PDFRouter (End-to-End)", test_pdf_router)


# ──────────────────────────────────────────────────────────────────────
# 12. ComplexProcessor — Direct Test with Mock Backends
# ──────────────────────────────────────────────────────────────────────
def test_complex_processor():
    sys.path.insert(0, "packages/ingestkit-pdf/tests")
    from conftest import (
        MockEmbeddingBackend,
        MockLLMBackend,
        MockStructuredDBBackend,
        MockVectorStoreBackend,
    )

    from ingestkit_pdf.processors.complex_processor import ComplexProcessor
    from ingestkit_pdf.models import (
        ClassificationResult,
        ClassificationStageResult,
        ClassificationTier,
        DocumentMetadata,
        DocumentProfile,
        ExtractionQuality,
        PageProfile,
        PageType,
        ParseStageResult,
        PDFType,
    )
    import hashlib

    mock_vector = MockVectorStoreBackend()
    mock_db = MockStructuredDBBackend()
    mock_embedder = MockEmbeddingBackend()
    mock_llm = MockLLMBackend()

    processor = ComplexProcessor(
        vector_store=mock_vector,
        structured_db=mock_db,
        embedder=mock_embedder,
        llm=mock_llm,
        config=config,
    )

    # Build profile with mixed page types to exercise routing
    pages_with_text = sum(1 for i in range(len(doc)) if len(doc[i].get_text().strip()) > 50)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        blocks = page.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b.get("type") == 1]
        total_area = page.rect.width * page.rect.height
        img_area = sum(
            (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
            for b in img_blocks
        ) if img_blocks else 0
        fonts = set()
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    fonts.add(span.get("font", "unknown"))

        page_quality = ExtractionQuality(
            printable_ratio=0.95,
            avg_words_per_page=len(text.split()),
            pages_with_text=pages_with_text,
            total_pages=len(doc),
            extraction_method="pymupdf",
        )
        pages.append(
            PageProfile(
                page_number=i + 1,
                text_length=len(text),
                word_count=len(text.split()),
                image_count=len(img_blocks),
                image_coverage_ratio=img_area / total_area if total_area > 0 else 0.0,
                table_count=0,
                font_count=len(fonts),
                font_names=sorted(fonts),
                has_form_fields=False,
                is_multi_column=False,
                page_type=PageType.TEXT,
                extraction_quality=page_quality,
            )
        )

    pdf_bytes = open(PDF_PATH, "rb").read()
    profile = DocumentProfile(
        file_path=PDF_PATH,
        file_size_bytes=len(pdf_bytes),
        page_count=len(doc),
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        metadata=DocumentMetadata(
            title=doc.metadata.get("title", ""),
            author=doc.metadata.get("author", ""),
            page_count=len(doc),
            file_size_bytes=len(pdf_bytes),
        ),
        pages=pages,
        page_type_distribution={"text": len(pages)},
        detected_languages=["en"],
        has_toc=False,
        overall_quality=ExtractionQuality(
            printable_ratio=0.95,
            avg_words_per_page=500,
            pages_with_text=pages_with_text,
            total_pages=len(doc),
            extraction_method="pymupdf",
        ),
        security_warnings=[],
    )

    classification = ClassificationResult(
        pdf_type=PDFType.COMPLEX,
        confidence=0.85,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="Test",
        per_page_types={i + 1: PageType.TEXT for i in range(len(doc))},
    )

    parse_result = ParseStageResult(
        pages_extracted=len(doc),
        pages_skipped=0,
        skipped_reasons={},
        extraction_method="pymupdf",
        overall_quality=ExtractionQuality(
            printable_ratio=0.95,
            avg_words_per_page=500,
            pages_with_text=pages_with_text,
            total_pages=len(doc),
            extraction_method="pymupdf",
        ),
        parse_duration_seconds=0.1,
    )

    classification_stage = ClassificationStageResult(
        pdf_type=PDFType.COMPLEX,
        confidence=0.85,
        tier_used=ClassificationTier.RULE_BASED,
        reasoning="Test",
        per_page_types={i + 1: PageType.TEXT for i in range(len(doc))},
        classification_duration_seconds=0.01,
    )

    result = processor.process(
        file_path=PDF_PATH,
        profile=profile,
        ingest_key="test-complex-key",
        ingest_run_id="test-complex-run",
        parse_result=parse_result,
        classification_result=classification_stage,
        classification=classification,
    )

    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Tables created: {result.tables_created}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Ingestion method: {result.ingestion_method}")
    if hasattr(mock_vector, 'total_chunks_upserted'):
        print(f"  Vector store chunks: {mock_vector.total_chunks_upserted}")
    if hasattr(mock_embedder, 'total_texts_embedded'):
        print(f"  Texts embedded: {mock_embedder.total_texts_embedded}")


run_module("ComplexProcessor (Direct)", test_complex_processor)


# ──────────────────────────────────────────────────────────────────────
# 13. PaddleOCR Engine (availability check)
# ──────────────────────────────────────────────────────────────────────
def test_paddleocr():
    from ingestkit_pdf.utils.ocr_engines import PaddleOCREngine

    try:
        engine = PaddleOCREngine(lang="en")
        print(f"  PaddleOCR available: True")
        print(f"  Engine name: {engine.name()}")
    except Exception as e:
        print(f"  PaddleOCR available: False ({type(e).__name__}: {e})")
        print(f"  (This is expected if paddleocr is not installed)")
        results["PaddleOCR Engine"] = "SKIP (not installed)"


run_module("PaddleOCR Engine", test_paddleocr)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
passed = sum(1 for v in results.values() if v == "PASS")
skipped = sum(1 for v in results.values() if v.startswith("SKIP"))
failed = sum(1 for v in results.values() if v.startswith("FAIL"))
for name, status in results.items():
    icon = "PASS" if status == "PASS" else ("SKIP" if "SKIP" in status else "FAIL")
    print(f"  [{icon}] {name}")
print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")

doc.close()
