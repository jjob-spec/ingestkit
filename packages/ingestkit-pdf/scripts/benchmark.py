#!/usr/bin/env python3
"""Standalone benchmark runner for the ingestkit-pdf pipeline.

Generates test PDFs programmatically, runs the full pipeline N iterations
per path, and outputs a benchmark-report-<date>.json file.

Usage:
    python scripts/benchmark.py --iterations 5
    python scripts/benchmark.py --iterations 10 --output-dir reports/
    python scripts/benchmark.py --paths A       # Path A only
    python scripts/benchmark.py --paths B       # Path B only
    python scripts/benchmark.py --paths A B     # Both (default)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# PDF Generation Helpers (extracted from conftest.py patterns)
# ---------------------------------------------------------------------------


def _create_text_native_pdf(path: Path) -> int:
    """Create a 3-page text-native PDF. Returns page count."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)

    chapters = [
        (
            "Chapter 1: Introduction",
            [
                "This document provides an overview of the project goals and methodology.",
                "The research was conducted over a period of six months with regular reviews.",
                "Key findings are summarized in the following chapters for stakeholder review.",
            ],
        ),
        (
            "Chapter 2: Methods",
            [
                "Data was collected from multiple sources including surveys and interviews.",
                "Statistical analysis was performed using standard regression techniques.",
                "All results were validated through cross-reference with existing literature.",
            ],
        ),
        (
            "Chapter 3: Results",
            [
                "The analysis revealed significant improvements across all measured metrics.",
                "Response rates exceeded expectations at ninety-two percent overall completion.",
                "Detailed tables and figures are provided in the appendices for reference.",
            ],
        ),
    ]

    for page_num, (heading, paragraphs) in enumerate(chapters, start=1):
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, 700, heading)
        c.setFont("Helvetica", 11)
        y = 660
        for para in paragraphs:
            c.drawString(72, y, para)
            y -= 20
        c.setFont("Helvetica", 9)
        c.drawString(280, 40, f"Page {page_num} of 3")
        c.showPage()

    c.save()
    return 3


def _create_scanned_pdf(path: Path) -> int:
    """Create a 2-page scanned (image-only) PDF. Returns page count."""
    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    page_width, page_height = letter

    page_texts = [
        [
            "Scanned Document - Page 1",
            "This is the first page of a scanned document.",
            "It contains text rendered as an image only.",
        ],
        [
            "Scanned Document - Page 2",
            "The second page has different content.",
            "No extractable text layer exists in this PDF.",
        ],
    ]

    dpi = 150
    img_w = int(8.5 * dpi)
    img_h = int(11 * dpi)

    for texts in page_texts:
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        y_pos = 150
        for line in texts:
            draw.text((100, y_pos), line, fill="black")
            y_pos += 40

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        c.drawImage(
            ImageReader(buf),
            0,
            0,
            width=page_width,
            height=page_height,
        )
        c.showPage()

    c.save()
    return 2


# ---------------------------------------------------------------------------
# Backend Availability Checks
# ---------------------------------------------------------------------------


def _check_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


def _check_ollama() -> bool:
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def _compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute p50, p95, and max for a list of values."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    p50_idx = int(n * 0.5)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "p50": sorted_v[p50_idx],
        "p95": sorted_v[p95_idx],
        "max": sorted_v[-1],
    }


def _run_path_benchmark(
    router,
    pdf_path: str,
    page_count: int,
    iterations: int,
    path_label: str,
) -> dict:
    """Run the pipeline N times and collect timing data."""
    total_times = []
    parse_times = []
    classify_times = []

    for i in range(iterations):
        start = time.monotonic()
        result = router.process(pdf_path)
        elapsed = time.monotonic() - start
        total_times.append(elapsed)

        # Collect per-stage timing from the result
        parse_times.append(result.parse_result.parse_duration_seconds)
        classify_times.append(
            result.classification_result.classification_duration_seconds
        )

        fatal = [e for e in result.errors if not e.startswith("W_")]
        status = "OK" if not fatal else f"ERRORS: {fatal}"
        print(
            f"  [{path_label}] iteration {i + 1}/{iterations}: "
            f"{elapsed:.3f}s ({status})"
        )

    total_pages = page_count * iterations
    total_time = sum(total_times)
    throughput = total_pages / total_time if total_time > 0 else 0

    return {
        "throughput_pages_per_sec": round(throughput, 2),
        "per_stage_latency": {
            "parse": _compute_percentiles(parse_times),
            "classification": _compute_percentiles(classify_times),
            "total": _compute_percentiles(total_times),
        },
        "iterations": iterations,
        "total_pages": total_pages,
        "avg_time_sec": round(statistics.mean(total_times), 4),
        "min_time_sec": round(min(total_times), 4),
        "max_time_sec": round(max(total_times), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ingestkit-pdf benchmarks and produce a JSON report."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per path (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for the benchmark report (default: current dir)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=["A", "B"],
        default=["A", "B"],
        help="Which paths to benchmark (default: A B)",
    )
    args = parser.parse_args()

    # Check backends
    print("Checking backend availability...")
    qdrant_ok = _check_qdrant()
    ollama_ok = _check_ollama()
    print(f"  Qdrant: {'OK' if qdrant_ok else 'NOT AVAILABLE'}")
    print(f"  Ollama: {'OK' if ollama_ok else 'NOT AVAILABLE'}")

    if not (qdrant_ok and ollama_ok):
        print("\nERROR: All backends must be available to run benchmarks.")
        print("Start Qdrant (localhost:6333) and Ollama (localhost:11434).")
        return 1

    # Build router
    from ingestkit_excel.backends import (
        OllamaEmbedding,
        OllamaLLM,
        QdrantVectorStore,
        SQLiteStructuredDB,
    )

    from ingestkit_pdf.config import PDFProcessorConfig
    from ingestkit_pdf.router import PDFRouter

    config = PDFProcessorConfig(
        tenant_id="benchmark",
        default_collection="test_benchmark",
    )
    router = PDFRouter(
        vector_store=QdrantVectorStore(url="http://localhost:6333"),
        structured_db=SQLiteStructuredDB(":memory:"),
        llm=OllamaLLM(base_url="http://localhost:11434"),
        embedder=OllamaEmbedding(
            base_url="http://localhost:11434", model="nomic-embed-text"
        ),
        config=config,
    )

    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "config": {
            "iterations": args.iterations,
            "paths": args.paths,
        },
    }

    slo_results: dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Path A benchmark
        if "A" in args.paths:
            print(f"\nRunning Path A benchmark ({args.iterations} iterations)...")
            pdf_a = Path(tmpdir) / "text_native.pdf"
            pages_a = _create_text_native_pdf(pdf_a)

            result_a = _run_path_benchmark(
                router, str(pdf_a), pages_a, args.iterations, "Path A"
            )
            slo_target_a = 50
            slo_pass_a = result_a["throughput_pages_per_sec"] >= slo_target_a
            result_a["slo_target"] = slo_target_a
            result_a["slo_pass"] = slo_pass_a
            report["path_a"] = result_a
            slo_results["path_a"] = slo_pass_a

        # Path B benchmark
        if "B" in args.paths:
            print(f"\nRunning Path B benchmark ({args.iterations} iterations)...")
            pdf_b = Path(tmpdir) / "scanned.pdf"
            pages_b = _create_scanned_pdf(pdf_b)

            result_b = _run_path_benchmark(
                router, str(pdf_b), pages_b, args.iterations, "Path B"
            )
            slo_target_b = 10
            slo_pass_b = result_b["throughput_pages_per_sec"] >= slo_target_b
            result_b["slo_target"] = slo_target_b
            result_b["slo_pass"] = slo_pass_b
            report["path_b"] = result_b
            slo_results["path_b"] = slo_pass_b

    # Phase 2 gate summary
    all_slos_pass = all(slo_results.values()) if slo_results else False
    report["phase2_gate"] = {
        "integration_tests_pass": None,  # Must be checked separately
        "benchmark_targets_met": all_slos_pass,
        **{f"{k}_slo": v for k, v in slo_results.items()},
    }

    # Write report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    report_path = output_dir / f"benchmark-report-{date_str}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if "path_a" in report:
        pa = report["path_a"]
        status = "PASS" if pa["slo_pass"] else "FAIL"
        print(
            f"  Path A: {pa['throughput_pages_per_sec']:.1f} pages/sec "
            f"(SLO >= {pa['slo_target']}) [{status}]"
        )

    if "path_b" in report:
        pb = report["path_b"]
        status = "PASS" if pb["slo_pass"] else "FAIL"
        print(
            f"  Path B: {pb['throughput_pages_per_sec']:.1f} pages/sec "
            f"(SLO >= {pb['slo_target']}) [{status}]"
        )

    overall = "PASS" if all_slos_pass else "FAIL"
    print(f"\n  Overall: {overall}")
    print(f"\n  Report: {report_path}")
    print("=" * 60)

    # Cleanup test collections
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333", timeout=2)
        collections = client.get_collections().collections
        for c in collections:
            if c.name.startswith("test_"):
                client.delete_collection(c.name)
    except Exception:
        pass

    return 0 if all_slos_pass else 1


if __name__ == "__main__":
    sys.exit(main())
