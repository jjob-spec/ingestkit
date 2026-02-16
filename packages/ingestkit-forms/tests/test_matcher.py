"""Tests for the form matcher module."""

from __future__ import annotations

import pytest
from PIL import Image

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.matcher import (
    FormMatcher,
    _compute_page_similarity,
    _deserialize_fingerprint,
    _quantize_fill_ratio,
    _windowed_match,
    compute_layout_fingerprint,
    compute_layout_fingerprint_from_file,
    compute_layout_similarity,
    detect_source_format,
)
from ingestkit_forms.models import FormIngestRequest, SourceFormat
from tests.conftest import (
    MockLayoutFingerprinter,
    make_fingerprint_bytes,
    make_page_bytes,
    make_template,
    make_uniform_page,
)


# ---------------------------------------------------------------------------
# Fingerprint computation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeLayoutFingerprint:
    """Tests for compute_layout_fingerprint."""

    def test_single_page(self, form_like_image, form_config):
        """Fingerprint from a single image has correct length."""
        fp = compute_layout_fingerprint([form_like_image], form_config)
        expected_len = form_config.fingerprint_grid_rows * form_config.fingerprint_grid_cols
        assert len(fp) == expected_len  # 20 * 16 = 320

    def test_blank_page_all_zeros(self, blank_page_image, form_config):
        """Fingerprint of a blank white image is all zeros (empty cells)."""
        fp = compute_layout_fingerprint([blank_page_image], form_config)
        assert all(b == 0 for b in fp)

    def test_multi_page(self, form_like_image, form_config):
        """Two-page fingerprint has length 2 * grid_rows * grid_cols."""
        fp = compute_layout_fingerprint(
            [form_like_image, form_like_image], form_config
        )
        expected_len = (
            2 * form_config.fingerprint_grid_rows * form_config.fingerprint_grid_cols
        )
        assert len(fp) == expected_len

    def test_deterministic(self, form_like_image, form_config):
        """Same image produces identical fingerprint on two calls."""
        fp1 = compute_layout_fingerprint([form_like_image], form_config)
        fp2 = compute_layout_fingerprint([form_like_image], form_config)
        assert fp1 == fp2

    def test_empty_pages_raises(self, form_config):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_layout_fingerprint([], form_config)

    def test_values_in_range(self, form_like_image, form_config):
        """All byte values in result are 0, 1, 2, or 3."""
        fp = compute_layout_fingerprint([form_like_image], form_config)
        assert all(b in (0, 1, 2, 3) for b in fp)

    def test_custom_grid(self, form_like_image):
        """Config with custom grid dimensions produces correct fingerprint length."""
        config = FormProcessorConfig(
            fingerprint_grid_rows=10, fingerprint_grid_cols=8
        )
        fp = compute_layout_fingerprint([form_like_image], config)
        assert len(fp) == 10 * 8  # 80

    def test_form_has_nonzero_cells(self, form_like_image, form_config):
        """Form-like image has some non-zero fingerprint cells (structural content)."""
        fp = compute_layout_fingerprint([form_like_image], form_config)
        assert any(b > 0 for b in fp)


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuantizeFillRatio:
    """Tests for _quantize_fill_ratio."""

    @pytest.mark.parametrize(
        "fill_ratio,expected",
        [
            (0.0, 0),
            (0.04, 0),
            (0.05, 1),
            (0.24, 1),
            (0.25, 2),
            (0.59, 2),
            (0.60, 3),
            (1.0, 3),
        ],
    )
    def test_boundary_values(self, fill_ratio, expected):
        """Quantization boundaries are correct."""
        assert _quantize_fill_ratio(fill_ratio) == expected


# ---------------------------------------------------------------------------
# Similarity tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeLayoutSimilarity:
    """Tests for compute_layout_similarity."""

    def test_identical(self):
        """Two identical fingerprints produce similarity 1.0."""
        fp = bytes([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0])
        sim = compute_layout_similarity(fp, fp, grid_cols=4, grid_rows=4)
        assert sim == 1.0

    def test_completely_different(self):
        """All-zeros vs all-threes produces 0.0."""
        fp_a = bytes([0] * 16)
        fp_b = bytes([3] * 16)
        sim = compute_layout_similarity(fp_a, fp_b, grid_cols=4, grid_rows=4)
        assert sim == 0.0

    def test_off_by_one(self):
        """Fingerprint where every cell is off-by-one produces 0.5."""
        fp_a = bytes([0] * 16)
        fp_b = bytes([1] * 16)
        sim = compute_layout_similarity(fp_a, fp_b, grid_cols=4, grid_rows=4)
        assert sim == 0.5

    def test_page_count_mismatch(self):
        """One-page vs two-page fingerprints produce 0.0."""
        fp_a = bytes([1] * 16)  # 1 page (4x4)
        fp_b = bytes([1] * 32)  # 2 pages (4x4)
        sim = compute_layout_similarity(fp_a, fp_b, grid_cols=4, grid_rows=4)
        assert sim == 0.0

    def test_empty_fingerprints(self):
        """Two empty bytes produce 0.0."""
        sim = compute_layout_similarity(b"", b"", grid_cols=4, grid_rows=4)
        assert sim == 0.0

    def test_mixed_diffs(self):
        """Mix of exact, off-by-one, off-by-two produces expected weighted score."""
        # 4 cells: exact(1.0), off-by-one(0.5), off-by-two(0.0), exact(1.0)
        fp_a = bytes([0, 1, 0, 3])
        fp_b = bytes([0, 2, 2, 3])
        sim = compute_layout_similarity(fp_a, fp_b, grid_cols=2, grid_rows=2)
        expected = (1.0 + 0.5 + 0.0 + 1.0) / 4
        assert sim == pytest.approx(expected)

    def test_invalid_length(self):
        """Fingerprints not divisible by cells_per_page return 0.0."""
        fp_a = bytes([1, 2, 3])  # 3 bytes, not divisible by 4*4=16
        fp_b = bytes([1, 2, 3])
        sim = compute_layout_similarity(fp_a, fp_b, grid_cols=4, grid_rows=4)
        assert sim == 0.0


# ---------------------------------------------------------------------------
# File-based fingerprint tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeFingerprintFromFile:
    """Tests for compute_layout_fingerprint_from_file."""

    def test_from_image_file(self, sample_image_file, form_config):
        """compute_layout_fingerprint_from_file with a .png file succeeds."""
        fp = compute_layout_fingerprint_from_file(sample_image_file, form_config)
        expected_len = (
            form_config.fingerprint_grid_rows * form_config.fingerprint_grid_cols
        )
        assert len(fp) == expected_len
        assert all(b in (0, 1, 2, 3) for b in fp)

    def test_unsupported_format_no_renderer(self, tmp_path, form_config):
        """.pdf without renderer raises FormIngestError with E_FORM_FINGERPRINT_FAILED."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")
        with pytest.raises(FormIngestException) as exc_info:
            compute_layout_fingerprint_from_file(str(pdf_path), form_config)
        assert exc_info.value.code == FormErrorCode.E_FORM_FINGERPRINT_FAILED
        assert "No renderer available" in exc_info.value.message

    def test_with_renderer(self, tmp_path, form_config):
        """.pdf with mock renderer produces fingerprint."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # Mock renderer returns a simple image
        mock_page = Image.new("L", (800, 600), color=128)

        def mock_renderer(path: str, dpi: int) -> list[Image.Image]:
            return [mock_page]

        fp = compute_layout_fingerprint_from_file(
            str(pdf_path), form_config, renderer=mock_renderer
        )
        expected_len = (
            form_config.fingerprint_grid_rows * form_config.fingerprint_grid_cols
        )
        assert len(fp) == expected_len


# ---------------------------------------------------------------------------
# Source format detection tests (issue #62)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectSourceFormat:
    """Tests for detect_source_format."""

    def test_detect_pdf(self):
        assert detect_source_format("form.pdf") == SourceFormat.PDF

    def test_detect_xlsx(self):
        assert detect_source_format("data.xlsx") == SourceFormat.XLSX

    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".jpg", SourceFormat.IMAGE),
            (".jpeg", SourceFormat.IMAGE),
            (".png", SourceFormat.IMAGE),
            (".tiff", SourceFormat.IMAGE),
            (".tif", SourceFormat.IMAGE),
        ],
    )
    def test_detect_image_variants(self, ext, expected):
        assert detect_source_format(f"scan{ext}") == expected

    def test_detect_case_insensitive(self):
        assert detect_source_format("FORM.PDF") == SourceFormat.PDF
        assert detect_source_format("data.XLSX") == SourceFormat.XLSX

    def test_detect_unknown_extension(self):
        with pytest.raises(FormIngestException) as exc_info:
            detect_source_format("file.doc")
        assert exc_info.value.code == FormErrorCode.E_FORM_UNSUPPORTED_FORMAT

    def test_detect_no_extension(self):
        with pytest.raises(FormIngestException) as exc_info:
            detect_source_format("noext")
        assert exc_info.value.code == FormErrorCode.E_FORM_UNSUPPORTED_FORMAT


# ---------------------------------------------------------------------------
# Fingerprint deserialization tests (issue #62)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeserializeFingerprint:
    """Tests for _deserialize_fingerprint."""

    def test_single_page(self):
        """320 bytes -> 1 page of 20x16."""
        fp = bytes(range(256)) + bytes(range(64))  # 320 bytes
        pages = _deserialize_fingerprint(fp, rows=20, cols=16)
        assert len(pages) == 1
        assert len(pages[0]) == 20
        assert len(pages[0][0]) == 16

    def test_multi_page(self):
        """640 bytes -> 2 pages of 20x16."""
        fp = bytes([1] * 640)
        pages = _deserialize_fingerprint(fp, rows=20, cols=16)
        assert len(pages) == 2

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="not a multiple"):
            _deserialize_fingerprint(bytes([0] * 100), rows=20, cols=16)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="not a multiple"):
            _deserialize_fingerprint(b"", rows=20, cols=16)


# ---------------------------------------------------------------------------
# Page similarity tests (issue #62)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputePageSimilarity:
    """Tests for _compute_page_similarity."""

    def test_identical_pages(self):
        page = make_uniform_page(2)
        assert _compute_page_similarity(page, page) == 1.0

    def test_off_by_one(self):
        page_a = make_uniform_page(1)
        page_b = make_uniform_page(2)
        assert _compute_page_similarity(page_a, page_b) == pytest.approx(0.5)

    def test_completely_different(self):
        page_a = make_uniform_page(0)
        page_b = make_uniform_page(3)
        assert _compute_page_similarity(page_a, page_b) == 0.0

    def test_mixed_similarity(self):
        """50% exact, 50% off-by-one => 0.75."""
        rows, cols = 20, 16
        page_a = [[0] * cols for _ in range(rows)]
        page_b = [[0] * cols for _ in range(rows)]
        # Make second half of rows off-by-one
        for r in range(rows // 2, rows):
            page_b[r] = [1] * cols
        sim = _compute_page_similarity(page_a, page_b)
        assert sim == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Windowed match tests (issue #62)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWindowedMatch:
    """Tests for _windowed_match."""

    def test_d_less_than_t(self):
        """D < T returns None."""
        doc = [make_uniform_page(1)]
        tmpl = [make_uniform_page(1), make_uniform_page(1)]
        assert _windowed_match(doc, tmpl, per_page_minimum=0.6, extra_page_penalty=0.02) is None

    def test_d_equals_t_match(self):
        """D == T, identical pages -> confidence ~1.0."""
        page = make_uniform_page(2)
        doc = [page, page]
        tmpl = [page, page]
        result = _windowed_match(doc, tmpl, per_page_minimum=0.6, extra_page_penalty=0.02)
        assert result is not None
        confidence, scores, start = result
        assert confidence == pytest.approx(1.0)
        assert start == 0
        assert len(scores) == 2

    def test_d_equals_t_no_match(self):
        """D == T but page similarity below minimum -> None."""
        doc = [make_uniform_page(0), make_uniform_page(0)]
        tmpl = [make_uniform_page(3), make_uniform_page(3)]
        result = _windowed_match(doc, tmpl, per_page_minimum=0.6, extra_page_penalty=0.02)
        assert result is None

    def test_d_greater_than_t_finds_best_window(self):
        """D=4, T=2, match at position 1."""
        bad_page = make_uniform_page(0)
        good_page = make_uniform_page(2)
        doc = [bad_page, good_page, good_page, bad_page]
        tmpl = [good_page, good_page]
        result = _windowed_match(doc, tmpl, per_page_minimum=0.6, extra_page_penalty=0.02)
        assert result is not None
        confidence, scores, start = result
        assert start == 1
        # Penalty = (4-2)*0.02 = 0.04
        assert confidence == pytest.approx(1.0 - 0.04)

    def test_extra_page_penalty_applied(self):
        """D=3, T=1 -> penalty = 2 * 0.02 = 0.04."""
        page = make_uniform_page(1)
        doc = [page, page, page]
        tmpl = [page]
        result = _windowed_match(doc, tmpl, per_page_minimum=0.6, extra_page_penalty=0.02)
        assert result is not None
        confidence, _, _ = result
        assert confidence == pytest.approx(1.0 - 0.04)

    def test_per_page_minimum_enforced(self):
        """All windows have at least one page below minimum -> None."""
        good_page = make_uniform_page(2)
        bad_page = make_uniform_page(0)  # will produce 0.0 similarity with good
        doc = [bad_page, bad_page]
        tmpl = [good_page, good_page]
        result = _windowed_match(doc, tmpl, per_page_minimum=0.6, extra_page_penalty=0.02)
        assert result is None

    def test_single_page_template_single_page_doc(self):
        """D=1, T=1 -> direct comparison."""
        page = make_uniform_page(1)
        result = _windowed_match([page], [page], per_page_minimum=0.6, extra_page_penalty=0.02)
        assert result is not None
        confidence, scores, start = result
        assert confidence == pytest.approx(1.0)
        assert start == 0
        assert len(scores) == 1


# ---------------------------------------------------------------------------
# FormMatcher.match_document tests (issue #62)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormMatcherMatchDocument:
    """Tests for FormMatcher.match_document."""

    def test_match_single_template_high_confidence(
        self, mock_template_store, form_config
    ):
        """Single template with identical fingerprint -> 1 match, confidence ~1.0."""
        page = make_uniform_page(2)
        page_bytes = make_page_bytes(page)
        fp_concat = make_fingerprint_bytes([page])

        tmpl = make_template(
            name="W-4",
            source_format=SourceFormat.PDF,
            layout_fingerprint=fp_concat,
            template_id="tmpl-1",
            status="approved",
        )
        mock_template_store.save_template(tmpl)

        fingerprinter = MockLayoutFingerprinter(pages=[page_bytes])
        matcher = FormMatcher(mock_template_store, fingerprinter, form_config)
        matches = matcher.match_document("form.pdf")

        assert len(matches) == 1
        assert matches[0].template_id == "tmpl-1"
        assert matches[0].confidence == pytest.approx(1.0)
        assert matches[0].matched_features == ["layout_grid"]

    def test_match_no_templates(self, mock_template_store, form_config):
        """Empty store -> empty result list."""
        page = make_uniform_page(1)
        fingerprinter = MockLayoutFingerprinter(pages=[make_page_bytes(page)])
        matcher = FormMatcher(mock_template_store, fingerprinter, form_config)
        matches = matcher.match_document("form.pdf")
        assert matches == []

    def test_match_below_warning_floor(self, mock_template_store, form_config):
        """Template with confidence < 0.5 is excluded."""
        # Template: all 2s. Doc: all 0s. Similarity per cell = 0.0.
        tmpl_page = make_uniform_page(2)
        doc_page = make_uniform_page(0)
        fp_concat = make_fingerprint_bytes([tmpl_page])

        tmpl = make_template(
            name="Low",
            source_format=SourceFormat.PDF,
            layout_fingerprint=fp_concat,
            template_id="tmpl-low",
        )
        mock_template_store.save_template(tmpl)

        fingerprinter = MockLayoutFingerprinter(pages=[make_page_bytes(doc_page)])
        matcher = FormMatcher(mock_template_store, fingerprinter, form_config)
        matches = matcher.match_document("form.pdf")
        assert matches == []

    def test_match_multiple_templates_sorted(self, mock_template_store, form_config):
        """Multiple templates are sorted by confidence descending."""
        # Exact match template
        exact_page = make_uniform_page(1)
        exact_fp = make_fingerprint_bytes([exact_page])
        tmpl_exact = make_template(
            name="Exact",
            source_format=SourceFormat.PDF,
            layout_fingerprint=exact_fp,
            template_id="tmpl-exact",
            status="approved",
        )

        # Off-by-one template (similarity = 0.5 per cell) won't pass
        # per_page_minimum=0.6 default.
        # Use a page that gives ~0.7 similarity instead.
        # Mix: 60% exact, 40% off-by-one => 0.6 + 0.2 = 0.8
        rows, cols = 20, 16
        mixed_page = [[1] * cols for _ in range(rows)]
        for r in range(int(rows * 0.4)):
            mixed_page[r] = [2] * cols
        mixed_fp = make_fingerprint_bytes([mixed_page])
        tmpl_mixed = make_template(
            name="Mixed",
            source_format=SourceFormat.PDF,
            layout_fingerprint=mixed_fp,
            template_id="tmpl-mixed",
            status="approved",
        )

        mock_template_store.save_template(tmpl_exact)
        mock_template_store.save_template(tmpl_mixed)

        doc_page = make_uniform_page(1)
        fingerprinter = MockLayoutFingerprinter(pages=[make_page_bytes(doc_page)])
        matcher = FormMatcher(mock_template_store, fingerprinter, form_config)
        matches = matcher.match_document("form.pdf")

        assert len(matches) >= 1
        # First match should have higher confidence
        if len(matches) > 1:
            assert matches[0].confidence >= matches[1].confidence
        # The exact match should be first
        assert matches[0].template_id == "tmpl-exact"
        assert matches[0].confidence == pytest.approx(1.0)

    def test_match_fingerprint_failure(self, mock_template_store, form_config):
        """Fingerprinter raising -> raises E_FORM_FINGERPRINT_FAILED."""
        page = make_uniform_page(1)
        fp_concat = make_fingerprint_bytes([page])
        tmpl = make_template(
            name="T",
            source_format=SourceFormat.PDF,
            layout_fingerprint=fp_concat,
            template_id="tmpl-fp-fail",
            status="approved",
        )
        mock_template_store.save_template(tmpl)

        class FailingFingerprinter:
            def compute_fingerprint(self, file_path: str) -> list[bytes]:
                raise RuntimeError("render failed")

        matcher = FormMatcher(mock_template_store, FailingFingerprinter(), form_config)
        with pytest.raises(FormIngestException) as exc_info:
            matcher.match_document("form.pdf")
        assert exc_info.value.code == FormErrorCode.E_FORM_FINGERPRINT_FAILED

    def test_match_invalid_template_fingerprint_skipped(
        self, mock_template_store, form_config
    ):
        """Template with invalid fingerprint bytes is skipped; valid ones still matched."""
        good_page = make_uniform_page(1)
        good_fp = make_fingerprint_bytes([good_page])
        tmpl_good = make_template(
            name="Good",
            source_format=SourceFormat.PDF,
            layout_fingerprint=good_fp,
            template_id="tmpl-good",
            status="approved",
        )

        # Bad fingerprint: wrong length
        tmpl_bad = make_template(
            name="Bad",
            source_format=SourceFormat.PDF,
            layout_fingerprint=bytes([0] * 100),  # not 320
            template_id="tmpl-bad",
            status="approved",
        )

        mock_template_store.save_template(tmpl_good)
        mock_template_store.save_template(tmpl_bad)

        fingerprinter = MockLayoutFingerprinter(pages=[make_page_bytes(good_page)])
        matcher = FormMatcher(mock_template_store, fingerprinter, form_config)
        matches = matcher.match_document("form.pdf")

        # Bad template skipped, good template matched
        assert len(matches) == 1
        assert matches[0].template_id == "tmpl-good"


# ---------------------------------------------------------------------------
# FormMatcher.resolve_manual_override tests (issue #62)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormMatcherResolveManualOverride:
    """Tests for FormMatcher.resolve_manual_override."""

    def _make_matcher(self, store, config):
        """Helper to create a matcher (fingerprinter not used for override)."""
        return FormMatcher(store, MockLayoutFingerprinter(), config)

    def test_override_happy_path(self, mock_template_store, form_config):
        """Template exists, format matches -> returns template."""
        tmpl = make_template(
            name="W-4",
            source_format=SourceFormat.PDF,
            template_id="tmpl-ok",
        )
        mock_template_store.save_template(tmpl)

        matcher = self._make_matcher(mock_template_store, form_config)
        request = FormIngestRequest(
            file_path="form.pdf",
            template_id="tmpl-ok",
        )
        result = matcher.resolve_manual_override(request)
        assert result.template_id == "tmpl-ok"

    def test_override_not_found(self, mock_template_store, form_config):
        """Template ID does not exist -> E_FORM_TEMPLATE_NOT_FOUND."""
        matcher = self._make_matcher(mock_template_store, form_config)
        request = FormIngestRequest(
            file_path="form.pdf",
            template_id="nonexistent",
        )
        with pytest.raises(FormIngestException) as exc_info:
            matcher.resolve_manual_override(request)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND
        assert "nonexistent" in exc_info.value.message

    def test_override_version_not_found(self, mock_template_store, form_config):
        """Specific version missing -> E_FORM_TEMPLATE_NOT_FOUND."""
        tmpl = make_template(
            name="W-4",
            source_format=SourceFormat.PDF,
            template_id="tmpl-v",
            version=1,
        )
        mock_template_store.save_template(tmpl)

        matcher = self._make_matcher(mock_template_store, form_config)
        request = FormIngestRequest(
            file_path="form.pdf",
            template_id="tmpl-v",
            template_version=99,
        )
        with pytest.raises(FormIngestException) as exc_info:
            matcher.resolve_manual_override(request)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND
        assert "version 99" in exc_info.value.message

    def test_override_format_mismatch(self, mock_template_store, form_config):
        """PDF template vs XLSX input -> E_FORM_FORMAT_MISMATCH."""
        tmpl = make_template(
            name="PDF Template",
            source_format=SourceFormat.PDF,
            template_id="tmpl-pdf",
        )
        mock_template_store.save_template(tmpl)

        matcher = self._make_matcher(mock_template_store, form_config)
        request = FormIngestRequest(
            file_path="data.xlsx",
            template_id="tmpl-pdf",
        )
        with pytest.raises(FormIngestException) as exc_info:
            matcher.resolve_manual_override(request)
        assert exc_info.value.code == FormErrorCode.E_FORM_FORMAT_MISMATCH
        assert "incompatible" in exc_info.value.message
        assert "'pdf'" in exc_info.value.message
        assert "'xlsx'" in exc_info.value.message

    def test_override_latest_version(self, mock_template_store, form_config):
        """No version specified -> returns latest version."""
        tmpl_v1 = make_template(
            name="W-4",
            source_format=SourceFormat.PDF,
            template_id="tmpl-latest",
            version=1,
        )
        tmpl_v2 = make_template(
            name="W-4",
            source_format=SourceFormat.PDF,
            template_id="tmpl-latest",
            version=2,
        )
        mock_template_store.save_template(tmpl_v1)
        mock_template_store.save_template(tmpl_v2)

        matcher = self._make_matcher(mock_template_store, form_config)
        request = FormIngestRequest(
            file_path="form.pdf",
            template_id="tmpl-latest",
        )
        result = matcher.resolve_manual_override(request)
        assert result.version == 2
