---
issue: 61
agent: PLAN-CHECK
date: 2026-02-15
status: PASS_WITH_ISSUES
depends_on: plan-61-021526.md
---

# PLAN-CHECK: Issue #61 -- Template System: CRUD, Versioning, and Fingerprinting

## 1. Executive Summary

The PLAN is thorough, well-structured, and covers the three main components (matcher, store, API) with correct implementation order. The fingerprinting algorithm matches spec section 5.4 (grid-based, 4-level quantization). Similarity scoring matches spec (exact=1.0, off-by-one=0.5, off-by-two+=0.0). The FileSystemTemplateStore implements all 6 protocol methods. Three issues require attention: (1) the `compute_layout_similarity` function signature in the PLAN adds `grid_cols` and `grid_rows` parameters not present in the spec's signature, (2) the `list_versions` method in the store returns ALL versions including soft-deleted ones which creates inconsistency with how `get_template` hides deleted versions, and (3) the PLAN's `_compute_otsu_threshold` is a reasonable adaptive thresholding approach but spec says "adaptive thresholding" which typically means local/block-based thresholding, not global Otsu.

---

## 2. Check Results

### 2.1 Does fingerprinting algorithm match spec section 5.4?

**PASS.** The PLAN correctly implements all 5 steps from spec section 5.4 lines 531-541:

| Spec Step | PLAN Section | Status |
|-----------|-------------|--------|
| 1. Render at 150 DPI | `compute_layout_fingerprint_from_file` uses `config.fingerprint_dpi` | Correct |
| 2. Convert to grayscale | Step 5a: `page.convert("L")` | Correct |
| 3. Adaptive thresholding | Step 5b: MedianFilter + Otsu threshold | See issue below |
| 4a. Divide into 16x20 grid | Step 5c-e: uses `config.fingerprint_grid_cols/rows` | Correct |
| 4b. Fill ratio per cell | Step 5f: `dark_count / total` | Correct |
| 4c. Quantize to 4 levels | `_quantize_fill_ratio` with thresholds 0.05/0.25/0.60 | Correct |
| 4d. NxM matrix IS the fingerprint | 1 byte per cell, `bytearray` | Correct |
| 5. Multi-page concatenation | Step 6: concatenates per-page fingerprints | Correct |

**MINOR ISSUE -- Thresholding approach:** The spec says "adaptive thresholding" (line 534), which in image processing typically means local/block-level thresholding (e.g., cv2.adaptiveThreshold). The PLAN uses global Otsu thresholding, which is a reasonable pure-Pillow approach but technically not "adaptive" in the classical sense. This is acceptable since Pillow lacks a built-in adaptive threshold and the PLAN avoids adding OpenCV as a dependency. **Recommendation:** Add a code comment explaining this design choice.

### 2.2 Does similarity scoring match spec?

**PASS.** PLAN section 1.6 correctly implements spec section 5.4 lines 547-560:

| Spec Rule | PLAN Implementation | Status |
|-----------|-------------------|--------|
| Exact match = 1.0 | `diff == 0: score_sum += 1.0` | Correct |
| Off-by-one = 0.5 | `diff == 1: score_sum += 0.5` | Correct |
| Off-by-two+ = 0.0 | `else: score_sum += 0.0` | Correct |
| Page count mismatch = 0.0 | `pages_a != pages_b: return 0.0` | Correct |
| Score = sum/total | `score_sum / total_cells` | Correct |

**MINOR ISSUE -- Function signature divergence:** The spec (line 547) defines `compute_layout_similarity(fp_a: bytes, fp_b: bytes) -> float` with only two parameters. The PLAN adds `grid_cols: int` and `grid_rows: int` as required parameters. This is a defensible design choice (the function needs grid dimensions to determine page boundaries), but it deviates from the spec's signature. **Recommendation:** Either match the spec signature and embed grid dimensions in the fingerprint bytes (e.g., a 4-byte header), or document the deviation clearly. The current approach is pragmatic and acceptable.

### 2.3 Does FileSystemTemplateStore implement all 6 FormTemplateStore protocol methods?

**PASS.** All 6 methods from `protocols.py:86-134` are implemented:

| Protocol Method | PLAN Section | Signature Match |
|----------------|-------------|-----------------|
| `save_template(template: FormTemplate) -> None` | 3.4 | Exact match |
| `get_template(template_id, version=None) -> FormTemplate \| None` | 3.5 | Exact match |
| `list_templates(tenant_id, source_format, active_only) -> list[FormTemplate]` | 3.6 | Exact match |
| `list_versions(template_id) -> list[FormTemplate]` | 3.7 | Exact match |
| `delete_template(template_id, version=None) -> None` | 3.8 | Exact match |
| `get_all_fingerprints(tenant_id, source_format) -> list[tuple[...]]` | 3.9 | Exact match |

### 2.4 Is version increment correct on update?

**PASS.** PLAN section 4.4, step 3: `new_version = existing.version + 1`. The updated template is built with this incremented version and saved as a new entry. The previous version is not modified (immutable once saved). This matches spec section 5.3 lines 499-500: "When an admin updates a template, the version number increments. The previous version is retained."

### 2.5 Is soft-delete correct (no hard delete)?

**PASS.** The PLAN correctly implements soft-delete:
- Files are NOT removed from disk (PLAN section 3.8: "Files are NOT removed from disk (audit trail preservation)")
- Deletion state tracked in `_meta.json` via `_TemplateMeta` model (not in `FormTemplate` model)
- Matches spec section 9.1 line 1298: "Hard deletion is not supported (audit trail preservation)"
- `delete_template(id)` with no version sets `all_deleted=True`
- `delete_template(id, version=N)` adds N to `deleted_versions`
- `get_template` returns `None` for deleted templates/versions

**MINOR ISSUE -- list_versions inconsistency:** PLAN section 3.7 says `list_versions` returns ALL versions "including soft-deleted versions." However, `get_template` hides deleted versions by returning `None`. This means `list_versions` can return templates that `get_template` would refuse to return. This is not necessarily wrong (the spec says "list all versions" at line 1262), but the PLAN should clarify this is intentional and document the behavior. The `_load_template_file` helper used by `list_versions` bypasses the deletion check, which is correct for this use case.

### 2.6 Does api.py cover all CRUD operations from spec section 9?

**PASS.** All 6 template CRUD operations from the `FormPluginAPI` protocol (spec section 9.1 lines 1233-1300) are covered:

| Spec Method | PLAN Section | Status |
|------------|-------------|--------|
| `list_templates(tenant_id, source_format)` | 4.7 | Covered |
| `get_template(template_id, version)` | 4.6 | Covered |
| `list_template_versions(template_id)` | 4.8 | Covered |
| `create_template(template_def)` | 4.3 | Covered |
| `update_template(template_id, template_def)` | 4.4 | Covered |
| `delete_template(template_id, version)` | 4.5 | Covered |

Non-CRUD operations from `FormPluginAPI` (`render_document`, `preview_extraction`, `match_document`, `extract_form`) are correctly out of scope for issue #61.

### 2.7 Are test cases comprehensive?

**PASS WITH NOTE.** Test count by file:

| Test File | Test Count | Coverage |
|-----------|-----------|----------|
| `test_matcher.py` | 17 | Fingerprint computation, quantization, similarity, file-based |
| `test_stores.py` | 18 | Protocol compliance, CRUD, filters, soft-delete, round-trip, persistence |
| `test_api.py` | 13 | All 6 API methods, error paths, fingerprint failure handling |
| **Total** | **48** | Meets target of ~48 tests |

The test coverage is thorough. Key scenarios verified:
- Deterministic fingerprinting
- Boundary values for quantization thresholds
- Multi-page fingerprints
- Page count mismatch in similarity
- Protocol isinstance check
- Bytes round-trip (hex serialization)
- Cross-instance persistence
- Soft-delete with get_template returning None
- Version increment on update
- Fingerprint failure graceful degradation

### 2.8 Does the stores/ subdirectory follow project patterns?

**PASS.** The PLAN creates `stores/__init__.py` with a public re-export of `FileSystemTemplateStore`, following the same pattern used in the project (e.g., `ingestkit_excel` uses subdirectory modules with `__init__.py` re-exports). The store uses structural subtyping (Protocol, not ABC), Pydantic v2 for the internal `_TemplateMeta` model, and the `ingestkit_forms` logger namespace.

---

## 3. Issues Summary

| # | Severity | Issue | Recommendation |
|---|----------|-------|----------------|
| 1 | LOW | `compute_layout_similarity` signature adds `grid_cols`/`grid_rows` params not in spec | Document deviation; pragmatic choice, acceptable |
| 2 | LOW | Otsu threshold is global, not "adaptive" per spec wording | Add code comment explaining Pillow limitation |
| 3 | LOW | `list_versions` returns soft-deleted versions but `get_template` hides them | Document intentional behavior; consider adding `include_deleted` param |

No blocking issues found. All three are low-severity and can be addressed with comments/documentation during PATCH.

---

## 4. Verdict

**PASS_WITH_ISSUES** -- The PLAN is ready for PATCH. The three low-severity issues should be noted but do not block implementation. The PLAN correctly implements the spec's fingerprinting algorithm, similarity scoring, all 6 store protocol methods, version incrementing, soft-delete semantics, and all 6 CRUD API operations. Test coverage is comprehensive at 48 tests across 3 files.

---

AGENT_RETURN: .agents/outputs/plan-check-61-021526.md
