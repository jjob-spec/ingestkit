# ingestkit-excel — Roadmap

Deferred items from team review and OSS benchmark analysis. These are valid concerns that don't belong in v1 of the `ingestkit-excel` package — either because they're system-level concerns owned by the caller/platform, or because they require operational data from production to implement well.

Reference: `docs/github-open-source-rag-pattern-benchmark-2026-02-11.md`

---

## System-Level (owned by VE-RAG-System or ingestkit-core)

These items require coordination beyond a single file-processing library. The package provides the data and hooks; the platform enforces the policy.

### 1. Durable State Machine / Dead-Letter Queue
**Review ref:** P0-3
**What:** Recoverable async workflow with stage-state tracking, timeout handling, stale-lease reclaim, and dead-letter semantics for permanently failed ingests.
**Why deferred:** `ingestkit-excel` is a library that processes one file and returns a `ProcessingResult`. Queue management, retry orchestration, and state persistence are the calling system's responsibility. Embedding this here couples the library to specific infrastructure (Redis, Celery, etc.) and makes it unusable outside the RAG system.
**What we provide instead:** `ProcessingResult` includes typed stage artifacts, error codes, and `WrittenArtifacts` — everything the caller needs to implement its own state machine.

### 2. Two-Phase Atomicity / Compensation
**Review ref:** P0-4
**What:** Transactional guarantees across vector store and structured DB writes — rollback vectors if DB fails, and vice versa.
**Why deferred:** Full two-phase commit requires an orchestrator with transaction coordination. The library can't own cross-backend transactions without making assumptions about the caller's infrastructure.
**What we provide instead:** `WrittenArtifacts` in `ProcessingResult` lists every vector point ID and DB table name created. The caller can use `delete_by_ids` and `drop_table` to implement compensating actions per its own policy. Protocols include these rollback methods.

### 3. Multi-Tenant Isolation Enforcement
**Review ref:** P0-8
**What:** Enforce tenant isolation in vector indexes, DB schemas, storage paths, and query filters.
**Why deferred:** Isolation enforcement (separate collections per tenant, row-level security, filtered queries) is a platform concern that spans all ingestkit packages and the query layer.
**What we provide instead:** `tenant_id` flows through `ExcelProcessorConfig` → `IngestKey` → `ChunkMetadata` → all persisted artifacts. The package propagates tenant identity everywhere; the platform enforces boundaries.

### 4. Ingestion Quality Dashboard
**Review ref:** P1-13
**What:** Dashboard showing parse coverage, duplicate ratio, chunk token variance, fallback frequency, per-type failure modes.
**Why deferred:** The library emits all the raw data (structured logs, typed stage artifacts, error codes). Building a dashboard is a visualization and aggregation concern for the platform.
**What we provide instead:** `ProcessingResult` with `ParseStageResult`, `ClassificationStageResult`, `EmbedStageResult`, normalized `ErrorCode` values — all structured and machine-readable.

### 5. Cost Controls / Token Budget Caps
**Review ref:** P1-14
**What:** Token and embedding budget caps, early-exit rules, per-file cost estimates.
**Why deferred:** Token budgets span the full RAG pipeline. Per-file cost tracking is useful but the budget enforcement belongs in the orchestrator that decides whether to continue processing.
**What we provide instead:** Stage artifacts include durations and counts (texts embedded, LLM calls made). The caller can implement budget enforcement using this data.

### 6. Concurrency / Locking for Same-File Re-ingest
**Review ref:** P1-15
**What:** Ingest lock keyed by idempotency key with safe reentry behavior.
**Why deferred:** File-level locking requires shared state (Redis lock, DB advisory lock, filesystem flock) that depends on deployment infrastructure the library shouldn't assume.
**What we provide instead:** Deterministic `ingest_key` computed from `content_hash + source_uri + parser_version`. The caller uses this key to check for in-progress or completed ingests before calling `process()`.

### 7. Schema Evolution / Versioning Strategy
**Review ref:** P1-17
**What:** Version classification, chunk, and metadata schemas. Support migration compatibility across versions.
**Why deferred:** This must be solved once across all ingestkit packages (excel, pdf, docx, image, audio, video). Solving it in ingestkit-excel alone means re-solving it 5 more times with potential inconsistencies.
**Where it belongs:** Future `ingestkit-core` shared package that defines versioned base schemas inherited by all format-specific packages.

---

## Strategic / Post-v1 Maturity

These items are valid but require operational experience or production data to implement well. Building them before v1 works end-to-end would be premature.

### 8. Cross-Parser A/B Evaluation
**Review ref:** P2-18
**What:** Run multiple parser strategies on the same corpus and compare quality/cost deltas.
**Why deferred:** You need a working parser before you can A/B test parsers. Build v1, measure quality on real files, then add experiment infrastructure.
**Prerequisite:** Working v1 + real client files from at least one pilot.

### 9. Policy-Controlled Fail-Open vs Fail-Closed
**Review ref:** P2-19
**What:** Deployment policy toggle for air-gapped security posture (strict mode rejects ambiguous files) vs availability mode (best-effort processing).
**Why deferred:** The v2 spec implements fail-closed as the default. Knowing which scenarios warrant fail-open requires real failure data from production. Premature policy configuration adds complexity without evidence.
**What we provide instead:** Fail-closed default with normalized error codes. The caller can implement its own policy layer on top of `ProcessingResult.errors`.

### 10. Benchmark Corpus for Target Verticals
**Review ref:** P2-20
**What:** Regression corpus for legal, manufacturing, financial services, agriculture, landscaping with expected parse/chunk outcomes.
**Why deferred:** Meaningful benchmarks require real client files from each vertical. The test fixtures in v1 cover structural coverage (Type A/B/C, edge cases). Vertical-specific benchmarks come from pilot deployments.
**Prerequisite:** Files from at least 2-3 pilots per vertical.

### 11. Model Governance Lifecycle
**Review ref:** P2-21
**What:** Pin model versions, maintain offline model manifest, run acceptance tests before model upgrades.
**Why deferred:** With one classification model (`qwen2.5:7b`) and one optional reasoning model (`deepseek-r1:14b`) running on local Ollama, this is overhead without payoff. Governance matters when managing multiple model versions across deployments.
**What we provide instead:** `parser_version` in config and metadata. Model names are configurable. The caller can implement version pinning in its deployment tooling.

### 12. Large Workbook Performance Profiling
**Review ref:** P2-22
**What:** Streaming/partial-load architecture for workbooks that exceed memory limits.
**Why deferred:** Full streaming requires rewriting the openpyxl parsing layer — significant effort that should be driven by real memory pressure data, not speculation.
**What we provide instead:** `max_rows_in_memory` config param (default 100k). Sheets exceeding this limit are skipped with `W_ROWS_TRUNCATED`. This prevents OOM without requiring architectural changes.
**Trigger to revisit:** When real production files consistently hit the limit, profile memory usage and decide whether streaming, partial reads, or chunked loading is the right approach.

### 13. Data Retention / Deletion Policy
**Review ref:** P2-23
**What:** Retention TTLs, cascade delete semantics (vector + DB + logs), audit proof of deletion.
**Why deferred:** Retention policies are system-level concerns that span all ingestkit packages and the RAG platform. The delete mechanism exists; the policy belongs elsewhere.
**What we provide instead:** `WrittenArtifacts` tracks everything written. Protocols include `delete_by_ids` and `drop_table`. The platform can implement TTL-based cleanup using `ingest_key` and creation timestamps.

### 14. Compliance / Audit Controls
**Review ref:** P2-24
**What:** Audit events for ingest lifecycle, config changes, operator actions.
**Why deferred:** An audit subsystem that indexes, retains, and makes events queryable is a separate service. It spans all ingestkit packages and the broader platform.
**What we provide instead:** Structured PII-safe logs with normalized error codes, stage durations, and typed artifacts. These are the raw audit events — the platform decides how to persist and query them.

---

## Code Quality / Technical Debt

Items identified during the February 2026 cross-project code review. These are small-to-medium improvements that should be addressed during v1 stabilization, before production use.

### 15. Extract Shared Retry Utility
**Review ref:** Code review 2026-02-14
**What:** Qdrant and Ollama backends both implement exponential backoff with nearly identical logic (sleep, retry count, backoff multiplier). Extract to a shared `retry_with_backoff()` utility in `ingestkit-core`.
**Why it matters:** Duplicated retry logic diverges over time. One backend gets a fix; the other doesn't. A shared utility ensures consistent behavior across all backends and future packages (pdf, docx).
**Effort:** Small — extract existing code, add tests, update both call sites.

### 16. Async I/O Support
**Review ref:** Code review 2026-02-14
**What:** All I/O is synchronous (`httpx.post()` for Ollama, blocking Qdrant calls). Add `httpx.AsyncClient` option for backends that make network calls.
**Why deferred from v1:** Synchronous is correct for a library that processes one file at a time. The caller can parallelize across files with threads or processes.
**Trigger to revisit:** When the VE-RAG-System orchestration layer needs to process multiple files concurrently within a single process (e.g., async FastAPI endpoint calling ingestkit).
**Effort:** Medium — requires async protocol variants or dual sync/async implementations.

### 17. Externalize LLM Prompt Templates
**Review ref:** Code review 2026-02-14
**What:** Classification and reasoning prompts are hardcoded strings in `llm_classifier.py`. Move to configurable prompt templates (Jinja2 or simple string templates) loadable from files.
**Why it matters:** Prompt tuning is the most common iteration cycle during pilot deployments. Editing Python source to change a prompt is error-prone and requires redeployment. External templates allow prompt iteration without code changes.
**Effort:** Small — extract strings to template files, add `prompt_template_dir` config parameter.

### 18. Configurable Logger Names
**Review ref:** Code review 2026-02-14
**What:** All modules use hardcoded `logging.getLogger("ingestkit_excel")`. When embedding the library in a larger application, callers may want logs under their own logger hierarchy (e.g., `"ve_rag.ingest.excel"`).
**Why deferred from v1:** Standard practice for libraries is `getLogger(__name__)` which produces `ingestkit_excel.parser_chain`, `ingestkit_excel.inspector`, etc. The current approach is functional but non-standard.
**Effort:** Small — replace hardcoded names with `__name__`, optionally add a `configure_logging(logger_name=)` setup function.

### 19. Standardize Test Mock Usage
**Review ref:** Code review 2026-02-14
**What:** `conftest.py` defines excellent Protocol-conforming mock classes (`MockVectorStore`, `MockStructuredDB`, `MockLLM`, `MockEmbedding`). However, some tests use `MagicMock()` instead. Standardize on the typed mocks everywhere.
**Why it matters:** `MagicMock()` accepts any method call without error — it can't catch Protocol violations. The typed mocks from conftest enforce correct method signatures and return types.
**Effort:** Small — audit tests for `MagicMock()` usage, replace with conftest fixtures.

### 20. Integration Test Markers
**Review ref:** Code review 2026-02-14
**What:** Add `@pytest.mark.integration` tests that run against real Qdrant, Ollama, and SQLite backends. Keep them skipped by default (require `--run-integration` flag).
**Why it matters:** 603 unit tests verify logic with mocks. Integration tests verify that real backends accept the data formats, handle edge cases, and return expected responses. Critical before production deployment.
**Effort:** Medium — write test fixtures that spin up or connect to local services, add pytest markers and CI configuration.

### 21. Performance Documentation
**Review ref:** Code review 2026-02-14
**What:** Document expected throughput, memory usage, and latency for representative workbooks (small/medium/large). Include parser chain timing, classification latency, and embedding throughput.
**Why it matters:** Without baseline numbers, operators can't capacity-plan or detect regressions. The stage artifacts already capture durations — just need to run and document them.
**Effort:** Small — run benchmarks on sample files, document in `docs/performance.md`.

---

## Implementation Priority (when ready)

| Priority | Items | Trigger |
|----------|-------|---------|
| **Immediate** (v1 stabilization) | Retry utility (#15), logger names (#18), mock standardization (#19) | Before production use |
| **First** (with ingestkit-core) | Schema versioning (#7), tenant isolation enforcement (#3), retry utility (#15) | When building second ingestkit package (ingestkit-pdf) |
| **Second** (with platform) | State machine (#1), atomicity (#2), concurrency locks (#6), retention (#13), audit (#14) | When building the VE-RAG-System orchestration layer |
| **Third** (pre-pilot) | Prompt templates (#17), integration tests (#20), performance docs (#21) | Before first pilot deployment |
| **Fourth** (post-pilot) | Dashboard (#4), cost controls (#5), benchmarks (#10), A/B eval (#8), model governance (#11) | After first paid pilot with real client files |
| **Fifth** (when needed) | Fail-open policy (#9), large workbook streaming (#12), async I/O (#16) | When production data reveals specific needs |
