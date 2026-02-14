# Tagging Systems Research (Readable Version)

## TL;DR
For RAG tagging and metadata filtering, the winning production pattern is:
1. Define a strict metadata schema.
2. Tag at document level before chunking.
3. Use pre-filtered vector search on indexed metadata.
4. Add optional query-time filter extraction only when useful.

## What Matters for ingestkit
- On-prem deployment.
- Pluggable backends and clean interfaces.
- Admin-controlled taxonomy/tag vocabulary.
- Multi-tenant isolation without exploding infrastructure.
- Predictable retrieval behavior under filters.

## Best Practices That Actually Work
- **Pre-filter before ANN search** (not post-filter).
- **Index every filterable metadata field**.
- **Single collection + `tenant_id` payload filter** for most cases.
- **Chunk metadata inheritance** from document-level tags.
- **Controlled vocabulary** with aliases/normalization.
- **Tiered auto-tagging**: rules -> lightweight classifier/NER -> LLM fallback.

## Anti-Patterns to Avoid
- Separate collection per tenant by default.
- Free-form unvalidated metadata keys.
- LLM-only tagging without allowed-value validation.
- Tagging after chunking (loses context quality).
- Exact string tag matching without alias normalization.

## Tool Shortlist

### Vector Stores (Filtering + Multi-Tenancy)
- **Qdrant**: strong payload filtering, `is_tenant`, good planner/index behavior.
- **Milvus**: strong filtered search and mature partition/isolation options.
- **Weaviate**: strong native multi-tenant shard model.
- **Chroma**: okay for prototypes, weaker at scale/tenant isolation.

### Framework Patterns to Reuse
- **LlamaIndex**: clean metadata filter abstractions and metadata extractors.
- **LangChain**: `SelfQueryRetriever` + metadata tagging flows.
- **Haystack**: metadata enrichment/routing pipeline style.
- **RAGFlow**: admin tag sets and constrained auto-tagging pattern.

### Auto-Tagging Choices
- **Rule-based**: cheapest, deterministic, high throughput.
- **GLiNER / lightweight classifiers**: scalable middle tier.
- **LLM structured extraction**: best for complex docs, gate with validation.

## Recommended v1 Architecture Pattern

```text
Ingest
  -> Document metadata extraction (schema validated)
  -> Normalize + alias resolution + vocabulary validation
  -> Chunking with metadata inheritance
  -> Vector upsert (vectors + metadata payload)

Query
  -> Optional query-time tag extraction (self-query)
  -> Build filter expression (tenant + tags + policy)
  -> Pre-filtered vector retrieval (hybrid optional)
  -> Rerank + answer
```

## Suggested v1 Stack (Pragmatic)
- Metadata schema: strict Pydantic model (required + optional fields).
- Taxonomy store: admin-managed config/table with aliases + allowed values.
- Vector backend: Qdrant or Milvus with indexed payload fields.
- Auto-tagging:
  - Tier 1: rules/regex/header heuristics.
  - Tier 2: GLiNER or small classifiers.
  - Tier 3: LLM structured tagging for hard cases.
- Query-time filters: conditional self-query (only when confidence is high).
- Multi-tenancy: `tenant_id` as mandatory filter field in retrieval path.

## Capability You’d Get with This
- Fast and precise filtered retrieval for department/type/security scoping.
- Reliable tenant isolation in one deployment.
- Admin-auditable taxonomy and tag behavior.
- Fewer hallucinated tags via constrained vocabulary validation.
- Better recall/precision balance through hybrid metadata + semantic retrieval.

## Original Extracted Report
- Full extracted report from the raw transcript:
  - `.agents/outputs/research-tagging-systems-021126-extracted.md`
