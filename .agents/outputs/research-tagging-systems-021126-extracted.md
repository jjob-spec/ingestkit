I now have comprehensive data. Let me compile the research report.

---

# Research Report: Tagging, Taxonomy, and Metadata-Based Filtering Systems for RAG Pipelines

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Vector Store Metadata Filtering Patterns](#2-vector-store-metadata-filtering-patterns)
3. [RAG Framework Projects with Tagging/Metadata](#3-rag-framework-projects)
4. [Document Ingestion and Classification Tools](#4-document-ingestion-tools)
5. [Query-Time Tag Resolution / Self-Querying](#5-query-time-tag-resolution)
6. [Multi-Tenant Document Organization](#6-multi-tenant-document-organization)
7. [Auto-Tagging and Classification Pipelines](#7-auto-tagging-and-classification)
8. [Supporting Tools and Models](#8-supporting-tools)
9. [Common Patterns and Anti-Patterns](#9-common-patterns-and-anti-patterns)
10. [Relevance to IngestKit](#10-relevance-to-ingestkit)

---

## 1. Executive Summary

The ecosystem for document tagging and metadata filtering in RAG systems has matured significantly. The dominant approach is **pre-filtering on indexed metadata fields** during vector search, with vector databases (Qdrant, Milvus, Weaviate) all converging on this strategy. For auto-tagging, two camps exist: **LLM-based structured extraction** (LangChain OpenAI Metadata Tagger, Haystack MetadataEnricher, RAGFlow auto-metadata) and **lightweight model-based NER** (GLiNER, Hugging Face classifiers). Multi-tenancy is universally handled via **tenant-scoped payload filtering** rather than separate collections. Query-time tag resolution is addressed by **self-querying retrievers** (LangChain, Haystack) that use an LLM to parse natural language into structured filters before search.

---

## 2. Vector Store Metadata Filtering Patterns

### 2.1 Qdrant

- **GitHub**: [qdrant/qdrant](https://github.com/qdrant/qdrant) | **28,757 stars** | Actively maintained (pushed 2026-02-12)
- **Filtering Strategy**: **Filterable HNSW index** -- a hybrid approach that adds specialized links between points that survive filtering, maintaining HNSW graph integrity. The query planner uses cardinality estimation to decide between:
  - **Payload index search** (when filter cardinality is below threshold -- scans fewer points)
  - **Filterable HNSW** (when cardinality is above threshold -- uses specialized graph links)
- **Filter Syntax**: JSON-based with nested conditions (`must`, `should`, `must_not`), supporting `match`, `range`, `geo_bounding_box`, `values_count`, and full-text match
- **Tag Schema**: Arbitrary JSON payload. No fixed schema enforced. Fields are indexed explicitly via `create_payload_index`. Types: keyword, integer, float, geo, text, uuid, datetime
- **Multi-tenancy**: `is_tenant: true` flag on payload index. Single collection, partitioned by tenant payload field. Tiered multitenancy in v1.16 allows promoting tenants from shared shard to dedicated shard
- **Tag Aliases / Fuzzy**: No native tag alias support. Fuzzy matching must be implemented at the application layer
- **Best Practices**: Index every field you filter by; use `uuid` index type for payload-heavy collections; avoid creating separate collections per tenant

**Key Reference**: [A Complete Guide to Filtering in Vector Search](https://qdrant.tech/articles/vector-search-filtering/)

### 2.2 Milvus

- **GitHub**: [milvus-io/milvus](https://github.com/milvus-io/milvus) | **42,733 stars** | Actively maintained
- **Filtering Strategy**: **Always pre-filtering** -- filters applied first, then ANN search follows on remaining points. Two categories: standard filtering (filter before search) and iterative filtering (progressively refine)
- **Filter Syntax**: SQL-like expressions via `expr` parameter. Example: `category == "electronics" and price < 500`
- **Tag Schema**: Typed scalar fields defined at collection creation. Supports int, float, varchar, bool, JSON, array
- **Multi-tenancy**: Four levels of isolation -- database-level (highest), collection-level, partition-level, and partition-key-level (most scalable). Partition key allows millions of tenants sharing a single collection with automatic partition management
- **RBAC**: Role-based access control with granular privileges (SELECT, INSERT) on specific collections/databases
- **Milvus 2.5**: Introduced full-text search, more powerful metadata filtering, and JSON/array field operators

**Key Reference**: [Designing Multi-Tenancy RAG with Milvus](https://milvus.io/blog/build-multi-tenancy-rag-with-milvus-best-practices-part-one.md)

### 2.3 Weaviate

- **GitHub**: [weaviate/weaviate](https://github.com/weaviate/weaviate) | **15,578 stars** | Actively maintained
- **Filtering Strategy**: Per-property inverted index buckets within each tenant's shard. Filtering and vector search run within the isolated tenant shard
- **Filter Syntax**: GraphQL-based with `where` clause supporting `And`, `Or`, `Equal`, `NotEqual`, `GreaterThan`, `LessThan`, `Like`, `ContainsAny`, `ContainsAll`
- **Tag Schema**: Properties defined per collection class. Each filterable/searchable property gets a dedicated inverted index bucket
- **Multi-tenancy**: Native per-tenant bucketed architecture. Each tenant gets a dedicated shard with physical isolation. Scales to 50,000+ active tenants per node. Tenant data is not visible across tenants
- **Tag Aliases / Fuzzy**: No native tag alias or fuzzy tag matching

**Key Reference**: [Weaviate Multi-Tenancy Architecture Explained](https://weaviate.io/blog/weaviate-multi-tenancy-architecture-explained)

### 2.4 Chroma

- **GitHub**: [chroma-core/chroma](https://github.com/chroma-core/chroma) | **26,091 stars** | Actively maintained
- **Filtering Strategy**: Basic in-memory metadata filtering with operators `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`
- **Filter Syntax**: Dictionary-based `where` clause and `where_document` for content filtering
- **Tag Schema**: Arbitrary key-value metadata dictionaries attached to documents at upsert time
- **Multi-tenancy**: No native multi-tenancy support. Must be implemented at the application layer (separate collections or metadata filtering)
- **Limitations**: Simpler filtering compared to Qdrant/Milvus/Weaviate. Primarily designed for prototyping and smaller-scale applications

### 2.5 Pinecone

- **Stars**: Proprietary (not open-source), but widely used
- **Filtering Strategy**: Pre-filtering. Searches with metadata filters retrieve exactly the specified number of nearest-neighbor results that match filters, often with lower latency than unfiltered searches
- **Multi-tenancy**: Two approaches:
  - **Namespaces**: Physical isolation per tenant. Each namespace stored separately in serverless architecture. Can delete entire namespace for tenant offboarding. Cannot query across namespaces
  - **Metadata filtering**: Logical isolation. Can query across tenants if needed. Recommended when cross-tenant queries are a future possibility
- **Tag Schema**: Key-value metadata. Supports string, number, boolean, and string arrays

**Key Reference**: [Namespaces vs. Metadata Filtering](https://docs.pinecone.io/troubleshooting/namespaces-vs-metadata-filtering)

---

## 3. RAG Framework Projects with Tagging/Metadata

### 3.1 RAGFlow

- **GitHub**: [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | **73,212 stars** | Very actively maintained
- **Auto-Metadata**: Uses LLMs to automatically extract metadata during file parsing (v0.23.0+). Users define custom metadata fields with names, descriptions, and examples to guide the LLM. Can restrict output to preset allowed values
- **Tag Sets**: Dedicated tag set feature for auto-tagging chunks. Tag sets are created as separate datasets (XLSX/CSV/TXT format), then applied to target datasets. Chunks are auto-tagged during parsing, and query-time tags boost matching chunks in retrieval
- **Metadata Filtering at Retrieval**: Chat assistants support metadata-based filtering with union logic operations
- **Document Classification**: DeepDoc component uses YOLOv8 for layout recognition, OCR for text extraction, and table structure recognition (TSR)
- **Architecture**: Monolithic platform with pluggable parsers (Docling, built-in). Python/FastAPI backend. Supports Elasticsearch and Infinity as vector stores
- **Relevance**: High. The tag set and auto-metadata features are directly relevant to admin-controlled taxonomy. Self-hosted deployment model aligns with on-premises requirements

### 3.2 LlamaIndex

- **GitHub**: [run-llama/llama_index](https://github.com/run-llama/llama_index) | **46,963 stars** | Very actively maintained
- **Metadata Filtering**: `MetadataFilter`, `MetadataFilters`, `FilterOperator` classes. Supports `AND`/`OR` filter combination. Works with Qdrant, Chroma, Pinecone, Weaviate, Milvus integrations (not the default in-memory store)
- **Auto-Retrieval**: LLM-powered feature that automatically tags incoming queries with metadata and applies filtering. The LLM infers which metadata filters to apply from the natural language query
- **Filter Operators**: `EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NIN`, `CONTAINS`, `TEXT_MATCH`, `ANY`, `ALL`
- **Metadata Extraction**: `MetadataExtractor` pipeline with built-in extractors: `TitleExtractor`, `QuestionsAnsweredExtractor`, `SummaryExtractor`, `KeywordExtractor`, `EntityExtractor`
- **Architecture**: Modular framework. Pluggable vector stores, LLMs, and embedding models. Strong protocol/interface-based design
- **Relevance**: Very high. The MetadataExtractor pipeline and auto-retrieval patterns are directly applicable. Backend-agnostic design matches IngestKit's protocol approach

**Key References**: [Metadata Filtering with LlamaIndex and Milvus](https://milvus.io/docs/llamaindex_milvus_metadata_filter.md), [Pinecone Vector Store Metadata Filter](https://docs.llamaindex.ai/en/stable/examples/vector_stores/pinecone_metadata_filter/)

### 3.3 LangChain

- **GitHub**: [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | **126,555 stars** | Very actively maintained
- **Metadata Tagging**: `OpenAIMetadataTagger` (now `create_metadata_tagger`) -- automates metadata extraction from documents using an LLM. Schema defined as a Pydantic model or JSON schema. Works best on whole documents before splitting
- **Self-Querying Retriever**: `SelfQueryRetriever` uses an LLM to parse natural language queries into structured filters + semantic query. Supports Chroma, Pinecone, Weaviate, Elasticsearch, Milvus, Qdrant, OpenSearch
- **Filter Syntax**: `Comparator` and `Operator` enums. Comparators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contain`, `like`, `in`, `nin`. Operators: `and`, `or`, `not`
- **Document Loaders**: All loaders populate `metadata` dict on `Document` objects. Common fields: `source`, `page`, `title`. Custom metadata added via transformers
- **Architecture**: Modular chain-based. LCEL (LangChain Expression Language) for composing retrieval pipelines
- **Relevance**: High. The metadata tagger pattern (schema-driven LLM extraction) and self-querying retriever pattern are reference implementations for what IngestKit could expose

**Key References**: [OpenAI Metadata Tagger](https://python.langchain.com/docs/integrations/document_transformers/openai_metadata_tagger/), [Self-Querying Retriever](https://towardsdatascience.com/how-to-build-a-rag-system-with-a-self-querying-retriever-in-langchain-16b4fa23e9ad/)

### 3.4 Haystack (deepset)

- **GitHub**: [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | **24,173 stars** | Very actively maintained
- **Metadata Filtering**: Dictionary-based filter syntax with comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`) and logical operators (`AND`, `OR`, `NOT`). Operator availability varies by Document Store integration
- **DocumentClassifier**: Automatically enriches documents with categories, sentiments, topics, or other metadata at index time. Adds classification label and score to `Document.meta`
- **MetadataRouter**: Routes documents to different pipeline branches based on metadata values. Used after classifiers to direct documents by language, topic, etc.
- **MetadataEnricher (Custom Component)**: Uses LLM structured outputs with Pydantic models to extract and assign metadata fields. Metadata inherits to chunks when using `DocumentSplitter`
- **Query Metadata Extraction**: LLM-based extraction of metadata filters from user queries (cookbook pattern). Conditional edges skip filter generation if no metadata found in query
- **Architecture**: DAG-based pipeline with typed components. Strong separation between indexing and query pipelines. Components communicate via typed inputs/outputs
- **Relevance**: Very high. The pipeline architecture (indexing-time classification -> metadata routing -> filtered retrieval) is the closest analog to IngestKit's Inspector -> Classifier -> Processor chain

**Key References**: [Metadata Filtering Tutorial](https://haystack.deepset.ai/tutorials/31_metadata_filtering), [Document Classification at Index Time](https://haystack.deepset.ai/tutorials/16_document_classifier_at_index_time), [Automated Metadata Enrichment](https://haystack.deepset.ai/cookbook/metadata_enrichment), [Extract Metadata Filters from Query](https://haystack.deepset.ai/cookbook/extracting_metadata_filters_from_a_user_query)

### 3.5 Dify

- **GitHub**: [langgenius/dify](https://github.com/langgenius/dify) | **129,499 stars** | Very actively maintained
- **Metadata as Knowledge Filter** (v1.1.0): Users add and manage metadata for documents in knowledge bases. Default metadata (filename, uploader, upload date) auto-assigned. Custom metadata fields with specific data types. Batch editing support
- **Knowledge Pipeline**: Swappable OCR, parsing, extraction, vector stores, and rerankers. Three chunking strategies: General, Parent-Child, Q&A
- **Auto-Tagging**: LLM and Code nodes for entity extraction, summarization, classification, redaction
- **Plugin Architecture**: Open plugin ecosystem. Enterprises select tools that fit their needs
- **Relevance**: Moderate. Dify is more of an application platform than a library. The metadata management UX and Knowledge Pipeline architecture are good reference designs, but IngestKit operates at a lower level

**Key Reference**: [Dify v1.1.0 Metadata Filtering](https://dify.ai/blog/dify-v1-1-0-filtering-knowledge-retrieval-with-customized-metadata)

### 3.6 txtai

- **GitHub**: [neuml/txtai](https://github.com/neuml/txtai) | **12,139 stars** | Actively maintained
- **Document Model**: Input as `(id, data, tags)` tuples. If data is a dictionary, all fields stored and queryable via SQL. Text field indexed for similarity search
- **Query Layer**: Joins relational store (SQL) and similarity index. `similar()` clause runs vector search, results fed to SQL query for filtering. Supports hybrid SQL + semantic queries
- **Pipeline Architecture**: Generic `__call__` interface. Pipelines for labeling, transcription, translation, summarization, entity extraction
- **Semantic Graph**: Builds knowledge graphs connecting related content, enabling graph-based traversal alongside vector search
- **Architecture**: All-in-one framework. Embedded database (SQLite + vector index). No external dependencies required
- **Relevance**: Moderate. The SQL + similarity hybrid query pattern is elegant. The `(id, data, tags)` tuple model is simple and applicable. Good reference for lightweight, self-contained deployments

**Key Reference**: [txtai Query Guide](https://neuml.github.io/txtai/embeddings/query/)

### 3.7 Verba (Weaviate)

- **GitHub**: [weaviate/Verba](https://github.com/weaviate/Verba) | **7,559 stars** | Last pushed July 2025
- **Smart Filtering**: Filter by document and document type. Customizable metadata settings
- **GLiNER Integration**: Uses GLiNER zero-shot NER to:
  1. **Tag chunks during ingestion**: Passes each chunk + target tags to GLiNER. Tags produced only if high confidence, reducing false positives
  2. **Parse queries at retrieval time**: Matches incoming queries against labeled metadata to filter results
- **Architecture**: Modular with ReaderManager, ChunkerManager, EmbeddingManager. Built on Weaviate vector store
- **Relevance**: High. The GLiNER-based auto-tagging pattern (lightweight NER model instead of full LLM) is highly relevant for on-premises deployment where LLM calls are expensive

### 3.8 Cognita (TrueFoundry)

- **GitHub**: [truefoundry/cognita](https://github.com/truefoundry/cognita) | **4,316 stars** | Last pushed November 2025
- **Metadata Store**: Separate metadata store for collection-level information. Vector DB stores embeddings + per-chunk metadata
- **Metadata Enrichment**: Language detection during loading. Presigned URLs and surrounding context added to chunk metadata before response
- **Architecture**: Fully modular. Pluggable data loaders, parsers, embedders, vector stores, retrievers
- **Relevance**: Moderate. The modular architecture and metadata store separation are good patterns. Less mature on auto-tagging

### 3.9 R2R (SciPhi-AI)

- **GitHub**: [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) | **7,677 stars** | Last pushed November 2025
- **RESTful API**: Full document management via REST. Upload, update, delete documents and metadata
- **Search**: Hybrid search with reciprocal rank fusion. Knowledge graph support
- **Metadata Filtering**: Customizable search settings with metadata filters. Still maturing (open issues on filtering in streaming mode)
- **Architecture**: Containerized. PostgreSQL + pgvector for vector storage. RESTful API layer
- **Relevance**: Moderate. The REST API design for document management is a good reference. Less relevant for library-level integration

---

## 4. Document Ingestion and Classification Tools

### 4.1 Unstructured.io

- **GitHub**: [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | **13,962 stars** | Actively maintained
- **Document Processing**: Partition documents into typed elements (NarrativeText, Title, Table, Image, etc.). 25+ file types supported
- **Auto-Metadata**: Automatically attaches metadata to each element: page number, file name, element type, hierarchy, XY coordinates, language
- **Classification**: Applies classifiers for document type, topic, sentiment. Entity extraction. PII redaction rules
- **Metadata Standardization**: Standardized metadata schema across document types. Reduces manual annotation
- **Architecture**: ETL pipeline. 30+ source/destination connectors. Pluggable parsers per document type
- **Relevance**: Very high. The element-level metadata and standardized schema across document types is exactly what IngestKit needs. The PII redaction integration is also relevant

### 4.2 Docling (IBM)

- **GitHub**: [docling-project/docling](https://github.com/docling-project/docling) | **52,840 stars** | Actively maintained
- **Document Parsing**: Layout-aware parsing. Identifies text, formulas, tables, images. Preserves document structure (headings, paragraphs, lists)
- **Metadata Extraction**: Structural metadata (document hierarchy, element types, bounding boxes). Integrates with LlamaIndex and LangChain
- **Architecture**: MIT-licensed toolkit. Focused on parsing, not end-to-end RAG. Clean API for integration into larger pipelines
- **Relevance**: High as a parsing component. Docling's structured output could feed IngestKit's inspector/classifier stage

### 4.3 Morphik

- **GitHub**: [morphik-org/morphik-core](https://github.com/morphik-org/morphik-core) | **3,474 stars** | Actively maintained
- **Features**: Fast metadata extraction including bounding boxes, labeling, classification. Multimodal document understanding (diagrams, CAD, tables, slides, scanned PDFs)
- **License**: Business Source License 1.1 (source-available, not fully open-source)
- **Relevance**: Low-moderate. BSL license limits usefulness. Interesting for multimodal classification patterns

---

## 5. Query-Time Tag Resolution / Self-Querying

### 5.1 LangChain SelfQueryRetriever

The most mature implementation of query-time tag resolution:

1. User submits natural language query (e.g., "Find horror movies made after 1980 with explosions")
2. **Query Constructor** (LLM-powered) decomposes into:
   - Semantic query: "explosions"
   - Metadata filters: `genre == "horror" AND year > 1980`
3. Filters applied to vector store, then similarity search on filtered subset
4. Conditional routing: if no metadata extracted, falls back to pure vector search

Supports 10+ vector store backends. Attribute descriptions provided at initialization guide the LLM in filter construction.

### 5.2 Haystack Query Metadata Extraction

Similar pattern but implemented as a pipeline component:

1. `MetadataExtractor` component uses LLM to parse query
2. Extracts structured filters matching defined metadata schema
3. `MetadataRouter` routes to appropriate retriever with filters applied
4. Conditional edge skips filter generation if no metadata detected

### 5.3 RAGFlow Query Tagging

RAGFlow applies tag sets to queries at retrieval time. Each query is tagged using the configured tag sets, and chunks carrying matching tags receive boosted retrieval scores. This is a term-matching approach rather than LLM-based extraction.

### 5.4 GLiNER for Query Parsing

Verba and Sease have explored using GLiNER as a lightweight alternative to LLMs for query parsing:
- Pass query text + target entity types to GLiNER
- Extract entities/tags without LLM API call
- Advantages: runs on CPU, sub-100ms latency, zero-shot (no training data needed), deterministic
- Disadvantage: less flexible than LLM-based parsing for complex queries

**Key Reference**: [GLiNER as an Alternative to LLMs for Query Parsing](https://sease.io/2025/10/gliner-as-an-alternative-to-llms-for-query-parsing-introduction.html)

---

## 6. Multi-Tenant Document Organization

### Patterns Observed (in order of isolation strength)

| Strategy | Isolation | Scalability | Cross-Tenant Query | Used By |
|----------|-----------|-------------|-------------------|---------|
| **Separate database/cluster** | Complete | Low (resource-heavy) | No | Milvus (database-level) |
| **Separate collection** | Strong | Low-medium | No | Anti-pattern in most systems |
| **Namespace** | Strong (physical) | Medium | No | Pinecone |
| **Dedicated shard** | Strong (physical) | Medium-high | No | Weaviate, Qdrant (promoted tenants) |
| **Partition key** | Moderate (logical) | Very high | Yes (filter-based) | Milvus, Qdrant |
| **Metadata filter** | Moderate (logical) | Very high | Yes | All vector stores |

### Best Practices

- **Anti-pattern**: Creating separate collections per tenant. Exhausts cluster resources, causes OOM errors, degrades performance
- **Recommended**: Use tenant-scoped payload filtering within a single collection, or namespace-based isolation if cross-tenant queries are not needed
- **RBAC**: Milvus provides role-based access control at database/collection level. Other systems rely on application-layer enforcement
- **JWT integration**: AWS and Azure patterns use JWT claims to map tenant identity to metadata filters at the API gateway level
- **Qdrant tiered multitenancy** (v1.16): Tenants start in shared shard, get promoted to dedicated shard when they grow -- best of both worlds

**Key References**: [Qdrant Multitenancy](https://qdrant.tech/articles/multitenancy/), [Milvus Multi-Tenancy Best Practices](https://milvus.io/blog/build-multi-tenancy-rag-with-milvus-best-practices-part-one.md), [Pinecone Multi-Tenancy](https://docs.pinecone.io/guides/index-data/implement-multitenancy), [The Right Approach to Authorization in RAG](https://www.osohq.com/post/right-approach-to-authorization-in-rag)

---

## 7. Auto-Tagging and Classification Pipelines

### 7.1 LLM-Based Auto-Tagging

| Tool | Approach | Schema | When Applied |
|------|----------|--------|-------------|
| **LangChain MetadataTagger** | OpenAI function calling with Pydantic schema | User-defined Pydantic model | Before splitting (whole doc) |
| **Haystack MetadataEnricher** | LLM structured output with Pydantic schema | User-defined Pydantic model | During indexing pipeline |
| **RAGFlow Auto-Metadata** | LLM extraction guided by field descriptions | Admin-defined fields with descriptions/examples | During parsing |
| **Dify Knowledge Pipeline** | LLM/Code nodes for extraction | Custom per-pipeline | During ingestion workflow |

**Common Pattern**: Define a Pydantic model (or equivalent schema) describing desired metadata fields. Pass document content + schema to an LLM configured for structured output. LLM returns populated fields. Fields attached to document metadata and inherited by chunks.

### 7.2 Classifier-Based Auto-Tagging

| Tool | Approach | Model Type |
|------|----------|-----------|
| **Haystack DocumentClassifier** | Hugging Face zero-shot/fine-tuned classifiers | Transformer (BERT-family) |
| **GLiNER** (via Verba) | Zero-shot NER with bidirectional transformer | BERT-like (lightweight) |
| **OnPrem.LLM** | Multiple backends: HF transformers, scikit-learn, SetFit | Various |
| **Unstructured.io** | Document type + topic + sentiment classifiers | Multiple |

### 7.3 Rule-Based Auto-Tagging

| Tool | Approach |
|------|----------|
| **RAGFlow Tag Sets** | Term matching from predefined tag vocabulary (XLSX/CSV) |
| **OpenMetadata** | Regex patterns, NLP via Presidio, custom recognizers |
| **Clade** | Taxonomy nodes with keyword sets, Boolean OR matching via Solr |

### 7.4 OpenMetadata

- **GitHub**: [open-metadata/OpenMetadata](https://github.com/open-metadata/OpenMetadata) | **8,688 stars** | Actively maintained
- **Auto-Classification**: PII detection using NLP (Presidio framework). Auto-tags data as Sensitive/Non-Sensitive
- **Custom Recognizers**: Customizable regex and script-based recognizers for classification
- **Taxonomy**: Hierarchical classification system. Supports ISO 25964 and W3C SKOS/OWL standards
- **Relevance**: Moderate. More focused on data governance than RAG document tagging, but the PII detection and taxonomy management patterns are transferable

---

## 8. Supporting Tools and Models

### 8.1 GLiNER

- **GitHub**: [urchade/GLiNER](https://github.com/urchade/GLiNER) | **2,811 stars** | Actively maintained
- **Purpose**: Zero-shot Named Entity Recognition. Extracts arbitrary entity types from text without training data
- **Model Size**: Lightweight (BERT-based). Runs on CPU. Sub-100ms inference
- **Use in RAG**: (1) Tag chunks during ingestion by extracting entities matching a target tag vocabulary. (2) Parse user queries to extract filter metadata at retrieval time
- **GLiNER2**: Multi-task extension supporting NER, relation extraction, summarization, QA, and open information extraction
- **Relevance**: High for on-premises deployment. Avoids LLM API calls for tagging. Could serve as IngestKit's auto-tagger for entity-based metadata

### 8.2 Semantic Tag Filtering (Research Pattern)

An approach where tag labels themselves are embedded into vector space, enabling fuzzy/semantic tag matching:
- Instead of exact tag string matching, embed both the query tag and the stored tags
- Use cosine similarity between tag embeddings to find semantically equivalent tags
- Handles synonyms and aliases naturally (e.g., "HR" matches "Human Resources")
- Reported sub-second performance on 40,000 samples

**Key Reference**: [Introducing Semantic Tag Filtering](https://medium.com/data-science/introducing-semantic-tag-filtering-enhancing-retrieval-with-tag-similarity-4f1b2d377a10)

---

## 9. Common Patterns and Anti-Patterns

### Patterns (What Works)

1. **Schema-driven metadata extraction**: Define the target metadata schema (Pydantic model, JSON schema) upfront. Use LLM structured output or classifiers to populate. This ensures consistency and enables validation

2. **Pre-filtering over post-filtering**: All mature vector stores (Qdrant, Milvus, Weaviate) converge on filtering before vector search. Post-filtering wastes compute and risks discarding relevant results

3. **Metadata inherited by chunks**: Enrich metadata on whole documents before splitting. Document-level metadata (author, department, classification) propagates to all chunks automatically

4. **Tenant-as-payload-field**: Single collection with tenant ID as an indexed payload field. Avoids resource exhaustion from per-tenant collections. Qdrant's `is_tenant` flag and Milvus's partition key formalize this

5. **Conditional self-querying**: Use LLM to extract metadata filters from user queries, but fall back to pure vector search when no metadata is detected. Avoid forcing filters that do not exist

6. **Lightweight taggers for high-throughput**: GLiNER or Hugging Face zero-shot classifiers for entity/category tagging. Reserve LLM calls for complex documents or low-confidence cases (mirrors IngestKit's Tier 1/2/3 escalation pattern)

7. **Tag sets as controlled vocabulary**: RAGFlow's pattern of maintaining tag sets as separate datasets that are applied to target data. This keeps taxonomy admin-controlled and auditable

8. **Hybrid search (vector + keyword + metadata)**: Combine semantic search with BM25/full-text and metadata filters. Reciprocal rank fusion for merging results

### Anti-Patterns (What Fails)

1. **Separate collection per tenant**: Exhausts RAM, causes OOM errors, degrades performance. Universal anti-pattern across all vector database documentation

2. **Free-form metadata without schema**: Allowing arbitrary metadata keys/types leads to inconsistent filtering, query failures, and silent data loss. Always define and validate schemas

3. **Post-filtering with low-cardinality filters**: Retrieving K vectors then filtering by a rare tag often returns zero results. Pre-filtering is essential for rare tags

4. **LLM-only auto-tagging without validation**: LLMs hallucinate tags that do not exist in the controlled vocabulary. Always validate against allowed values (RAGFlow's "restrict to defined values" mode)

5. **Exact string matching for tags without normalization**: "Human Resources" vs "HR" vs "human resources" will miss matches. Solutions: case normalization, alias tables, or semantic tag filtering

6. **Overloading vector dimensions with metadata**: Encoding metadata into the embedding vector (e.g., concatenating tag embeddings with content embeddings) pollutes the semantic space. Keep metadata in payload/scalar fields, content in vectors

7. **Ignoring payload index creation**: In Qdrant, failing to create payload indexes prevents cardinality estimation, causing the query planner to choose suboptimal strategies. Always index filterable fields

8. **Applying metadata tagger after chunking**: LangChain's metadata tagger documentation explicitly warns this produces inferior results. Tag whole documents first, then split

---

## 10. Relevance to IngestKit

Based on this research, here are the most directly applicable patterns for IngestKit's on-premises, pluggable-backend, admin-controlled use case:

### Highest-Priority Patterns to Adopt

1. **Schema-driven metadata model** (Pydantic): Define a `DocumentMetadata` model with fields like `department`, `document_type`, `sensitivity_level`, `tags`, etc. This becomes the contract between the processor stage and downstream vector stores. Haystack and LangChain both use this pattern

2. **Tiered auto-tagging** (already built into IngestKit's Tier 1/2/3 design):
   - Tier 1 (rule-based): Extract metadata from file properties, sheet names, header patterns (already in Inspector)
   - Tier 2 (lightweight NER): GLiNER-based entity extraction for content-derived tags. Runs on CPU, no LLM needed
   - Tier 3 (LLM-based): Structured extraction via LLM for complex documents (already in LLM Classifier)

3. **Tag vocabulary as configuration** (RAGFlow pattern): Admin-defined tag sets loaded from config files. Auto-taggers constrained to this vocabulary. Prevents hallucinated tags

4. **Tenant-scoped payload filtering**: `tenant_id` (already in IngestKit's config and `IngestKey`) becomes a payload filter field. Single collection per deployment, partitioned by tenant

5. **Metadata pass-through protocol**: Define a `MetadataBackend` protocol that processors populate and vector store backends consume. This keeps IngestKit's core metadata-aware without coupling to any specific vector store's filtering syntax

### Reference Implementations to Study

| Pattern | Best Reference |
|---------|---------------|
| Pydantic metadata schema + LLM extraction | Haystack `MetadataEnricher` cookbook |
| Self-querying / query-time filter extraction | LangChain `SelfQueryRetriever` |
| Lightweight NER auto-tagging | Verba + GLiNER integration |
| Admin-controlled tag vocabulary | RAGFlow tag sets |
| Tenant isolation in vector store | Qdrant `is_tenant` + tiered multitenancy |
| Document-level metadata -> chunk inheritance | LangChain `OpenAIMetadataTagger` (apply before splitting) |
| Conditional filter routing | Haystack `MetadataRouter` |

---

## Sources

### Vector Stores
- [A Complete Guide to Filtering in Vector Search - Qdrant](https://qdrant.tech/articles/vector-search-filtering/)
- [Qdrant Multitenancy Guide](https://qdrant.tech/articles/multitenancy/)
- [Qdrant 1.16 - Tiered Multitenancy](https://qdrant.tech/blog/qdrant-1.16.x/)
- [Milvus Multi-Tenancy RAG Best Practices](https://milvus.io/blog/build-multi-tenancy-rag-with-milvus-best-practices-part-one.md)
- [Milvus Filtered Search](https://milvus.io/docs/filtered-search.md)
- [Milvus 2.5 Full-Text Search and Metadata Filtering](https://milvus.io/blog/introduce-milvus-2-5-full-text-search-powerful-metadata-filtering-and-more.md)
- [Weaviate Multi-Tenancy Architecture Explained](https://weaviate.io/blog/weaviate-multi-tenancy-architecture-explained)
- [Pinecone Namespaces vs. Metadata Filtering](https://docs.pinecone.io/troubleshooting/namespaces-vs-metadata-filtering)
- [Pinecone Implement Multitenancy](https://docs.pinecone.io/guides/index-data/implement-multitenancy)
- [Pinecone RAG with Access Control](https://www.pinecone.io/learn/rag-access-control/)

### RAG Frameworks
- [RAGFlow GitHub](https://github.com/infiniflow/ragflow)
- [RAGFlow Auto-Metadata Documentation](https://www.ragflow.io/docs/auto_metadata)
- [RAGFlow Tag Sets Documentation](https://ragflow.io/docs/use_tag_sets)
- [LlamaIndex Metadata Filtering with Milvus](https://milvus.io/docs/llamaindex_milvus_metadata_filter.md)
- [LlamaIndex Pinecone Metadata Filter](https://docs.llamaindex.ai/en/stable/examples/vector_stores/pinecone_metadata_filter/)
- [LangChain Metadata Filtering - GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/metadata-filtering-in-langchain/)
- [LangChain Self-Querying Retriever](https://towardsdatascience.com/how-to-build-a-rag-system-with-a-self-querying-retriever-in-langchain-16b4fa23e9ad/)
- [Haystack Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- [Haystack Document Classification at Index Time](https://haystack.deepset.ai/tutorials/16_document_classifier_at_index_time)
- [Haystack Automated Metadata Enrichment](https://haystack.deepset.ai/cookbook/metadata_enrichment)
- [Haystack Extract Metadata Filters from Query](https://haystack.deepset.ai/cookbook/extracting_metadata_filters_from_a_user_query)
- [Dify v1.1.0 Metadata Filtering](https://dify.ai/blog/dify-v1-1-0-filtering-knowledge-retrieval-with-customized-metadata)
- [Dify Knowledge Pipeline](https://dify.ai/blog/introducing-knowledge-pipeline)
- [txtai GitHub](https://github.com/neuml/txtai)
- [txtai Query Guide](https://neuml.github.io/txtai/embeddings/query/)
- [Verba GitHub](https://github.com/weaviate/Verba)
- [Cognita GitHub](https://github.com/truefoundry/cognita)
- [R2R GitHub](https://github.com/SciPhi-AI/R2R)

### Document Ingestion Tools
- [Unstructured.io GitHub](https://github.com/Unstructured-IO/unstructured)
- [Unstructured Metadata for RAG](https://unstructured.io/insights/how-to-use-metadata-in-rag-for-better-contextual-results)
- [Docling GitHub](https://github.com/docling-project/docling)
- [Morphik Core GitHub](https://github.com/morphik-org/morphik-core)

### Auto-Tagging and Classification
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [GLiNER as Alternative to LLMs for Query Parsing - Sease](https://sease.io/2025/10/gliner-as-an-alternative-to-llms-for-query-parsing-introduction.html)
- [OpenMetadata GitHub](https://github.com/open-metadata/OpenMetadata)
- [OpenMetadata Auto-Classification](https://docs.open-metadata.org/latest/how-to-guides/data-governance/classification/auto-classification)
- [Semantic Tag Filtering - TDS](https://medium.com/data-science/introducing-semantic-tag-filtering-enhancing-retrieval-with-tag-similarity-4f1b2d377a10)
- [Automated Content Tagging with Local LLMs](https://dasroot.net/posts/2026/02/automated-content-tagging-local-llms/)

### Multi-Tenant RAG
- [Building Successful Multi-Tenant RAG Applications - Nile](https://www.thenile.dev/blog/multi-tenant-rag)
- [The Right Approach to Authorization in RAG - Oso](https://www.osohq.com/post/right-approach-to-authorization-in-rag)
- [HoneyBee: RBAC for Vector Databases via Dynamic Partitioning](https://arxiv.org/html/2505.01538v1)
- [Multi-Tenant RAG with Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/multi-tenant-rag-implementation-with-amazon-bedrock-and-amazon-opensearch-service-for-saas-using-jwt/)

### Overviews
- [15 Best Open-Source RAG Frameworks in 2026 - Firecrawl](https://www.firecrawl.dev/blog/best-open-source-rag-frameworks)
- [2025 Guide to Open-Source RAG Frameworks - Morphik](https://www.morphik.ai/blog/guide-to-oss-rag-frameworks-for-developers)
- [RAG Frameworks You Should Know - DataCamp](https://www.datacamp.com/blog/rag-framework)
- [Optimizing Vector Search with Metadata Filtering and Fuzzy Filtering - KX Systems](https://medium.com/kx-systems/optimizing-vector-search-with-metadata-filtering-41276e1a7370)
