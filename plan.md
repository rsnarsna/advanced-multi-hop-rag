# Advanced Multi-Hop RAG Architecture and Implementation Plan

This system integrates a Hybrid Search architecture (Vector Semantics via Neon PGVector + Knowledge Graphs via Neo4j AuraDB) and recursive LangGraph agent reasoning to resolve complex, multi-hop queries.

## 1. Correct Architecture

The state-of-the-art architecture for Multi-Hop RAG combines scalable **Vector Storage** with structured **Graph Storage** connected via shared metadata, orchestrated by an **Agentic Planner** with strict boundaries and evaluations.

### Components

#### A. Ingestion & Knowledge Construction Pipeline (Async & Scalable)
1.  **Document Parsers & Cleaners**: Extract and sanitize text and structured data.
2.  **Semantic Chunker**: Intelligent chunking boundaries using LangChain splitters.
3.  **Knowledge Extractor (Hybrid Ontology via LLM & Pydantic)**:
    *   **Base Ontology Validation**: Enforces strict extraction of foundational node types like `Person`, `Company`, `Event`, `Concept`, `Document` using Pydantic and JSON validation.
    *   **Dynamic Domain Expansion**: Allows the LLM to apply custom relationship labels (e.g., `ACQUIRED`, `INVESTED_IN`) within the boundaries of the base node schema to handle domain-specific nuances.
    *   Includes retry/fallback logic for extraction hallucinations.
4.  **Hybrid Cloud Storage Layer (Tightly Linked & Batched)**:
    *   **Vector Database (Neon PGVector)**: High-throughput async vector store via connection pooling (PgBouncer/asyncpg).
    *   **Graph Database (Neo4j Cloud / AuraDB)**: Strict relational database.
    *   **Metadata Integration Layer**: Exact `doc_id` and `chunk_id` metadata is shared between PGVector records and Neo4j nodes to enable seamless, bidirectional "retrieval fusion" when traversing from vectors to graphs and vice versa.

#### B. Query & Retrieval Engine (Bounded & Parallelized)
1.  **Query Analyzer / Router**: A LangGraph node that classifies questions into Simple vs. Complex.
2.  **Query Decomposer**: Breaks complex questions into actionable sub-queries.
3.  **Iterative Multi-Hop Retriever Engine (LangGraph)**:
    *   **Parallel Execution**: Sub-queries are executed as parallel branches within LangGraph for latency optimization.
    *   **Step 1:** Semantic vector search (PGVector) for sub-queries.
    *   **Step 2:** Deep retrieval traversal (Neo4j Cypher) linking off extracted semantic entities based on shared metadata.
    *   **Step 3 (The Evaluator Node):** Checks if the aggregated context can answer the sub-query.
    *   **Hop Boundaries**: Strict `max_iterations (3-4)` are enforced at the graph level to prevent explosive latency, runaway costs, and endless loops. If the hop limit is hit, the agent synthesizes the best available partial answer.
4.  **Re-Ranker & Context Compressor**: A cross-encoder dynamically filters and deduplicates the merged vector/graph sub-networks to map accurately into the synthesis prompt.

#### C. Generation, Synthesis & Dual-Strategy Evaluation
1.  **Response Synthesizer**: An LLM Node synthesizing context and attaching precise inline citations to root `doc_id` chunks.
2.  **Evaluation Telemetry (Dual Strategy)**:
    *   **Real-time Dev/Prod Traceability (LangSmith)**: Attached directly to the agent pipeline for immediate visibility into node latencies, prompt performance, and trace-level debugging.
    *   **Batch Offline Benchmarking (RAGAS)**: A separated CI/CD workflow testing massive offline datasets against Faithfulness, Answer Relevance, and Context Precision to quantitatively validate new embedding models or chunking strategies.

---

## 2. Implementation Plan

Here is the phased, actionable breakdown for building the Agent using LangChain, LangGraph, Neon PGVector, and Neo4j.

### Phase 1: Robust Scaffolding, Connecting Pooling & Ingestion Foundations
- **Goal:** Set up scalable connections, parallel data ingestion, and base metadata architectures.
- **Tasks:**
  - Setup `.env` and connection libraries (`langchain`, `asyncpg`, `neo4j`).
  - Configure **Neon PGVector** pooling and database schema linking `doc_id` metadata.
  - Implement basic parsing, semantic chunking, and async bulk vector insertion.

### Phase 2: Hybrid Knowledge Graph Extraction & Metatdata Linking
- **Goal:** Extract mixed static/dynamic entities safely and fuse them directly to Vector chunks.
- **Tasks:**
  - Define the base Pydantic schemas (Person, Company, Event, Concept, Document).
  - Configure LLM prompt extraction to strictly fill base nodes but allow dynamic relationship mapping (`LLMGraphTransformer`).
  - Execute batched, error-handled `UNWIND` Cypher ingestions to Neo4j, rigidly enforcing the `doc_id` link payload between the SQL rows and Graph nodes.

### Phase 3: LangGraph Setup & Parallel Sub-Querying
- **Goal:** Build the agent state machine ensuring sub-queries run fast.
- **Tasks:**
  - Define LangGraph `State` tracking queries, parallel contexts, and iteration counts.
  - Implement Router and Decomposer Nodes.
  - Setup `Send` API or parallel branching in LangGraph for independent sub-query isolated workflows.

### Phase 4: Bounded Hybrid Retrieval & Fusion
- **Goal:** Hit vectors, traverse graphs, link the data, and strictly control the iteration loop.
- **Tasks:**
  - Implement Retriever Nodes targeting specific data sources (Neo4j and Neon).
  - Write the Fusion logic: Use extracted entities from Neon vector chunks to look up 2-hop neighbor relationships in Neo4j using the aligned `doc_id`.
  - Add the `Evaluation Node` edge logic checking context sufficiency.
  - **CRITICAL:** Add the `hop_count` variable to graph state. Force the graph to route to the Synthesizer if `hop_count >= 4`.
  - Add Re-ranking.

### Phase 5: Synthesis, LangSmith Tracing & RAGAS
- **Goal:** Produce answers, capture live traces, and build offline tests.
- **Tasks:**
  - Implement the generation node to cite both the text chunks and relationship edges.
  - Wrap in a robust FastAPI service containing the LangGraph pipeline invocation.
  - Enable **LangSmith** in the `.env` configuration to ensure all chains and LLM calls in Phase 3/4 are traced automatically out of the box.
  - Build a separate Python script `/tests/benchmark_ragas.py` mapping retrieved contexts against synthetic datasets for offline A/B testing of the prompt structures and vector configurations.
