# 🤝 AI Talent Matcher: LangGraph Resume Matching Agent

An intelligent, multi-stage recruitment and candidate screening agent powered by **LangGraph**, **Hybrid RAG Search** (BM25 + Bi-Encoder RRF), and **Cross-Encoder Reranking**. Features a dynamic Streamlit chat interface for real-time requirement adjustments, candidate head-to-head comparisons, and automated match explainability reports.

---

## 🌟 Key Features

* **🤖 LangGraph State Architecture:** Directed state machine controlling end-to-end recruitment workflows with human-in-the-loop feedback loops.
* **🔍 Hybrid RAG Retrieval:** Combines sparse keyword matching (**BM25**) with dense semantic search (**MiniLM Bi-Encoder**) fused via **Reciprocal Rank Fusion (RRF)**.
* **🎯 High-Precision Reranking:** Fine-grained candidate re-ranking using a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to capture deep requirement-resume interactions.
* **💬 Mid-Conversation Refinement:** Interactive chat interface allows hiring managers to adjust criteria mid-flight and immediately re-evaluate candidates.
* **📊 Multi-Round Screening & Explainability:** Automatically generates candidate match reports, highlights strengths/gaps, produces head-to-head comparisons, and suggests tailored interview questions.

---

## 🏗️ Agent Workflow Topology

```text
                  ┌───────────────────────────────┐
                  │            START              │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Parse JD & Extract Req     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      Search & Hybrid RAG      │
                  │   (BM25 + Bi-Encoder + RRF)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Rank & Rerank Candidates   │
                  │        (Cross-Encoder)        │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │     Generate Match Report     │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Human Feedback Loop /      │
                  │    Refinement Routing         │
                  └──────────────┬────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
          [Refine Query]                  [Approved]
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │             END               │
                  └───────────────────────────────┘

🛠️ Tech Stack
Agent Orchestration: LangGraph, LangChain

LLM Engine: OpenRouter API (openai/gpt-4o-mini) via OpenAI SDK

Search & Embeddings: sentence-transformers, rank-bm25, pydantic

Interface: Streamlit
