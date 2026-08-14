# 🤝 AI Talent Matcher: LangGraph Resume Matching Agent

An intelligent, multi-stage recruitment and candidate screening agent powered by **LangGraph**, **Hybrid RAG Search (ChromaDB + BM25 + Bi-Encoder RRF)**, and **Cross-Encoder Reranking**. Features a dynamic Streamlit chat interface for real-time requirement adjustments, candidate head-to-head comparisons, and automated match explainability reports.

## 🌟 Key Features
* **🤖 LangGraph State Architecture:** Directed state machine controlling end-to-end recruitment workflows with human-in-the-loop feedback loops.
* **🔍 Hybrid RAG Retrieval:** Combines local ChromaDB vector database, sparse keyword matching (BM25), and dense semantic search fused via Reciprocal Rank Fusion (RRF).
* **🎯 High-Precision Reranking:** Fine-grained candidate re-ranking using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to capture deep requirement-resume interactions.
* **💬 Mid-Conversation Refinement:** Interactive chat interface allows hiring managers to adjust criteria mid-flight and immediately re-evaluate candidates.
* **📊 Multi-Round Screening & Explainability:** Automatically generates candidate match reports, highlights strengths/gaps, produces head-to-head comparisons, and suggests tailored interview questions.

## 🛠️ Tech Stack
* **Agent Orchestration:** LangGraph, LangChain
* **LLM Engine:** OpenRouter API (`openai/gpt-4o-mini`) via OpenAI SDK
* **Vector Store & Retrieval:** ChromaDB, `sentence-transformers`, `rank-bm25`
* **Validation & Tools:** Pydantic, FileSystem Tools (`fs_tools.py`)
* **Interface:** Streamlit

## 📁 Repository Structure
```text
agentic_profile_match/
├── data/resumes/         # PDF & text resume files
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── matching_agent.py # LangGraph State Machine (Choice A)
│   └── tools/
│       ├── __init__.py
│       ├── fs_tools.py      # File system operations (Milestone 1)
│       ├── resume_rag.py    # Ingestion & ChromaDB (Milestone 2)
│       ├── job_matcher.py   # Hybrid Search & Cross-Encoder (Milestone 2)
│       └── llm_tools.py     # Pydantic schemas & comparison tools
├── app.py                   # Streamlit Interactive UI
├── test_scenarios.py        # 5 Evaluation Flows
├── requirements.txt
└── .env
---
```
## 🏗️ Agent Workflow Topology


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
Interface: Streamlit
