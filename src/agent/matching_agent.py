import os
import json
from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from langgraph.graph import StateGraph, END

# Imports from your Milestone 1 and Milestone 2 modules
from src.tools.fs_tools import list_files, read_file
from src.tools.job_matcher import hybrid_search, rank_candidates_cross_encoder

# Initialize OpenAI Client directly
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o-mini"

llm_client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# ==========================================
# 1. PYDANTIC SCHEMAS
# ==========================================
class ExtractedRequirements(BaseModel):
    must_have: List[str] = Field(description="Non-negotiable skills or minimum experience required")
    nice_to_have: List[str] = Field(description="Preferred or nice-to-have capabilities")


# ==========================================
# 2. AGENT STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    raw_jd: str
    must_have_reqs: List[str]
    nice_have_reqs: List[str]
    candidate_pool: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]]
    report: str
    human_feedback: Optional[str]
    screening_round: int


# ==========================================
# 3. HELPER & LLM TOOL FUNCTIONS
# ==========================================
def extract_requirements(jd: str) -> ExtractedRequirements:
    """Parses raw JD into must-have and nice-to-have requirements."""
    messages = [
        {"role": "system", "content": "You are an AI recruitment specialist. Parse job descriptions into structured requirements."},
        {"role": "user", "content": f"Extract requirements from this Job Description:\n\n{jd}"}
    ]
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_requirements",
                "strict": True,
                "schema": ExtractedRequirements.model_json_schema()
            }
        },
        temperature=0
    )
    return ExtractedRequirements.model_validate_json(response.choices[0].message.content)


def compare_candidates(candidate_a: Dict[str, Any], candidate_b: Dict[str, Any]) -> str:
    """Head-to-head candidate comparison."""
    prompt = f"""Compare these two candidate profiles for the role:
Candidate A: {json.dumps(candidate_a)}
Candidate B: {json.dumps(candidate_b)}

Highlight key trade-offs, strengths, weaknesses, and make a clear recommendation."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


def generate_interview_questions(candidate: Dict[str, Any], requirements: List[str]) -> str:
    """Creates tailored screening interview questions."""
    prompt = f"""Candidate Details: {json.dumps(candidate)}
Requirements: {requirements}

Generate 3 to 5 targeted technical and behavioral screening questions to evaluate skill gaps and experience."""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


# ==========================================
# 4. LANGGRAPH NODES
# ==========================================
def parse_jd_node(state: AgentState) -> Dict[str, Any]:
    """Extracts structured requirements from raw JD."""
    extracted = extract_requirements(state["raw_jd"])
    return {
        "must_have_reqs": extracted.must_have,
        "nice_have_reqs": extracted.nice_to_have
    }


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    """Uses Milestone 2 Hybrid RAG search to pull top resume matches."""
    query = " ".join(state["must_have_reqs"])
    if state.get("human_feedback"):
        query += f" {state['human_feedback']}"

    top_k = 10 if state.get("screening_round", 1) == 1 else 3

    retrieved = hybrid_search(
        query=query,
        documents=state["candidate_pool"],
        top_k=top_k
    )
    return {"shortlist": retrieved}


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    """Uses Cross-Encoder to perform deep semantic reranking."""
    query = " ".join(state["must_have_reqs"])
    reranked = rank_candidates_cross_encoder(query, state["shortlist"])
    return {"shortlist": reranked}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """Produces match report with strengths, gaps, and recommendations."""
    shortlist_summary = json.dumps(state["shortlist"], indent=2)
    prompt = f"""Generate an executive candidate evaluation report.
Must-Have Requirements: {state['must_have_reqs']}
Candidates: {shortlist_summary}

Structure the output as follows:
1. Executive Summary
2. Detailed Candidate Match Breakdown (Strengths & Gaps for each)
3. Borderline Candidate Improvement Recommendations
4. Final Hire / No-Hire Verdicts"""

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return {"report": response.choices[0].message.content}


def human_feedback_routing(state: AgentState) -> str:
    """Determines whether to loop back for refinement or end execution."""
    feedback = state.get("human_feedback")
    if feedback and feedback.strip().lower() not in ["approved", "no", "exit", "done"]:
        return "refine"
    return "end"


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================
def build_matching_agent():
    """Compiles Choice A LangGraph pipeline."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("parse_jd", parse_jd_node)
    workflow.add_node("search_resumes", search_resumes_node)
    workflow.add_node("rank_candidates", rank_candidates_node)
    workflow.add_node("generate_report", generate_report_node)

    # Wire Edges
    workflow.set_entry_point("parse_jd")
    workflow.add_edge("parse_jd", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")

    # Conditional Routing for Human Feedback
    workflow.add_conditional_edges(
        "generate_report",
        human_feedback_routing,
        {
            "refine": "search_resumes",
            "end": END
        }
    )

    return workflow.compile()