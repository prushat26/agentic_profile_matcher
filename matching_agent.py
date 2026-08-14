"""
matching_agent.py - Proactively Fixed LangGraph State Machine
"""

import json
import os
from typing import Annotated, Any, Dict, List, Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import existing module functionalities
from fs_tools import read_file, list_files, write_file, search_in_file
from job_matcher import hybrid_search, rank_candidates_cross_encoder, run_job_matcher

load_dotenv()

# ==========================================
# 1. STATE DEFINITION
# ==========================================

class Candidate(TypedDict):
    id: str
    name: str
    score: float
    matched_skills: List[str]
    content: str
    reasoning: str
    recommendation: str

class Requirements(TypedDict):
    title: str
    must_have: List[str]
    nice_to_have: List[str]
    min_years_experience: float

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    jd_raw: str
    requirements: Requirements
    candidate_pool: List[Dict[str, Any]]
    shortlist: List[Candidate]
    final_report: str
    screening_round: int
    human_feedback: str
    feedback_turns: int

# ==========================================
# 2. STRUCTURED PYDANTIC SCHEMAS
# ==========================================

class ExtractedRequirements(BaseModel):
    title: str = Field(default="Position", description="Target job title")
    must_have: List[str] = Field(default_factory=list, description="Essential technical and soft skills required")
    nice_to_have: List[str] = Field(default_factory=list, description="Preferred but non-mandatory skills")
    min_years_experience: float = Field(default=0.0, description="Minimum required years of experience")

# ==========================================
# 3. HELPER & TOOL FUNCTIONS
# ==========================================

def extract_requirements(jd_text: str) -> Dict[str, Any]:
    """Parse job description safely into structured requirements."""
    if not jd_text or not jd_text.strip():
        return ExtractedRequirements().model_dump()
        
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ExtractedRequirements)
        prompt = f"Parse the following Job Description into structured requirements:\n\n{jd_text}"
        res = structured_llm.invoke(prompt)
        if isinstance(res, ExtractedRequirements):
            return res.model_dump()
        return ExtractedRequirements().model_dump()
    except Exception as e:
        print(f"⚠️ Warning during extract_requirements: {e}")
        return ExtractedRequirements(title="Software Role", must_have=[], nice_to_have=[], min_years_experience=0.0).model_dump()


def compare_candidates(candidate_ids: List[str], shortlist: List[Candidate]) -> str:
    """Head-to-head comparison between specified candidates from the shortlist."""
    if not shortlist:
        return "No candidates available in shortlist to compare."
        
    targets = [c for c in shortlist if any(cid.lower() in c.get("name", "").lower() for cid in candidate_ids) or c.get("id") in candidate_ids]
    
    # Fallback to top 2 if targets not explicitly resolved
    if len(targets) < 2:
        targets = shortlist[:2]
        
    if len(targets) < 2:
        return f"Insufficient candidates for head-to-head comparison (Found {len(targets)})."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Perform a strict head-to-head evaluation between these candidate profiles:\n\n{json.dumps(targets, indent=2)}"
    res = llm.invoke([SystemMessage(content="You are an expert HR recruiter comparing candidate profiles."), HumanMessage(content=prompt)])
    return res.content


def generate_interview_questions(candidate: Candidate, requirements: Requirements) -> List[str]:
    """Create targeted screening questions based on candidate skills and identified gaps."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = f"""
    Generate 5 screening questions for candidate '{candidate.get('name', 'Candidate')}' based on requirements:
    {json.dumps(requirements)}

    Candidate Matched Skills: {candidate.get('matched_skills', [])}
    Candidate Overview: {candidate.get('reasoning', '')}
    """
    res = llm.invoke(prompt)
    return [q.strip() for q in res.content.split("\n") if q.strip()]

# ==========================================
# 4. GRAPH NODES (WORKFLOW STEPS)
# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def parse_jd_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    raw_jd = state.get("jd_raw", "")
    if not raw_jd and messages:
        raw_jd = messages[-1].content
    return {"jd_raw": raw_jd, "feedback_turns": state.get("feedback_turns", 0)}


def extract_requirements_node(state: AgentState) -> Dict[str, Any]:
    jd_text = state.get("jd_raw", "")
    reqs = extract_requirements(jd_text)
    return {"requirements": reqs}


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    jd_text = state.get("jd_raw", "")
    reqs = state.get("requirements", {})
    min_exp = reqs.get("min_years_experience", 0.0)
    
    raw_res = run_job_matcher(jd_text=jd_text, min_years_required=min_exp, top_k=10)
    parsed = json.loads(raw_res)
    matches = parsed.get("top_matches", [])
    
    candidate_pool = []
    for idx, m in enumerate(matches):
        candidate_pool.append({
            "id": f"cand_{idx+1}",
            "name": m.get("candidate_name", f"Candidate_{idx+1}"),
            "score": m.get("match_score", 0.0),
            "matched_skills": m.get("matched_skills", []),
            "content": m.get("reasoning", "") + "\nExcerpts: " + " ".join(m.get("relevant_excerpts", [])),
            "reasoning": m.get("reasoning", ""),
            "recommendation": "Borderline"
        })
    return {"candidate_pool": candidate_pool, "screening_round": 1}


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    pool = state.get("candidate_pool", [])
    reqs = state.get("requirements", {})
    round_num = state.get("screening_round", 1)
    
    ranked = rank_candidates_cross_encoder(query=json.dumps(reqs), candidates=pool)
    
    for c in ranked:
        if c.get("score", 0) >= 75.0:
            c["recommendation"] = "Hire"
        elif c.get("score", 0) >= 50.0:
            c["recommendation"] = "Borderline"
        else:
            c["recommendation"] = "No-Hire"
            
    return {"shortlist": ranked[:10], "screening_round": min(3, round_num + 1)}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    shortlist = state.get("shortlist", [])
    reqs = state.get("requirements", {})
    
    prompt = f"""
    Generate an executive Candidate Match Report based on requirements:
    {json.dumps(reqs, indent=2)}

    Top Candidate Shortlist:
    {json.dumps(shortlist, indent=2)}

    For each candidate include:
    1. Overall Match Score & Recommendation (Hire / No-Hire / Borderline)
    2. Key Strengths & Matched Skills
    3. Identified Skill Gaps
    4. Actionable Improvement Suggestions (for Borderline candidates)
    """
    report_res = llm.invoke([SystemMessage(content="You are a senior recruitment agent."), HumanMessage(content=prompt)])
    
    return {
        "final_report": report_res.content,
        "messages": [AIMessage(content=report_res.content)]
    }


def should_continue(state: AgentState) -> Literal["human_feedback_loop", "end"]:
    feedback = state.get("human_feedback", "")
    turns = state.get("feedback_turns", 0)
    
    # Stop loop if feedback is empty, exit keyword received, or max turns reached (safeguard against infinite recursion)
    if not feedback or feedback.strip().lower() in ["approve", "exit", "done", "none"] or turns >= 5:
        return "end"
    return "human_feedback_loop"


def process_feedback_node(state: AgentState) -> Dict[str, Any]:
    feedback = state.get("human_feedback", "")
    current_reqs = state.get("requirements", {})
    turns = state.get("feedback_turns", 0)
    
    prompt = f"""
    Original Requirements: {json.dumps(current_reqs)}
    User Refinement Command: "{feedback}"
    
    Update the requirements JSON schema reflecting the user's modifications.
    """
    try:
        updated_reqs = llm.with_structured_output(ExtractedRequirements).invoke(prompt)
        reqs_dict = updated_reqs.model_dump() if isinstance(updated_reqs, ExtractedRequirements) else current_reqs
    except Exception:
        reqs_dict = current_reqs

    return {
        "requirements": reqs_dict,
        "human_feedback": "",  # Clear feedback buffer
        "feedback_turns": turns + 1,
        "messages": [AIMessage(content=f"Updated requirements: {json.dumps(reqs_dict, indent=2)}")]
    }

# ==========================================
# 5. BUILD & COMPILE GRAPH
# ==========================================

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("parse_jd", parse_jd_node)
    builder.add_node("extract_requirements", extract_requirements_node)
    builder.add_node("search_resumes", search_resumes_node)
    builder.add_node("rank_candidates", rank_candidates_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("process_feedback", process_feedback_node)
    
    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "extract_requirements")
    builder.add_edge("extract_requirements", "search_resumes")
    builder.add_edge("search_resumes", "rank_candidates")
    builder.add_edge("rank_candidates", "generate_report")
    
    builder.add_conditional_edges(
        "generate_report",
        should_continue,
        {
            "human_feedback_loop": "process_feedback",
            "end": END
        }
    )
    builder.add_edge("process_feedback", "search_resumes")
    
    return builder.compile()

agent_graph = build_graph()