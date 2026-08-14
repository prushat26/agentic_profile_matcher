"""
app.py - Streamlit Interface with Automatic Resume Ingestion
"""

import streamlit as st
import chromadb
import json

# Import functions from existing untouched modules
from resume_rag import build_vector_db
from openai import OpenAI
from matching_agent import agent_graph, compare_candidates, generate_interview_questions, process_feedback_node, rank_candidates_node, generate_report_node

# Direct OpenAI client initialization (uses OPENAI_API_KEY from .env)
openai_client = OpenAI()

st.set_page_config(page_title="RAG Candidate Matching Agent", layout="wide")

# ==========================================
# AUTOMATIC INGESTION CHECK ON APP STARTUP
# ==========================================
@st.cache_resource
def initialize_system():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="resumes")
    
    # Auto-run build_vector_db from your resume_rag.py if collection is empty
    if collection.count() == 0:
        with st.spinner("Initializing vector store & ingesting PDF resumes from resumes/..."):
            build_vector_db(resumes_dir="resumes")
            
    return collection.count()

indexed_count = initialize_system()

st.title("🤖 Candidate Matching & Screening Agent")
st.caption(f"Status: **{indexed_count} Resume(s) active in Vector Database**")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

# Sidebar Controls
st.sidebar.header("Job Specification Input")
sample_jd = st.sidebar.text_area(
    "Paste Job Description:", 
    height=250, 
    value="Looking for a Python Developer with Machine Learning experience, React, and at least 3 years of experience."
)

if st.sidebar.button("Run Full Agent Pipeline"):
    if indexed_count == 0:
        st.error("⚠️ No resumes indexed! Please ensure PDF files exist in 'resumes/' directory.")
    else:
        with st.spinner("Executing LangGraph agent workflow..."):
            initial_input = {
                "messages": [("user", sample_jd)],
                "jd_raw": sample_jd,
                "human_feedback": "",
                "feedback_turns": 0
            }
            res = agent_graph.invoke(initial_input)
            st.session_state.agent_state = res
            st.session_state.messages.append({"role": "assistant", "content": res.get("final_report", "No report generated.")})

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Natural Language Query Input
if user_prompt := st.chat_input("Ask candidate queries, request comparisons, or adjust requirements..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if st.session_state.agent_state:
        state = st.session_state.agent_state
        shortlist = state.get("shortlist", [])
        candidate_pool = state.get("candidate_pool", [])
        reqs = state.get("requirements", {})

        # Build full candidate context for GPT
        candidates_context = json.dumps(shortlist if shortlist else candidate_pool, indent=2)

        # Call OpenAI directly for chat turns
        with st.spinner("Thinking..."):
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert HR recruitment assistant chatting with a hiring manager. "
                            "You have full access to the processed candidate pool and job requirements provided in the context. "
                            "If the user requests updated criteria, new skill priorities, or adjustments, YOU MUST RESCORE AND RE-RANK "
                            "Maintain state and memory across the entire chat conversation. "
                            "When the user adjusts criteria or asks to re-rank, calculate NEW revised scores/ranks and "
                            "REMEMBER those new scores/ranks for all future follow-up questions in this session. "
                            "Never silently revert to initial baseline scores once a re-ranking or criteria adjustment has occurred."
                            "Answer user questions dynamically and flexibly. You can compare specific candidates side by side, "
                            "explain why candidate A beat candidate B, answer deep questions about specific individuals, "
                            "generate tailored interview questions, or re-rank candidates based on updated requirements. "
                            "DO NOT output an unprompted 10-candidate shortlist template unless specifically asked. "
                            "Address the user's explicit question directly and conversationally."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Current Job Requirements:\n{json.dumps(reqs, indent=2)}\n\n"
                            f"Candidate Data Pool:\n{candidates_context}\n\n"
                            f"User Prompt: {user_prompt}"
                        ),
                    },
                ],
                temperature=0.2,
            )
            response = completion.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        st.warning("Please run the initial agent pipeline first from the sidebar!")