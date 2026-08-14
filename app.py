import os
import json
import streamlit as st

# Import LangGraph compiled workflow and tool helpers from src/agent/matching_agent
from src.agent.matching_agent import (
    build_matching_agent,
    compare_candidates,
    generate_interview_questions,
    AgentState
)
from src.tools.fs_tools import list_files, read_file

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Talent Matcher",
    page_icon="🤝",
    layout="wide"
)

st.title("🤝 AI Talent Matcher: LangGraph Recruitment Agent")
st.caption("Powered by LangGraph, Hybrid RAG Search (ChromaDB + BM25), and Cross-Encoder Reranking")

# Initialize Agent Graph
@st.cache_resource
def get_compiled_agent():
    return build_matching_agent()

agent_app = get_compiled_agent()


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [],
        "raw_jd": "",
        "must_have_reqs": [],
        "nice_have_reqs": [],
        "candidate_pool": [],
        "shortlist": [],
        "reasoning": {},
        "report": "",
        "human_feedback": None,
        "screening_round": 1
    }

if "candidate_pool_loaded" not in st.session_state:
    # Load candidate resumes from data/resumes directory using Milestone 1 tools
    try:
        resume_files = list_files("data/resumes")
        candidates = []
        for idx, file_path in enumerate(resume_files):
            content = read_file(file_path)
            candidate_name = os.path.basename(file_path).replace(".pdf", "").replace(".txt", "").replace("_", " ").title()
            candidates.append({
                "id": f"cand_{idx+1}",
                "name": candidate_name,
                "file_path": file_path,
                "content": content
            })
        st.session_state.agent_state["candidate_pool"] = candidates
        st.session_state.candidate_pool_loaded = True
    except Exception as e:
        st.warning(f"Could not automatically load resumes from data/resumes: {e}")
        st.session_state.agent_state["candidate_pool"] = []


# ==========================================
# 3. SIDEBAR: CONTROL & CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    st.subheader("1. Job Description")
    default_jd = """Senior Full Stack Engineer
Must have:
- 3+ years experience with Python and React
- Demonstrated experience in building RAG or AI workflows
- Strong knowledge of PostgreSQL and vector databases

Nice to have:
- Experience with LangGraph and Streamlit
- Docker and Cloud Deployment (AWS)"""
    
    raw_jd_input = st.text_area("Paste Job Description:", value=default_jd, height=200)
    
    st.markdown("---")
    st.subheader("2. Multi-Round Screening")
    current_round = st.session_state.agent_state.get("screening_round", 1)
    st.info(f"**Current Round:** Round {current_round}")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("Round 1 (Top 10)"):
            st.session_state.agent_state["screening_round"] = 1
            st.rerun()
    with col_r2:
        if st.button("Round 2 (Deep 3)"):
            st.session_state.agent_state["screening_round"] = 2
            st.rerun()

    st.markdown("---")
    if st.button("🚀 Run Agent Pipeline", type="primary"):
        st.session_state.agent_state["raw_jd"] = raw_jd_input
        with st.spinner("Executing LangGraph Workflow..."):
            # Run the compiled graph state machine
            final_output = agent_app.invoke(st.session_state.agent_state)
            st.session_state.agent_state.update(final_output)
            st.success("Matching Complete!")
            st.rerun()


# ==========================================
# 4. MAIN DASHBOARD TABS
# ==========================================
tab_chat, tab_report, tab_compare, tab_questions = st.tabs([
    "💬 Conversational Screening",
    "📊 Match Report & Explainability",
    "⚔️ Head-to-Head Comparison",
    "❓ Interview Question Generator"
])


# ------------------------------------------
# TAB 1: Conversational Screening & Refinement
# ------------------------------------------
with tab_chat:
    st.subheader("Conversational Candidate Screening")
    st.caption("Ask natural language queries, refine criteria mid-flight, or inquire about specific candidate rankings.")

    # Display shortlists if available
    shortlist = st.session_state.agent_state.get("shortlist", [])
    if shortlist:
        st.markdown("### 🏆 Current Top Candidate Matches")
        for rank, cand in enumerate(shortlist, 1):
            cand_name = cand.get("name", f"Candidate {rank}")
            reasoning = st.session_state.agent_state.get("reasoning", {}).get(cand_name, "Evaluated by Hybrid RAG & Cross-Encoder")
            with st.expander(f"#{rank} - {cand_name}"):
                st.write(f"**Reasoning:** {reasoning}")
                st.text(cand.get("content", "")[:300] + "...")

    st.markdown("---")
    
    # Chat History Container
    for msg in st.session_state.agent_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User Input Chat Prompt
    if user_prompt := st.chat_input("e.g. Find me candidates with React and 3+ years experience or compare top matches"):
        # Append user message
        st.session_state.agent_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # Trigger iterative refinement back into graph loop
        st.session_state.agent_state["human_feedback"] = user_prompt
        
        with st.chat_message("assistant"):
            with st.spinner("Re-evaluating pipeline..."):
                updated_state = agent_app.invoke(st.session_state.agent_state)
                st.session_state.agent_state.update(updated_state)
                st.rerun()


# ------------------------------------------
# TAB 2: Match Report & Explainability
# ------------------------------------------
with tab_report:
    st.subheader("Executive Candidate Evaluation Report")
    report = st.session_state.agent_state.get("report", "")
    if report:
        st.markdown(report)
    else:
        st.info("Run the agent pipeline from the sidebar to generate an executive match report.")


# ------------------------------------------
# TAB 3: Head-to-Head Comparison
# ------------------------------------------
with tab_compare:
    st.subheader("Head-to-Head Candidate Comparison")
    candidate_pool = st.session_state.agent_state.get("candidate_pool", [])
    
    if candidate_pool:
        options = [c.get("name", c.get("id")) for c in candidate_pool]
        selected_candidates = st.multiselect("Select 2 or more candidates to compare:", options)
        
        if st.button("Compare Selected Candidates"):
            if len(selected_candidates) >= 2:
                with st.spinner("Generating side-by-side comparison..."):
                    comparison_result = compare_candidates(
                        candidate_ids=selected_candidates,
                        candidate_pool=candidate_pool
                    )
                    st.markdown("### Comparison Results")
                    st.markdown(comparison_result)
            else:
                st.warning("Please select at least 2 candidates for comparison.")
    else:
        st.info("No candidate profiles loaded.")


# ------------------------------------------
# TAB 4: Interview Question Generator
# ------------------------------------------
with tab_questions:
    st.subheader("Tailored Interview Question Generator")
    candidate_pool = st.session_state.agent_state.get("candidate_pool", [])
    reqs = st.session_state.agent_state.get("must_have_reqs", [])

    if candidate_pool:
        options = [c.get("name", c.get("id")) for c in candidate_pool]
        selected_cand = st.selectbox("Select Candidate:", options)
        
        if st.button("Generate Screening Questions"):
            with st.spinner("Creating interview questions..."):
                questions = generate_interview_questions(
                    candidate_id=selected_cand,
                    candidate_pool=candidate_pool,
                    requirements=reqs
                )
                st.markdown(f"### Screening Questions for {selected_cand}")
                st.markdown(questions)
    else:
        st.info("No candidate profiles loaded.")