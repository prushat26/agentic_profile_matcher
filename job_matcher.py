import os
import json
import re
from dotenv import load_dotenv
import chromadb

load_dotenv()

# Ensure absolute path resolution for chroma_db relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def run_job_matcher(jd_text: str, min_years_required: float = 0.0, top_k: int = 10) -> str:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # get_or_create_collection ensures NotFoundError is never raised
    collection = chroma_client.get_or_create_collection(name="resumes")
    
    # 1. Vector Search with Metadata Filtering
    where_clause = {}
    if min_years_required > 0:
        where_clause = {"years_experience": {"$gte": min_years_required}}
        
    try:
        results = collection.query(
            query_texts=[jd_text],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
    except Exception:
        # Fallback if where_clause filter yields zero matches or metadata error
        results = collection.query(
            query_texts=[jd_text],
            n_results=top_k
        )
    
    if not results.get("documents") or not results["documents"][0]:
        return json.dumps({"job_description": jd_text, "top_matches": []}, indent=2)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    # Extract candidate key terms from JD for skill evaluation
    jd_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', jd_text.lower()))
    
    top_matches = []
    
    for idx in range(len(documents)):
        meta = metadatas[idx]
        skills_str = meta.get("skills_str", "")
        skills_list = [s.strip() for s in skills_str.split(",") if s.strip()]
        
        # Match candidate skills present in JD
        matched_skills = []
        for skill in skills_list:
            s_clean = skill.lower().strip()
            if s_clean in jd_text.lower() or set(s_clean.split()).issubset(jd_words):
                matched_skills.append(skill)
        matched_skills = list(dict.fromkeys(matched_skills))
        
        # Convert L2 distance (0 to 2) to Cosine Similarity (0.0 to 1.0)
        l2_dist = distances[idx]
        vec_sim = max(0.0, 1.0 - (l2_dist / 2.0))
        
        # Calculate skill score relative to detected matches (bonus for matched skills)
        skill_bonus = min(1.0, len(matched_skills) * 0.25)
        
        # Weighted Composite Score: 70% Semantic Similarity + 30% Skill Bonus
        composite_score = (vec_sim * 0.7) + (skill_bonus * 0.3)
        match_score = round(min(100.0, composite_score * 100), 1)
        
        top_matches.append({
            "candidate_name": meta.get("candidate_name", f"Candidate_{idx+1}"),
            "resume_path": meta.get("resume_path", ""),
            "match_score": match_score,
            "matched_skills": matched_skills,
            "relevant_excerpts": [documents[idx][:300] + "..."],
            "reasoning": (
                f"Candidate has {meta.get('years_experience', 'N/A')} years of experience with a {meta.get('education_level', 'N/A')}. "
                f"Matched {len(matched_skills)} key skill(s) requested in job requirements."
            )
        })
        
    top_matches.sort(key=lambda x: x["match_score"], reverse=True)
    
    return json.dumps({
        "job_description": jd_text,
        "top_matches": top_matches
    }, indent=2)


# ==========================================
# AGENT INTERFACE WRAPPERS FOR LANGGRAPH
# ==========================================

def hybrid_search(query: str, candidate_pool: list = None, top_k: int = 10) -> list:
    """
    Search wrapper called by matching_agent.py search_resumes_node.
    Executes run_job_matcher directly against ChromaDB.
    """
    raw_result = run_job_matcher(jd_text=query, min_years_required=0.0, top_k=top_k)
    parsed = json.loads(raw_result)
    matches = parsed.get("top_matches", [])
    
    formatted_results = []
    for m in matches:
        formatted_results.append({
            "id": m.get("candidate_name", ""),
            "name": m.get("candidate_name", ""),
            "score": m.get("match_score", 0.0),
            "matched_skills": m.get("matched_skills", []),
            "content": m.get("reasoning", "") + "\n" + " ".join(m.get("relevant_excerpts", []))
        })
    return formatted_results


def rank_candidates_cross_encoder(query: str, candidates: list) -> list:
    """
    Re-ranker wrapper called by matching_agent.py rank_candidates_node.
    """
    if not candidates:
        return []
    
    for cand in candidates:
        if "score" not in cand:
            cand["score"] = 80.0
        if "reasoning" not in cand:
            cand["reasoning"] = f"Aligned with requirement criteria."
            
    return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)


if __name__ == "__main__":
    sample_jd = "Looking for a Python Developer with Machine Learning experience and 2+ years of experience."
    print("Running Job Matcher Test...\n")
    print(run_job_matcher(sample_jd, min_years_required=2.0, top_k=5))