import os
import json
import re
from dotenv import load_dotenv
import chromadb

load_dotenv()

def run_job_matcher(jd_text: str, min_years_required: float = 0.0, top_k: int = 10) -> str:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="resumes")
    
    # 1. Vector Search with Metadata Filtering
    where_clause = {}
    if min_years_required > 0:
        where_clause = {"years_experience": {"$gte": min_years_required}}
        
    results = collection.query(
        query_texts=[jd_text],
        n_results=top_k,
        where=where_clause if where_clause else None
    )
    
    if not results["documents"] or not results["documents"][0]:
        return json.dumps({"job_description": jd_text, "top_matches": []}, indent=2)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    # Extract candidate key terms from JD for skill evaluation
    jd_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', jd_text.lower()))
    
    top_matches = []
    
    for idx in range(len(documents)):
        meta = metadatas[idx]
        skills_list = [s.strip() for s in meta["skills_str"].split(",") if s.strip()]
        
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
            "candidate_name": meta["candidate_name"],
            "resume_path": meta["resume_path"],
            "match_score": match_score,
            "matched_skills": matched_skills,
            "relevant_excerpts": [documents[idx][:300] + "..."],
            "reasoning": (
                f"Candidate has {meta['years_experience']} years of experience with a {meta['education_level']}. "
                f"Matched {len(matched_skills)} key skill(s) requested in job requirements."
            )
        })
        
    top_matches.sort(key=lambda x: x["match_score"], reverse=True)
    
    return json.dumps({
        "job_description": jd_text,
        "top_matches": top_matches
    }, indent=2)

if __name__ == "__main__":
    sample_jd = "Looking for a Python Developer with Machine Learning experience and 2+ years of experience."
    print("Running Job Matcher Test...\n")
    print(run_job_matcher(sample_jd, min_years_required=2.0, top_k=5))