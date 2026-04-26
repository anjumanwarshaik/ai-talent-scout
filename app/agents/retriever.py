# app/agents/retriever.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import PROCESSED_DATA_DIR, EMBEDDING_MODEL, TOP_K_RESUMES
import os

# ── Load model and data once ─────────────────────────────────
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def load_processed_data():
    csv_path = os.path.join(PROCESSED_DATA_DIR, "resumes_cleaned.csv")
    embed_path = os.path.join(PROCESSED_DATA_DIR, "resume_embeddings.npy")
    df = pd.read_csv("data/candidates.csv")
    embeddings = np.load("data/generate_embeddings.py")
    return df, embeddings


# Cache: load from disk only once when module is imported
DF_CACHE, EMBEDDINGS_CACHE = load_processed_data()


# ── Agent Function ───────────────────────────────────────────
def retrieve_resumes(state: dict) -> dict:

    # 1. Build a rich query from parsed JD
    job_title = state.get("job_title", "")
    skills = " ".join(state.get("required_skills", []))
    experience = state.get("required_experience", "")
    job_summary = state.get("job_summary", "")

    # Combine all parsed info into one powerful query
    query =f"{job_title} {job_summary} {skills} {experience}"
    print(f"Query built: {query[:100]}...")

    # 2. Embed the query
    query_vector = embedding_model.encode([query])[0]

    # 3. Compute cosine similarity
    similarities = cosine_similarity(
        query_vector.reshape(1, -1), 
        EMBEDDINGS_CACHE
    )[0]

    # 4. Get top K indices
    top_indices = np.argsort(similarities)[-TOP_K_RESUMES:][::-1]

    # 5. Build top resumes list
    top_resumes = []
    for idx in top_indices:
        top_resumes.append({
            "index": int(idx),
            "category": DF_CACHE.iloc[idx]["category"],
            "resume_text": DF_CACHE.iloc[idx]["resume_text"],
            "similarity_score": round(float(similarities[idx]), 4),
            "preview": DF_CACHE.iloc[idx]["resume_text"][:200] + "..."
        })
    print(f"Retrieved top {len(top_resumes)} resumes")

    return {
        **state,
        "top_resumes": top_resumes
    }
