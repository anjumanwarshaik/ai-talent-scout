# utils/similarity.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_top_matches(query_embedding, resume_embeddings, df, top_n = 5):
    """
    Finds the most similar resumes to a given query vector.
    """
    # Compute cosine similarity
    # query_embedding needs to be 2D, so we ensure it is
    similarities = cosine_similarity(query_embedding.reshape(1, -1), resume_embeddings)[0]
    
    # Get top N indices
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "category": df.iloc[idx]['category'],
            "score": round(float(similarities[idx]), 4),
            'preview': df.iloc[idx]['resume_text'][:150] + "..."
        })
    return results