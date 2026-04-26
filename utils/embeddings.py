# utils/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once (Global within this module)
# This model is small, fast, and great for resumes.
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_list):
    """
    Converts a list of strings into a list of numerical vectors
    """
    print(f"Generate embeddings for {len(text_list)} items. . .")
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings
