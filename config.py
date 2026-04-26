# config.py
import os

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR = r"C:\Users\ADMIN\Desktop\DS_projects\resume_matcher"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ── Gemini LLM Config ────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBTfSWfer5TpgW77leebiM5SiSV0Wp9urU")
GEMINI_MODEL = "gemini-2.0-flash"    # update if needed

# ── Embedding Config ─────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Retrieval Config ─────────────────────────────────────────
TOP_K_RESUMES = 10  # how many resumes to retrieve before scoring