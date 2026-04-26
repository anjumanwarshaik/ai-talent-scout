# main.py

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes import router

# ── App Initialization ───────────────────────────────────────
app = FastAPI(
    title="Agentic Resume Matcher",
    description="Hybrid LLM + classical ML recruitment pipeline with explainable AI scoring",
    version="1.1.0"
)

# ── CORS (important for frontend later) ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health Check Endpoint ────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Agentic Resume Matcher API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── Include Routes ───────────────────────────────────────────
app.include_router(router, prefix="/api", tags=["Resume Matching"])

# ── Run Server ───────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )