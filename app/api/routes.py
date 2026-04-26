# app/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.graph.workflow import workflow

router = APIRouter()

# ── Shared Models ─────────────────────────────────────────────

class JobDescriptionRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    debug: bool = False

# ── Scout (Agentic) Models ────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str
    message: str

class ShortlistedCandidate(BaseModel):
    rank: int
    name: str
    email: str
    linkedin: str
    current_title: str
    current_company: str
    years_experience: int
    skills: List[str]
    location: str
    notice_period: str
    match_score: float
    interest_score: int
    interest_label: str
    interest_reason: str
    combined_score: float
    conversation: List[ConversationTurn]

class ScoutResponse(BaseModel):
    job_title: str
    job_summary: str
    required_skills: List[str]
    required_experience: str
    shortlist: List[ShortlistedCandidate]
    final_report: str

# ── Classic Match Models ──────────────────────────────────────

class CandidateResult(BaseModel):
    rank: int
    category: str
    final_score: float
    confidence: float
    base_score: float
    penalty: float
    skill_score: float
    experience_score: float
    resume_years: float
    matched_skills: List[str]
    missing_skills: List[str]
    llm_skill_match: int
    llm_exp_match: int
    llm_domain_fit: int
    llm_reason: str
    preview: str

class MatchResponse(BaseModel):
    job_title: str
    job_summary: str
    required_skills: List[str]
    required_experience: str
    candidates: List[CandidateResult]
    final_report: str

# ── Routes ───────────────────────────────────────────────────

@router.post("/scout", response_model=ScoutResponse)
async def scout_candidates(request: JobDescriptionRequest):
    """
    Runs the full 6-agent pipeline: Parser -> Retriever -> Scorer -> Outreach -> Ranker -> Reporter.
    Uses the modern agentic flow with simulated engagement.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty")

    try:
        initial_state = {
            "job_description": request.job_description,
            "required_skills": [],
            "required_experience": "",
            "job_title": "",
            "job_summary": "",
            "top_resumes": [],
            "scored_resumes": [],
            "outreach_results": [],
            "shortlist": [],
            "final_report": ""
        }

        # Run the LangGraph Workflow
        result = workflow.invoke(initial_state)
        raw_shortlist = result.get("shortlist", [])

        if not raw_shortlist:
            raise HTTPException(status_code=404, detail="No candidates found")

        # Apply score filtering and top_k slicing
        filtered_shortlist = [
            c for c in raw_shortlist
            if c.get("combined_score", 0) >= request.min_score
        ][:request.top_k]

        candidates_out = []
        for c in filtered_shortlist:
            candidates_out.append(ShortlistedCandidate(
                rank=c["rank"],
                name=c["name"],
                email=c["email"],
                linkedin=c["linkedin"],
                current_title=c["current_title"],
                current_company=c["current_company"],
                years_experience=c["years_experience"],
                skills=c["skills"],
                location=c["location"],
                notice_period=c["notice_period"],
                match_score=round(c["match_score"], 3),
                interest_score=c["interest_score"],
                interest_label=c["interest_label"],
                interest_reason=c["interest_reason"],
                combined_score=round(c["combined_score"], 3),
                conversation=[
                    ConversationTurn(**turn)
                    for turn in c.get("conversation", [])
                ]
            ))

        return ScoutResponse(
            job_title=result.get("job_title", ""),
            job_summary=result.get("job_summary", ""),
            required_skills=result.get("required_skills", []),
            required_experience=result.get("required_experience", ""),
            shortlist=candidates_out,
            final_report=result.get("final_report", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scout Error: {str(e)}")


@router.post("/match", response_model=MatchResponse)
async def match_resumes(request: JobDescriptionRequest):
    """
    Legacy matching route focused on technical scoring of uploaded resumes.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty")

    try:
        initial_state = {
            "job_description": request.job_description,
            "required_skills": [],
            "required_experience": "",
            "job_title": "",
            "job_summary": "",
            "top_resumes": [],
            "scored_resumes": [],
            "final_report": ""
        }

        result = workflow.invoke(initial_state)
        scored = result.get("scored_resumes", [])

        if not scored:
            raise HTTPException(status_code=404, detail="No candidates found")

        # Filtering logic
        filtered = [
            r for r in scored if r.get("final_score", 0) >= request.min_score
        ][:request.top_k]

        candidates = [
            CandidateResult(
                rank=i + 1,
                category=r["category"],
                final_score=r["final_score"],
                confidence=r.get("confidence", 0.0),
                base_score=r["base_score"],
                penalty=r["penalty"],
                skill_score=r["skill_score"],
                experience_score=r["experience_score"],
                resume_years=r.get("resume_years", 0.0),
                matched_skills=r["matched_skills"],
                missing_skills=r["missing_skills"],
                llm_skill_match=r["llm_skill_match"],
                llm_exp_match=r["llm_exp_match"],
                llm_domain_fit=r["llm_domain_fit"],
                llm_reason=r["llm_reason"],
                preview=r.get("preview", "")[:200]
            ) for i, r in enumerate(filtered)
        ]

        return MatchResponse(
            job_title=result.get("job_title", ""),
            job_summary=result.get("job_summary", ""),
            required_skills=result.get("required_skills", []),
            required_experience=result.get("required_experience", ""),
            candidates=candidates,
            final_report=result.get("final_report", "") if not request.debug else str(result)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Match Error: {str(e)}")