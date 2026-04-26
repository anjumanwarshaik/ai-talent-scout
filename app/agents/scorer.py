# app/agents/scorer.py
import re
import numpy as np
import json
import hashlib
from app.llm_client import call_llm

# ── Scoring Weights ──────────────────────────────────────────
WEIGHTS = {
    "similarity": 0.5,
    "skills":     0.3,
    "experience": 0.2
}

LLM_BLEND_WEIGHT = 0.3   # how much LLM influences final score
TOP_K_LLM = 5            # only top K resumes get LLM evaluation

# Critical skills (can expand later dynamically)
CRITICAL_SKILLS = {"python", "machine learning", "sql"}

# ── Helpers ──────────────────────────────────────────────────
def extract_experience_years(text: str) -> float:
    text = text.lower()

    patterns = [
        r'(\d+(\.\d+)?)\s*(years?|yrs?)',
        r'over\s*(\d+)\s*years',
        r'(\d+)\+?\s*years',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return 0.0


def score_skill_match(resume_text: str, required_skills: list):
    if not required_skills:
        return 0.0, [], []

    resume_lower = resume_text.lower()
    matched, missing = [], []

    total_weight = 0
    score = 0

    for skill in required_skills:
        skill_lower = skill.lower()
        weight = 2 if skill_lower in CRITICAL_SKILLS else 1
        total_weight += weight

        if skill_lower in resume_lower:
            score += weight
            matched.append(skill)
        else:
            missing.append(skill)

    final_score = round(score / total_weight, 4) if total_weight > 0 else 0.0
    return final_score, matched, missing


def score_experience_match(resume_text: str, required_experience: str):
    match = re.search(r'(\d+)', required_experience)
    if not match:
        return 0.5, 0, 0

    required_years = int(match.group(1))
    resume_years = extract_experience_years(resume_text)

    if resume_years == 0:
        return 0.3, resume_years, required_years
    elif resume_years >= required_years:
        return 1.0, resume_years, required_years
    else:
        return round(resume_years / required_years, 4), resume_years, required_years


def deduplicate_resumes(resumes: list):
    seen = set()
    unique = []

    for r in resumes:
        fingerprint = hashlib.md5(r["resume_text"].encode()).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(r)

    return unique


# ── LLM Evaluation ───────────────────────────────────────────
def llm_evaluate(resume_text, job_title, required_skills, required_experience):
    system = (
        "You are an expert technical recruiter. "
        "Evaluate candidates strictly and return only valid JSON."
    )

    prompt = f"""
Evaluate this resume against the job requirements.

JOB TITLE: {job_title}
REQUIRED SKILLS: {', '.join(required_skills)}
REQUIRED EXPERIENCE: {required_experience}

RESUME (first 800 chars):
{resume_text[:800]}

Return ONLY this JSON:
{{
  "skill_match": <0-100>,
  "experience_match": <0-100>,
  "domain_fit": <0-100>,
  "missing_skills": [],
  "reason": "<one-line explanation>"
}}
"""

    raw = call_llm(prompt=prompt, system=system)

    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except:
        return {
            "skill_match": 50,
            "experience_match": 50,
            "domain_fit": 50,
            "missing_skills": [],
            "reason": "LLM parsing failed"
        }


# ── Agent Function ───────────────────────────────────────────
def score_resumes(state: dict) -> dict:
    top_resumes = state.get("top_resumes", [])
    required_skills = state.get("required_skills", [])
    required_experience = state.get("required_experience", "")
    job_title = state.get("job_title", "")

    # Step 1: Deduplicate
    top_resumes = deduplicate_resumes(top_resumes)
    print(f"After deduplication: {len(top_resumes)} unique resumes")

    scored = []

    # ── First pass: classical scoring ────────────────────────
    for resume in top_resumes:
        resume_text = resume["resume_text"]

        sim_score = resume["similarity_score"]

        skill_score, matched_skills, missing_skills = score_skill_match(
            resume_text, required_skills
        )

        exp_score, resume_years, required_years = score_experience_match(
            resume_text, required_experience
        )

        base_score = round(
            (sim_score   * WEIGHTS["similarity"]) +
            (skill_score * WEIGHTS["skills"]) +
            (exp_score   * WEIGHTS["experience"]),
            4
        )

        # ── Dynamic penalties ────────────────────────────────
        penalty = 0.0

        if required_years > 0 and resume_years < required_years:
            penalty += 0.15

        for skill in missing_skills:
            if skill.lower() in CRITICAL_SKILLS:
                penalty += 0.1
            else:
                penalty += 0.03

        scored.append({
            **resume,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_score": skill_score,
            "experience_score": exp_score,
            "resume_years": resume_years,
            "base_score": base_score,
            "penalty": round(penalty, 4),
        })

    # Sort before LLM
    scored = sorted(scored, key=lambda x: (x["base_score"] - x["penalty"]), reverse=True)

    # ── Second pass: LLM only for top K ──────────────────────
    for i, resume in enumerate(scored):
        if i < TOP_K_LLM:
            print(f"Calling LLM for resume {resume['index']}...")
            llm_eval = llm_evaluate(
                resume["resume_text"], job_title, required_skills, required_experience
            )

            llm_score = (
                0.4 * (llm_eval["skill_match"] / 100) +
                0.3 * (llm_eval["experience_match"] / 100) +
                0.3 * (llm_eval["domain_fit"] / 100)
            )
        else:
            llm_eval = {
                "skill_match": 0,
                "experience_match": 0,
                "domain_fit": 0,
                "missing_skills": [],
                "reason": "LLM skipped"
            }
            llm_score = 0

        final_score = round(
            (1 - LLM_BLEND_WEIGHT) * (resume["base_score"] - resume["penalty"]) +
            (LLM_BLEND_WEIGHT * llm_score),
            4
        )

        confidence = round(1 - resume["penalty"], 3)

        resume.update({
            "final_score": final_score,
            "confidence": confidence,
            "llm_skill_match": llm_eval["skill_match"],
            "llm_exp_match": llm_eval["experience_match"],
            "llm_domain_fit": llm_eval["domain_fit"],
            "llm_missing": llm_eval["missing_skills"],
            "llm_reason": llm_eval["reason"]
        })

    # Final sort
    scored = sorted(scored, key=lambda x: x["final_score"], reverse=True)

    print(f"\nScored and ranked {len(scored)} resumes")
    print(f"Top score: {scored[0]['final_score']}")
    print(f"Reason: {scored[0]['llm_reason']}")

    return {
        **state,
        "scored_resumes": scored
    }