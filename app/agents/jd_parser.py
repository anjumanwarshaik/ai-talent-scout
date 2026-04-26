# app/agents/jd_parser.py
import json
from app.llm_client import call_llm

# ── Prompt ───────────────────────────────────────────────────
JD_SYSTEM_PROMPT = "You are a strict JSON generator. Output only valid JSON, no markdown, no backticks, no explanations."

JD_USER_TEMPLATE = """
TASK:
Extract structured information from the job description.

RULES:
- Output MUST be valid JSON
- Do NOT include markdown, explanations, or extra text
- Do NOT wrap in backticks
- If unsure, use empty string "" or empty list []

OUTPUT FORMAT:
{{
    "job_title": "",
    "job_summary": "",
    "required_skills": [],
    "required_experience": ""
}}

JOB DESCRIPTION:
{job_description}
"""

# ── Agent Function ───────────────────────────────────────────
def parse_jd(state: dict) -> dict:
    job_description = state["job_description"]

    # Call LLM
    raw = call_llm(
        prompt=JD_USER_TEMPLATE.format(job_description=job_description),
        system=JD_SYSTEM_PROMPT
    )

    # Parse Response
    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)

        return {
            **state,
            "job_title": parsed.get("job_title", ""),
            "job_summary": parsed.get("job_summary", ""),
            "required_skills": parsed.get("required_skills", []),
            "required_experience": parsed.get("required_experience", "")
        }
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON: {e}")
        return {
            **state,
            "job_title": "Unknown",
            "job_summary": raw,
            "required_skills": [],
            "required_experience": ""
        }