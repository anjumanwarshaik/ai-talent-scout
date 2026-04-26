# app/agents/outreach.py
import json
import os
from app.llm_client import call_llm

OUTREACH_SYSTEM = (
    "You are a Recruiter AI specialized in talent engagement. "
    "Your goal is to simulate a realistic interaction and accurately assess interest. "
    "Output ONLY valid JSON, no markdown, no backticks, no explanations."
)

OUTREACH_TEMPLATE = """
You are simulating a 3-message outreach exchange between a Recruiter and a Candidate.

JOB CONTEXT:
- Title: {job_title}
- Required Skills: {skills}
- Experience Needed: {experience}
- Summary: {summary}

CANDIDATE PROFILE:
- Name: {name}
- Current Title: {current_title} at {company}
- Skills: {candidate_skills}
- Experience: {years} years
- Bio: {bio}
- Notice Period: {notice_period}

TASK:
1. Generate a realistic 3-message exchange (Recruiter -> Candidate -> Recruiter -> Candidate -> Recruiter -> Candidate).
2. IMPORTANT: Do not make everyone 'Highly Interested'. To make this realistic:
   - Some candidates should be happy where they are and say "No".
   - Some should ask hard questions about "Remote Work", "Salary", or "Notice Period buyouts".
   - Some should be very excited and open to a chat.
3. Score the 'Interest Level' (0-100) based on the candidate's final stance in the conversation.

OUTPUT FORMAT (Strict JSON):
{{
  "conversation": [
    {{"role": "recruiter", "message": "..."}},
    {{"role": "candidate", "message": "..."}},
    {{"role": "recruiter", "message": "..."}},
    {{"role": "candidate", "message": "..."}},
    {{"role": "recruiter", "message": "..."}},
    {{"role": "candidate", "message": "..."}}
  ],
  "interest_score": <int 0-100>,
  "interest_label": "<Cold|Lukewarm|Interested|Very Interested>",
  "interest_reason": "<one sentence explaining why they got this score, e.g., 'Candidate insisted on 100% remote work' or 'Very excited about the tech stack'>"
}}
"""

# Path to your new candidates.json
CANDIDATES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "candidates.json"
)

def load_candidates() -> list:
    if not os.path.exists(CANDIDATES_PATH):
        print(f"Error: {CANDIDATES_PATH} not found!")
        return []
    with open(CANDIDATES_PATH, "r") as f:
        return json.load(f)

def run_outreach_for_candidate(candidate: dict, state: dict) -> dict:
    prompt = OUTREACH_TEMPLATE.format(
        job_title=state.get("job_title", "AI Engineer"),
        skills=", ".join(state.get("required_skills", [])),
        experience=state.get("required_experience", "Not specified"),
        summary=state.get("job_summary", ""),
        name=candidate.get("name", "Unknown"),
        current_title=candidate.get("current_title", ""),
        company=candidate.get("current_company", ""),
        candidate_skills=", ".join(candidate.get("skills", [])),
        years=candidate.get("years_experience", 0),
        bio=candidate.get("bio", ""),
        notice_period=candidate.get("notice_period", "Not specified")
    )

    raw = call_llm(prompt=prompt, system=OUTREACH_SYSTEM)

    try:
        # Clean potential markdown backticks
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM response for {candidate['name']}")
        result = {
            "conversation": [],
            "interest_score": 50,
            "interest_label": "Lukewarm",
            "interest_reason": "Failed to parse simulation data"
        }

    return {
        **candidate,
        "interest_score": result.get("interest_score", 50),
        "interest_label": result.get("interest_label", "Lukewarm"),
        "interest_reason": result.get("interest_reason", ""),
        "conversation": result.get("conversation", [])
    }

def run_outreach(state: dict) -> dict:
    """
    Agent Node: Discovers candidates, runs engagement simulation, 
    and returns state with interest data.
    """
    all_candidates = load_candidates()
    required_skills = state.get("required_skills", [])

    print(f"Processing discovery for {len(all_candidates)} candidates...")

    # Filter/Rank top candidates to save LLM costs (Processing top 8 for the demo)
    def calculate_skill_overlap(c):
        c_skills = set(s.lower() for s in c.get("skills", []))
        j_skills = set(s.lower() for s in required_skills)
        return len(c_skills & j_skills)

    # Sort by overlap so we 'engage' the most relevant ones first
    ranked = sorted(all_candidates, key=calculate_skill_overlap, reverse=True)
    top_8 = ranked[:8] 
    others = ranked[8:]

    engaged_results = []

    # Engage top matches via LLM
    for i, candidate in enumerate(top_8):
        print(f"  Engaging {i+1}/8: {candidate['name']}...")
        result = run_outreach_for_candidate(candidate, state)
        engaged_results.append(result)

    # Passive interest for the rest (to avoid unnecessary API calls)
    for candidate in others:
        engaged_results.append({
            **candidate,
            "interest_score": 30,
            "interest_label": "Not Engaged",
            "interest_reason": "Low initial match; skipped automated outreach.",
            "conversation": []
        })

    return {
        **state,
        "outreach_results": engaged_results
    }