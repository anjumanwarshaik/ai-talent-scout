# app/agents/reporter.py
from app.llm_client import call_llm

# ── Prompt ───────────────────────────────────────────────────
REPORT_SYSTEM_PROMPT = "You are an expert HR recruiter writing professional candidate shortlist reports based on technical fit and engagement simulation."

REPORT_USER_TEMPLATE = """
Job Title: {job_title}
Required Skills: {required_skills}
Required Experience: {required_experience}

Top Candidates (Ranked by Match + Interest):
{top_candidates}

Write a professional recruitment report with:
1. Executive Summary: Overview of the search and engagement success.
2. Top 3 Candidate Deep-Dives: Include their technical fit, location advantage, and a summary of their 'Outreach' behavior (interest level).
3. Engagement Analysis: Identify who is a 'Hot' lead vs who has reservations (salary, notice period, etc.).
4. Final Recommendation: Which candidate should the hiring manager interview first?

Be professional, concise, and highlight the 'Interest Score' as a key decision factor.
"""

# ── Helper ───────────────────────────────────────────────────
def format_candidates_for_prompt(shortlist: list) -> str:
    output = ""
    # We take the top 3 from the final ranked shortlist
    for i, c in enumerate(shortlist[:3]):
        output += f"""
Rank {c['rank']}: {c['name']}
- Location: {c.get('location', 'Not specified')}
- Combined Score: {c['combined_score']}
- Technical Match: {c['match_score']*100}%
- Interest Score: {c['interest_score']}/100 ({c['interest_label']})
- Engagement Reason: {c.get('interest_reason', 'N/A')}
- Skills: {', '.join(c['skills'])}
- Experience: {c['years_experience']} years
- Bio: {c['bio']}
"""
    return output

# ── Agent Function ───────────────────────────────────────────
def generate_report(state: dict) -> dict:
    # We now pull from 'shortlist' created by Agent 5 (Ranker)
    shortlist = state.get("shortlist", [])

    if not shortlist:
        return {
            **state,
            "final_report": "No candidates were successfully ranked for this report."
        }

    # Format the data for the LLM
    top_candidates_formatted = format_candidates_for_prompt(shortlist)

    # Generate the professional summary
    final_report = call_llm(
        prompt=REPORT_USER_TEMPLATE.format(
            job_title=state.get("job_title", "Position"),
            required_skills=", ".join(state.get("required_skills", [])),
            required_experience=state.get("required_experience", "N/A"),
            top_candidates=top_candidates_formatted
        ),
        system=REPORT_SYSTEM_PROMPT
    )

    final_report = final_report.strip()

    print("Agent 6: Final Report generated successfully!")
    
    return {
        **state,
        "final_report": final_report
    }