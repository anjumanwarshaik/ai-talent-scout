# app/agents/ranker.py
import re

# Weighting constants for the combined score
MATCH_WEIGHT = 0.6
INTEREST_WEIGHT = 0.4

def rank_candidates(state: dict) -> dict:
    """
    Agent 4: Merges technical 'Match Scores' with simulated 'Interest Scores'.
    Calculates a final unified rank using Skills, Experience, Location, and Interest.
    """
    outreach_results = state.get("outreach_results", [])
    required_skills = set(s.lower() for s in state.get("required_skills", []))
    
    # Pre-process JD location for the bonus logic
    # We check both the explicit 'location' key and the raw 'job_description'
    job_loc_target = (state.get("location", "") + " " + state.get("job_description", "")).lower()

    def compute_match_score(candidate: dict) -> float:
        # 1. Skill Overlap Score (0.0 - 1.0)
        c_skills = set(s.lower() for s in candidate.get("skills", []))
        overlap = len(c_skills & required_skills)
        total_req = len(required_skills) if required_skills else 1
        skill_ratio = min(overlap / total_req, 1.0)

        # 2. Experience Match Score (0.0 - 1.0)
        years = candidate.get("years_experience", 0)
        exp_req_str = str(state.get("required_experience", "0"))
        # Extract first number found in string (e.g., "5+ years" -> 5)
        match = re.search(r"(\d+)", exp_req_str)
        required_years = int(match.group(1)) if match else 0
        
        if required_years > 0:
            exp_ratio = min(years / required_years, 1.0)
        else:
            exp_ratio = 0.8  # Default baseline if no exp required

        # 3. Location Bonus (Optional +5% boost)
        # If the candidate is in the same city mentioned in the JD
        candidate_loc = candidate.get("location", "").lower()
        location_bonus = 0.0
        if candidate_loc and any(city in job_loc_target for city in candidate_loc.split(',')):
            location_bonus = 0.05

        # Calculate Technical Match (Weighted Skills + Experience + Bonus)
        technical_match = (0.6 * skill_ratio) + (0.4 * exp_ratio) + location_bonus
        return round(min(technical_match, 1.0), 4)

    shortlist = []

    for candidate in outreach_results:
        # Get the technical score
        match_score = compute_match_score(candidate)
        
        # Normalize the interest score from outreach (0-100 -> 0.0-1.0)
        interest_score_normalized = candidate.get("interest_score", 50) / 100.0

        # Final Combined Calculation
        combined_score = round(
            (MATCH_WEIGHT * match_score) + (INTEREST_WEIGHT * interest_score_normalized),
            4
        )

        shortlist.append({
            **candidate,
            "match_score": match_score,
            "interest_score_normalized": interest_score_normalized,
            "combined_score": combined_score,
        })

    # Sort by combined score descending
    shortlist = sorted(shortlist, key=lambda x: x["combined_score"], reverse=True)

    # Assign final rank position
    for i, c in enumerate(shortlist):
        c["rank"] = i + 1

    if shortlist:
        print(f"Ranking Complete. Top Candidate: {shortlist[0]['name']} | Score: {shortlist[0]['combined_score']}")
    else:
        print("Warning: Shortlist is empty.")

    return {
        **state,
        "shortlist": shortlist
    }