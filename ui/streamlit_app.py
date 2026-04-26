# ui/streamlit_app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from app.graph.workflow import workflow

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Talent Scout",
    page_icon="🎯",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🎯 AI Talent Scouting & Engagement Agent")
st.caption("Paste a Job Description → AI parses it, discovers candidates, simulates outreach, and ranks by Match + Interest.")

st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top K candidates to show", 1, 20, 5)
    min_score = st.slider("Minimum combined score", 0.0, 1.0, 0.0, 0.05)
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. 🔍 JD Parser")
    st.markdown("2. 📋 Candidate Matcher")
    st.markdown("3. 🤖 Outreach Simulator")
    st.markdown("4. 🏆 Ranker")
    st.markdown("5. 📝 Report Generator")

# ── JD Input ─────────────────────────────────────────────────
st.subheader("📄 Job Description")

sample_jd = """We are looking for a Senior Data Scientist with 4+ years of experience in machine learning and NLP.

Requirements:
- Strong Python skills (Pandas, Scikit-learn, PyTorch or TensorFlow)
- Experience with NLP, LLMs, or transformer models
- Familiarity with LangChain or LangGraph is a plus
- SQL proficiency
- Experience deploying models to production
"""

jd_input = st.text_area("Paste JD", value=sample_jd, height=220)

col1, col2 = st.columns([1,1])
run_button = col1.button("🚀 Run Scout", use_container_width=True)
clear_button = col2.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# ── Helpers ──────────────────────────────────────────────────
def interest_badge(label):
    return {
        "Very Interested": "🔥",
        "Interested": "✅",
        "Lukewarm": "🤔",
        "Cold": "❄️"
    }.get(label, "❓")

def score_bar(score, max_val=1.0):
    pct = int((score/max_val)*10)
    return "█"*pct + "░"*(10-pct)

# ── Run Pipeline ─────────────────────────────────────────────
if run_button:

    with st.spinner("Running agents..."):

        state = {
            "job_description": jd_input,
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

        result = workflow.invoke(state)
        shortlist = result.get("shortlist", [])

        shortlist = [c for c in shortlist if c.get("combined_score",0) >= min_score][:top_k]

    st.success("Pipeline complete")

    col_a, col_b = st.columns(2)

    # ── LEFT: JD DETAILS ─────────────────────────────
    with col_a:
        st.subheader("📌 Parsed JD")
        st.write("Role:", result.get("job_title"))
        st.write("Experience:", result.get("required_experience"))
        st.write("Skills:", result.get("required_skills"))

    # ── RIGHT: SUMMARY + STATUS ─────────────────────
    with col_b:
        st.subheader("📊 Pipeline Summary")
        st.metric("Candidates", len(result.get("outreach_results", [])))
        st.metric("Shortlisted", len(shortlist))
        top_score = f"{shortlist[0]['combined_score']:.1%}" if shortlist else "N/A"
        st.metric("Top Score", top_score)

        # ✅ STATUS EXPANDER (NEW)
        with st.expander("⚙️ Pipeline Status / Debug"):
            st.write("JD Parsed:", bool(result.get("job_title")))
            st.write("Skills Extracted:", result.get("required_skills"))

            outreach = result.get("outreach_results", [])
            st.write("Outreach Completed:", len(outreach))

            if outreach:
                st.write("Sample Candidate:", outreach[0]["name"])
                st.write("Interest Score:", outreach[0]["interest_score"])

            # Optional debug
            st.write("State Keys:", list(result.keys()))

    st.divider()

    # ── SHORTLIST TABLE ─────────────────────────────
    st.subheader("🏆 Shortlist")

    for c in shortlist:
        st.markdown(f"### #{c['rank']} {c['name']} ({c['combined_score']:.1%})")

        st.write("Role:", c["current_title"])
        st.write("Company:", c["current_company"])
        st.write("Match:", f"{c['match_score']:.1%}")
        st.write("Interest:", c["interest_score"], interest_badge(c["interest_label"]))

        st.write("Reason:", c["interest_reason"])

        # Conversation
        if c.get("conversation"):
            with st.expander("Conversation"):
                for turn in c["conversation"]:
                    # This uses the built-in avatars for 'assistant' and 'user'
                    with st.chat_message("assistant" if turn['role'] == "recruiter" else "user"):
                        st.write(f"**{'Recruiter' if turn['role'] == 'recruiter' else c['name']}:** {turn['message']}")

        st.divider()

    # ── REPORT ──────────────────────────────────────
    st.subheader("📝 Report")
    st.write(result.get("final_report", "No report generated"))