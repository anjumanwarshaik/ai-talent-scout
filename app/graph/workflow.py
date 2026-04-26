# app/graph/workflow.py
import os
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# ── State shared across all agents ───────────────────────────
class ResumeMatcherState(TypedDict):
    job_description: str
    # Agent 1 output: parsed JD
    required_skills: List[str]
    required_experience: str
    job_title: str
    job_summary: str
    # Agent 2 output: retrieved resumes
    top_resumes: List[Dict[str, Any]]
    # Agent 3 output: scored resumes
    scored_resumes: List[Dict[str, Any]]
    # Agent 4 output: outreach results
    outreach_results: List[Dict[str, Any]]
    # Agent 5 output: final ranked shortlist
    shortlist: List[Dict[str, Any]]
    # Agent 6 output: final report
    final_report: str

# ── Import Agents ───────────────────────────────────────────
from app.agents.jd_parser import parse_jd
from app.agents.retriever import retrieve_resumes
from app.agents.scorer import score_resumes
from app.agents.outreach import run_outreach
from app.agents.ranker import rank_candidates
from app.agents.reporter import generate_report

# ── Node Wrappers ───────────────────────────────────────────
def jd_parser_node(state: ResumeMatcherState) -> Dict:
    print("Agent 1: Parsing Job Description...")
    return parse_jd(state)

def retriever_node(state: ResumeMatcherState) -> Dict:
    print("Agent 2: Retrieving Top Resumes...")
    return retrieve_resumes(state)

def scorer_node(state: ResumeMatcherState) -> Dict:
    print("Agent 3: Scoring and Ranking Resumes...")
    return score_resumes(state)

def outreach_node(state: ResumeMatcherState) -> Dict:
    print("Agent 4: Running Outreach Simulation...")
    return run_outreach(state)

def ranker_node(state: ResumeMatcherState) -> Dict:
    print("Agent 5: Ranking by Match + Interest...")
    return rank_candidates(state)

def reporter_node(state: ResumeMatcherState) -> Dict:
    print("Agent 6: Generating Final Report...")
    return generate_report(state)

# ── Build the Graph ──────────────────────────────────────────
def build_workflow():
    graph = StateGraph(ResumeMatcherState)

    graph.add_node("jd_parser", jd_parser_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("scorer",    scorer_node)
    graph.add_node("outreach",  outreach_node)
    graph.add_node("ranker",    ranker_node)
    graph.add_node("reporter",  reporter_node)

    graph.set_entry_point("jd_parser")
    graph.add_edge("jd_parser", "retriever")
    graph.add_edge("retriever", "scorer")
    graph.add_edge("scorer",    "outreach")
    graph.add_edge("outreach",  "ranker")
    graph.add_edge("ranker",    "reporter")
    graph.add_edge("reporter",  END)

    return graph.compile()

workflow = build_workflow()