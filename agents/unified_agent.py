"""
unified_agent.py — FINAL VERSION

Reviewer Recommendation — Unified Agent Layer

Coordinates:
✔ Agent A — Quality (Q)
✔ Agent B — Acceptance (A)
✔ Agent C — Reliability (R)

Provides:
✔ Batch scoring
✔ Single-paper scoring
✔ Cached outputs
"""

import json
import os

from .llm_client import (
    score_quality,
    score_acceptance,
    score_reliability
)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

OUTPUT_DIR = "outputs"

QUALITY_PATH     = os.path.join(OUTPUT_DIR, "quality_scores.json")
ACCEPT_PATH      = os.path.join(OUTPUT_DIR, "acceptance_scores.json")
RELIABILITY_PATH = os.path.join(OUTPUT_DIR, "reliability_scores.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────
# BATCH MODE (OFFLINE)
# ─────────────────────────────────────────

def run_batch_agents(manuscripts, reviewers, force=False):

    print("\n=== UNIFIED AGENT — BATCH MODE ===")

    # load cached if exists
    quality = load_json(QUALITY_PATH)
    accept  = load_json(ACCEPT_PATH)
    reliability = load_json(RELIABILITY_PATH)

    if quality and accept and reliability and not force:
        print("Using cached LLM scores")
        return quality, accept, reliability

    quality = {}
    accept  = {}
    reliability = {}

    # ── Agent A + B (papers)
    for m in manuscripts:
        pid = m["paper_id"]

        quality[pid] = score_quality(m)
        accept[pid]  = score_acceptance(m)

    # ── Agent C (reviewers)
    for r in reviewers:
        rid = r["reviewer_id"]
        reliability[rid] = score_reliability(r)

    # save
    save_json(quality, QUALITY_PATH)
    save_json(accept, ACCEPT_PATH)
    save_json(reliability, RELIABILITY_PATH)

    print("LLM scores computed and saved")

    return quality, accept, reliability


# ─────────────────────────────────────────
# SINGLE PAPER MODE (UI DEMO)
# ─────────────────────────────────────────

def run_single_paper_agents(paper, reviewers):

    print("\n=== UNIFIED AGENT — SINGLE PAPER ===")

    # paper-level
    q = score_quality(paper)
    a = score_acceptance(paper)

    # reviewer-level
    reliability = {}

    for r in reviewers:
        rid = r["reviewer_id"]
        reliability[rid] = score_reliability(r)

    return {
        "quality": q,
        "acceptance": a,
        "reliability": reliability
    }


# ─────────────────────────────────────────
# FAST ACCESS (FOR FUSION)
# ─────────────────────────────────────────

def load_agent_scores():

    quality = load_json(QUALITY_PATH) or {}
    accept  = load_json(ACCEPT_PATH) or {}
    reliability = load_json(RELIABILITY_PATH) or {}

    return quality, accept, reliability