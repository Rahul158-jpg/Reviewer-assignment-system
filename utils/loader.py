"""
loader.py — FINAL VERSION

Reviewer Recommendation — Data Loader

Features:
✔ Safe JSON loading (fixes encoding issues)
✔ Pre-built dictionaries (fast lookup)
✔ COI mapping
✔ Clean reusable interface
"""

import json
import os

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

MANUSCRIPTS_PATH = os.path.join(DATA_DIR, "manuscripts.json")
REVIEWERS_PATH   = os.path.join(DATA_DIR, "reviewers.json")
GROUND_TRUTH     = os.path.join(DATA_DIR, "ground_truth.json")
COI_PATH         = os.path.join(DATA_DIR, "coi_graph.json")
ACCEPT_PATH      = os.path.join(DATA_DIR, "acceptance_labels.json")


# ─────────────────────────────────────────
# SAFE JSON LOADER (CRITICAL FIX)
# ─────────────────────────────────────────

def load_json(path):
    """
    Safe loader to avoid UnicodeDecodeError
    """
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return None

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        # fallback (Windows issue fix)
        with open(path, encoding="latin-1") as f:
            return json.load(f)


# ─────────────────────────────────────────
# BUILD LOOKUP STRUCTURES
# ─────────────────────────────────────────

def build_manuscript_dict(manuscripts):
    return {
        m["paper_id"]: m
        for m in manuscripts
    }


def build_reviewer_dict(reviewers):
    return {
        r["reviewer_id"]: r
        for r in reviewers
    }


def build_ground_truth(gt):
    return {
        g["paper_id"]: g["assigned_reviewers"]
        for g in gt
    }


def build_coi_dict(coi):
    return {
        (c["paper_id"], c["reviewer_id"]): c
        for c in coi
    }


def build_accept_dict(acc):
    return {
        a["paper_id"]: a.get("label", 0)
        for a in acc
    }


# ─────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────

def load_all():

    print("\n=== LOADING DATA ===")

    manuscripts = load_json(MANUSCRIPTS_PATH) or []
    reviewers   = load_json(REVIEWERS_PATH) or []
    gt          = load_json(GROUND_TRUTH) or []
    coi         = load_json(COI_PATH) or []
    acc         = load_json(ACCEPT_PATH) or []

    # build fast lookup structures
    ms_dict = build_manuscript_dict(manuscripts)
    rev_dict = build_reviewer_dict(reviewers)
    gt_dict = build_ground_truth(gt)
    coi_dict = build_coi_dict(coi)
    acc_dict = build_accept_dict(acc)

    print("\nSummary:")
    print(f"  Manuscripts: {len(manuscripts)}")
    print(f"  Reviewers:   {len(reviewers)}")
    print(f"  GT pairs:    {len(gt)}")
    print(f"  COI pairs:   {len(coi)}")
    print(f"  Accept labels: {len(acc)}")

    return {
        "manuscripts": manuscripts,
        "reviewers": reviewers,
        "ms_dict": ms_dict,
        "rev_dict": rev_dict,
        "gt": gt_dict,
        "coi": coi_dict,
        "accept": acc_dict
    }