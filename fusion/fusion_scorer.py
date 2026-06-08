"""
fusion_scorer.py — FINAL STABLE VERSION

Reviewer Recommendation — Fusion Scoring Engine
"""

import json
import os
import numpy as np

# ─────────────────────────────────────────
# PATHS (FIXED — PROJECT ROOT)
# ─────────────────────────────────────────

# Go to project root (one level up from fusion/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SIM_PATH = os.path.join(OUTPUT_DIR, "similarity_combined.json")
KG_PATH  = os.path.join(OUTPUT_DIR, "kg_similarity.json")
COI_PATH = os.path.join(DATA_DIR, "coi_graph.json")

QUALITY_PATH     = os.path.join(OUTPUT_DIR, "quality_scores.json")
ACCEPT_PATH      = os.path.join(OUTPUT_DIR, "acceptance_scores.json")
RELIABILITY_PATH = os.path.join(OUTPUT_DIR, "reliability_scores.json")

OUT_PATH = os.path.join(OUTPUT_DIR, "fused_scores.json")


# ─────────────────────────────────────────
# LOAD JSON (SAFE)
# ─────────────────────────────────────────

def load_json(path, required=False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"❌ Required file missing: {path}")
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────

def normalize(scores):
    if not scores:
        return scores

    values = np.array(list(scores.values()))
    min_v, max_v = values.min(), values.max()

    if max_v - min_v < 1e-8:
        return {k: 0 for k in scores}

    return {
        k: (v - min_v) / (max_v - min_v)
        for k, v in scores.items()
    }


# ─────────────────────────────────────────
# FUSION CORE
# ─────────────────────────────────────────

def fuse():

    print("\n=== FUSION SCORING ENGINE ===")

    # Debug (optional — remove later)
    print("Looking for:", SIM_PATH)
    print("Exists?", os.path.exists(SIM_PATH))

    # REQUIRED
    sim_data = load_json(SIM_PATH, required=True)

    # OPTIONAL
    kg_data  = load_json(KG_PATH)
    coi_data = load_json(COI_PATH)

    quality     = load_json(QUALITY_PATH) or {}
    accept      = load_json(ACCEPT_PATH) or {}
    reliability = load_json(RELIABILITY_PATH) or {}

    # SAFE COI DICT
    coi_dict = {}
    if coi_data:
        coi_dict = {
            (c["paper_id"], c["reviewer_id"]): c
            for c in coi_data
        }

    fused = {}

    for pid, ranked in sim_data.items():

        if not ranked:
            continue

        fused[pid] = []

        # Normalize similarity
        sim_scores = {str(rid): score for rid, score in ranked}
        sim_scores = normalize(sim_scores)

        for rid, _ in ranked:

            sim = sim_scores.get(rid, 0.0)

            kg = 0.0
            if kg_data:
                kg = kg_data.get(pid, {}).get(rid, 0.0)

            # LLM signals
            q = quality.get(pid, 0.5)
            a = accept.get(pid, 0.5)
            r = reliability.get(rid, 0.5)

            llm_score = (q + a + r) / 3

            # COI is a hard filter, not a positive or negative scoring term.
            coi_entry = coi_dict.get((pid, rid), {})
            coi = float(coi_entry.get("coi_score", 0) or 0)
            if coi_entry.get("flagged") or coi >= 0.5:
                continue

            # FINAL SCORE
            score = (
                0.85 * sim +
                0.05 * kg +
                0.10 * llm_score
            )

            fused[pid].append([str(rid), round(score, 6)])

        # Sort
        fused[pid].sort(key=lambda x: x[1], reverse=True)

    return fused


# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

def save(fused):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fused, f, indent=2)

    print(f"\n✅ Saved → {OUT_PATH}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():

    try:
        fused = fuse()
        save(fused)

        print("\n🎯 Done — fusion scoring complete.\n")

    except Exception as e:
        print("\n❌ ERROR:", str(e))


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    main()
