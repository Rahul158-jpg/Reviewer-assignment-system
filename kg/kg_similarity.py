"""
kg_similarity.py — FINAL VERSION

Reviewer Recommendation — Knowledge Graph Similarity

Computes similarity between:
    - Paper topics (GLiNER + CSO)
    - Reviewer topics (GLiNER + CSO)

Features:
✔ Weighted similarity (not binary overlap)
✔ Supports GLiNER + CSO
✔ Fast computation
✔ Plug-and-play for fusion stage

Input:
    gliner_manuscript_labels.json
    gliner_reviewer_labels.json
    (optional) cso_manuscript_topics.json
    (optional) cso_reviewer_topics.json

Output:
    kg_similarity[paper_id][reviewer_id] = score
"""

import json
import os
from tqdm import tqdm

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

GLINER_MS_PATH = os.path.join(OUTPUT_DIR, "gliner_manuscript_labels.json")
GLINER_REV_PATH = os.path.join(OUTPUT_DIR, "gliner_reviewer_labels.json")

CSO_MS_PATH = os.path.join(OUTPUT_DIR, "cso_manuscript_topics.json")
CSO_REV_PATH = os.path.join(OUTPUT_DIR, "cso_reviewer_topics.json")


# ─────────────────────────────────────────
# LOAD UTILS
# ─────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────
# CORE SIMILARITY (WEIGHTED JACCARD)
# ─────────────────────────────────────────

def weighted_similarity(paper_w, reviewer_w):
    """
    Compute weighted overlap between two topic distributions

    paper_w: dict {topic: weight}
    reviewer_w: dict {topic: weight}

    Returns:
        float (0–1)
    """

    if not paper_w or not reviewer_w:
        return 0.0

    score = 0.0

    # intersection (min weights)
    for topic in paper_w:
        if topic in reviewer_w:
            score += min(paper_w[topic], reviewer_w[topic])

    return score


# ─────────────────────────────────────────
# MERGE GLINER + CSO
# ─────────────────────────────────────────

def merge_topics(gliner_data, cso_data, entity_id):
    """
    Combine GLiNER weights + CSO topics into unified representation
    """

    merged = {}

    # GLiNER weights
    if gliner_data and entity_id in gliner_data:
        weights = gliner_data[entity_id].get("weights", {})
        for k, v in weights.items():
            merged[k] = merged.get(k, 0) + v

    # CSO topics (uniform weight)
    if cso_data and entity_id in cso_data:
        for topic in cso_data[entity_id]:
            merged[topic] = merged.get(topic, 0) + 0.3  # 🔥 small boost

    # normalize
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}

    return merged


# ─────────────────────────────────────────
# MAIN KG SIMILARITY
# ─────────────────────────────────────────

def compute_kg_similarity():

    print("\n=== KG SIMILARITY COMPUTATION ===")

    gliner_ms = load_json(GLINER_MS_PATH)
    gliner_rev = load_json(GLINER_REV_PATH)

    cso_ms = load_json(CSO_MS_PATH)
    cso_rev = load_json(CSO_REV_PATH)

    if not gliner_ms or not gliner_rev:
        print("⚠️ Missing GLiNER data — KG disabled")
        return {}

    kg_scores = {}

    paper_ids = list(gliner_ms.keys())
    reviewer_ids = list(gliner_rev.keys())

    for pid in tqdm(paper_ids, desc="KG Similarity"):

        paper_topics = merge_topics(gliner_ms, cso_ms, pid)

        kg_scores[pid] = {}

        for rid in reviewer_ids:

            reviewer_topics = merge_topics(gliner_rev, cso_rev, rid)

            score = weighted_similarity(paper_topics, reviewer_topics)

            kg_scores[pid][rid] = round(score, 6)

    return kg_scores


# ─────────────────────────────────────────
# FAST LOOKUP VERSION (FOR PIPELINE)
# ─────────────────────────────────────────

def get_kg_score(pid, rid, kg_scores):
    """
    Safe lookup
    """
    return kg_scores.get(pid, {}).get(rid, 0.0)


# ─────────────────────────────────────────
# SINGLE PAPER VERSION (FOR PDF DEMO)
# ─────────────────────────────────────────

def compute_kg_for_single_paper(paper_labels, reviewer_labels):

    paper_w = paper_labels.get("weights", {})
    reviewer_w = reviewer_labels.get("weights", {})

    return weighted_similarity(paper_w, reviewer_w)


# ─────────────────────────────────────────
# MAIN (OPTIONAL)
# ─────────────────────────────────────────

if __name__ == "__main__":

    kg_scores = compute_kg_similarity()

    out_path = os.path.join(OUTPUT_DIR, "kg_similarity.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kg_scores, f, indent=2)

    print(f"\nSaved → {out_path}")
    print("Done — KG similarity ready for fusion.")