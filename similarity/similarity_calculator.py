import json
import os
import numpy as np
import re
from tqdm import tqdm

from similarity.embedding_model import (
    build_paper_embeddings,
    build_reviewer_embeddings
)

from kg.kg_similarity import compute_kg_similarity


# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MANUSCRIPTS_PATH = os.path.join(DATA_DIR, "manuscripts.json")
REVIEWERS_PATH   = os.path.join(DATA_DIR, "reviewers.json")

OUT_PATH = os.path.join(OUTPUT_DIR, "similarity_combined.json")


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# SAFE COSINE SIMILARITY
# ─────────────────────────────────────────

def cosine_similarity_safe(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


# ─────────────────────────────────────────
# CLEAN TOKENIZATION
# ─────────────────────────────────────────

STOPWORDS = {
    "the", "is", "and", "of", "in", "to", "a", "for", "on", "with", "as", "by"
}

def tokenize(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def keyword_overlap(text1, text2):
    s1 = tokenize(text1)
    s2 = tokenize(text2)

    if not s1 or not s2:
        return 0.0

    return len(s1 & s2) / len(s1 | s2)


def build_text(item):
    return (item.get("title", "") + " " + item.get("abstract", "")).lower()


# ─────────────────────────────────────────
# MAIN SIMILARITY
# ─────────────────────────────────────────

def compute_similarity():

    print("\n=== FINAL HYBRID SIMILARITY (FIXED) ===")

    manuscripts = load_json(MANUSCRIPTS_PATH)
    reviewers   = load_json(REVIEWERS_PATH)

    # ── EMBEDDINGS ────────────────────────
    paper_embs = build_paper_embeddings(manuscripts)
    reviewer_embs = build_reviewer_embeddings(reviewers)

    # ── KG SIMILARITY ────────────────────
    kg_scores = compute_kg_similarity()

    # ── TEXT PREP ────────────────────────
    paper_texts = {
        m["paper_id"]: build_text(m)
        for m in manuscripts
    }

    reviewer_texts = {}

    for r in reviewers:
        text = ""
        for pub in r.get("publications", [])[:5]:
            text += pub.get("title", "") + " "
            text += pub.get("abstract", "") + " "

        reviewer_texts[r["reviewer_id"]] = text.lower()

    # ── COMPUTE SIMILARITY ───────────────
    results = {}

    for pid in tqdm(paper_embs, desc="Computing similarity"):

        results[pid] = []
        p_emb = paper_embs[pid]
        p_text = paper_texts[pid]

        for rid in reviewer_embs:

            r_emb = reviewer_embs[rid]

            # ✅ 1. COSINE (SAFE)
            emb_score = cosine_similarity_safe(p_emb, r_emb)

            # ✅ 2. KG (already normalized)
            kg_score = kg_scores.get(pid, {}).get(rid, 0.0)

            # ✅ 3. KEYWORD (cleaned)
            kw_score = keyword_overlap(
                p_text,
                reviewer_texts.get(rid, "")
            )

            # 🔥 FINAL HYBRID SCORE (CALIBRATED)
            final_score = (
                0.6 * emb_score +
                0.3 * kg_score +
                0.1 * kw_score
            )

            results[pid].append([
                rid,
                round(final_score, 6)
            ])

        # sort reviewers
        results[pid].sort(key=lambda x: x[1], reverse=True)

    return results


# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

def save_results(results):

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {OUT_PATH}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    results = compute_similarity()
    save_results(results)

    print("\nDone — FINAL hybrid similarity ready.")