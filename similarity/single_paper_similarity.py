"""
single_paper_similarity.py — FINAL VERSION

Reviewer Recommendation — Single Paper Inference

Used for:
✔ PDF upload demo
✔ Real-time reviewer recommendation

Pipeline:
    Paper → Embedding + KG → Similarity → Ranked reviewers
"""

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import util

# ─────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────

from similarity.embedding_model import (
    encode_single,
    build_reviewer_embeddings
)

from kg.kg_similarity import (
    compute_kg_similarity,
    compute_kg_for_single_paper
)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

REVIEWERS_PATH = os.path.join(DATA_DIR, "reviewers.json")

GLINER_REV_PATH = os.path.join(OUTPUT_DIR, "gliner_reviewer_labels.json")


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────
# BUILD REVIEWER TEXTS
# ─────────────────────────────────────────

def build_reviewer_texts(reviewers):
    texts = {}

    for r in reviewers:
        rid = str(r["reviewer_id"])

        # stronger reviewer representation: area + keywords + top publication titles
        area = r.get("area", "") or r.get("areas", "") or ""
        kws = r.get("keywords", "")
        if isinstance(kws, list):
            kws = " ".join(kws)

        titles = " ".join([p.get("title", "") for p in r.get("publications", [])[:5]])
        text = " ".join([area, kws, titles]).strip().lower()

        texts[rid] = text

    return texts


# ─────────────────────────────────────────
# KEYWORD OVERLAP
# ─────────────────────────────────────────

def keyword_overlap(text1, text2):
    s1 = set(text1.split())
    s2 = set(text2.split())

    if not s1 or not s2:
        return 0.0

    return len(s1 & s2) / len(s1 | s2)


# ─────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────

def rank_reviewers_for_paper(paper, reviewers=None, sim_data=None):

    print("\n=== SINGLE PAPER SIMILARITY ===")

    # allow passing preloaded data (used by the web UI)
    if reviewers is None:
        reviewers = load_json(REVIEWERS_PATH)

    gliner_rev = load_json(GLINER_REV_PATH)

    # ── PAPER TEXT ────────────────────────
    paper_text = (
        paper.get("title", "") + " " +
        paper.get("abstract", "")
    ).lower()

    # ── EMBEDDINGS ───────────────────────
    print("Encoding paper...")
    paper_emb = encode_single(paper_text)

    print("Loading reviewer embeddings...")
    reviewer_embs = build_reviewer_embeddings(reviewers)

    # ── REVIEWER TEXTS ───────────────────
    reviewer_texts = build_reviewer_texts(reviewers)

    results = []

    for r in tqdm(reviewers, desc="Ranking reviewers"):

        rid = str(r["reviewer_id"])

        # 🔥 1. EMBEDDING
        # stronger cosine similarity using sentence-transformers util
        try:
            emb_score = float(util.cos_sim(paper_emb, reviewer_embs[rid]).item())
        except Exception:
            emb_score = float(np.dot(paper_emb, reviewer_embs[rid]))

        # 🔥 2. KG (if available)
        kg_score = 0.0
        if gliner_rev and rid in gliner_rev:
            kg_score = compute_kg_for_single_paper(
                {"weights": {}},  # paper GLiNER can be added later
                gliner_rev[rid]
            )

        # 🔥 3. KEYWORD
        kw_score = keyword_overlap(
            paper_text,
            reviewer_texts.get(rid, "")
        )

        # 🔥 FINAL SCORE
        final_score = (
            0.7 * emb_score +
            0.2 * kg_score +
            0.1 * kw_score
        )

        # cheap re-ranking boost for clear keyword matches (small bias)
        if kw_score and kw_score > 0.15:
            final_score += 0.05

        results.append({
            "reviewer_id": rid,
            "score": round(final_score, 6)
        })

    # sort
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# ─────────────────────────────────────────
# TEST (DEMO)
# ─────────────────────────────────────────

if __name__ == "__main__":

    sample_paper = {
        "title": "Transformer Models for Text Classification",
        "abstract": "This paper explores transformer architectures for NLP tasks including classification and generation."
    }

    ranked = rank_reviewers_for_paper(sample_paper)

    print("\nTop 5 Reviewers:")
    for r in ranked[:5]:
        print(r)