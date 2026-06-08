"""
gliner_extractor.py — FINAL PRODUCTION VERSION

Reviewer Recommendation — Knowledge Graph Entity Extractor

Features:
✔ GLiNER-based entity extraction
✔ Domain-specific labels (stronger than generic)
✔ Weighted label outputs (for KG similarity)
✔ Persistent caching (fast re-runs)
✔ Fallback mode (no crash if model fails)
✔ UI-ready outputs (topics + weights)

Outputs:
    outputs/gliner_manuscript_labels.json
    outputs/gliner_reviewer_labels.json
"""

import json
import os
import re
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MS_PATH    = os.path.join(DATA_DIR, "manuscripts.json")
REV_PATH   = os.path.join(DATA_DIR, "reviewers.json")

CACHE_PATH = os.path.join(OUTPUT_DIR, "gliner_cache.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔥 Improved domain-specific labels (VERY IMPORTANT)
GLINER_LABELS = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "knowledge graph",
    "information retrieval",
    "data mining",
    "transformer",
    "neural network",
    "text classification",
    "recommendation system",
]

THRESHOLD = 0.2


def simple_extract(text):
    return list(set(str(text or "").lower().split()[:20]))


def extract_entities_dict(data):
    out = {}
    for _id, text in (data or {}).items():
        out[_id] = set(simple_extract(text))
    return out

# ─────────────────────────────────────────
# MODEL LOADING (SAFE)
# ─────────────────────────────────────────

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

_model = None
_model_failed = False


def get_model():
    global _model, _model_failed

    if _model_failed:
        return None

    if _model is None:
        try:
            if GLiNER is None:
                raise ImportError("GLiNER not installed")

            print("Loading GLiNER model...")
            _model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            print("GLiNER ready.")

        except Exception as e:
            print("⚠️ GLiNER failed:", e)
            print("Using fallback mode...")
            _model_failed = True
            return None

    return _model


# ─────────────────────────────────────────
# FALLBACK (important for demo stability)
# ─────────────────────────────────────────

def fallback_extract(text):
    text = text.lower()
    labels = []

    for label in GLINER_LABELS:
        if label in text:
            labels.append(label)

    return labels


# ─────────────────────────────────────────
# SINGLE TEXT CLASSIFICATION
# ─────────────────────────────────────────

def classify_text(text):

    if not text or not text.strip():
        return []

    model = get_model()

    if model:
        try:
            results = model.predict_entities(
                text,
                labels=GLINER_LABELS,
                threshold=THRESHOLD
            )

            return list(set([r["label"] for r in results]))

        except Exception:
            return fallback_extract(text)

    return fallback_extract(text)


# ─────────────────────────────────────────
# CACHE SYSTEM
# ─────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def classify_batch(texts):

    cache = load_cache()

    new_texts = [t for t in texts if t not in cache]

    print(f"New texts: {len(new_texts)}")

    for text in tqdm(new_texts, desc="GLiNER"):
        cache[text] = classify_text(text)

    save_cache(cache)

    return cache


# ─────────────────────────────────────────
# WEIGHTED LABEL BUILDER (VERY IMPORTANT)
# ─────────────────────────────────────────

def build_weighted_labels(labels):

    if not labels:
        return {}

    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    total = sum(counts.values())

    return {k: v / total for k, v in counts.items()}


# ─────────────────────────────────────────
# MANUSCRIPT PROCESSING
# ─────────────────────────────────────────

def process_manuscripts(manuscripts):

    texts = set()

    for m in manuscripts:
        texts.add((m.get("title", "") or "").strip())
        texts.add((m.get("abstract", "") or "").strip())

    cache = classify_batch(list(texts))

    results = {}

    for m in manuscripts:

        pid = m["paper_id"]

        title = (m.get("title", "") or "").strip()
        abstract = (m.get("abstract", "") or "").strip()

        labels = cache.get(title, []) + cache.get(abstract, [])

        weighted = build_weighted_labels(labels)

        results[pid] = {
            "labels": list(set(labels)),
            "weights": weighted
        }

    return results


# ─────────────────────────────────────────
# REVIEWER PROCESSING
# ─────────────────────────────────────────

def process_reviewers(reviewers):

    texts = set()

    for r in reviewers:
        for pub in r.get("publications", []):
            texts.add((pub.get("title", "") or "").strip())
            texts.add((pub.get("abstract", "") or "").strip())

    cache = classify_batch(list(texts))

    results = {}

    for r in reviewers:

        rid = r["reviewer_id"]
        labels = []

        for pub in r.get("publications", []):
            title = (pub.get("title", "") or "").strip()
            abstract = (pub.get("abstract", "") or "").strip()

            labels += cache.get(title, [])
            labels += cache.get(abstract, [])

        weighted = build_weighted_labels(labels)

        results[rid] = {
            "labels": list(set(labels)),
            "weights": weighted
        }

    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("\n=== GLiNER KG EXTRACTION ===")

    # Load data
    with open(MS_PATH, encoding="utf-8") as f:
        manuscripts = json.load(f)

    with open(REV_PATH, encoding="utf-8") as f:
        reviewers = json.load(f)

    # Process
    ms_results = process_manuscripts(manuscripts)
    rev_results = process_reviewers(reviewers)

    # Save outputs
    ms_out = os.path.join(OUTPUT_DIR, "gliner_manuscript_labels.json")
    rev_out = os.path.join(OUTPUT_DIR, "gliner_reviewer_labels.json")

    with open(ms_out, "w", encoding="utf-8") as f:
        json.dump(ms_results, f, indent=2)

    with open(rev_out, "w", encoding="utf-8") as f:
        json.dump(rev_results, f, indent=2)

    print("\nSaved:")
    print(ms_out)
    print(rev_out)

    print("\nDone — GLiNER KG ready for fusion.")
