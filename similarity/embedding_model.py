"""
embedding_model.py — FINAL VERSION

Reviewer Recommendation — Embedding Engine

Features:
✔ SPECTER (default — best for research papers)
✔ MiniLM fallback (fast mode)
✔ Persistent caching (CRITICAL)
✔ Batch + single inference
✔ Safe loading (no repeated model init)
"""

import os
import pickle
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CACHE_PATH = os.path.join(OUTPUT_DIR, "embedding_cache.pkl")

try:
    from config import EMBEDDING_MODEL as MODEL_NAME
except Exception:
    MODEL_NAME = "all-mpnet-base-v2"

FAST_MODEL = "all-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

_model = None


# ─────────────────────────────────────────
# LOAD MODEL (SINGLETON)
# ─────────────────────────────────────────

def load_model(use_fast=False):
    global _model

    if _model is None:
        try:
            name = FAST_MODEL if use_fast else MODEL_NAME
            print(f"Loading embedding model: {name}")
            _model = SentenceTransformer(name)
        except Exception:
            print("⚠️ Falling back to MiniLM")
            _model = SentenceTransformer(FAST_MODEL)

    return _model


# ─────────────────────────────────────────
# CACHE SYSTEM
# ─────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


# ─────────────────────────────────────────
# TEXT BUILDER (IMPORTANT)
# ─────────────────────────────────────────

def build_text(title="", abstract=""):
    return (title + " " + abstract).strip()


# ─────────────────────────────────────────
# BATCH ENCODING (SMART CACHE)
# ─────────────────────────────────────────

def encode_texts(texts, use_cache=True):

    model = load_model()
    cache = load_cache() if use_cache else {}

    to_encode = []
    keys = []

    for text in texts:
        if use_cache and text in cache:
            continue
        to_encode.append(text)
        keys.append(text)

    # compute missing embeddings
    if to_encode:
        print(f"Encoding {len(to_encode)} new texts...")

        embeddings = model.encode(
            to_encode,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for k, emb in zip(keys, embeddings):
            cache[k] = emb

        save_cache(cache)

    # return ordered embeddings
    return [cache[t] for t in texts]


# ─────────────────────────────────────────
# SINGLE TEXT ENCODING (FOR UI)
# ─────────────────────────────────────────

def encode_single(text):

    model = load_model()
    cache = load_cache()

    if text in cache:
        return cache[text]

    emb = model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    cache[text] = emb
    save_cache(cache)

    return emb


# ─────────────────────────────────────────
# BUILD PAPER EMBEDDINGS
# ─────────────────────────────────────────

def build_paper_embeddings(manuscripts):

    print("\nBuilding paper embeddings...")

    texts = [
        build_text(m.get("title", ""), m.get("abstract", ""))
        for m in manuscripts
    ]

    embeddings = encode_texts(texts)

    return {
        str(m["paper_id"]): emb
        for m, emb in zip(manuscripts, embeddings)
    }


# ─────────────────────────────────────────
# BUILD REVIEWER EMBEDDINGS
# ─────────────────────────────────────────

def build_reviewer_embeddings(reviewers):

    print("\nBuilding reviewer embeddings...")

    texts = []

    for r in reviewers:
        combined = ""

        # limit publications (important for speed)
        for pub in r.get("publications", [])[:5]:
            combined += pub.get("title", "") + " "
            combined += pub.get("abstract", "") + " "

        texts.append(combined.strip())

    embeddings = encode_texts(texts)

    return {
        str(r.get("reviewer_id")): emb
        for r, emb in zip(reviewers, embeddings)
    }