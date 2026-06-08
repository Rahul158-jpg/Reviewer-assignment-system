"""
preprocessing.py — FINAL VERSION

Reviewer Recommendation — Text Preprocessing

Features:
✔ Clean text (remove noise)
✔ Normalize tokens
✔ Lightweight stopword removal
✔ Keyword extraction (for similarity)
✔ Consistent across pipeline
"""

import re

# ─────────────────────────────────────────
# STOPWORDS (carefully selected)
# ─────────────────────────────────────────

STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might',
    'this','that','these','those','it','its','we','our','they','their',
    'which','who','by','from','as','if','not','so','all','can','also',
    'such','more','into','over','each','both','about','than','then',
    'when','where','how','what'
}

# IMPORTANT: do NOT remove domain words like:
# model, learning, network, data, etc.


# ─────────────────────────────────────────
# BASIC CLEANING
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    # remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # remove numbers (optional)
    text = re.sub(r'\d+', ' ', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ─────────────────────────────────────────
# TOKENIZATION + STOPWORD REMOVAL
# ─────────────────────────────────────────

def tokenize(text: str):
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


# ─────────────────────────────────────────
# FULL PREPROCESS PIPELINE
# ─────────────────────────────────────────

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


# ─────────────────────────────────────────
# KEYWORD EXTRACTION (LIGHTWEIGHT)
# ─────────────────────────────────────────

def extract_keywords(text: str, top_k=20):
    """
    Simple frequency-based keyword extraction
    """
    text = preprocess_text(text)
    tokens = text.split()

    freq = {}

    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    # sort by frequency
    keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return [k for k, _ in keywords[:top_k]]


# ─────────────────────────────────────────
# BUILD PAPER TEXT
# ─────────────────────────────────────────

def build_paper_text(paper):
    return preprocess_text(
        (paper.get("title", "") + " " +
         paper.get("abstract", ""))
    )


# ─────────────────────────────────────────
# BUILD REVIEWER TEXT
# ─────────────────────────────────────────

def build_reviewer_text(reviewer, max_pubs=5):

    text = ""

    for pub in reviewer.get("publications", [])[:max_pubs]:
        text += pub.get("title", "") + " "
        text += pub.get("abstract", "") + " "

    return preprocess_text(text)