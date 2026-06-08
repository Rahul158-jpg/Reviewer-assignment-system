import numpy as np


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_kg_similarity(paper_id, reviewer_id, embeddings):
    paper_key = f"p_{paper_id}"
    reviewer_key = f"r_{reviewer_id}"

    if paper_key not in embeddings or reviewer_key not in embeddings:
        return 0.0

    score = cosine(embeddings[paper_key], embeddings[reviewer_key])
    return max(0.0, min(1.0, score))
