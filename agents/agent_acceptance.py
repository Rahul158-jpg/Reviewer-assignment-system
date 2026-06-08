# agents/agent_acceptance.py
# ================================================
# Agent B — Acceptance Likelihood Prediction
# Model: Mistral
#
# Predicts the likelihood that a manuscript will
# be accepted at a peer reviewed venue.
# Returns acceptance score A(p) in [0.0, 1.0]
# ================================================

import ollama
import time
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AGENT_ACCEPTANCE_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_TIMEOUT
)

# ------------------------------------------------
# ACCEPTANCE PREDICTION PROMPT
# Mistral is used here for its strong analytical
# judgment on structured academic assessment tasks
# ------------------------------------------------

ACCEPTANCE_PROMPT_TEMPLATE = """
You are an experienced academic program chair with 
expertise in evaluating research submissions.

Assess the following manuscript abstract and predict 
the likelihood it would be accepted at a competitive 
peer reviewed venue.

Evaluate based on these four dimensions:

1. Originality: Does it present a genuinely new idea
   or significant advancement over existing work?

2. Technical Soundness: Is the methodology rigorous,
   well defined, and reproducible?

3. Impact: Would acceptance advance the field
   meaningfully?

4. Presentation Quality: Is the abstract well 
   written, precise, and professionally structured?

Abstract:
\"\"\"{abstract}\"\"\"

Scoring Instructions:
- Score each dimension from 0.0 to 1.0
- Compute the weighted average:
  Originality x 0.30
  Technical Soundness x 0.35
  Impact x 0.20
  Presentation x 0.15
- Return ONLY the final weighted average as a
  single decimal number between 0.0 and 1.0
- Do not explain. Do not add text. Just the number.

Example valid response: 0.81
"""

# ------------------------------------------------
# CORE AGENT FUNCTION
# ------------------------------------------------

def predict_acceptance_likelihood(abstract: str) -> float:
    """
    Sends manuscript abstract to Mistral for
    acceptance likelihood prediction.

    Args:
        abstract: Raw abstract text of the manuscript

    Returns:
        float: Acceptance score A(p) in range [0.0, 1.0]
               Returns 0.5 as neutral default on failure
    """

    if not abstract or not abstract.strip():
        print("[Agent B] Warning: Empty abstract received.")
        return 0.5

    prompt = ACCEPTANCE_PROMPT_TEMPLATE.format(
        abstract=abstract.strip()
    )

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=AGENT_ACCEPTANCE_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                options=OLLAMA_OPTIONS
            )

            raw = response['message']['content'].strip()

            # Parse float from response
            score = parse_score(raw)

            if score is not None:
                print(f"[Agent B — Mistral] "
                      f"Acceptance Score: {score:.4f}")
                return score
            else:
                print(f"[Agent B] Attempt {attempt}: "
                      f"Could not parse score from: '{raw}'")

        except Exception as e:
            print(f"[Agent B] Attempt {attempt} failed: {e}")
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(2)

    # If all retries fail return neutral score
    print("[Agent B] All retries failed. "
          "Returning default score 0.5")
    return 0.5


# ------------------------------------------------
# BATCH PROCESSING
# Useful when running on full manuscript dataset
# ------------------------------------------------

def predict_batch(abstracts: dict) -> dict:
    """
    Runs acceptance prediction on a batch of
    manuscripts.

    Args:
        abstracts: dict of {paper_id: abstract_text}

    Returns:
        dict of {paper_id: acceptance_score}
    """
    results = {}
    total = len(abstracts)

    for idx, (paper_id, abstract) in enumerate(
        abstracts.items(), 1
    ):
        print(f"\n[Agent B] Processing {idx}/{total} "
              f"— Paper: {paper_id}")
        score = predict_acceptance_likelihood(abstract)
        results[paper_id] = score

    return results


# ------------------------------------------------
# SCORE PARSING UTILITIES
# ------------------------------------------------

def parse_score(raw_text: str) -> float:
    """
    Extracts a float score from Mistral raw output.
    Handles cases where model adds surrounding text.

    Args:
        raw_text: Raw string output from Mistral

    Returns:
        float in [0.0, 1.0] or None if unparseable
    """

    # Try direct float parse first
    try:
        score = float(raw_text.strip())
        return clamp(score)
    except ValueError:
        pass

    # Try extracting first float from text
    matches = re.findall(
        r"\b0?\.\d+|\b1\.0\b|\b0\b|\b1\b",
        raw_text
    )
    if matches:
        try:
            score = float(matches[0])
            return clamp(score)
        except ValueError:
            pass

    return None


def clamp(score: float) -> float:
    """Ensures score stays within [0.0, 1.0]"""
    return max(0.0, min(1.0, score))


# ------------------------------------------------
# STANDALONE TEST
# Run: python agents/agent_acceptance.py
# ------------------------------------------------

if __name__ == "__main__":

    test_cases = {
        "paper_001": """
        We propose a novel knowledge graph-enhanced 
        framework for automated reviewer assignment 
        in scholarly peer review. Our approach integrates 
        structured semantic representations with large 
        language model reasoning to assess manuscript 
        quality, reviewer reliability, and ethical 
        constraints simultaneously. Experiments on 
        real-world conference datasets demonstrate 
        significant improvements over TF-IDF and 
        embedding-based baselines, achieving a 
        Precision@3 of 0.78 and NDCG@3 of 0.82.
        """,
        "paper_002": """
        This paper presents a study of sorting 
        algorithms. We compare bubble sort and 
        insertion sort on small arrays. Results 
        show both work similarly on datasets 
        under 100 elements.
        """
    }

    print("=" * 50)
    print("Agent B — Acceptance Likelihood Prediction")
    print("Model: Mistral")
    print("=" * 50)

    for paper_id, abstract in test_cases.items():
        print(f"\nTesting: {paper_id}")
        score = predict_acceptance_likelihood(abstract)
        print(f"Acceptance Score A(p): {score:.4f}")

    print("\n" + "=" * 50)
    print("Batch Processing Test")
    print("=" * 50)

    batch_results = predict_batch(test_cases)
    for pid, score in batch_results.items():
        print(f"{pid}: {score:.4f}")