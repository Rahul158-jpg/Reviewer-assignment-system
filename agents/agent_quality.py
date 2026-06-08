import ollama
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AGENT_QUALITY_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_TIMEOUT
)

# ------------------------------------------------
# RUBRIC PROMPT
# Structured scoring criteria sent to LLaMA 3.2
# Mirrors academic peer review quality assessment
# ------------------------------------------------

QUALITY_PROMPT_TEMPLATE = """
You are a scientific peer review assistant.
Evaluate the following manuscript abstract using 
this rubric:

Rubric Criteria (each worth 0.0 to 1.0):
1. Clarity: Is the research problem clearly stated?
2. Novelty: Does it propose something new?
3. Methodology: Is the approach technically sound?
4. Contribution: Are the contributions significant?
5. Evidence: Are results or validation mentioned?

Abstract:
\"\"\"{abstract}\"\"\"

Instructions:
- Score each criterion from 0.0 to 1.0
- Compute the average of all five scores
- Return ONLY the final average as a single float
- Do not explain. Do not add text. Just the number.

Example valid response: 0.76
"""

# ------------------------------------------------
# CORE AGENT FUNCTION
# ------------------------------------------------

def assess_manuscript_quality(abstract: str) -> float:
    """
    Sends manuscript abstract to LLaMA 3.2 for
    rubric-based quality scoring.

    Args:
        abstract: Raw abstract text of the manuscript

    Returns:
        float: Quality score Q(p) in range [0.0, 1.0]
               Returns 0.5 as neutral default on failure
    """

    if not abstract or not abstract.strip():
        print("[Agent A] Warning: Empty abstract received.")
        return 0.5

    prompt = QUALITY_PROMPT_TEMPLATE.format(
        abstract=abstract.strip()
    )

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=AGENT_QUALITY_MODEL,
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
                print(f"[Agent A — LLaMA 3.2] "
                      f"Quality Score: {score:.4f}")
                return score
            else:
                print(f"[Agent A] Attempt {attempt}: "
                      f"Could not parse score from: '{raw}'")

        except Exception as e:
            print(f"[Agent A] Attempt {attempt} failed: {e}")
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(2)

    # If all retries fail return neutral score
    print("[Agent A] All retries failed. "
          "Returning default score 0.5")
    return 0.5


def parse_score(raw_text: str) -> float:
    """
    Extracts a float score from LLM raw output.
    Handles cases where model adds extra text.

    Args:
        raw_text: Raw string output from LLaMA 3.2

    Returns:
        float in [0.0, 1.0] or None if unparseable
    """
    import re

    # Try direct float parse first
    try:
        score = float(raw_text.strip())
        return clamp(score)
    except ValueError:
        pass

    # Try extracting first float from text
    matches = re.findall(r"\b0?\.\d+|\b1\.0\b|\b0\b|\b1\b",
                         raw_text)
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
# Run: python agents/agent_quality.py
# ------------------------------------------------

if __name__ == "__main__":
    test_abstract = """
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
    """

    print("=" * 50)
    print("Agent A — Manuscript Quality Assessment")
    print("Model: LLaMA 3.2")
    print("=" * 50)

    score = assess_manuscript_quality(test_abstract)

    print(f"\nFinal Quality Score Q(p): {score:.4f}")
    print("=" * 50)