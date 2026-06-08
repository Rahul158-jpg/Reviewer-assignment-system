# agents/agent_reliability.py
# ================================================
# Agent C — Reviewer Reliability Modeling
# Model: Phi3
#
# Models reviewer behavioral reliability based
# on their publication record and review history.
# Returns reliability score R(r) in [0.0, 1.0]
# ================================================

import ollama
import time
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AGENT_RELIABILITY_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_TIMEOUT
)

# ------------------------------------------------
# RELIABILITY ASSESSMENT PROMPT
# Phi3 is used here for its strong behavioral
# pattern recognition from structured text inputs
# ------------------------------------------------

RELIABILITY_PROMPT_TEMPLATE = """
You are an expert editorial manager assessing 
reviewer reliability for a peer review system.

Based on the following reviewer profile, estimate 
how reliable this reviewer would be if assigned 
a manuscript to review.

Reviewer Profile:
- Name: {reviewer_name}
- Number of Publications: {pub_count}
- Research Areas: {research_areas}
- Recent Publication Titles:
{recent_titles}

Evaluate reliability across four dimensions:

1. Research Activity: Is the reviewer actively 
   publishing in relevant areas?
   (More recent publications = higher activity)

2. Domain Depth: Does the reviewer have deep 
   expertise in a focused research area?
   (Focused expertise = higher depth)

3. Publication Consistency: Is their publication 
   record consistent over time?
   (Regular output = higher consistency)

4. Profile Completeness: Is enough known about 
   this reviewer to make a confident assignment?
   (More information = higher confidence)

Scoring Instructions:
- Score each dimension from 0.0 to 1.0
- Compute the weighted average:
  Research Activity    x 0.30
  Domain Depth         x 0.35
  Publication Consistency x 0.25
  Profile Completeness x 0.10
- Return ONLY the final weighted average as a
  single decimal number between 0.0 and 1.0
- Do not explain. Do not add text. Just the number.

Example valid response: 0.73
"""

# ------------------------------------------------
# CORE AGENT FUNCTION
# ------------------------------------------------

def assess_reviewer_reliability(reviewer: dict) -> float:
    """
    Sends reviewer profile to Phi3 for behavioral
    reliability scoring.

    Args:
        reviewer: dict containing reviewer profile
                  with keys: reviewer_id, name,
                  publications, behavior_score, tier

    Returns:
        float: Reliability score R(r) in [0.0, 1.0]
               Returns 0.5 as neutral default on failure
    """

    if not reviewer:
        print("[Agent C] Warning: Empty reviewer "
              "profile received.")
        return 0.5

    # Extract reviewer profile components
    reviewer_name = reviewer.get(
        'name', 
        reviewer.get('reviewer_id', 'Unknown')
    )

    publications = reviewer.get('publications', [])
    pub_count    = len(publications)

    # Extract research areas from publication titles
    research_areas = extract_research_areas(publications)

    # Get recent publication titles (last 5)
    recent_titles = extract_recent_titles(
        publications, limit=5
    )

    if pub_count == 0:
        print(f"[Agent C] Warning: No publications "
              f"found for reviewer {reviewer_name}. "
              f"Returning default 0.3")
        return 0.3

    prompt = RELIABILITY_PROMPT_TEMPLATE.format(
        reviewer_name    = reviewer_name,
        pub_count        = pub_count,
        research_areas   = research_areas,
        recent_titles    = recent_titles
    )

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=AGENT_RELIABILITY_MODEL,
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
                print(f"[Agent C — Phi3] Reviewer: "
                      f"{reviewer_name} | "
                      f"Reliability Score: {score:.4f}")
                return score
            else:
                print(f"[Agent C] Attempt {attempt}: "
                      f"Could not parse score "
                      f"from: '{raw}'")

        except Exception as e:
            print(f"[Agent C] Attempt {attempt} "
                  f"failed: {e}")
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(2)

    print(f"[Agent C] All retries failed for "
          f"{reviewer_name}. Returning default 0.5")
    return 0.5


# ------------------------------------------------
# BATCH PROCESSING
# Precomputes reliability scores for all reviewers
# before the main assignment pipeline runs
# ------------------------------------------------

def precompute_reliability(reviewers: dict) -> dict:
    """
    Precomputes reliability scores for all reviewers.
    Run this once before the assignment pipeline
    to avoid repeated LLM calls per paper.

    Args:
        reviewers: dict of {reviewer_id: reviewer_dict}

    Returns:
        dict of {reviewer_id: reliability_score}
    """
    reliability_scores = {}
    total = len(reviewers)

    print(f"\n[Agent C] Precomputing reliability "
          f"scores for {total} reviewers...")

    for idx, (reviewer_id, reviewer) in enumerate(
        reviewers.items(), 1
    ):
        print(f"\n[Agent C] Reviewer {idx}/{total} "
              f"— ID: {reviewer_id}")

        # Use behavior_score from JSON as fallback
        # if LLM scoring fails
        fallback = reviewer.get('behavior_score', 0.5)

        score = assess_reviewer_reliability(reviewer)

        # Blend LLM score with behavior_score
        # for robustness
        # 70% LLM judgment + 30% historical behavior
        blended = round(0.7 * score + 0.3 * fallback, 4)

        reliability_scores[reviewer_id] = blended
        print(f"[Agent C] Blended R(r): {blended:.4f} "
              f"(LLM: {score:.4f} | "
              f"Historical: {fallback:.4f})")

    return reliability_scores


# ------------------------------------------------
# PROFILE EXTRACTION UTILITIES
# ------------------------------------------------

def extract_research_areas(
    publications: list,
    limit: int = 10
) -> str:
    """
    Extracts key research topics from publication
    titles to build a reviewer research profile.
    """
    if not publications:
        return "No publications available"

    titles = []
    for pub in publications[:limit]:
        title = pub.get('title', '')
        if title:
            titles.append(title)

    if not titles:
        return "No publication titles available"

    # Simple keyword extraction from titles
    common_words = {
        'a', 'an', 'the', 'of', 'in', 'on',
        'for', 'and', 'or', 'with', 'using',
        'based', 'via', 'towards', 'from',
        'to', 'is', 'are', 'by', 'an', 'at'
    }

    words = []
    for title in titles:
        tokens = title.lower().split()
        words.extend([
            t for t in tokens
            if t not in common_words
            and len(t) > 3
        ])

    # Get most frequent meaningful words
    from collections import Counter
    freq = Counter(words)
    top_areas = [
        word for word, _ in freq.most_common(8)
    ]

    return ", ".join(top_areas) if top_areas \
        else "General Computer Science"


def extract_recent_titles(
    publications: list,
    limit: int = 5
) -> str:
    """
    Formats recent publication titles for the prompt.
    """
    if not publications:
        return "  - No publications available"

    titles = []
    for pub in publications[:limit]:
        title = pub.get('title', '').strip()
        if title:
            titles.append(f"  - {title}")

    return "\n".join(titles) if titles \
        else "  - No titles available"


# ------------------------------------------------
# SCORE PARSING UTILITIES
# ------------------------------------------------

def parse_score(raw_text: str) -> float:
    """
    Extracts a float score from Phi3 raw output.

    Args:
        raw_text: Raw string output from Phi3

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
# Run: python agents/agent_reliability.py
# ------------------------------------------------

if __name__ == "__main__":

    test_reviewers = {
        "reviewer_001": {
            "reviewer_id": "reviewer_001",
            "name":        "Dr. Sarah Chen",
            "tier":        "Expert",
            "behavior_score": 0.88,
            "publications": [
                {
                    "title": "Knowledge Graph Enhanced"
                             " Semantic Reasoning for"
                             " Scientific Documents",
                    "abstract": "We propose a novel KG"
                                " approach..."
                },
                {
                    "title": "Large Language Models for"
                             " Information Extraction",
                    "abstract": "This paper presents..."
                },
                {
                    "title": "Graph Neural Networks in"
                             " Scholarly Recommendation",
                    "abstract": "We demonstrate..."
                },
                {
                    "title": "Automated Peer Review"
                             " Quality Assessment",
                    "abstract": "A rubric-based system..."
                },
                {
                    "title": "Multi-hop Reasoning over"
                             " Academic Knowledge Graphs",
                    "abstract": "We introduce..."
                }
            ]
        },
        "reviewer_002": {
            "reviewer_id": "reviewer_002",
            "name":        "Dr. James Wilson",
            "tier":        "Fresher",
            "behavior_score": 0.42,
            "publications": [
                {
                    "title": "A Survey of Sorting"
                             " Algorithms",
                    "abstract": "We survey..."
                }
            ]
        }
    }

    print("=" * 55)
    print("Agent C — Reviewer Reliability Modeling")
    print("Model: Phi3")
    print("=" * 55)

    # Test individual assessment
    for rid, reviewer in test_reviewers.items():
        print(f"\nTesting reviewer: "
              f"{reviewer['name']} [{reviewer['tier']}]")
        score = assess_reviewer_reliability(reviewer)
        print(f"Reliability Score R(r): {score:.4f}")

    print("\n" + "=" * 55)
    print("Batch Precomputation Test")
    print("=" * 55)

    batch_scores = precompute_reliability(test_reviewers)
    for rid, score in batch_scores.items():
        name = test_reviewers[rid]['name']
        print(f"{name}: {score:.4f}")