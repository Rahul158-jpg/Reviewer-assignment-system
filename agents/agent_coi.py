# agents/agent_coi.py
# ================================================
# Agent D — Conflict of Interest Detection
# Model: Gemma2
#
# Detects potential conflicts of interest between
# manuscripts and reviewers using relational
# reasoning over scholarly network signals.
# Returns COI risk score E(p,r) in [0.0, 1.0]
# Higher score = higher conflict risk
# ================================================

import ollama
import time
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AGENT_COI_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_TIMEOUT
)

# ------------------------------------------------
# COI DETECTION PROMPT
# Gemma2 is used here for its superior relational
# and entity level reasoning capabilities
# ------------------------------------------------

COI_PROMPT_TEMPLATE = """
You are an ethics officer for a peer review system.
Your task is to detect conflicts of interest between
a manuscript and a potential reviewer.

Manuscript Information:
- Title: {paper_title}
- Authors: {paper_authors}
- Institutions: {paper_institutions}

Reviewer Information:
- Name: {reviewer_name}
- Institution: {reviewer_institution}
- Recent Co-authors: {reviewer_coauthors}
- Recent Publication Titles:
{reviewer_titles}

Known Conflict Signals:
- Co-authorship overlap: {coauthor_overlap}
- Institutional match: {institution_match}
- Citation dependence: {citation_dependence}

Evaluate conflict of interest risk across 
these four dimensions:

1. Direct Co-authorship: Has this reviewer 
   co-authored papers with any manuscript author?
   (Direct overlap = very high risk)

2. Institutional Proximity: Do the reviewer and
   any manuscript author share the same institution?
   (Same institution = high risk)

3. Citation Dependence: Does the reviewer heavily
   cite or get cited by the manuscript authors?
   (Strong citation link = moderate risk)

4. Competitive Conflict: Is the reviewer working
   on directly competing research that could bias
   their judgment either positively or negatively?
   (Direct competitor = moderate risk)

Scoring Instructions:
- Score each dimension from 0.0 to 1.0
- Compute the weighted average:
  Direct Co-authorship    x 0.40
  Institutional Proximity x 0.30
  Citation Dependence     x 0.20
  Competitive Conflict    x 0.10
- Return ONLY the final weighted average as a
  single decimal number between 0.0 and 1.0
- Higher score means higher conflict of interest
- Do not explain. Do not add text. Just the number.

Example valid response: 0.23
"""

# ------------------------------------------------
# CORE AGENT FUNCTION
# ------------------------------------------------

def detect_coi(
    paper: dict,
    reviewer: dict,
    coi_data: dict = None
) -> float:
    """
    Detects conflict of interest between a manuscript
    and a reviewer using Gemma2 relational reasoning
    combined with structured COI graph signals.

    Args:
        paper:    dict with paper metadata including
                  title, authors, institutions
        reviewer: dict with reviewer profile including
                  name, institution, publications
        coi_data: optional pre-computed COI signals
                  from coi_graph.json

    Returns:
        float: COI risk score E(p,r) in [0.0, 1.0]
               Returns 0.0 as safe default on failure
    """

    if not paper or not reviewer:
        print("[Agent D] Warning: Missing paper "
              "or reviewer data.")
        return 0.0

    # If COI graph has a pre-flagged entry use it
    # as a strong signal combined with LLM reasoning
    if coi_data:
        graph_score = coi_data.get('coi_score', 0.0)
        flagged     = coi_data.get('flagged', False)

        # If already flagged in graph return
        # high risk immediately without LLM call
        if flagged:
            print(f"[Agent D — Gemma2] "
                  f"Pre-flagged COI detected. "
                  f"Graph score: {graph_score:.4f}")
            return min(1.0, graph_score + 0.2)

    # Extract paper information
    paper_title        = paper.get('title', 'Unknown')
    paper_authors      = extract_authors(paper)
    paper_institutions = extract_institutions(paper)

    # Extract reviewer information
    reviewer_name        = reviewer.get(
        'name',
        reviewer.get('reviewer_id', 'Unknown')
    )
    reviewer_institution = reviewer.get(
        'institution', 'Unknown'
    )
    reviewer_coauthors   = extract_coauthors(reviewer)
    reviewer_titles      = extract_recent_titles(
        reviewer.get('publications', []), limit=5
    )

    # Compute structural overlap signals
    coauthor_overlap    = check_coauthor_overlap(
        paper_authors, reviewer_coauthors
    )
    institution_match   = check_institution_match(
        paper_institutions, reviewer_institution
    )
    citation_dependence = get_citation_signal(
        coi_data
    )

    prompt = COI_PROMPT_TEMPLATE.format(
        paper_title          = paper_title,
        paper_authors        = paper_authors,
        paper_institutions   = paper_institutions,
        reviewer_name        = reviewer_name,
        reviewer_institution = reviewer_institution,
        reviewer_coauthors   = reviewer_coauthors,
        reviewer_titles      = reviewer_titles,
        coauthor_overlap     = coauthor_overlap,
        institution_match    = institution_match,
        citation_dependence  = citation_dependence
    )

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=AGENT_COI_MODEL,
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
                # Blend LLM score with graph signal
                final_score = blend_scores(
                    llm_score   = score,
                    graph_score = coi_data.get(
                        'coi_score', 0.0
                    ) if coi_data else 0.0
                )
                print(f"[Agent D — Gemma2] "
                      f"Reviewer: {reviewer_name} | "
                      f"COI Risk: {final_score:.4f}")
                return final_score
            else:
                print(f"[Agent D] Attempt {attempt}: "
                      f"Could not parse score "
                      f"from: '{raw}'")

        except Exception as e:
            print(f"[Agent D] Attempt {attempt} "
                  f"failed: {e}")
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(2)

    print(f"[Agent D] All retries failed. "
          f"Returning safe default 0.0")
    return 0.0


# ------------------------------------------------
# BATCH COI DETECTION
# Computes COI scores for all reviewer candidates
# for a given manuscript
# ------------------------------------------------

def detect_coi_batch(
    paper: dict,
    reviewers: dict,
    coi_lookup: dict = None
) -> dict:
    """
    Runs COI detection for all reviewers
    against a single manuscript.

    Args:
        paper:      manuscript dict
        reviewers:  dict of {reviewer_id: reviewer}
        coi_lookup: dict of {(paper_id, reviewer_id):
                    coi_signals}

    Returns:
        dict of {reviewer_id: coi_risk_score}
    """
    coi_scores = {}
    paper_id   = paper.get('paper_id', '')
    total      = len(reviewers)

    for idx, (reviewer_id, reviewer) in enumerate(
        reviewers.items(), 1
    ):
        print(f"\n[Agent D] COI check "
              f"{idx}/{total} — "
              f"Reviewer: {reviewer_id}")

        # Get pre-computed COI signals if available
        coi_data = None
        if coi_lookup:
            coi_data = coi_lookup.get(
                (paper_id, reviewer_id), None
            )

        score = detect_coi(paper, reviewer, coi_data)
        coi_scores[reviewer_id] = score

    return coi_scores


# ------------------------------------------------
# STRUCTURAL SIGNAL EXTRACTORS
# These provide hard evidence to supplement
# Gemma2 relational reasoning
# ------------------------------------------------

def extract_authors(paper: dict) -> str:
    """Extracts formatted author list from paper."""
    authors = paper.get('authors', [])
    if isinstance(authors, list):
        return ", ".join(authors) \
            if authors else "Unknown"
    if isinstance(authors, str):
        return authors
    return "Unknown"


def extract_institutions(paper: dict) -> str:
    """Extracts institution list from paper."""
    institutions = paper.get('institutions', [])
    if isinstance(institutions, list):
        return ", ".join(institutions) \
            if institutions else "Unknown"
    if isinstance(institutions, str):
        return institutions
    return "Unknown"


def extract_coauthors(reviewer: dict) -> str:
    """
    Extracts co-author names from reviewer
    publication records.
    """
    publications = reviewer.get('publications', [])
    coauthors    = set()

    for pub in publications[:10]:
        authors = pub.get('authors', [])
        if isinstance(authors, list):
            coauthors.update(authors)
        elif isinstance(authors, str):
            for a in authors.split(','):
                coauthors.add(a.strip())

    # Remove the reviewer themselves
    reviewer_name = reviewer.get('name', '')
    coauthors.discard(reviewer_name)

    return ", ".join(list(coauthors)[:10]) \
        if coauthors else "None found"


def extract_recent_titles(
    publications: list,
    limit: int = 5
) -> str:
    """Formats recent publication titles."""
    if not publications:
        return "  - No publications available"

    titles = []
    for pub in publications[:limit]:
        title = pub.get('title', '').strip()
        if title:
            titles.append(f"  - {title}")

    return "\n".join(titles) \
        if titles else "  - No titles available"


def check_coauthor_overlap(
    paper_authors: str,
    reviewer_coauthors: str
) -> str:
    """
    Checks for direct co-authorship overlap
    between manuscript authors and reviewer
    co-author history.
    """
    if (paper_authors == "Unknown" or
            reviewer_coauthors == "None found"):
        return "No overlap detected"

    paper_names    = set(
        a.strip().lower()
        for a in paper_authors.split(',')
    )
    reviewer_names = set(
        a.strip().lower()
        for a in reviewer_coauthors.split(',')
    )

    overlap = paper_names & reviewer_names

    if overlap:
        return (f"OVERLAP DETECTED: "
                f"{', '.join(overlap)}")
    return "No direct overlap detected"


def check_institution_match(
    paper_institutions: str,
    reviewer_institution: str
) -> str:
    """
    Checks for institutional proximity between
    manuscript authors and reviewer.
    """
    if (paper_institutions == "Unknown" or
            reviewer_institution == "Unknown"):
        return "Unknown"

    paper_inst    = paper_institutions.lower()
    reviewer_inst = reviewer_institution.lower()

    if reviewer_inst in paper_inst:
        return (f"MATCH DETECTED: "
                f"{reviewer_institution}")
    return "No institutional match"


def get_citation_signal(coi_data: dict) -> str:
    """
    Extracts citation dependence signal from
    pre-computed COI graph data.
    """
    if not coi_data:
        return "No citation data available"

    coi_score = coi_data.get('coi_score', 0.0)
    name_overlap = coi_data.get(
        'name_overlap', False
    )

    signals = []
    if coi_score > 0.5:
        signals.append(
            f"High COI score: {coi_score:.2f}"
        )
    if name_overlap:
        signals.append("Name overlap detected")

    return (", ".join(signals)
            if signals
            else "No strong citation signals")


# ------------------------------------------------
# SCORE UTILITIES
# ------------------------------------------------

def blend_scores(
    llm_score: float,
    graph_score: float,
    llm_weight: float = 0.6,
    graph_weight: float = 0.4
) -> float:
    """
    Blends LLM relational reasoning score with
    structural COI graph signal.
    60% LLM judgment + 40% graph evidence
    """
    blended = (llm_weight   * llm_score +
               graph_weight * graph_score)
    return round(clamp(blended), 4)


def parse_score(raw_text: str) -> float:
    """
    Extracts float score from Gemma2 raw output.

    Returns:
        float in [0.0, 1.0] or None if unparseable
    """
    try:
        score = float(raw_text.strip())
        return clamp(score)
    except ValueError:
        pass

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
# Run: python agents/agent_coi.py
# ------------------------------------------------

if __name__ == "__main__":

    test_paper = {
        "paper_id":     "paper_001",
        "title":        "Knowledge Graph Enhanced"
                        " Reviewer Assignment",
        "authors":      ["Dr. Sarah Chen",
                         "Dr. James Wilson"],
        "institutions": ["MIT", "Stanford University"]
    }

    test_reviewers = {
        "reviewer_001": {
            "reviewer_id":   "reviewer_001",
            "name":          "Dr. Sarah Chen",
            "institution":   "MIT",
            "publications":  [
                {
                    "title": "Knowledge Graphs in"
                             " Peer Review",
                    "authors": ["Dr. Sarah Chen",
                                "Dr. James Wilson"]
                }
            ]
        },
        "reviewer_002": {
            "reviewer_id":   "reviewer_002",
            "name":          "Dr. Ali Hassan",
            "institution":   "University of Edinburgh",
            "publications":  [
                {
                    "title": "Graph Neural Networks"
                             " for Recommendation",
                    "authors": ["Dr. Ali Hassan",
                                "Dr. Maria Garcia"]
                }
            ]
        }
    }

    test_coi_lookup = {
        ("paper_001", "reviewer_001"): {
            "coi_score":    0.95,
            "flagged":      True,
            "name_overlap": True
        },
        ("paper_001", "reviewer_002"): {
            "coi_score":    0.05,
            "flagged":      False,
            "name_overlap": False
        }
    }

    print("=" * 55)
    print("Agent D — Conflict of Interest Detection")
    print("Model: Gemma2")
    print("=" * 55)

    for rid, reviewer in test_reviewers.items():
        print(f"\nChecking COI for: {reviewer['name']}")
        coi_data = test_coi_lookup.get(
            ("paper_001", rid), None
        )
        score = detect_coi(
            test_paper, reviewer, coi_data
        )
        print(f"COI Risk Score E(p,r): {score:.4f}")
        if score > 0.5:
            print(f"⚠️  HIGH RISK — "
                  f"Reviewer should be excluded")
        else:
            print(f"✅ LOW RISK — "
                  f"Reviewer is safe to assign")

    print("\n" + "=" * 55)
    print("Batch COI Detection Test")
    print("=" * 55)

    batch_scores = detect_coi_batch(
        test_paper,
        test_reviewers,
        test_coi_lookup
    )

    for rid, score in batch_scores.items():
        name = test_reviewers[rid]['name']
        flag = "⚠️  HIGH" if score > 0.5 else "✅ SAFE"
        print(f"{name}: {score:.4f} — {flag}")