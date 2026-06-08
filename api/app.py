# ==========================================
# Reviewer Assignment  — FINAL STABLE APP
# ==========================================

from flask import Flask, render_template, request
import os
import sys
import copy
import json
import hashlib
import time
import math
import random
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.pdf_extractor import extract_pdf_metadata
from assignment.assigner import (
    assign_reviewers,
    build_paper_text,
    build_reviewer_text,
    classify_reviewer,
    compute_authority,
    compute_stats,
    prepare_reviewers_for_ranking,
    reviewer_capacity_status,
)
from utils.loader import load_json

from evaluation.system_metrics import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_coverage,
    compute_diversity,
    compute_load_balance
)

app = Flask(__name__)

# -------------------------------
# PATH SETUP
# -------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MANUSCRIPTS_PATH = os.path.join(DATA_DIR, "manuscripts.json")
REVIEWERS_PATH   = os.path.join(DATA_DIR, "reviewers.json")
GROUND_TRUTH     = os.path.join(DATA_DIR, "ground_truth.json")
SIM_PATH         = os.path.join(OUTPUT_DIR, "similarity_combined.json")
ASSIGN_PATH      = os.path.join(OUTPUT_DIR, "assignments.json")
METRICS_PATH     = os.path.join(OUTPUT_DIR, "metrics.json")
FAIRNESS_PATH    = os.path.join(RESULTS_DIR, "fairness_values.json")
LIVE_WORKLOAD_PATH = os.path.join(OUTPUT_DIR, "live_workload.json")

# -------------------------------
# LOAD DATA
# -------------------------------
manuscripts = load_json(MANUSCRIPTS_PATH) or []
reviewers   = load_json(REVIEWERS_PATH) or []
gt_raw      = load_json(GROUND_TRUTH) or []
sim_data    = load_json(SIM_PATH) or {}
assignments = load_json(ASSIGN_PATH) or {}
saved_metrics = load_json(METRICS_PATH) or {}
_RANK_CACHE = {}
_SCORED_ASSIGNMENT_CACHE = {}
_ABLATION_RESULT_CACHE = {}
_BASELINE_RESULT_CACHE = {}


def load_optional_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _reviewer_ids_from_assignment_payload(assigned):
    if isinstance(assigned, dict):
        if isinstance(assigned.get("full"), list):
            return _reviewer_ids_from_assignment_payload(assigned.get("full"))
        if assigned.get("reviewer_id"):
            return [str(assigned.get("reviewer_id"))]
        reviewer_ids = []
        for value in assigned.values():
            if isinstance(value, list):
                reviewer_ids.extend(_reviewer_ids_from_assignment_payload(value))
            elif isinstance(value, dict):
                reviewer_ids.extend(_reviewer_ids_from_assignment_payload(value))
            elif value:
                reviewer_ids.append(str(value))
        return reviewer_ids
    if isinstance(assigned, list):
        reviewer_ids = []
        for item in assigned:
            if isinstance(item, dict):
                rid = item.get("reviewer_id")
                if rid:
                    reviewer_ids.append(str(rid))
            elif item:
                reviewer_ids.append(str(item))
        return reviewer_ids
    return []


def load_assignment_history_counts():
    history = load_optional_json(ASSIGN_PATH, {}) or {}
    counts = Counter()
    paper_ids = set()
    if not isinstance(history, dict):
        return counts, paper_ids
    for paper_id, assigned in history.items():
        reviewer_ids = _reviewer_ids_from_assignment_payload(assigned)
        if reviewer_ids:
            paper_ids.add(str(paper_id))
            counts.update(reviewer_ids)
    return counts, paper_ids


def load_workload_summary():
    payload = load_optional_json(FAIRNESS_PATH, {}) or {}
    live_payload = load_optional_json(LIVE_WORKLOAD_PATH, {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    if not isinstance(live_payload, dict):
        live_payload = {}
    counts = payload.get("assignment_counts") or {}
    if not isinstance(counts, dict):
        counts = {}
    live_counts = live_payload.get("assignment_counts") or {}
    if not isinstance(live_counts, dict):
        live_counts = {}
    assignment_counts, assignment_paper_ids = load_assignment_history_counts()
    if assignment_counts:
        merged_counter = Counter({str(key): int(value or 0) for key, value in assignment_counts.items()})
    else:
        merged_counter = Counter({str(key): int(value or 0) for key, value in counts.items()})
    live_by_paper = live_payload.get("assignments_by_paper") or {}
    if isinstance(live_by_paper, dict):
        for paper_id, reviewer_ids in live_by_paper.items():
            if str(paper_id) in assignment_paper_ids:
                continue
            merged_counter.update(str(rid) for rid in (reviewer_ids or []) if rid)
    elif not assignment_counts:
        merged_counter.update({str(key): int(value or 0) for key, value in live_counts.items()})
    merged_counts = dict(merged_counter)
    if not payload and not counts and not assignment_counts:
        if not merged_counts:
            return None
    loads = [int(merged_counts.get(str(reviewer.get("reviewer_id")), 0) or 0) for reviewer in reviewers]
    total_reviewers = len(loads) or int(payload.get("total_reviewers", 0) or 0)
    assigned_reviewers = sum(1 for load in loads if load > 0)
    total_assignments = sum(loads)
    avg_load = (total_assignments / total_reviewers) if total_reviewers else 0.0
    variance = (sum((load - avg_load) ** 2 for load in loads) / total_reviewers) if total_reviewers else 0.0
    return {
        "total_reviewers": total_reviewers,
        "assigned_reviewers": assigned_reviewers,
        "total_assignments": total_assignments,
        "avg_load": avg_load,
        "variance": variance,
        "std_load": variance ** 0.5,
        "min_load": min(loads) if loads else 0,
        "max_load": max(loads) if loads else 0,
        "gini": _gini(loads),
        "assignment_counts": merged_counts,
    }


def _gini(values):
    values = sorted(float(value or 0.0) for value in values)
    if not values or sum(values) <= 0:
        return 0.0
    n = len(values)
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2 * weighted) / (n * sum(values)) - (n + 1) / n


def load_live_workload_counts():
    summary = load_workload_summary() or {}
    return summary.get("assignment_counts", {})


def update_live_workload(paper_id, reviewer_ids):
    payload = load_optional_json(LIVE_WORKLOAD_PATH, {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    by_paper = payload.get("assignments_by_paper") or {}
    if not isinstance(by_paper, dict):
        by_paper = {}
    by_paper[str(paper_id)] = [str(rid) for rid in reviewer_ids]
    counts = Counter()
    for ids in by_paper.values():
        for rid in ids or []:
            counts[str(rid)] += 1
    payload["assignments_by_paper"] = by_paper
    payload["assignment_counts"] = dict(counts)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LIVE_WORKLOAD_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def reviewer_workload_info(reviewer_id, summary=None):
    summary = summary or load_workload_summary() or {}
    counts = summary.get("assignment_counts") or {}
    current = int(counts.get(str(reviewer_id), 0) or 0)
    avg_load = float(summary.get("avg_load", 0.0) or 0.0)
    std_load = float(summary.get("std_load", 0.0) or 0.0)
    high_threshold = avg_load + max(std_load, 1.0)

    if current <= 0:
        status = "Available"
    elif current <= avg_load:
        status = "Light Load"
    elif current <= high_threshold:
        status = "Moderate Load"
    else:
        status = "High Load"

    return {
        "current_assignments": current,
        "capacity_status": status,
        "avg_load": avg_load,
        "max_load": int(summary.get("max_load", 0) or 0),
    }


def _utilization_label(load, capacity):
    capacity = max(1, int(capacity or 1))
    load = int(load or 0)
    pct = (load / capacity) * 100
    if load <= 0:
        return "Unused", "low"
    if load >= capacity:
        return "At Capacity", "critical"
    if pct < 40:
        return "Low", "low"
    if pct <= 70:
        return "Moderate", "moderate"
    if pct <= 90:
        return "High", "high"
    return "Near Capacity", "critical"


def _topic_cluster(reviewer):
    area = reviewer.get("area") or reviewer.get("areas") or reviewer.get("expertise") or reviewer.get("topics")
    if area:
        if isinstance(area, list):
            return ", ".join(str(item) for item in area[:2])
        return str(area)[:80]

    publications = reviewer.get("publications") or []
    if publications:
        title = str(publications[0].get("title", "") or "")
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", title)
        return " ".join(words[:4]) if words else "Publication profile"
    return "General profile"


def build_workload_registry(summary=None):
    summary = summary or load_workload_summary() or {}
    counts = summary.get("assignment_counts") or {}
    stats = compute_stats(reviewers)
    rows = []

    for reviewer in reviewers:
        rid = str(reviewer.get("reviewer_id", ""))
        current_load = int(counts.get(rid, 0) or 0)
        row = dict(reviewer)
        authority = compute_authority(row, stats)
        row["authority_score"] = authority
        row["role"] = classify_reviewer(row, stats)
        capacity_info = reviewer_capacity_status(row, current_load, stats)
        capacity = max(int(capacity_info["capacity"]), current_load)
        utilization = round((current_load / max(1, capacity)) * 100, 1)
        status, status_level = _utilization_label(current_load, capacity)
        rows.append({
            "reviewer_id": rid,
            "current_load": current_load,
            "assigned_papers": current_load,
            "capacity": capacity,
            "utilization": utilization,
            "utilization_text": f"{current_load}/{capacity} ({utilization:.1f}%)",
            "status": status,
            "status_level": status_level,
            "assigned": current_load > 0,
            "authority_score": round(float(authority or 0.0), 3),
            "topic_cluster": _topic_cluster(row),
        })

    rows.sort(key=lambda item: (-item["current_load"], item["reviewer_id"]))
    return rows


def normalize_ground_truth(rows):
    if isinstance(rows, list):
        return {
            str(row.get("paper_id")): [str(rid) for rid in row.get("assigned_reviewers", [])]
            for row in rows
            if row.get("paper_id") is not None
        }
    if isinstance(rows, dict):
        normalized = {}
        for paper_id, reviewer_ids in rows.items():
            values = reviewer_ids.get("assigned_reviewers", []) if isinstance(reviewer_ids, dict) else reviewer_ids
            normalized[str(paper_id)] = [str(rid) for rid in (values or [])]
        return normalized
    return {}


GT_MAP = normalize_ground_truth(gt_raw)


def attach_ground_truth(paper):
    if not paper:
        return paper
    paper_id = str(paper.get("paper_id", ""))
    if paper_id in GT_MAP:
        paper["ground_truth_reviewer_ids"] = list(GT_MAP[paper_id])
    return paper


def paper_catalog():
    return sorted(
        manuscripts,
        key=lambda item: str(item.get("paper_id", "")).zfill(8)
    )

# Normalize authors
for m in manuscripts:
    a = m.get("authors", [])
    if isinstance(a, str):
        m["authors"] = [x.strip() for x in a.split(",")] if "," in a else [a]
    elif isinstance(a, list):
        m["authors"] = [str(x) for x in a]
    else:
        m["authors"] = []

ms_dict = {str(m["paper_id"]): m for m in manuscripts}
for manuscript in manuscripts:
    attach_ground_truth(manuscript)

# -------------------------------
# REVIEWER MAP
# -------------------------------
rev_dict = {}
for r in reviewers:
    rid = str(r.get("reviewer_id")).strip()
    rev_dict[rid] = r

# -------------------------------
# 🔥 FIX: NORMALIZATION FUNCTION
# -------------------------------
def normalize_to_dict(data):
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {str(i): v for i, v in enumerate(data)}
    return {}

# -------------------------------
# FAST RANK
# -------------------------------
def fast_rank(paper_id):
    paper_id = str(paper_id)

    if paper_id not in sim_data:
        return []

    data = sim_data[paper_id]

    if not data:
        return []

    if isinstance(data[0], dict):
        return sorted(data, key=lambda x: x.get("score", 0), reverse=True)

    elif isinstance(data[0], (list, tuple)):
        return [
            {"reviewer_id": str(r[0]), "score": float(r[1])}
            for r in sorted(data, key=lambda x: float(x[1]), reverse=True)
        ]

    return []


def paper_cache_key(paper, reviewer_rows=None, version="ui_scored_assignment_v2"):
    reviewer_ids = []
    for reviewer in reviewer_rows or []:
        reviewer_ids.append(str(reviewer.get("reviewer_id", "")))

    payload = {
        "paper_id": str((paper or {}).get("paper_id", "")),
        "title": str((paper or {}).get("title", "")),
        "abstract": str((paper or {}).get("abstract", "")),
        "reviewer_ids": reviewer_ids,
        "version": version,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def cached_fast_rank(paper):
    paper_id = str((paper or {}).get("paper_id", ""))
    key = paper_cache_key(paper)
    if key in _RANK_CACHE:
        return copy.deepcopy(_RANK_CACHE[key])

    ranked = fast_rank(paper_id) if paper_id else []
    if not ranked:
        from similarity.single_paper_similarity import rank_reviewers_for_paper
        ranked = rank_reviewers_for_paper(paper, reviewers, sim_data)

    _RANK_CACHE[key] = copy.deepcopy(ranked)
    return ranked

# -------------------------------
# ENRICH REVIEWERS
# -------------------------------
def build_top_k_reviewers(ranked, k=20):
    enriched = []

    for r in ranked[:k]:
        rid = str(r["reviewer_id"]).strip()
        base = copy.deepcopy(rev_dict.get(rid))

        if not base:
            continue

        base["similarity_score"] = r.get("score", 0)
        base["semantic_score"] = base["similarity_score"]
        base["_prefetched_similarity"] = True

        base["h_index"] = (
            base.get("h_index") or
            base.get("hIndex") or
            base.get("h_index_proxy") or
            0
        )

        base["citations"] = (
            base.get("citations") or
            base.get("numCitations") or
            base.get("total_citations") or
            0
        )

        base["behavior_score"] = base.get("behavior_score", 0.3)
        base["publications"] = base.get("publications", [])

        enriched.append(base)

    return enriched

# -------------------------------
# HELPERS
# -------------------------------
def convert_to_role_dict(selected):
    return {f"Reviewer {idx}": str(r["reviewer_id"]) for idx, r in enumerate(selected, start=1)}


def reviewer_policy_label(role):
    role = str(role or "").lower()
    if role == "expert":
        return "Assignment Role: Primary Reviewer"
    if role == "moderate":
        return "Assignment Role: Secondary Reviewer"
    return "Assignment Role: Supporting Reviewer"


def score_display(value, low_label=None, calibrated=False):
    raw_value = float(value or 0)
    value = max(0.0, min(1.0, raw_value))
    if calibrated:
        value = 0.20 + 0.74 * value
    pct = round(value * 100, 1)
    bar = max(1.0, min(100.0, pct))
    if low_label and value <= 0:
        return {"pct": pct, "label": low_label, "bar": bar}
    return {"pct": pct, "label": f"{pct}%", "bar": pct}


def why_selected_label(role, details):
    role = str(role).lower()
    sim = float(details.get("similarity_normalized", details.get("similarity", 0)) or 0)
    auth = float(details.get("authority", 0) or 0)
    kg = float(details.get("kg", 0) or 0)
    if role == "expert":
        if sim >= 0.70 and auth >= sim:
            return "Strong authority reinforced by high topic relevance"
        return "High-authority reviewer with strong semantic alignment" if auth >= sim else "Authority profile with strong topic fit"
    if role == "moderate":
        if float(details.get("coverage_gain", 0) or 0) > 0.05:
            return "Selected for semantic strength with marginal topic coverage"
        return "Selected for semantic reinforcement under panel constraints"
    return "Selected for complementary topical coverage under reliability constraints"


def assignment_status_label(role, details):
    reasons = details.get("rejection_reasons") or []
    if reasons:
        return "Selection Status: Accepted with reliability caveat"
    return "Selection Status: Accepted"


def add_role_priority_notes(cards):
    expert_card = next((card for card in cards if str(card.get("role", "")).lower() == "expert"), None)
    moderate_card = next((card for card in cards if str(card.get("role", "")).lower() == "moderate"), None)
    if not expert_card or not moderate_card:
        return cards

    expert_final = float(expert_card["details"].get("final", 0) or 0)
    moderate_final = float(moderate_card["details"].get("final", 0) or 0)
    expert_auth = float(expert_card["details"].get("authority", 0) or 0)
    moderate_auth = float(moderate_card["details"].get("authority", 0) or 0)

    if moderate_final > expert_final and expert_auth >= moderate_auth:
        expert_card["details"]["selection_override_note"] = (
            "Positioned earlier due to authority prioritization despite lower composite score."
        )
        moderate_card["details"]["selection_override_note"] = (
            "Higher composite score but positioned later due to lower authority."
        )
    return cards


def consistency_badges(details):
    badges = []
    sim = float(details.get("similarity_normalized", details.get("similarity", 0)) or 0)
    auth = float(details.get("authority", 0) or 0)
    kg = float(details.get("kg", 0) or 0)
    if auth >= sim and auth >= kg:
        badges.append("Authority-led")
    if sim >= auth and sim >= kg:
        badges.append("Similarity-led")
    return badges or ["Similarity-led"]


def _quality_label(value, high=0.70, medium=0.45):
    value = float(value or 0.0)
    if value >= high:
        return "High"
    if value >= medium:
        return "Stable"
    return "Limited"


def _alignment_label(value):
    value = float(value or 0.0)
    if value >= 0.25:
        return "Strong Match"
    if value >= 0.06:
        return "Moderate Match"
    return "Limited Match"


def _panel_focus_label(redundancy):
    redundancy = float(redundancy or 0.0)
    if redundancy < 0.35:
        return "Broad Coverage"
    if redundancy < 0.65:
        return "Balanced Focus"
    return "Specialized Panel"


def build_details_map(reviewers):
    return {
        str(r["reviewer_id"]): {
            "similarity": max(0.0, min(1.0, float(r.get("similarity_score", 0) or 0))),
            "similarity_normalized": max(0.0, min(1.0, float(r.get("normalized_similarity", r.get("similarity_score", 0)) or 0))),
            "authority": max(0.0, min(1.0, float(r.get("authority_score", 0) or 0))),
            "authority_display": max(0.05, max(0.0, min(1.0, float(r.get("authority_score", 0) or 0)))),
            "authority_note": "Low authority (limited evidence)" if float(r.get("authority_score", 0) or 0) <= 0 else "",
            "final": max(0.0, min(1.0, float(r.get("final_score", 0) or 0))),
            "kg": max(0.0, min(1.0, float(r.get("kg_score", 0) or 0))),
            "kg_effective": max(0.0, min(1.0, float(r.get("score_debug", {}).get("kg_effective", 0) or 0))),
            "kg_confidence": max(0.0, min(1.0, float(r.get("kg_confidence", 0) or 0))),
            "llm_match": max(0.0, min(1.0, float(r.get("llm_match_score", 0) or 0))),
            "llm_profile_overlap": r.get("llm_profile_overlap", []),
            "manuscript_profile": r.get("manuscript_profile", {}),
            "reviewer_profile": r.get("reviewer_profile", {}),
            "workload_score": max(0.0, min(1.0, float(r.get("score_debug", {}).get("workload_score", r.get("availability_score", 1.0)) or 0))),
            "role_bias": max(0.0, min(1.0, float(r.get("role_bias", 0) or 0))),
            "base_final": max(0.0, min(1.0, float(r.get("base_final_score", 0) or 0))),
            "initial_rank": r.get("initial_semantic_rank"),
            "final_rank": r.get("final_rank"),
            "rank_margin": max(0.0, min(1.0, float(r.get("rank_margin", 0) or 0))),
            "rank_percentile": max(0.0, min(1.0, float(r.get("rank_percentile", r.get("rank_margin", 0)) or 0))),
            "constraint_label": "Panel constraint checked",
            "constraint_satisfied": bool(r.get("constraint_satisfied", False)),
            "diversity_gain": max(0.0, min(1.0, float(r.get("diversity_gain", 0) or 0))),
            "diversity_penalty": max(0.0, min(1.0, float(r.get("diversity_penalty", 0) or 0))),
            "coverage_gain": max(0.0, min(1.0, float(r.get("coverage_gain", 0) or 0))),
            "relevance_confidence": max(0.0, min(1.0, float(r.get("relevance_confidence", 0) or 0))),
            "relevance_confidence_label": r.get("score_debug", {}).get("relevance_confidence_label", ""),
            "signals": r.get("routing_signals", ["semantic", "kg", "authority"]),
            "score_debug": r.get("score_debug", {}),
            "rejection_reasons": r.get("rejection_reasons", []),
            "threshold_checks": r.get("threshold_checks", []),
            "status_label": r.get("status_label", ""),
            "policy_label": reviewer_policy_label(r.get("role")),
            "reason": r.get("reason", ""),
            "weakness": r.get("weakness", ""),
            "top_papers": [p.get("title", "") for p in r.get("publications", [])[:3]]
        }
        for r in reviewers
    }


def build_panel_summary(cards):
    if not cards:
        return {}
    details = [card["details"] for card in cards]
    avg_semantic = sum(float(d.get("similarity_normalized", 0) or 0) for d in details) / len(details)
    avg_kg = sum(float(d.get("kg_effective", 0) or 0) for d in details) / len(details)
    avg_llm = sum(float(d.get("llm_match", 0) or 0) for d in details) / len(details)
    avg_confidence = sum(float(d.get("relevance_confidence", 0) or 0) for d in details) / len(details)
    coverage = sum(float(d.get("coverage_gain", 0) or 0) for d in details[1:]) / max(1, len(details) - 1)
    redundancy = sum(float(d.get("diversity_penalty", 0) or 0) for d in details[1:]) / max(1, len(details) - 1)
    authorities = [float(d.get("authority", 0) or 0) for d in details]
    avg_authority = sum(authorities) / len(authorities)
    spread = (sum((value - avg_authority) ** 2 for value in authorities) / len(authorities)) ** 0.5
    near_capacity = sum(1 for d in details if str(d.get("capacity_status", "")).lower() in ("near capacity", "at capacity"))

    return {
        "average_semantic": score_display(avg_semantic),
        "kg_contribution": score_display(avg_kg),
        "expertise_alignment": _alignment_label(avg_llm),
        "authority_contribution": score_display(avg_authority),
        "confidence": score_display(avg_confidence),
        "topic_coverage": _quality_label(coverage, high=0.35, medium=0.10),
        "panel_focus": _panel_focus_label(redundancy),
        "selected_for_current_paper": len(cards),
        "authority_balance": "Stable" if spread < 0.25 else "Uneven",
        "workload_balance": "Good" if near_capacity == 0 else "Watch",
    }


def build_enriched_map(reviewers):
    return {str(r["reviewer_id"]): r for r in reviewers}


def build_assignment_cards(selected, enriched):
    rev_map = build_enriched_map(enriched)
    details_map = build_details_map(selected)
    workload_summary = load_workload_summary()
    cards = []

    for index, reviewer in enumerate(selected, start=1):
        rid = str(reviewer["reviewer_id"])
        role = reviewer.get("role", "reviewer")
        details = details_map.get(rid, {})
        info = rev_map.get(rid, reviewer)
        details.update(reviewer_workload_info(rid, workload_summary))
        details["why_selected"] = why_selected_label(role, details)
        details["status_label"] = assignment_status_label(role, details)
        details["badges"] = consistency_badges(details)
        details["similarity_display"] = score_display(details.get("similarity_normalized"))
        details["authority_display_meta"] = score_display(details.get("authority"))
        details["kg_display"] = score_display(details.get("kg"))
        details["kg_effective_display"] = score_display(details.get("kg_effective"), low_label="Minimal graph contribution")
        details["llm_match_display"] = score_display(details.get("llm_match"), low_label="Minimal expertise-profile alignment")
        details["final_display"] = score_display(details.get("final"), calibrated=True)
        details["diversity_gain_display"] = round(details.get("diversity_gain", 0), 2)
        details["coverage_gain_display"] = round(details.get("coverage_gain", 0), 2)
        details["relevance_confidence_display"] = round(details.get("relevance_confidence", 0), 2)
        if index == 1:
            details["panel_contribution"] = "Initial panel anchor"
            details["diversity_gain_text"] = "Not applicable for first reviewer"
            details["coverage_gain_text"] = "Baseline topic profile"
        else:
            diversity_gain = float(details.get("diversity_gain", 0) or 0)
            coverage_gain = float(details.get("coverage_gain", 0) or 0)
            details["panel_contribution"] = (
                "Complementary coverage reviewer" if coverage_gain >= 0.20
                else "Semantic reinforcement reviewer"
            )
            details["diversity_gain_text"] = (
                f"+{diversity_gain:.2f} marginal diversity"
                if diversity_gain > 0
                else "No measurable marginal diversity"
            )
            details["coverage_gain_text"] = (
                f"+{coverage_gain:.2f} new-topic coverage"
                if coverage_gain > 0
                else "No additional topic coverage detected"
            )

        cards.append({
            "index": index,
            "role": role,
            "display_label": f"Reviewer {index}",
            "reviewer_id": rid,
            "info": info,
            "details": details,
        })

    return add_role_priority_notes(cards)


def build_ablation_results(enriched, paper):
    cache_key = paper_cache_key(paper, enriched, version="ablation_results_v2")
    if cache_key in _ABLATION_RESULT_CACHE:
        return copy.deepcopy(_ABLATION_RESULT_CACHE[cache_key])

    modes = [
        ("semantic", "Semantic only"),
        ("semantic_kg", "Semantic + KG"),
        ("full", "Full Hybrid"),
    ]
    results = {}

    for mode, label in modes:
        reviewers_copy = copy.deepcopy(enriched)
        selected = assign_reviewers(reviewers_copy, paper, scoring_mode=mode)
        results[mode] = {
            "label": label,
            "cards": build_assignment_cards(selected, reviewers_copy),
        }

    _ABLATION_RESULT_CACHE[cache_key] = copy.deepcopy(results)
    return results


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _baseline_quality(selected_ids, gt_ids):
    selected_ids = [str(rid) for rid in selected_ids[:3]]
    gt_ids = {str(rid) for rid in (gt_ids or [])}
    if not gt_ids:
        return {"precision_at_3": None, "recall_at_3": None, "mrr": None, "ndcg_at_3": None}
    hits = [rid in gt_ids for rid in selected_ids]
    reciprocal_rank = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / idx
            break
    dcg = sum((1.0 / math.log2(idx + 1)) for idx, hit in enumerate(hits, start=1) if hit)
    ideal_hits = min(len(gt_ids), len(selected_ids), 3)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return {
        "precision_at_3": round(sum(hits) / max(1, min(3, len(selected_ids))), 4),
        "recall_at_3": round(sum(hits) / len(gt_ids), 4),
        "mrr": round(reciprocal_rank, 4),
        "ndcg_at_3": round(dcg / idcg, 4) if idcg else 0.0,
    }


def build_baseline_results(enriched, paper):
    cache_key = paper_cache_key(paper, enriched, version="baseline_results_v2")
    if cache_key in _BASELINE_RESULT_CACHE:
        return copy.deepcopy(_BASELINE_RESULT_CACHE[cache_key])

    gt_ids = paper.get("ground_truth_reviewer_ids") or paper.get("reviewer_ids") or []
    rows = []

    semantic_ranked = sorted(
        enriched,
        key=lambda r: float(r.get("similarity_score", r.get("semantic_score", 0.0)) or 0.0),
        reverse=True,
    )[:3]
    semantic_ids = [str(r.get("reviewer_id")) for r in semantic_ranked]
    rows.append({
        "has_ground_truth": bool(gt_ids),
        "method": "Semantic cosine",
        "description": "Embedding similarity baseline",
        "reviewer_ids": semantic_ids,
        **_baseline_quality(semantic_ids, gt_ids),
    })

    paper_tokens = _tokens(build_paper_text(paper))
    query_counts = Counter(paper_tokens)
    docs = []
    doc_freq = Counter()
    for reviewer in enriched:
        tokens = _tokens(build_reviewer_text(reviewer))
        counts = Counter(tokens)
        docs.append((reviewer, counts, len(tokens)))
        doc_freq.update(set(tokens))
    avgdl = (sum(length for _reviewer, _counts, length in docs) / len(docs)) if docs else 1.0
    total_docs = max(1, len(docs))

    def bm25_score(counts, doc_len):
        score = 0.0
        for term, qf in query_counts.items():
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + 1.5 * (1.0 - 0.75 + 0.75 * (doc_len / max(avgdl, 1e-9)))
            score += idf * ((tf * 2.5) / max(denom, 1e-9)) * max(1, qf)
        return score

    bm25_ranked = sorted(docs, key=lambda row: bm25_score(row[1], row[2]), reverse=True)[:3]
    bm25_ids = [str(row[0].get("reviewer_id")) for row in bm25_ranked]
    rows.append({
        "has_ground_truth": bool(gt_ids),
        "method": "BM25 lexical",
        "description": "Lexical retrieval baseline over reviewer profiles",
        "reviewer_ids": bm25_ids,
        **_baseline_quality(bm25_ids, gt_ids),
    })

    rng = random.Random(42)
    random_pool = list(enriched)
    rng.shuffle(random_pool)
    random_ids = [str(r.get("reviewer_id")) for r in random_pool[:3]]
    rows.append({
        "has_ground_truth": bool(gt_ids),
        "method": "Random",
        "description": "Deterministic random baseline",
        "reviewer_ids": random_ids,
        **_baseline_quality(random_ids, gt_ids),
    })

    heuristic_ranked = sorted(
        enriched,
        key=lambda r: (
            float(r.get("h_index", r.get("hIndex", r.get("h_index_proxy", 0))) or 0.0),
            float(r.get("citations", r.get("total_citations", 0)) or 0.0),
        ),
        reverse=True,
    )[:3]
    heuristic_ids = [str(r.get("reviewer_id")) for r in heuristic_ranked]
    rows.append({
        "has_ground_truth": bool(gt_ids),
        "method": "Authority heuristic",
        "description": "Citation/h-index prioritization baseline",
        "reviewer_ids": heuristic_ids,
        **_baseline_quality(heuristic_ids, gt_ids),
    })
    _BASELINE_RESULT_CACHE[cache_key] = copy.deepcopy(rows)
    return rows


def build_proposed_validation_row(full_cards, paper):
    reviewer_ids = [str(card.get("reviewer_id")) for card in (full_cards or [])[:3]]
    gt_ids = paper.get("ground_truth_reviewer_ids") or paper.get("reviewer_ids") or []
    return {
        "has_ground_truth": bool(gt_ids),
        "method": "Proposed hybrid",
        "description": "Hybrid panel optimization with semantic, authority, KG, workload, and complementary coverage scoring",
        "reviewer_ids": reviewer_ids,
        **_baseline_quality(reviewer_ids, gt_ids),
    }


def save_assignment_artifacts(paper, ablation_results, baseline_results=None):
    if not paper:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paper_id = str(paper.get("paper_id", paper.get("title", "uploaded")))

    assignments_payload = load_json(ASSIGN_PATH) or {}
    if not isinstance(assignments_payload, dict):
        assignments_payload = {}
    assignments_payload[paper_id] = {
        mode: [
            {
                "rank": card["index"],
                "position": card["display_label"],
                "reviewer_id": card["reviewer_id"],
                "final_score": card["details"]["final_display"]["pct"],
                "initial_semantic_rank": card["details"].get("initial_rank"),
                "final_rank": card["details"].get("final_rank"),
                "constraint": card["details"].get("constraint_label"),
                "diversity_gain": card["details"].get("diversity_gain_display"),
                "coverage_gain": card["details"].get("coverage_gain_display"),
                "relevance_confidence": card["details"].get("relevance_confidence_display"),
                "current_assignments": card["details"].get("current_assignments"),
                "capacity_status": card["details"].get("capacity_status"),
            }
            for card in result["cards"]
        ]
        for mode, result in ablation_results.items()
    }

    metrics_payload = load_json(METRICS_PATH) or {}
    if not isinstance(metrics_payload, dict):
        metrics_payload = {}
    metrics_payload.setdefault("reproducibility", {})
    metrics_payload["reproducibility"].update({
        "random_seed": 42,
        "numpy_seed": 42,
        "reviewer_embeddings_cached": True,
        "kg_embeddings_cached": True,
        "assignments_cached": True,
        "normalization": "Semantic scores are min-max scaled per paper; mode switches reuse cached semantic signals",
    })
    metrics_payload["last_ablation"] = {
        mode: [card["reviewer_id"] for card in result["cards"]]
        for mode, result in ablation_results.items()
    }
    gt_ids = paper.get("ground_truth_reviewer_ids") or paper.get("reviewer_ids") or []
    gt_ids = {str(rid) for rid in gt_ids}

    def _mode_quality(cards):
        if not cards:
            return {}
        selected_ids = [str(card["reviewer_id"]) for card in cards[:3]]
        if gt_ids:
            precision_at_3 = sum(1 for rid in selected_ids if rid in gt_ids) / min(3, len(selected_ids))
            recall_at_3 = sum(1 for rid in selected_ids if rid in gt_ids) / len(gt_ids)
            reciprocal_rank = 0.0
            for idx, rid in enumerate(selected_ids, start=1):
                if rid in gt_ids:
                    reciprocal_rank = 1.0 / idx
                    break
            hits = [rid in gt_ids for rid in selected_ids]
            dcg = sum((1.0 / math.log2(idx + 1)) for idx, hit in enumerate(hits, start=1) if hit)
            ideal_hits = min(len(gt_ids), len(selected_ids), 3)
            idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
            ndcg_at_3 = dcg / idcg if idcg else 0.0
        else:
            precision_at_3 = None
            recall_at_3 = None
            reciprocal_rank = None
            ndcg_at_3 = None
        coverage = sum(float(card["details"].get("coverage_gain", 0) or 0) for card in cards) / len(cards)
        relevance = sum(float(card["details"].get("relevance_confidence", 0) or 0) for card in cards) / len(cards)
        authorities = [float(card["details"].get("authority", 0) or 0) for card in cards]
        avg_authority = sum(authorities) / len(authorities)
        variance = sum((value - avg_authority) ** 2 for value in authorities) / len(authorities)
        authority_balance = max(0.0, 1.0 - (variance ** 0.5))
        return {
            "precision_at_3": None if precision_at_3 is None else round(precision_at_3, 4),
            "recall_at_3": None if recall_at_3 is None else round(recall_at_3, 4),
            "mrr": None if reciprocal_rank is None else round(reciprocal_rank, 4),
            "ndcg_at_3": None if ndcg_at_3 is None else round(ndcg_at_3, 4),
            "mean_relevance_confidence": round(relevance, 4),
            "topic_coverage": round(coverage, 4),
            "authority_balance": round(authority_balance, 4),
        }

    metrics_payload["last_assignment_quality"] = {
        mode: _mode_quality(result["cards"])
        for mode, result in ablation_results.items()
    }
    metrics_payload["last_ablation_study"] = {
        "semantic_only": metrics_payload["last_assignment_quality"].get("semantic", {}),
        "semantic_plus_kg": metrics_payload["last_assignment_quality"].get("semantic_kg", {}),
        "semantic_authority_kg": metrics_payload["last_assignment_quality"].get("full", {}),
    }
    metrics_payload["last_baseline_comparison"] = baseline_results or []

    with open(ASSIGN_PATH, "w", encoding="utf-8") as f:
        json.dump(assignments_payload, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)


def render_assignment(paper, enriched):
    start = time.time()
    if not all(row.get("_prefetched_similarity") for row in enriched):
        enriched = prepare_reviewers_for_ranking(enriched)
    ablation_results = build_ablation_results(enriched, paper)
    full_cards = ablation_results["full"]["cards"]
    assigned_reviewer_ids = [str(card["reviewer_id"]) for card in full_cards]
    update_live_workload(paper.get("paper_id", paper.get("title", "uploaded")), assigned_reviewer_ids)
    updated_workload_summary = load_workload_summary()
    for card in full_cards:
        card["details"].update(
            reviewer_workload_info(card["reviewer_id"], updated_workload_summary)
        )
    baseline_results = [
        build_proposed_validation_row(full_cards, paper),
        *build_baseline_results(enriched, paper),
    ]
    save_assignment_artifacts(paper, ablation_results, baseline_results)

    response = render_template(
        "upload.html",
        paper=paper,
        reviewers={card["display_label"]: card["reviewer_id"] for card in full_cards},
        rev_map=build_enriched_map(enriched),
        details_map={card["reviewer_id"]: card["details"] for card in full_cards},
        ablation_results=ablation_results,
        baseline_results=baseline_results,
        panel_summary=build_panel_summary(full_cards),
        workload_summary=updated_workload_summary,
        workload_registry=build_workload_registry(updated_workload_summary),
        papers=paper_catalog(),
        active_mode="full",
    )
    print("render_assignment seconds:", round(time.time() - start, 3))
    return response


def assign_and_render_paper(paper, timing_label, start_time):
    ranked = cached_fast_rank(paper)
    enriched = build_top_k_reviewers(ranked)

    if not enriched:
        enriched = copy.deepcopy(reviewers)

    response = render_assignment(paper, enriched)
    print(f"{timing_label} seconds:", round(time.time() - start_time, 3))
    return response

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", papers=paper_catalog())


@app.route("/upload", methods=["GET", "POST"])
def upload():
    start = time.time()

    pid = request.args.get("paper_id")

    if request.method == "GET" and pid:

        paper = attach_ground_truth(ms_dict.get(str(pid)))
        if not paper:
            return "Paper not found", 404

        print("FAST ranking")

        return assign_and_render_paper(paper, "upload GET", start)

    if request.method == "POST":

        file = request.files.get("file")
        if not file:
            return "No file uploaded"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        paper = attach_ground_truth(extract_pdf_metadata(filepath))

        return assign_and_render_paper(paper, "upload POST", start)

    return render_template("upload.html", papers=paper_catalog())


@app.route("/dashboard")
def dashboard():
    gt_fixed = GT_MAP

    # Ensure sim keys are strings (preserve structure of values)
    if isinstance(sim_data, dict):
        sim_fixed = {str(k): v for k, v in sim_data.items()}
    elif isinstance(sim_data, list):
        sim_fixed = {str(i): v for i, v in enumerate(sim_data)}
    else:
        sim_fixed = {}

    total_reviewers = len(reviewers)
    metrics = saved_metrics or {
        "mrr": round(compute_mrr(sim_fixed, gt_fixed), 4),
        "precision_at_5": round(compute_precision_at_k(sim_fixed, gt_fixed, 5), 4),
        "recall_at_5": round(compute_recall_at_k(sim_fixed, gt_fixed, 5), 4),
        "coverage": round(compute_coverage(sim_fixed, total_reviewers), 4),
        "diversity": round(compute_diversity(sim_fixed), 4),
    }

    return render_template(
        "dashboard.html",
        metrics={
            "mrr": round(metrics.get("mrr", 0.0), 4),
            "precision": round(metrics.get("precision_at_5", metrics.get("precision", 0.0)), 4),
            "recall": round(metrics.get("recall_at_5", metrics.get("recall", 0.0)), 4),
            "coverage": round(metrics.get("coverage", 0.0), 4),
            "diversity": round(metrics.get("diversity", 0.0), 4),
            "load_balance": round(compute_load_balance(assignments), 4)
        },
        stats={
            "total_papers": len(manuscripts),
            "total_reviewers": total_reviewers,
            "total_assignments": len(assignments)
        },
        workload_summary=load_workload_summary(),
        workload_registry=build_workload_registry(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
