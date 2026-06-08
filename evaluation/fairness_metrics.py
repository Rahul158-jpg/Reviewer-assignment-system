"""
fairness_metrics.py
===================
Reviewer Recommendation — Fairness and Ethics Evaluation

Metrics computed:
    1. Gini Coefficient        — workload equality across reviewers
    2. COI Violation Rate      — fraction of assignments with COI flags
    3. Tier Distribution       — Expert / Moderate / Fresher balance
    4. Workload Statistics     — min, max, avg, std of assignments
    5. Overload Rate           — fraction of reviewers exceeding max load
"""

import json
import math
from collections import defaultdict, Counter
from typing import Optional


# ─────────────────────────────────────────
# 1. GINI COEFFICIENT
# ─────────────────────────────────────────

def gini_coefficient(workloads: list[int]) -> float:
    """
    Compute Gini coefficient over reviewer workload distribution.

    Gini = 0.0  → perfect equality (every reviewer has same load)
    Gini = 1.0  → maximum inequality (one reviewer has everything)

    Args:
        workloads: list of assignment counts per reviewer

    Returns:
        float: Gini coefficient in [0.0, 1.0]
    """
    if not workloads:
        return 0.0

    arr = sorted(workloads)
    n   = len(arr)
    total = sum(arr)

    if total == 0:
        return 0.0

    cumulative = 0.0
    for i, val in enumerate(arr, 1):
        cumulative += val * (2 * i - n - 1)

    return round(cumulative / (n * total), 4)


# ─────────────────────────────────────────
# 2. COI VIOLATION RATE
# ─────────────────────────────────────────

def coi_violation_rate(
    assignments:   dict,         # { paper_id: PaperAssignment }
    coi_lookup:    dict,         # { (paper_id, reviewer_id): coi_dict }
    coi_threshold: float = 0.5,
) -> dict:
    """
    Compute COI violation rate across all assignments.

    Args:
        assignments:   output from three_tier_assignment.assign_all_papers()
        coi_lookup:    dict keyed by (paper_id, reviewer_id)
        coi_threshold: score above which a pair is a violation

    Returns:
        dict with total, violations, rate, and flagged pairs
    """
    total      = 0
    violations = 0
    flagged    = []

    for paper_id, assignment in assignments.items():
        reviewers = (
            assignment.assigned_reviewers
            if hasattr(assignment, "assigned_reviewers")
            else assignment.get("reviewers", [])
        )

        for r in reviewers:
            rid = r.reviewer_id if hasattr(r, "reviewer_id") else r.get("reviewer_id")
            total += 1

            key = (paper_id, rid)
            coi_entry = coi_lookup.get(key, {})
            coi_score = coi_entry.get("coi_score", 0.0)
            flagged_in_graph = coi_entry.get("flagged", False)

            if coi_score >= coi_threshold or flagged_in_graph:
                violations += 1
                flagged.append({
                    "paper_id":    paper_id,
                    "reviewer_id": rid,
                    "coi_score":   coi_score,
                    "flagged":     flagged_in_graph,
                })

    rate = round(violations / total, 4) if total else 0.0

    return {
        "total_assignments": total,
        "violations":        violations,
        "violation_rate":    rate,
        "violation_pct":     round(rate * 100, 2),
        "flagged_pairs":     flagged,
    }


# ─────────────────────────────────────────
# 3. TIER DISTRIBUTION
# ─────────────────────────────────────────

def tier_distribution(assignments: dict) -> dict:
    """
    Count how many reviewers were assigned per tier.

    Args:
        assignments: output from three_tier_assignment

    Returns:
        dict with counts and percentages per tier
    """
    counts  = Counter()
    fill_count = 0
    total   = 0

    for paper_id, assignment in assignments.items():
        reviewers = (
            assignment.assigned_reviewers
            if hasattr(assignment, "assigned_reviewers")
            else assignment.get("reviewers", [])
        )

        for r in reviewers:
            tier    = r.tier    if hasattr(r, "tier")    else r.get("tier", "Unknown")
            is_fill = r.is_fill if hasattr(r, "is_fill") else r.get("is_fill", False)
            counts[tier] += 1
            if is_fill:
                fill_count += 1
            total += 1

    result = {"total_assignments": total, "fill_assignments": fill_count}
    for tier in ["Expert", "Moderate", "Fresher"]:
        count = counts.get(tier, 0)
        result[tier] = {
            "count":   count,
            "percent": round(count / total * 100, 2) if total else 0.0,
        }

    return result


# ─────────────────────────────────────────
# 4. WORKLOAD STATISTICS
# ─────────────────────────────────────────

def workload_stats(workload_tracker: dict) -> dict:
    """
    Compute descriptive statistics over reviewer workloads.

    Args:
        workload_tracker: { reviewer_id: assignment_count }

    Returns:
        dict with min, max, mean, std, median, gini
    """
    if not workload_tracker:
        return {}

    loads = list(workload_tracker.values())
    n     = len(loads)
    mean  = sum(loads) / n
    var   = sum((x - mean) ** 2 for x in loads) / n
    std   = round(math.sqrt(var), 4)

    sorted_loads = sorted(loads)
    mid = n // 2
    median = (
        sorted_loads[mid]
        if n % 2 != 0
        else (sorted_loads[mid - 1] + sorted_loads[mid]) / 2
    )

    return {
        "total_reviewers":    n,
        "total_assignments":  sum(loads),
        "min_load":           min(loads),
        "max_load":           max(loads),
        "mean_load":          round(mean, 4),
        "median_load":        round(median, 4),
        "std_load":           std,
        "gini_coefficient":   gini_coefficient(loads),
    }


# ─────────────────────────────────────────
# 5. OVERLOAD RATE
# ─────────────────────────────────────────

def overload_rate(workload_tracker: dict, max_load: int = 15) -> dict:
    """
    Fraction of reviewers that exceeded the maximum load cap.

    Args:
        workload_tracker: { reviewer_id: assignment_count }
        max_load:         the cap used during assignment

    Returns:
        dict with overloaded count, total, and rate
    """
    if not workload_tracker:
        return {}

    total      = len(workload_tracker)
    overloaded = sum(1 for v in workload_tracker.values() if v >= max_load)

    return {
        "total_reviewers": total,
        "overloaded":      overloaded,
        "overload_rate":   round(overloaded / total, 4) if total else 0.0,
        "overload_pct":    round(overloaded / total * 100, 2) if total else 0.0,
        "max_load_cap":    max_load,
    }


# ─────────────────────────────────────────
# 6. FULL FAIRNESS REPORT
# ─────────────────────────────────────────

def compute_fairness_report(
    assignments:      dict,
    workload_tracker: dict,
    coi_lookup:       dict,
    max_load:         int   = 15,
    coi_threshold:    float = 0.5,
) -> dict:
    """
    Compute full fairness and ethics report.

    Args:
        assignments:      output from three_tier_assignment
        workload_tracker: { reviewer_id: count }
        coi_lookup:       { (paper_id, reviewer_id): coi_dict }
        max_load:         workload cap used during assignment
        coi_threshold:    COI flag threshold

    Returns:
        dict with all fairness metrics
    """
    coi    = coi_violation_rate(assignments, coi_lookup, coi_threshold)
    tiers  = tier_distribution(assignments)
    wstats = workload_stats(workload_tracker)
    oload  = overload_rate(workload_tracker, max_load)

    return {
        "gini_coefficient":   wstats.get("gini_coefficient", 0.0),
        "coi_violation":      coi,
        "tier_distribution":  tiers,
        "workload_stats":     wstats,
        "overload_rate":      oload,
    }


def save_fairness_report(report: dict, output_path: str = "outputs/fairness_report.json"):
    """Save fairness report to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Fairness report saved to: {output_path}")


# ─────────────────────────────────────────
# PRINT FORMATTED TABLE
# ─────────────────────────────────────────

def print_fairness_table(report: dict):
    """Print a clean formatted fairness table to terminal."""

    print()
    print("=" * 58)
    print("  Table 2: Fairness and Ethics Metrics — Reviewer Recommendation")
    print("=" * 58)

    # Gini
    gini = report.get("gini_coefficient", 0.0)
    gini_label = (
        "Excellent" if gini < 0.2 else
        "Good"      if gini < 0.4 else
        "Fair"      if gini < 0.6 else
        "Poor"
    )
    print(f"\n  Workload Fairness")
    print(f"  {'Gini Coefficient':<30} {gini:.4f}   ({gini_label})")

    # Workload stats
    ws = report.get("workload_stats", {})
    print(f"  {'Mean Load per Reviewer':<30} {ws.get('mean_load', 0):.2f}")
    print(f"  {'Max Load':<30} {ws.get('max_load', 0)}")
    print(f"  {'Min Load':<30} {ws.get('min_load', 0)}")
    print(f"  {'Std Dev':<30} {ws.get('std_load', 0):.4f}")

    # Overload
    ol = report.get("overload_rate", {})
    print(f"  {'Overload Rate':<30} {ol.get('overload_pct', 0):.2f}%")

    # COI
    coi = report.get("coi_violation", {})
    coi_pct = coi.get("violation_pct", 0.0)
    coi_label = "✅ Clean" if coi_pct == 0.0 else f"⚠️  {coi_pct}%"
    print(f"\n  Ethics")
    print(f"  {'COI Violation Rate':<30} {coi_label}")
    print(f"  {'Total Assignments Checked':<30} {coi.get('total_assignments', 0)}")
    print(f"  {'Violations Found':<30} {coi.get('violations', 0)}")

    # Tier distribution
    tiers = report.get("tier_distribution", {})
    print(f"\n  Tier Distribution")
    for tier in ["Expert", "Moderate", "Fresher"]:
        t = tiers.get(tier, {})
        count = t.get("count", 0)
        pct   = t.get("percent", 0.0)
        bar   = "█" * int(pct / 5)
        print(f"  {tier:<12} {count:>5} assignments  ({pct:.1f}%)  {bar}")

    fill = tiers.get("fill_assignments", 0)
    print(f"  {'Fallback fills':<30} {fill}")
    print("=" * 58)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    from types import SimpleNamespace

    # Simulate assignment output from three_tier_assignment
    def make_assignment(paper_id, reviewers_data):
        reviewers = [
            SimpleNamespace(
                reviewer_id=r[0], tier=r[1],
                fused_score=r[2], reliability=r[3],
                coi_risk=r[4], workload=0, is_fill=False
            )
            for r in reviewers_data
        ]
        return SimpleNamespace(
            paper_id=paper_id,
            assigned_reviewers=reviewers,
            tiers_filled=[r.tier for r in reviewers],
            tiers_missing=[],
        )

    # Simulate 5 papers assigned
    assignments = {
        "paper_001": make_assignment("paper_001", [
            ("reviewer_001", "Expert",   0.4032, 0.8940, 0.1040),
            ("reviewer_002", "Moderate", 0.3841, 0.7760, 0.0740),
            ("reviewer_003", "Fresher",  0.3592, 0.6200, 0.0320),
        ]),
        "paper_002": make_assignment("paper_002", [
            ("reviewer_004", "Expert",   0.3900, 0.8100, 0.0500),
            ("reviewer_002", "Moderate", 0.3700, 0.7760, 0.0600),
            ("reviewer_005", "Fresher",  0.3400, 0.5900, 0.0200),
        ]),
        "paper_003": make_assignment("paper_003", [
            ("reviewer_001", "Expert",   0.4100, 0.8940, 0.0900),
            ("reviewer_006", "Moderate", 0.3600, 0.7100, 0.0400),
            ("reviewer_003", "Fresher",  0.3300, 0.6200, 0.0300),
        ]),
    }

    # Simulate workload tracker
    workload_tracker = {
        "reviewer_001": 2,
        "reviewer_002": 2,
        "reviewer_003": 2,
        "reviewer_004": 1,
        "reviewer_005": 1,
        "reviewer_006": 1,
    }

    # Simulate COI lookup — one flagged pair
    coi_lookup = {
        ("paper_001", "reviewer_001"): {"coi_score": 0.1040, "flagged": False},
        ("paper_001", "reviewer_002"): {"coi_score": 0.0740, "flagged": False},
        ("paper_001", "reviewer_003"): {"coi_score": 0.0320, "flagged": False},
        ("paper_002", "reviewer_004"): {"coi_score": 0.0500, "flagged": False},
        ("paper_002", "reviewer_002"): {"coi_score": 0.0600, "flagged": False},
        ("paper_002", "reviewer_005"): {"coi_score": 0.0200, "flagged": False},
        ("paper_003", "reviewer_001"): {"coi_score": 0.0900, "flagged": False},
        ("paper_003", "reviewer_006"): {"coi_score": 0.0400, "flagged": False},
        ("paper_003", "reviewer_003"): {"coi_score": 0.0300, "flagged": False},
    }

    print("=" * 58)
    print("  Fairness Metrics — Reviewer Recommendation")
    print("=" * 58)

    report = compute_fairness_report(
        assignments=assignments,
        workload_tracker=workload_tracker,
        coi_lookup=coi_lookup,
        max_load=15,
        coi_threshold=0.5,
    )

    print_fairness_table(report)

    # Individual metric tests
    print("\n  Individual metric checks:")
    loads = list(workload_tracker.values())
    print(f"  Gini({loads}) = {gini_coefficient(loads)}")
    print(f"  Expected: ~0.0 (all reviewers have 1-2 papers — fairly equal)")