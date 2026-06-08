# ==========================================
# RANKING METRICS (FINAL CLEAN VERSION)
# ==========================================

import json
import os

# -------------------------------
# PATHS (FIXED)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIM_PATH = os.path.join(BASE_DIR, "outputs", "similarity_combined.json")
GT_PATH  = os.path.join(BASE_DIR, "data", "ground_truth.json")


# -------------------------------
# LOAD
# -------------------------------
def load_json(path):
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# METRICS
# -------------------------------

def compute_mrr(sim_data, gt, k=5):
    def _normalize_ranked(ranked):
        if not ranked:
            return []
        first = ranked[0]
        out = []
        if isinstance(first, dict):
            for item in ranked:
                rid = str(item.get("reviewer_id") or item.get("rid") or item.get("id"))
                score = item.get("score") or item.get("final") or item.get("similarity") or 0
                out.append((rid, float(score)))
        else:
            for pair in ranked:
                try:
                    rid = str(pair[0])
                    score = pair[1]
                except Exception:
                    continue
                out.append((rid, float(score)))
        return out

    total = 0
    count = 0

    for pid, gt_reviewers in gt.items():
        pid_s = str(pid)
        if pid_s not in sim_data:
            continue

        ranked = _normalize_ranked(sim_data[pid_s])

        for i, (rid, _) in enumerate(ranked[:k], 1):
            if rid in map(str, gt_reviewers):
                total += 1 / i
                break

        count += 1

    return total / count if count else 0


def compute_precision_at_k(sim_data, gt, k=5):
    def _normalize_ranked(ranked):
        if not ranked:
            return []
        first = ranked[0]
        out = []
        if isinstance(first, dict):
            for item in ranked:
                rid = str(item.get("reviewer_id") or item.get("rid") or item.get("id"))
                score = item.get("score") or item.get("final") or item.get("similarity") or 0
                out.append((rid, float(score)))
        else:
            for pair in ranked:
                try:
                    rid = str(pair[0])
                    score = pair[1]
                except Exception:
                    continue
                out.append((rid, float(score)))
        return out

    total = 0
    count = 0

    for pid, gt_reviewers in gt.items():
        pid_s = str(pid)
        if pid_s not in sim_data:
            continue

        ranked = _normalize_ranked(sim_data[pid_s])[:k]
        preds = [rid for rid, _ in ranked]

        relevant = sum(1 for r in preds if r in map(str, gt_reviewers))

        total += relevant / k
        count += 1

    return total / count if count else 0


def compute_recall_at_k(sim_data, gt, k=5):
    def _normalize_ranked(ranked):
        if not ranked:
            return []
        first = ranked[0]
        out = []
        if isinstance(first, dict):
            for item in ranked:
                rid = str(item.get("reviewer_id") or item.get("rid") or item.get("id"))
                score = item.get("score") or item.get("final") or item.get("similarity") or 0
                out.append((rid, float(score)))
        else:
            for pair in ranked:
                try:
                    rid = str(pair[0])
                    score = pair[1]
                except Exception:
                    continue
                out.append((rid, float(score)))
        return out

    total = 0
    count = 0

    for pid, gt_reviewers in gt.items():
        pid_s = str(pid)
        if pid_s not in sim_data or not gt_reviewers:
            continue

        ranked = _normalize_ranked(sim_data[pid_s])[:k]
        preds = [rid for rid, _ in ranked]

        relevant = sum(1 for r in preds if r in map(str, gt_reviewers))

        total += relevant / len(gt_reviewers)
        count += 1

    return total / count if count else 0


def compute_map(sim_data, gt, k=5):

    total = 0
    count = 0

    def _normalize_ranked(ranked):
        if not ranked:
            return []
        first = ranked[0]
        out = []
        if isinstance(first, dict):
            for item in ranked:
                rid = str(item.get("reviewer_id") or item.get("rid") or item.get("id"))
                score = item.get("score") or item.get("final") or item.get("similarity") or 0
                out.append((rid, float(score)))
        else:
            for pair in ranked:
                try:
                    rid = str(pair[0])
                    score = pair[1]
                except Exception:
                    continue
                out.append((rid, float(score)))
        return out

    for pid, gt_reviewers in gt.items():
        pid_s = str(pid)
        if pid_s not in sim_data or not gt_reviewers:
            continue

        ranked = _normalize_ranked(sim_data[pid_s])[:k]

        hits = 0
        score = 0

        for i, (rid, _) in enumerate(ranked, 1):
            if rid in map(str, gt_reviewers):
                hits += 1
                score += hits / i

        if hits > 0:
            total += score / min(len(gt_reviewers), k)
            count += 1

    return total / count if count else 0


# -------------------------------
# MAIN
# -------------------------------
def main():

    print("\n=== RANKING METRICS ===")

    sim_data = load_json(SIM_PATH)
    gt_raw = load_json(GT_PATH)

    if sim_data is None or gt_raw is None:
        print("❌ Missing required data")
        return

    # convert GT to dict
    gt = {
        g["paper_id"]: g["assigned_reviewers"]
        for g in gt_raw
    }

    mrr = compute_mrr(sim_data, gt, k=5)
    p5  = compute_precision_at_k(sim_data, gt, k=5)
    r5  = compute_recall_at_k(sim_data, gt, k=5)
    map_score = compute_map(sim_data, gt, k=5)

    print("\nRESULTS")
    print("MRR:", round(mrr, 4))
    print("Precision@5:", round(p5, 4))
    print("Recall@5:", round(r5, 4))
    print("MAP@5:", round(map_score, 4))


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()