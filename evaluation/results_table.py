"""
results_table.py — Results table and summary metrics exporter

Generates a CSV and JSON table of the top-K ranked reviewers per
paper, merges assignment information and ground-truth flags, and
computes basic ranking metrics (MRR / Precision@K / Recall@K).

Outputs written to `outputs/results_table.csv`,
`outputs/results_table.json`, and `outputs/metrics_results.json`.
"""

import os
import json
import csv

from evaluation.ranking_metrics import (
	compute_mrr,
	compute_precision_at_k,
	compute_recall_at_k,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SIM_PATH = os.path.join(OUT_DIR, "similarity_combined.json")
ASSIGN_PATH = os.path.join(OUT_DIR, "assignments.json")
MANUSCRIPTS_PATH = os.path.join(DATA_DIR, "manuscripts.json")
REVIEWERS_PATH = os.path.join(DATA_DIR, "reviewers.json")
GT_PATH = os.path.join(DATA_DIR, "ground_truth.json")


def load_json(path):
	if not os.path.exists(path):
		return None
	with open(path, encoding="utf-8") as f:
		return json.load(f)


def build_rows(sim_data, assignments, ms_dict, rev_dict, gt_dict, top_k=10):
	rows = []

	for pid, ranked in sim_data.items():
		title = ms_dict.get(pid, {}).get("title", "")
		assigned = assignments.get(pid, {}) if assignments else {}
		gt_reviewers = gt_dict.get(pid, [])

		for rank, entry in enumerate(ranked[:top_k], 1):
			# entry format: [reviewer_id, score]
			try:
				rid, score = entry[0], entry[1]
			except Exception:
				# fallback for unexpected formats
				continue

			reviewer_name = rev_dict.get(rid, {}).get("name", "")

			assigned_role = ""
			for role, arid in (assigned or {}).items():
				if arid == rid:
					assigned_role = role
					break

			in_gt = rid in gt_reviewers if gt_reviewers else False

			rows.append({
				"paper_id": pid,
				"title": title,
				"rank": rank,
				"reviewer_id": rid,
				"reviewer_name": reviewer_name,
				"score": float(score),
				"assigned_role": assigned_role,
				"in_ground_truth": bool(in_gt),
			})

	return rows


def save_csv(rows, path):
	if not rows:
		print("No rows to save — CSV skipped.")
		return

	keys = [
		"paper_id",
		"title",
		"rank",
		"reviewer_id",
		"reviewer_name",
		"score",
		"assigned_role",
		"in_ground_truth",
	]

	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=keys)
		writer.writeheader()
		for r in rows:
			writer.writerow(r)

	print(f"Saved CSV → {path}")


def save_json(obj, path):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2)
	print(f"Saved JSON → {path}")


def main(top_k=10):
	sim_data = load_json(SIM_PATH) or {}
	assignments = load_json(ASSIGN_PATH) or {}
	manuscripts = load_json(MANUSCRIPTS_PATH) or []
	reviewers = load_json(REVIEWERS_PATH) or []
	gt_raw = load_json(GT_PATH) or []

	ms_dict = {str(m["paper_id"]): m for m in manuscripts}
	rev_dict = {str(r["reviewer_id"]): r for r in reviewers}
	gt_dict = {str(g["paper_id"]): [str(x) for x in g.get("assigned_reviewers", [])] for g in gt_raw}

	rows = build_rows(sim_data, assignments, ms_dict, rev_dict, gt_dict, top_k=top_k)

	csv_path = os.path.join(OUT_DIR, "results_table.csv")
	json_path = os.path.join(OUT_DIR, "results_table.json")

	save_csv(rows, csv_path)
	save_json(rows, json_path)

	# Summary metrics
	mrr = compute_mrr(sim_data, gt_dict, k=top_k)
	p_k = compute_precision_at_k(sim_data, gt_dict, k=top_k)
	r_k = compute_recall_at_k(sim_data, gt_dict, k=top_k)

	metrics = {
		"mrr": round(mrr, 4),
		f"precision@{top_k}": round(p_k, 4),
		f"recall@{top_k}": round(r_k, 4),
		"rows": len(rows),
	}

	metrics_path = os.path.join(OUT_DIR, "metrics_results.json")
	save_json(metrics, metrics_path)

	print("Results table generation complete.")
	return rows, metrics


if __name__ == "__main__":
	main()

