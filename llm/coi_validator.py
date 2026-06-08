import os

from llm.profile_utils import OUTPUT_DIR, extract_json_object, load_json, optional_llm_json, save_json


COI_PATH = os.path.join(OUTPUT_DIR, "coi_validations.json")


def _key(paper, reviewer):
    return f"{str((paper or {}).get('paper_id', 'uploaded'))}:{str((reviewer or {}).get('reviewer_id', ''))}"


def validate_coi(paper, reviewer, base_status=None):
    base_status = base_status or {}
    validations = load_json(COI_PATH, {}) or {}
    key = _key(paper, reviewer)
    if key in validations:
        return validations[key]

    if base_status.get("flagged"):
        risk = "high"
        reason = "; ".join(base_status.get("reasons") or ["structured COI signal"])
    else:
        risk = "low"
        reason = "no structured conflict signal detected"

    prompt = f"""
Return JSON only with keys risk and reason. risk must be low, medium, or high.
Check conflict of interest for peer review.

Paper authors: {(paper or {}).get('authors', [])}
Paper affiliation: {(paper or {}).get('affiliation', '') or (paper or {}).get('institution', '')}
Reviewer id: {(reviewer or {}).get('reviewer_id', '')}
Reviewer affiliation: {(reviewer or {}).get('affiliation', '') or (reviewer or {}).get('institution', '')}
Known structured status: {base_status}
"""
    llm_payload = optional_llm_json(prompt, models=["gemma2", "mistral"], enabled_env="ENABLE_LLM_COI_VALIDATION")
    if isinstance(llm_payload, dict):
        raw_risk = str(llm_payload.get("risk") or risk).lower().strip()
        risk = raw_risk if raw_risk in {"low", "medium", "high"} else risk
        reason = str(llm_payload.get("reason") or reason).strip()

    payload = {
        "risk": risk,
        "reason": reason,
        "source": "gemma-structured" if os.getenv("ENABLE_LLM_COI_VALIDATION", "false").lower() in ("1", "true", "yes") else "deterministic-structured",
    }
    validations[key] = payload
    save_json(COI_PATH, validations)
    return payload
