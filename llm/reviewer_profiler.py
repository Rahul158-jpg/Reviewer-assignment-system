import os

from llm.profile_utils import (
    OUTPUT_DIR,
    as_list,
    keyword_profile,
    load_json,
    optional_llm_json,
    profiling_models,
    save_json,
)


PROFILE_PATH = os.path.join(OUTPUT_DIR, "reviewer_profiles.json")
_REGENERATED_THIS_RUN = set()


def reviewer_text(reviewer, publication_limit=3):
    parts = [
        str((reviewer or {}).get("area", "") or ""),
        " ".join((reviewer or {}).get("keywords", []) or []),
    ]
    for pub in (reviewer or {}).get("publications", [])[:publication_limit]:
        parts.append(str(pub.get("title", "")))
        parts.append(str(pub.get("abstract", "")))
    return " ".join(part for part in parts if part).strip()


def deterministic_profile(reviewer):
    text = reviewer_text(reviewer)
    terms = keyword_profile(text, limit=14)
    h_index = int((reviewer or {}).get("h_index") or (reviewer or {}).get("h_index_proxy") or 0)
    seniority = "senior" if h_index >= 8 else ("mid-level" if h_index >= 4 else "junior")
    return {
        "expertise_domains": terms[:7],
        "methods": [term for term in terms if any(key in term for key in ("model", "graph", "learning", "retrieval", "classification", "semantic"))][:6],
        "review_strengths": terms[:6],
        "research_strengths": terms[:7],
        "publication_focus": terms[:8],
        "seniority_level": seniority,
        "source": "deterministic-keyphrase",
        "model": "fallback",
    }


def generate_reviewer_profile(reviewer):
    profiles = load_json(PROFILE_PATH, {}) or {}
    rid = str((reviewer or {}).get("reviewer_id", ""))
    cached = profiles.get(rid)
    llm_enabled = os.getenv("ENABLE_LLM_STRUCTURED_PROFILING", "false").lower() in ("1", "true", "yes")
    force_regen = os.getenv("FORCE_LLM_PROFILE_REGEN", "false").lower() in ("1", "true", "yes")
    if cached and not force_regen:
        if not (llm_enabled and cached.get("source") == "deterministic-keyphrase"):
            upgraded = dict(cached)
            fallback_for_upgrade = deterministic_profile(reviewer)
            upgraded.setdefault("research_strengths", fallback_for_upgrade["research_strengths"])
            upgraded.setdefault("publication_focus", fallback_for_upgrade["publication_focus"])
            if upgraded != cached:
                profiles[rid] = upgraded
                save_json(PROFILE_PATH, profiles)
            return upgraded
    if cached and force_regen and rid in _REGENERATED_THIS_RUN:
        return cached

    fallback = deterministic_profile(reviewer)
    corpus = reviewer_text(
        reviewer,
        publication_limit=int(os.getenv("LLM_REVIEWER_PROFILE_PUBLICATIONS", "3")),
    )[: int(os.getenv("LLM_REVIEWER_PROFILE_CHARS", "1200"))]
    prompt = f"""
Return compact JSON only. No markdown.
Keys: expertise_domains, methods, review_strengths, research_strengths, publication_focus, seniority_level.
Use short phrases, max 5 list items per key.

Reviewer id: {rid}
Publication corpus:
{corpus}
"""
    llm_payload = optional_llm_json(prompt, models=profiling_models(["phi3", "llama3.2"]))
    if isinstance(llm_payload, dict):
        profile = {
            "expertise_domains": as_list(llm_payload.get("expertise_domains"), limit=8) or fallback["expertise_domains"],
            "methods": as_list(llm_payload.get("methods"), limit=6) or fallback["methods"],
            "review_strengths": as_list(llm_payload.get("review_strengths"), limit=6) or fallback["review_strengths"],
            "research_strengths": as_list(llm_payload.get("research_strengths"), limit=8) or fallback["research_strengths"],
            "publication_focus": as_list(llm_payload.get("publication_focus"), limit=8) or fallback["publication_focus"],
            "seniority_level": str(llm_payload.get("seniority_level") or fallback["seniority_level"]).strip().lower(),
            "source": "llm-assisted",
            "model": ",".join(profiling_models(["phi3", "llama3.2"])),
        }
    else:
        profile = fallback

    profiles[rid] = profile
    save_json(PROFILE_PATH, profiles)
    _REGENERATED_THIS_RUN.add(rid)
    return profile
