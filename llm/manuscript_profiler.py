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


PROFILE_PATH = os.path.join(OUTPUT_DIR, "manuscript_profiles.json")
_REGENERATED_THIS_RUN = set()


def _paper_id(paper):
    return str((paper or {}).get("paper_id") or (paper or {}).get("title") or "uploaded")


def _paper_text(paper):
    return " ".join(
        str(part)
        for part in [
            (paper or {}).get("title", ""),
            (paper or {}).get("abstract", ""),
            " ".join((paper or {}).get("keywords", []) or []),
        ]
        if part
    ).strip()


def deterministic_profile(paper):
    text = _paper_text(paper)
    terms = keyword_profile(text, limit=12)
    return {
        "research_domain": terms[0] if terms else "general computer science",
        "main_topics": terms[:6],
        "methods": [term for term in terms if any(key in term for key in ("model", "graph", "learning", "retrieval", "classification", "interface"))][:5],
        "required_expertise": terms[:8],
        "research_area": terms[:6],
        "source": "deterministic-keyphrase",
        "model": "fallback",
    }


def generate_manuscript_profile(paper):
    profiles = load_json(PROFILE_PATH, {}) or {}
    pid = _paper_id(paper)
    cached = profiles.get(pid)
    llm_enabled = os.getenv("ENABLE_LLM_STRUCTURED_PROFILING", "false").lower() in ("1", "true", "yes")
    force_regen = os.getenv("FORCE_LLM_PROFILE_REGEN", "false").lower() in ("1", "true", "yes")
    if cached and not force_regen:
        if not (llm_enabled and cached.get("source") == "deterministic-keyphrase"):
            upgraded = dict(cached)
            fallback_for_upgrade = deterministic_profile(paper)
            upgraded.setdefault("research_area", fallback_for_upgrade["research_area"])
            if upgraded != cached:
                profiles[pid] = upgraded
                save_json(PROFILE_PATH, profiles)
            return upgraded
    if cached and force_regen and pid in _REGENERATED_THIS_RUN:
        return cached

    fallback = deterministic_profile(paper)
    abstract = str((paper or {}).get("abstract", ""))[:1200]
    prompt = f"""
Return compact JSON only. No markdown.
Keys: research_domain, main_topics, methods, required_expertise, research_area.
Use short phrases, max 5 list items per key.

Title: {(paper or {}).get("title", "")}
Abstract: {abstract}
"""
    llm_payload = optional_llm_json(prompt, models=profiling_models(["phi3", "llama3.2"]))
    if isinstance(llm_payload, dict):
        profile = {
            "research_domain": str(llm_payload.get("research_domain") or fallback["research_domain"]).strip().lower(),
            "main_topics": as_list(llm_payload.get("main_topics"), limit=8) or fallback["main_topics"],
            "methods": as_list(llm_payload.get("methods"), limit=6) or fallback["methods"],
            "required_expertise": as_list(llm_payload.get("required_expertise"), limit=8) or fallback["required_expertise"],
            "research_area": as_list(llm_payload.get("research_area"), limit=8) or fallback["research_area"],
            "source": "llm-assisted",
            "model": ",".join(profiling_models(["phi3", "llama3.2"])),
        }
    else:
        profile = fallback

    profiles[pid] = profile
    save_json(PROFILE_PATH, profiles)
    _REGENERATED_THIS_RUN.add(pid)
    return profile
