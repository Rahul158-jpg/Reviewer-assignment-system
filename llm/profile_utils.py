import json
import os
import re

from agents.llm_client import call_llm
from utils.keywords import clean_keywords, enrich_keywords, extract_keyphrases


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def as_list(value, limit=8):
    if isinstance(value, str):
        value = re.split(r"[,;\n]", value)
    if not isinstance(value, list):
        value = []
    out = []
    seen = set()
    for item in value:
        text = str(item).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def extract_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def keyword_profile(text, limit=10):
    terms = enrich_keywords(clean_keywords(extract_keyphrases(text or "")))
    return as_list(terms, limit=limit)


def profiling_models(defaults=None, env_key="LLM_PROFILE_MODELS"):
    configured = os.getenv(env_key, "")
    raw_models = configured.split(",") if configured else (defaults or ["phi3", "llama3.2"])
    models = []
    seen = set()
    for model in raw_models:
        name = str(model or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        models.append(name)
    return models or ["phi3", "llama3.2"]


def optional_llm_json(prompt, models, enabled_env="ENABLE_LLM_STRUCTURED_PROFILING"):
    if os.getenv(enabled_env, "false").lower() not in ("1", "true", "yes"):
        return None
    timeout = int(os.getenv("LLM_PROFILE_TIMEOUT", "45"))
    retries = int(os.getenv("LLM_PROFILE_RETRIES", "1"))
    options = {
        "num_predict": int(os.getenv("LLM_PROFILE_NUM_PREDICT", "180")),
        "num_ctx": int(os.getenv("LLM_PROFILE_NUM_CTX", "2048")),
    }
    response = call_llm(prompt, models=models, timeout=timeout, retries=retries, temperature=0.0, options=options)
    return extract_json_object(response)
