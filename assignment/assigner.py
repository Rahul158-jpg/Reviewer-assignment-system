import math
import os
import threading
import re
import json
import hashlib
import time
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config import (
    CROSS_UNCERTAIN_MARGIN,
    DEFAULT_REVIEWER_CAPACITY,
    ENABLE_HARD_CAPACITY_EXCLUSION,
    EMBEDDING_MODEL,
    AUTHORITY_WEIGHT,
    KG_WEIGHT,
    LLM_SUITABILITY_TOP,
    LLM_MATCH_WEIGHT,
    WORKLOAD_WEIGHT,
    JUNIOR_REVIEWER_CAPACITY,
    MID_REVIEWER_CAPACITY,
    NEAR_CAPACITY_RATIO,
    RANKING_SCORING_MODE,
    RERANK_TOP,
    ASSIGNMENT_CANDIDATE_LIMIT,
    RRF_K,
    RRF_WEIGHT,
    SECONDARY_EMBEDDING_MODEL,
    SECONDARY_EMBEDDING_WEIGHT,
    SEMANTIC_RERANK_TOP,
    SENIOR_AUTHORITY_CAPACITY,
    SENIOR_REVIEWER_CAPACITY,
    TOP_K,
    USE_CROSS,
    USE_KG,
    USE_RRF_FUSION,
)

try:
    import numpy as np
    np.random.seed(42)
except Exception:
    np = None
random.seed(42)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    nx = None
    NETWORKX_AVAILABLE = False
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    SENTS_AVAILABLE = True

    def _load_sentence_transformer(name):
        try:
            return SentenceTransformer(name, local_files_only=True)
        except Exception as local_error:
            try:
                return SentenceTransformer(name)
            except Exception:
                raise local_error

    try:
        bi_model = _load_sentence_transformer(os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL))
    except Exception as _e:
        print('Could not load configured SentenceTransformer model:', _e)
        bi_model = None
except Exception:
    SENTS_AVAILABLE = False
    bi_model = None
    cos_sim = None

_ACTIVE_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL) if bi_model is not None else None
_SECONDARY_MODEL_NAME = os.getenv("SECONDARY_EMBEDDING_MODEL", SECONDARY_EMBEDDING_MODEL)
_SECONDARY_WEIGHT = max(0.0, min(1.0, float(os.getenv("SECONDARY_EMBEDDING_WEIGHT", SECONDARY_EMBEDDING_WEIGHT))))
_PRIMARY_WEIGHT = 1.0 - _SECONDARY_WEIGHT
secondary_bi_model = None
_ACTIVE_SECONDARY_EMBEDDING_MODEL = None
profile_bi_model = None
_ACTIVE_PROFILE_EMBEDDING_MODEL = None


def _embedding_model_candidates():
    configured = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    prefer_specter2 = os.getenv("PREFER_SPECTER2", "false").lower() in ("1", "true", "yes")
    candidates = [
        configured,
        "allenai-specter",
        "sentence-transformers/allenai-specter",
        "allenai/specter2_base" if prefer_specter2 else "",
        "BAAI/bge-large-en-v1.5" if prefer_specter2 else "",
        "intfloat/e5-large-v2" if prefer_specter2 else "",
        "all-mpnet-base-v2",
        "all-MiniLM-L6-v2",
    ]
    seen = set()
    unique = []
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


if SENTS_AVAILABLE:
    for _model_name in _embedding_model_candidates():
        if _model_name == _ACTIVE_EMBEDDING_MODEL:
            break
        try:
            bi_model = _load_sentence_transformer(_model_name)
            _ACTIVE_EMBEDDING_MODEL = _model_name
            break
        except Exception:
            continue

if SENTS_AVAILABLE and _SECONDARY_MODEL_NAME:
    try:
        secondary_bi_model = _load_sentence_transformer(_SECONDARY_MODEL_NAME)
        _ACTIVE_SECONDARY_EMBEDDING_MODEL = _SECONDARY_MODEL_NAME
    except Exception as _e:
        print("Secondary embedding disabled:", _e)
        secondary_bi_model = None
        _ACTIVE_SECONDARY_EMBEDDING_MODEL = None


def _profile_embedding_model_candidates():
    configured = os.getenv("LLM_PROFILE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    candidates = [
        configured,
        "intfloat/e5-base-v2",
        "BAAI/bge-base-en-v1.5",
        _ACTIVE_EMBEDDING_MODEL,
    ]
    seen = set()
    unique = []
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


if SENTS_AVAILABLE:
    for _profile_model_name in _profile_embedding_model_candidates():
        try:
            profile_bi_model = _load_sentence_transformer(_profile_model_name)
            _ACTIVE_PROFILE_EMBEDDING_MODEL = _profile_model_name
            break
        except Exception:
            continue
    if profile_bi_model is None:
        profile_bi_model = bi_model
        _ACTIVE_PROFILE_EMBEDDING_MODEL = _ACTIVE_EMBEDDING_MODEL

try:
    from sentence_transformers import CrossEncoder
    CROSS_AVAILABLE = bool(USE_CROSS)
    cross_model = None
except Exception:
    CROSS_AVAILABLE = False
    cross_model = None


def _get_cross_model():
    global cross_model
    global CROSS_AVAILABLE

    if not USE_CROSS or not CROSS_AVAILABLE:
        return None
    if cross_model is not None:
        return cross_model

    try:
        cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
    except Exception as exc:
        print("CrossEncoder disabled:", exc)
        CROSS_AVAILABLE = False
        cross_model = None
    return cross_model

def _embedding_dimension(model=None):
    model = bi_model if model is None else model
    if model is None:
        return None

    getter = getattr(model, "get_embedding_dimension", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass

    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass

    return None

MODEL_EMBEDDING_DIM = _embedding_dimension()
SECONDARY_EMBEDDING_DIM = _embedding_dimension(secondary_bi_model)
PROFILE_EMBEDDING_DIM = _embedding_dimension(profile_bi_model)

CROSS_BATCH_SIZE = int(os.getenv("CROSS_BATCH_SIZE", "64"))
CROSS_SCORE_THRESHOLD = float(os.getenv("CROSS_SCORE_THRESHOLD", "-999.0"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
CROSS_TOP_K = int(os.getenv("CROSS_TOP_K", str(RERANK_TOP)))
CROSS_MARGIN_THRESHOLD = float(os.getenv("CROSS_UNCERTAIN_MARGIN", str(CROSS_UNCERTAIN_MARGIN)))
CROSS_ONLY_UNCERTAIN = os.getenv("CROSS_ONLY_UNCERTAIN", "false").lower() in ("1", "true", "yes")
CROSS_DEBUG_TIMING = os.getenv("CROSS_DEBUG_TIMING", "false").lower() in ("1", "true", "yes")
MCP_DEBUG = os.getenv("MCP_DEBUG", "false").lower() in ("1", "true", "yes")

SEM_THRESHOLD = float(os.getenv("SEM_THRESHOLD", "0.2"))

from agents.llm_client import generate_reviewer_reason
from llm.coi_validator import validate_coi
from llm.manuscript_profiler import generate_manuscript_profile
from llm.reviewer_profiler import generate_reviewer_profile
from utils.keywords import extract_keyphrases, GENERIC_TERMS, clean_keywords, enrich_keywords
import utils.keywords as kw

try:
    if os.getenv("USE_NODE2VEC_KG", "true" if USE_KG else "false").lower() in ("1", "true", "yes"):
        from kg.gliner_extractor import extract_entities_dict
        from kg.kg_builder import build_graph, graph_overlap_score, load_or_build_persistent_graph
        from kg.node2vec_embedder import generate_embeddings
        from similarity.kg_similarity import compute_kg_similarity as compute_node2vec_kg_similarity
        KG_NODE2VEC_AVAILABLE = True
    else:
        raise ImportError("node2vec KG disabled for fast interactive mode")
except Exception:
    KG_NODE2VEC_AVAILABLE = False
    extract_entities_dict = None
    build_graph = None
    generate_embeddings = None
    compute_node2vec_kg_similarity = None
    graph_overlap_score = None
    load_or_build_persistent_graph = None


def safe_get(r, key, default=0):
    v = r.get(key, default)
    return v if v is not None else default


def get_h_index(r):
    v = r.get("h_index")
    if v is None:
        v = r.get("h_index_proxy") or r.get("hindex") or r.get("hIndex") or 0
    try:
        return int(v)
    except Exception:
        return 0


def get_citations(r):
    v = r.get("citations")
    if v is None:
        v = r.get("total_citations") or r.get("citations_total") or r.get("numCitations") or 0
    try:
        return int(v)
    except Exception:
        return 0


def _has_metric_value(r, keys):
    for key in keys:
        value = r.get(key)
        if value is None or value == "":
            continue
        try:
            float(value)
            return True
        except Exception:
            continue
    return False


def normalize(v, min_v, max_v):
    if max_v == min_v:
        return 0.0
    return (v - min_v) / (max_v - min_v)


def log_normalize(v, min_v, max_v):
    v = math.log1p(v)
    min_v = math.log1p(min_v)
    max_v = math.log1p(max_v)
    return normalize(v, min_v, max_v)


def _to_text(value):
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    if value is None:
        return ""
    return str(value)


def build_paper_text(paper):
    if not paper:
        return ""

    return " ".join(
        part for part in [
            _to_text(paper.get("title", "")),
            _to_text(paper.get("abstract", "")),
            _to_text(paper.get("keywords", [])),
        ] if part
    ).strip()


def _representative_publications(reviewer, limit=5):
    publications = sorted(
        reviewer.get("publications", []),
        key=lambda pub: (
            float(pub.get("citations", 0) or 0),
            -float(pub.get("rank", 10**9) or 10**9),
        ),
        reverse=True,
    )
    return publications[: max(1, int(limit or 5))]


def _reviewer_top_venues(reviewer, limit=2):
    venues = []
    seen = set()
    for venue in reviewer.get("venues", []) or []:
        text = _to_text(venue).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            venues.append(text)
        if len(venues) >= limit:
            return venues

    for pub in _representative_publications(reviewer, limit=20):
        text = _to_text(pub.get("venue", "") or pub.get("conference", "")).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            venues.append(text)
        if len(venues) >= limit:
            break
    return venues


def _reviewer_keywords(reviewer, limit=10):
    raw_keywords = reviewer.get("keywords", [])
    if isinstance(raw_keywords, str):
        keywords = [item.strip() for item in re.split(r"[,;]", raw_keywords) if item.strip()]
    else:
        keywords = [_to_text(item).strip() for item in raw_keywords or [] if _to_text(item).strip()]

    if len(keywords) < limit:
        text = " ".join(
            " ".join([_to_text(pub.get("title", "")), _to_text(pub.get("abstract", ""))])
            for pub in _representative_publications(reviewer, limit=5)
        )
        for keyword in clean_keywords(extract_keyphrases(text)):
            if keyword and keyword.lower() not in {item.lower() for item in keywords}:
                keywords.append(keyword)
            if len(keywords) >= limit:
                break

    return keywords[:limit]


def build_reviewer_text(reviewer):
    area = _to_text(reviewer.get("area", "") or reviewer.get("areas", ""))
    interests = _to_text(
        reviewer.get("research_interests", "")
        or reviewer.get("interests", "")
        or reviewer.get("expertise", "")
        or reviewer.get("topics", "")
    )
    keywords = _to_text(_reviewer_keywords(reviewer, limit=20))
    venues = _to_text(_reviewer_top_venues(reviewer, limit=5) or reviewer.get("venues", []) or reviewer.get("venue", ""))
    max_representative = int(os.getenv("REVIEWER_REPRESENTATIVE_PAPERS", "12"))
    publications = _representative_publications(reviewer, limit=max_representative)

    return " ".join(
        part
        for part in [
            area,
            interests,
            keywords,
            venues,
            " ".join(
                " ".join([_to_text(pub.get("title", "")), _to_text(pub.get("abstract", ""))])
                for pub in publications
            ),
        ]
        if part
    ).strip()


def _semantic_query_text(text):
    return "Represent this research paper for reviewer matching: " + (text or "")


def _semantic_document_text(text):
    return "Represent this reviewer expertise profile: " + (text or "")


_TEXT_EMBED_CACHE = {}
_SECONDARY_TEXT_EMBED_CACHE = {}
_PROFILE_TEXT_EMBED_CACHE = {}
_REVIEWER_EMBEDDINGS = None
_REVIEWER_INDEX = None
_SECONDARY_REVIEWER_EMBEDDINGS = None
_SECONDARY_REVIEWER_INDEX = None
_REVIEWER_CACHE_INFO = {"status": "uninitialized"}
_KG_EMBEDDINGS = None
_KG_CACHE_INFO = {"status": "uninitialized"}
_KG_LOCK = threading.Lock()
_KG_RESULT_CACHE = {}
_PERSISTENT_KG = None
_PERSISTENT_KG_META = {}
_COI_LOOKUP_CACHE = None


def _cache_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _cross_cache_path():
    return os.path.join(_cache_dir(), "cross_encoder_scores.json")


_CROSS_SCORE_CACHE = None


def _load_cross_score_cache():
    global _CROSS_SCORE_CACHE
    if _CROSS_SCORE_CACHE is not None:
        return _CROSS_SCORE_CACHE

    _CROSS_SCORE_CACHE = _load_json_if_exists(_cross_cache_path())
    if not isinstance(_CROSS_SCORE_CACHE, dict):
        _CROSS_SCORE_CACHE = {}
    return _CROSS_SCORE_CACHE


def _save_cross_score_cache():
    if _CROSS_SCORE_CACHE is None:
        return
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        with open(_cross_cache_path(), "w", encoding="utf-8") as f:
            json.dump(_CROSS_SCORE_CACHE, f)
    except Exception:
        pass


def cross_cache_key(paper_id, reviewer_id, paper_text=None, reviewer_text=None):
    payload = {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "paper_id": str(paper_id or ""),
        "reviewer_id": str(reviewer_id or ""),
        "paper_text_hash": hashlib.sha256((paper_text or "").encode("utf-8")).hexdigest(),
        "reviewer_text_hash": hashlib.sha256((reviewer_text or "").encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def get_cached_cross_score(key):
    return _load_cross_score_cache().get(key)


def set_cached_cross_score(key, score):
    _load_cross_score_cache()[key] = float(score)


def _reviewer_embedding_cache_path():
    return os.path.join(_cache_dir(), "reviewer_emb.npy")


def _reviewer_embedding_meta_path():
    return os.path.join(_cache_dir(), "reviewer_emb_meta.json")


def _secondary_reviewer_embedding_cache_path():
    return os.path.join(_cache_dir(), "reviewer_emb_secondary.npy")


def _secondary_reviewer_embedding_meta_path():
    return os.path.join(_cache_dir(), "reviewer_emb_secondary_meta.json")


def _output_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _get_persistent_kg():
    global _PERSISTENT_KG
    global _PERSISTENT_KG_META

    if _PERSISTENT_KG is not None:
        return _PERSISTENT_KG
    if load_or_build_persistent_graph is None:
        _PERSISTENT_KG_META = {"status": "unavailable", "reason": "persistent KG builder unavailable"}
        return None

    try:
        _PERSISTENT_KG, _PERSISTENT_KG_META = load_or_build_persistent_graph()
        _PERSISTENT_KG_META = dict(_PERSISTENT_KG_META or {})
        _PERSISTENT_KG_META["status"] = "loaded"
    except Exception as exc:
        _PERSISTENT_KG = None
        _PERSISTENT_KG_META = {"status": "unavailable", "reason": str(exc)}
    return _PERSISTENT_KG


def _load_coi_lookup():
    global _COI_LOOKUP_CACHE

    if _COI_LOOKUP_CACHE is not None:
        return _COI_LOOKUP_CACHE
    rows = _load_json_if_exists(os.path.join(_data_dir(), "coi_graph.json"))
    lookup = {}
    if isinstance(rows, list):
        for row in rows:
            paper_id = str(row.get("paper_id", ""))
            reviewer_id = str(row.get("reviewer_id", ""))
            if paper_id and reviewer_id:
                lookup[(paper_id, reviewer_id)] = row
    _COI_LOOKUP_CACHE = lookup
    return _COI_LOOKUP_CACHE


def _norm_name(value):
    return re.sub(r"[^a-z0-9]+", " ", _to_text(value).lower()).strip()


def _coi_status(paper, reviewer):
    paper_id = str((paper or {}).get("paper_id", ""))
    reviewer_id = str((reviewer or {}).get("reviewer_id", ""))
    entry = _load_coi_lookup().get((paper_id, reviewer_id), {}) if paper_id and reviewer_id else {}
    score = float(entry.get("coi_score", 0.0) or 0.0)
    reasons = []
    if entry.get("flagged") or score >= 0.5:
        reasons.append("precomputed COI graph")

    paper_authors = {_norm_name(author) for author in (paper or {}).get("authors", []) or [] if _norm_name(author)}
    reviewer_names = {
        _norm_name((reviewer or {}).get("name", "")),
        _norm_name((reviewer or {}).get("full_name", "")),
    }
    reviewer_names = {name for name in reviewer_names if name}
    if paper_authors and reviewer_names and paper_authors & reviewer_names:
        score = max(score, 1.0)
        reasons.append("author/reviewer identity overlap")

    paper_affiliation = _norm_name((paper or {}).get("affiliation", "") or (paper or {}).get("institution", ""))
    reviewer_affiliation = _norm_name((reviewer or {}).get("affiliation", "") or (reviewer or {}).get("institution", ""))
    if paper_affiliation and reviewer_affiliation and paper_affiliation == reviewer_affiliation:
        score = max(score, 0.6)
        reasons.append("same affiliation")

    status = {
        "score": max(0.0, min(1.0, score)),
        "flagged": bool(reasons),
        "reasons": reasons,
        "source": "coi_graph+metadata",
    }
    validation = validate_coi(paper, reviewer, status)
    status["llm_validation"] = validation
    if validation.get("risk") == "high":
        status["flagged"] = True
        status["score"] = max(status["score"], 0.75)
        status["reasons"].append(f"Gemma COI validation: {validation.get('reason', 'high risk')}")
    return status


def _profile_terms(profile, keys):
    terms = set()
    for key in keys:
        value = profile.get(key, []) if isinstance(profile, dict) else []
        if isinstance(value, str):
            value = [value]
        for item in value or []:
            text = kw.normalize(_to_text(item)).lower().strip()
            if text:
                terms.add(text)
    return terms


def _profile_values(profile, keys):
    values = []
    seen = set()
    if not isinstance(profile, dict):
        return values
    for key in keys:
        value = profile.get(key, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            continue
        for item in value or []:
            text = kw.normalize(_to_text(item)).lower().strip()
            if text and text not in seen:
                seen.add(text)
                values.append(text)
    return values


def _profile_document(profile, kind="paper"):
    if not isinstance(profile, dict):
        return ""

    if kind == "paper":
        sections = [
            ("Domain", ("research_domain", "domain", "research_area")),
            ("Topics", ("main_topics", "topics", "research_area")),
            ("Methods", ("methods",)),
            ("Required Expertise", ("required_expertise", "required_skills")),
        ]
    else:
        sections = [
            ("Expertise", ("expertise_domains", "expertise", "research_strengths", "review_strengths")),
            ("Methods", ("methods",)),
            ("Publication Focus", ("publication_focus", "topics", "keywords")),
            ("Seniority", ("seniority_level",)),
        ]

    lines = []
    for label, keys in sections:
        values = _profile_values(profile, keys)
        if values:
            lines.append(f"{label}: " + "; ".join(values))
    return "\n".join(lines).strip()


def _profile_embedding_score(paper_profile, reviewer_profile):
    paper_doc = _profile_document(paper_profile, "paper")
    reviewer_doc = _profile_document(reviewer_profile, "reviewer")
    if not paper_doc or not reviewer_doc:
        return None, paper_doc, reviewer_doc

    paper_emb = _embed_profile_text(paper_doc, "query")
    reviewer_emb = _embed_profile_text(reviewer_doc, "document")
    if paper_emb is None or reviewer_emb is None:
        return None, paper_doc, reviewer_doc

    score = _cosine_from_embeddings(paper_emb, reviewer_emb)
    return max(0.0, min(1.0, float(score or 0.0))), paper_doc, reviewer_doc


def _llm_profile_match(paper_profile, reviewer_profile):
    paper_terms = _profile_terms(paper_profile, ("main_topics", "methods", "required_expertise", "research_domain"))
    reviewer_terms = _profile_terms(reviewer_profile, ("expertise_domains", "methods", "review_strengths", "seniority_level"))
    if not paper_terms or not reviewer_terms:
        return 0.0, []

    overlap = set()
    for p_term in paper_terms:
        for r_term in reviewer_terms:
            if p_term == r_term or p_term in r_term or r_term in p_term:
                overlap.add(p_term if len(p_term) <= len(r_term) else r_term)

    score = len(overlap) / max(3, min(len(paper_terms), 10))
    return max(0.0, min(1.0, score)), sorted(overlap)[:6]


def _term_overlap_score(paper_terms, reviewer_terms, denominator_floor=3, cap=1.0):
    if not paper_terms or not reviewer_terms:
        return 0.0, []

    overlap = set()
    for p_term in paper_terms:
        for r_term in reviewer_terms:
            if p_term == r_term or p_term in r_term or r_term in p_term:
                overlap.add(p_term if len(p_term) <= len(r_term) else r_term)

    denominator = max(denominator_floor, min(len(paper_terms), 10))
    score = len(overlap) / denominator
    return max(0.0, min(cap, score)), sorted(overlap)[:8]


def _soft_term_similarity(left, right):
    left_norm = kw.normalize(str(left or "")).lower().strip()
    right_norm = kw.normalize(str(right or "")).lower().strip()
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.85

    left_tokens = _expanded_profile_tokens(left_norm)
    right_tokens = _expanded_profile_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(left_tokens & right_tokens)
    if intersection == 0:
        return 0.0
    union = len(left_tokens | right_tokens)
    containment = intersection / max(1, min(len(left_tokens), len(right_tokens)))
    jaccard = intersection / max(1, union)
    return max(jaccard, 0.75 * containment)


def _expanded_profile_tokens(text):
    tokens = set(_lexical_token_set(text))
    aliases = {
        "ai": ("artificial", "intelligence"),
        "bert": ("language", "model", "transformer"),
        "cnn": ("convolutional", "neural", "network"),
        "dl": ("deep", "learning"),
        "gnn": ("graph", "neural", "network"),
        "kg": ("knowledge", "graph"),
        "kgs": ("knowledge", "graph"),
        "llm": ("large", "language", "model"),
        "llms": ("large", "language", "model"),
        "ml": ("machine", "learning"),
        "nlp": ("natural", "language", "processing"),
        "rag": ("retrieval", "augmented", "generation"),
        "rl": ("reinforcement", "learning"),
    }
    expanded = set(tokens)
    for token in tokens:
        if token in aliases:
            expanded.update(aliases[token])
        if token.endswith("ies") and len(token) > 5:
            expanded.add(token[:-3] + "y")
        elif token.endswith("es") and len(token) > 5:
            expanded.add(token[:-2])
        elif token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])
    return expanded


def _soft_profile_score(paper_terms, reviewer_terms, denominator_floor=3):
    paper_terms = sorted(set(paper_terms or []))
    reviewer_terms = sorted(set(reviewer_terms or []))
    if not paper_terms or not reviewer_terms:
        return 0.0, []

    best_scores = []
    evidence = []
    for p_term in paper_terms:
        best_score = 0.0
        best_reviewer_term = ""
        for r_term in reviewer_terms:
            score = _soft_term_similarity(p_term, r_term)
            if score > best_score:
                best_score = score
                best_reviewer_term = r_term
        best_scores.append(best_score)
        if best_score >= 0.35:
            evidence.append(p_term if len(str(p_term)) <= len(str(best_reviewer_term)) else best_reviewer_term)

    denominator = max(denominator_floor, min(len(paper_terms), 10))
    score = sum(sorted(best_scores, reverse=True)[:denominator]) / denominator
    return max(0.0, min(1.0, score)), sorted(set(evidence))[:8]


def _profile_component_score(paper_terms, reviewer_terms, denominator_floor=3):
    exact_score, exact_overlap = _term_overlap_score(
        paper_terms,
        reviewer_terms,
        denominator_floor=denominator_floor,
    )
    soft_score, soft_overlap = _soft_profile_score(
        paper_terms,
        reviewer_terms,
        denominator_floor=denominator_floor,
    )
    paper_tokens = set()
    reviewer_tokens = set()
    for term in paper_terms or []:
        paper_tokens.update(_expanded_profile_tokens(term))
    for term in reviewer_terms or []:
        reviewer_tokens.update(_expanded_profile_tokens(term))

    token_coverage = 0.0
    token_jaccard = 0.0
    if paper_tokens and reviewer_tokens:
        overlap = paper_tokens & reviewer_tokens
        token_coverage = len(overlap) / max(1, min(len(paper_tokens), len(reviewer_tokens)))
        token_jaccard = len(overlap) / max(1, len(paper_tokens | reviewer_tokens))

    continuity_floor = 0.03 if paper_terms and reviewer_terms else 0.0
    score = max(
        exact_score,
        soft_score,
        0.65 * token_coverage,
        0.35 * token_jaccard,
        continuity_floor,
    )
    evidence = sorted(set(exact_overlap + soft_overlap))[:8]
    return max(0.0, min(1.0, score)), evidence


def _llm_suitability_reasoning(paper_profile, reviewer_profile):
    """
    Structured R_LLM(p,r) signal for LLM profile-based expertise matching.

    The primary score is cosine similarity between semantic documents built
    from LLM-assisted manuscript and reviewer profiles. Overlap components are
    retained as interpretable evidence and as a fallback when profile embeddings
    are unavailable.
    """
    paper_required = _profile_terms(paper_profile, ("required_expertise", "main_topics"))
    reviewer_expertise = _profile_terms(reviewer_profile, ("expertise_domains", "review_strengths", "research_strengths"))
    paper_methods = _profile_terms(paper_profile, ("methods",))
    reviewer_methods = _profile_terms(reviewer_profile, ("methods",))
    paper_domain = _profile_terms(paper_profile, ("research_domain", "domain", "research_area", "main_topics"))
    reviewer_domain = _profile_terms(reviewer_profile, ("expertise_domains", "review_strengths", "research_strengths", "publication_focus"))
    paper_topics = _profile_terms(paper_profile, ("main_topics", "research_domain", "research_area"))
    reviewer_topics = _profile_terms(reviewer_profile, ("expertise_domains", "review_strengths", "research_strengths", "publication_focus"))

    domain_expertise, domain_overlap = _profile_component_score(paper_domain, reviewer_domain)
    methodology_expertise, methodology_overlap = _profile_component_score(
        paper_methods,
        reviewer_methods,
        denominator_floor=2,
    )
    publication_relevance, publication_overlap = _profile_component_score(paper_required, reviewer_expertise)
    topic_overlap, topic_evidence = _profile_component_score(paper_topics, reviewer_topics)

    if not paper_methods or not reviewer_methods:
        methodology_expertise = min(methodology_expertise or topic_overlap, topic_overlap)

    overlap_suitability = (
        0.30 * domain_expertise
        + 0.30 * methodology_expertise
        + 0.20 * publication_relevance
        + 0.20 * topic_overlap
    )
    source_bonus = 0.10 if (
        str((paper_profile or {}).get("source", "")).startswith("llm")
        or str((reviewer_profile or {}).get("source", "")).startswith("llm")
    ) else 0.0
    overlap_terms = sorted(set(
        domain_overlap
        + methodology_overlap
        + publication_overlap
        + topic_evidence
    ))[:10]
    evidence_count = len(overlap_terms)
    confidence = max(0.35, min(0.95, 0.45 + 0.07 * evidence_count + source_bonus))
    overlap_suitability = overlap_suitability * (0.85 + 0.15 * confidence)
    profile_embedding_score, paper_profile_document, reviewer_profile_document = _profile_embedding_score(
        paper_profile,
        reviewer_profile,
    )
    final_suitability = (
        profile_embedding_score
        if profile_embedding_score is not None
        else overlap_suitability
    )

    return {
        "domain_expertise": max(0.0, min(1.0, domain_expertise)),
        "methodology_expertise": max(0.0, min(1.0, methodology_expertise)),
        "publication_relevance": max(0.0, min(1.0, publication_relevance)),
        "topic_overlap": max(0.0, min(1.0, topic_overlap)),
        "expertise_match": max(0.0, min(1.0, domain_expertise)),
        "methodology_match": max(0.0, min(1.0, methodology_expertise)),
        "domain_match": max(0.0, min(1.0, topic_overlap)),
        "publication_alignment": max(0.0, min(1.0, publication_relevance)),
        "confidence": max(0.0, min(1.0, confidence)),
        "final_suitability": max(0.0, min(1.0, final_suitability)),
        "profile_embedding_score": profile_embedding_score,
        "profile_embedding_model": _ACTIVE_PROFILE_EMBEDDING_MODEL,
        "overlap_suitability": max(0.0, min(1.0, overlap_suitability)),
        "paper_profile_document": paper_profile_document,
        "reviewer_profile_document": reviewer_profile_document,
        "overlap_terms": overlap_terms,
        "source": "llm-profile-embedding-v3" if profile_embedding_score is not None else "llm-profile-overlap-fallback-v2",
    }


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _labels_from_payload(value):
    if isinstance(value, dict):
        labels = value.get("labels") or value.get("all_labels")
        if not labels and isinstance(value.get("weights"), dict):
            labels = list(value["weights"].keys())
        value = labels or []
    elif isinstance(value, str):
        value = [value]

    labels = []
    seen = set()
    for item in value or []:
        label = str(item).strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _normalize_topic_map(data):
    return {
        str(entity_id): _labels_from_payload(payload)
        for entity_id, payload in (data or {}).items()
    }


def _merge_topic_maps(primary, secondary):
    merged = {str(key): list(value) for key, value in (primary or {}).items()}
    for key, values in (secondary or {}).items():
        key = str(key)
        existing = set(merged.get(key, []))
        merged.setdefault(key, [])
        for value in values:
            if value not in existing:
                merged[key].append(value)
                existing.add(value)
    return merged


def _get_kg_embeddings():
    global _KG_EMBEDDINGS
    global _KG_CACHE_INFO

    if _KG_EMBEDDINGS is not None:
        return _KG_EMBEDDINGS

    with _KG_LOCK:
        if _KG_EMBEDDINGS is not None:
            return _KG_EMBEDDINGS

        if not KG_NODE2VEC_AVAILABLE:
            _KG_EMBEDDINGS = {}
            _KG_CACHE_INFO = {"status": "unavailable", "reason": "kg/node2vec imports failed"}
            return _KG_EMBEDDINGS

        out_dir = _output_dir()
        gliner_papers = _normalize_topic_map(
            _load_json_if_exists(os.path.join(out_dir, "gliner_manuscript_labels.json"))
        )
        gliner_reviewers = _normalize_topic_map(
            _load_json_if_exists(os.path.join(out_dir, "gliner_reviewer_labels.json"))
        )
        cso_papers = _normalize_topic_map(
            _load_json_if_exists(os.path.join(out_dir, "cso_manuscript_topics.json"))
        )
        cso_reviewers = _normalize_topic_map(
            _load_json_if_exists(os.path.join(out_dir, "cso_reviewer_topics.json"))
        )

        paper_topics = _merge_topic_maps(gliner_papers, cso_papers)
        reviewer_topics = _merge_topic_maps(gliner_reviewers, cso_reviewers)
        graph = build_graph(paper_topics, reviewer_topics)
        try:
            _KG_EMBEDDINGS = generate_embeddings(graph, stable_cache=True)
        except Exception as exc:
            _KG_EMBEDDINGS = {}
            _KG_CACHE_INFO = {
                "status": "unavailable",
                "reason": str(exc),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            }
            return _KG_EMBEDDINGS

        _KG_CACHE_INFO = {
            "status": "built" if _KG_EMBEDDINGS else "empty",
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "embeddings": len(_KG_EMBEDDINGS),
        }
        return _KG_EMBEDDINGS


def _reviewer_cache_signature(reviewer_texts):
    payload = {
        "model": _ACTIVE_EMBEDDING_MODEL,
        "secondary_model": _ACTIVE_SECONDARY_EMBEDDING_MODEL,
        "secondary_weight": _SECONDARY_WEIGHT,
        "profile_version": "instruction_profile_v9_scientific_profile",
        "dim": MODEL_EMBEDDING_DIM,
        "secondary_dim": SECONDARY_EMBEDDING_DIM,
        "count": len(reviewer_texts),
        "texts": reviewer_texts,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cosine_from_embeddings(a, b):
    if a is None or b is None:
        return 0.0

    try:
        if len(a) != len(b):
            return 0.0
    except Exception:
        return 0.0

    if np is not None:
        return float(np.dot(a, b))

    denom = math.sqrt(sum(float(x) * float(x) for x in a)) * math.sqrt(sum(float(x) * float(x) for x in b))
    if not denom:
        return 0.0
    return float(sum(float(x) * float(y) for x, y in zip(a, b)) / denom)


def _is_compatible_embedding(embedding):
    if embedding is None:
        return False

    if MODEL_EMBEDDING_DIM is None:
        return True

    try:
        return len(embedding) == MODEL_EMBEDDING_DIM
    except Exception:
        return False


def _embed_text_with_model(text, model, cache, expected_dim=None):
    text = (text or "").strip()
    if not text:
        return None

    cached = cache.get(text)
    if cached is not None and (expected_dim is None or len(cached) == expected_dim):
        return cached
    if cached is not None:
        cache.pop(text, None)

    if not (SENTS_AVAILABLE and model is not None):
        return None

    try:
        emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        cache[text] = emb
        return emb
    except Exception:
        return None


def _embed_text(text):
    return _embed_text_with_model(text, bi_model, _TEXT_EMBED_CACHE, MODEL_EMBEDDING_DIM)


def _embed_paper_text(text):
    return _embed_text(_semantic_query_text(text))


def _embed_reviewer_text(text):
    return _embed_text(_semantic_document_text(text))


def _embed_secondary_text(text):
    return _embed_text_with_model(
        text,
        secondary_bi_model,
        _SECONDARY_TEXT_EMBED_CACHE,
        SECONDARY_EMBEDDING_DIM,
    )


def _profile_embedding_text(text, kind="document"):
    text = (text or "").strip()
    model_name = str(_ACTIVE_PROFILE_EMBEDDING_MODEL or "").lower()
    if "e5-" in model_name or "e5_" in model_name:
        prefix = "query: " if kind == "query" else "passage: "
        return prefix + text
    if "bge-" in model_name or "bge_" in model_name:
        prefix = "Represent this research manuscript profile for reviewer matching: " if kind == "query" else "Represent this reviewer expertise profile for manuscript matching: "
        return prefix + text
    return (_semantic_query_text(text) if kind == "query" else _semantic_document_text(text))


def _embed_profile_text(text, kind="document"):
    return _embed_text_with_model(
        _profile_embedding_text(text, kind),
        profile_bi_model,
        _PROFILE_TEXT_EMBED_CACHE,
        PROFILE_EMBEDDING_DIM,
    )


def _embed_secondary_paper_text(text):
    return _embed_text_with_model(
        _semantic_query_text(text),
        secondary_bi_model,
        _SECONDARY_TEXT_EMBED_CACHE,
        SECONDARY_EMBEDDING_DIM,
    )


def _cosine_scores_against_reviewers(paper_text, reviewer_embeddings):
    if not paper_text or reviewer_embeddings is None or np is None:
        return None

    paper_emb = _embed_paper_text(paper_text)
    if paper_emb is None:
        return None

    try:
        if cos_sim is not None:
            scores = cos_sim(paper_emb, reviewer_embeddings)[0]
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            else:
                scores = np.asarray(scores)
        else:
            scores = np.dot(reviewer_embeddings, paper_emb)
        return np.asarray(scores, dtype=float)
    except Exception:
        return None


def _keyword_set_from_text(text):
    keywords = clean_keywords(extract_keyphrases(text or ""))
    return {kw.lower() for kw in keywords if kw}


def _lexical_token_set(text):
    stop = {
        "this", "that", "with", "from", "using", "based", "their", "these",
        "those", "into", "over", "under", "between", "among", "paper",
        "study", "approach", "method", "methods", "result", "results",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", str(text or "").lower())
        if token not in stop and len(token) > 3
    }


def _lexical_overlap_score(paper_tokens, paper_keywords, reviewer):
    reviewer_tokens = reviewer.get("_reviewer_token_set") or set()
    reviewer_keywords = reviewer.get("_reviewer_keyword_set") or set()
    token_score = (
        len(paper_tokens & reviewer_tokens) / max(1, len(paper_tokens | reviewer_tokens))
        if paper_tokens and reviewer_tokens
        else 0.0
    )
    keyword_score = (
        len(paper_keywords & reviewer_keywords) / max(1, len(paper_keywords | reviewer_keywords))
        if paper_keywords and reviewer_keywords
        else 0.0
    )
    return (0.45 * token_score) + (0.55 * keyword_score)


def _sigmoid(score):
    score = max(min(float(score), 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-score))


def _smooth_score(score):
    score = max(0.0, float(score or 0.0))
    return score / (1.0 + score)


def _rank_map(items, key):
    ranked = sorted(
        items,
        key=lambda row: float(key(row) or 0.0),
        reverse=True,
    )
    return {str(row.get("reviewer_id")): rank for rank, row in enumerate(ranked, start=1)}


def _normalized_rrf_score(*ranks, k=None):
    k = max(1, int(k or RRF_K))
    usable = [int(rank) for rank in ranks if rank]
    if not usable:
        return 0.0
    raw = sum(1.0 / (k + rank) for rank in usable)
    best = len(usable) * (1.0 / (k + 1))
    return max(0.0, min(1.0, raw / max(best, 1e-12)))


def _apply_rrf_fusion(candidates, weight=None, k=None):
    if not candidates:
        return
    weight = max(0.0, min(1.0, float(RRF_WEIGHT if weight is None else weight)))
    if weight <= 0.0:
        return

    semantic_rank = _rank_map(candidates, lambda row: row.get("normalized_similarity", row.get("similarity_score", 0.0)))
    authority_rank = _rank_map(candidates, lambda row: row.get("authority_score", 0.0))
    kg_rank = _rank_map(
        candidates,
        lambda row: float(row.get("kg_score", 0.0) or 0.0) * float(row.get("kg_confidence", 0.0) or 0.0),
    )
    hybrid_rank = _rank_map(candidates, lambda row: row.get("final_score", 0.0))

    for row in candidates:
        reviewer_id = str(row.get("reviewer_id"))
        base_score = max(0.0, min(1.0, float(row.get("final_score", 0.0) or 0.0)))
        rrf_score = _normalized_rrf_score(
            semantic_rank.get(reviewer_id),
            authority_rank.get(reviewer_id),
            kg_rank.get(reviewer_id),
            hybrid_rank.get(reviewer_id),
            k=k,
        )
        row["rrf_score"] = rrf_score
        row["pre_rrf_final_score"] = base_score
        row["final_score"] = max(0.0, min(1.0, ((1.0 - weight) * base_score) + (weight * rrf_score)))
        row.setdefault("score_debug", {})
        row["score_debug"]["rrf_score"] = rrf_score
        row["score_debug"]["rrf_weight"] = weight


SCORING_WEIGHTS = {
    "similarity": max(0.0, float(os.getenv("SIM_WEIGHT", "0.40"))),
    "authority": AUTHORITY_WEIGHT,
    "diversity": 0.0,
    "kg": KG_WEIGHT,
    "llm": LLM_MATCH_WEIGHT,
    "workload": WORKLOAD_WEIGHT,
}

ROLE_SCORING_WEIGHTS = {
    "expert": {"similarity": SCORING_WEIGHTS["similarity"], "authority": AUTHORITY_WEIGHT, "kg": KG_WEIGHT, "llm": LLM_MATCH_WEIGHT, "workload": WORKLOAD_WEIGHT},
    "moderate": {"similarity": SCORING_WEIGHTS["similarity"], "authority": AUTHORITY_WEIGHT, "kg": KG_WEIGHT, "llm": LLM_MATCH_WEIGHT, "workload": WORKLOAD_WEIGHT},
    "fresher": {"similarity": SCORING_WEIGHTS["similarity"], "authority": AUTHORITY_WEIGHT, "kg": KG_WEIGHT, "llm": LLM_MATCH_WEIGHT, "workload": WORKLOAD_WEIGHT},
}

KG_TOP_K = int(os.getenv("KG_TOP_K", "12"))
SEMANTIC_REJECTION_THRESHOLD = float(os.getenv("SEMANTIC_REJECTION_THRESHOLD", "0.40"))
KG_CONFIDENCE_USE_THRESHOLD = float(os.getenv("KG_CONFIDENCE_USE_THRESHOLD", "0.12"))
PRIMARY_SIMILARITY_FLOOR = float(os.getenv("PRIMARY_SIMILARITY_FLOOR", "0.55"))
SUPPORTING_SIMILARITY_FLOOR = float(os.getenv("SUPPORTING_SIMILARITY_FLOOR", "0.45"))
SUPPORTING_AUTHORITY_FLOOR = float(os.getenv("SUPPORTING_AUTHORITY_FLOOR", "0.30"))
PANEL_MIN_SIMILARITY = float(os.getenv("PANEL_MIN_SIMILARITY", "0.50"))
PANEL_MIN_FINAL_SCORE = float(os.getenv("PANEL_MIN_FINAL_SCORE", "0.25"))
USE_NODE2VEC_KG = os.getenv("USE_NODE2VEC_KG", "true" if USE_KG else "false").lower() in ("1", "true", "yes")

ROLE_SCORE_BIAS = {
    "expert": 0.0,
    "moderate": 0.0,
    "fresher": 0.0,
}

ABLATION_MODES = {
    "semantic": {"similarity": 1.0, "authority": 0.0, "diversity": 0.0, "kg": 0.0, "llm": 0.0, "workload": 0.0},
    "semantic_kg": {"similarity": 0.50, "authority": 0.0, "diversity": 0.0, "kg": 0.50, "llm": 0.0, "workload": 0.0},
    "semantic_llm": {"similarity": 0.50, "authority": 0.0, "diversity": 0.0, "kg": 0.0, "llm": 0.50, "workload": 0.0},
    "semantic_kg_llm": {"similarity": 0.34, "authority": 0.0, "diversity": 0.0, "kg": 0.33, "llm": 0.33, "workload": 0.0},
    "full": SCORING_WEIGHTS,
}


def _role_score_bonus(role):
    return ROLE_SCORE_BIAS.get(str(role).lower(), 0.0)


def _scoring_weights(mode):
    return ABLATION_MODES.get(mode, SCORING_WEIGHTS)


def _role_scoring_weights(role):
    return dict(ROLE_SCORING_WEIGHTS.get(str(role or "").lower(), SCORING_WEIGHTS))


def _effective_scoring_weights(mode, kg_confidence=1.0, role=None):
    weights = dict(_scoring_weights(mode))
    if mode == "full":
        weights = _role_scoring_weights(role)
        weights["base_kg"] = float(weights.get("kg", 0.15) or 0.0)
        weights["authority"] = float(weights.get("authority", 0.0) or 0.0)
    return weights


def _effective_kg_score(kg_score, similarity, kg_confidence, mode="full"):
    kg_raw = max(0.0, min(1.0, float(kg_score or 0.0)))
    similarity = max(0.0, min(1.0, float(similarity or 0.0)))
    kg_confidence = max(0.0, min(1.0, float(kg_confidence or 0.0)))
    if mode == "semantic_kg":
        return kg_raw * kg_confidence, False, False
    capped = min(kg_raw, similarity + 0.25) if mode == "full" else kg_raw
    conditioned = capped * kg_confidence
    return conditioned, False, capped < kg_raw


def _hybrid_score(similarity, authority, kg_score=0.0, mode="full", diversity=0.0, kg_confidence=1.0, role=None, llm_match=0.0, workload_score=1.0):
    weights = _effective_scoring_weights(mode, kg_confidence, role)
    similarity = max(0.0, min(1.0, float(similarity or 0.0)))
    authority = max(0.0, min(1.0, float(authority or 0.0)))
    llm_match = max(0.0, min(1.0, float(llm_match or 0.0)))
    workload_score = max(0.0, min(1.0, float(workload_score if workload_score is not None else 1.0)))
    kg_score, _kg_disabled, _kg_capped = _effective_kg_score(kg_score, similarity, kg_confidence, mode)
    return _cap_assignment_score(
        weights["similarity"] * similarity
        + weights["authority"] * authority
        + weights.get("diversity", 0.0) * max(0.0, min(1.0, float(diversity or 0.0)))
        + weights.get("kg", 0.0) * kg_score
        + weights.get("llm", 0.0) * llm_match
        + weights.get("workload", 0.0) * workload_score
    )


def _score_contributions(similarity, authority, kg_score=0.0, mode="full", kg_confidence=1.0, role=None, llm_match=0.0, workload_score=1.0):
    weights = _effective_scoring_weights(mode, kg_confidence, role)
    similarity = max(0.0, min(1.0, float(similarity or 0.0)))
    authority = max(0.0, min(1.0, float(authority or 0.0)))
    llm_match = max(0.0, min(1.0, float(llm_match or 0.0)))
    workload_score = max(0.0, min(1.0, float(workload_score if workload_score is not None else 1.0)))
    kg_raw = max(0.0, min(1.0, float(kg_score or 0.0)))
    kg_score, kg_disabled, kg_capped = _effective_kg_score(kg_raw, similarity, kg_confidence, mode)
    weighted = {
        "semantic": weights["similarity"] * similarity,
        "authority": weights.get("authority", 0.0) * authority,
        "kg": weights.get("kg", 0.0) * kg_score,
        "llm": weights.get("llm", 0.0) * llm_match,
        "workload": weights.get("workload", 0.0) * workload_score,
    }
    total = sum(weighted.values())
    normalized = {
        key: (value / total if total > 1e-12 else 0.0)
        for key, value in weighted.items()
    }
    return {
        "semantic": normalized["semantic"],
        "authority": normalized["authority"],
        "kg": normalized["kg"],
        "llm": normalized["llm"],
        "workload": normalized["workload"],
        "weighted": weighted,
        "weights": weights,
        "kg_disabled": kg_disabled,
        "kg_capped": kg_capped,
        "kg_effective": kg_score,
        "semantic_rejected": mode == "full" and similarity < SEMANTIC_REJECTION_THRESHOLD,
    }


def _apply_reviewer_score_policy(reviewer, score):
    adjusted = float(score or 0.0)
    if reviewer.get("semantic_threshold_failed"):
        reviewer["fresher_penalty_applied"] = get_h_index(reviewer) <= 2
        reviewer["expert_bonus_applied"] = False
        adjusted *= 0.75
    if get_h_index(reviewer) <= 2:
        adjusted *= 0.68
        reviewer["fresher_penalty_applied"] = True
    else:
        reviewer["fresher_penalty_applied"] = False

    if str(reviewer.get("role", "")).lower() == "fresher":
        sem = max(0.0, min(1.0, float(reviewer.get("normalized_similarity", 0.0) or 0.0)))
        adjusted = max(adjusted, sem * 0.58)

    if str(reviewer.get("role", "")).lower() == "expert":
        adjusted += 0.05
        reviewer["expert_bonus_applied"] = True
    else:
        reviewer["expert_bonus_applied"] = False
    return max(0.0, min(1.0, adjusted))


def _rejection_reasons(reviewer, scoring_mode="full"):
    reasons = []
    role = str(reviewer.get("role", "")).lower()
    sem = max(0.0, min(1.0, float(reviewer.get("normalized_similarity", 0.0) or 0.0)))
    kg_conf = max(0.0, min(1.0, float(reviewer.get("kg_confidence", 0.0) or 0.0)))
    authority = max(0.0, min(1.0, float(reviewer.get("authority_score", 0.0) or 0.0)))

    if scoring_mode == "full" and sem < SEMANTIC_REJECTION_THRESHOLD:
        reasons.append(f"semantic relevance below {SEMANTIC_REJECTION_THRESHOLD:.2f}")
    if sem < PANEL_MIN_SIMILARITY:
        reasons.append(f"panel similarity floor below {PANEL_MIN_SIMILARITY:.2f}")
    candidate_score = float(
        reviewer.get(
            "final_score",
            reviewer.get("raw_final_score", reviewer.get("score_debug", {}).get("final", 0.0)),
        ) or 0.0
    )
    if candidate_score < PANEL_MIN_FINAL_SCORE:
        reasons.append(f"panel quality floor below {PANEL_MIN_FINAL_SCORE:.2f}")
    return reasons


def _threshold_checks(reviewer, scoring_mode="full"):
    role = str(reviewer.get("role", "")).lower()
    sem = max(0.0, min(1.0, float(reviewer.get("normalized_similarity", 0.0) or 0.0)))
    authority = max(0.0, min(1.0, float(reviewer.get("authority_score", 0.0) or 0.0)))
    kg_conf = max(0.0, min(1.0, float(reviewer.get("kg_confidence", 0.0) or 0.0)))

    checks = []
    if scoring_mode == "full":
        checks.append({
            "label": f"Semantic >= {SEMANTIC_REJECTION_THRESHOLD:.2f}",
            "passed": sem >= SEMANTIC_REJECTION_THRESHOLD,
            "value": round(sem, 3),
        })
        if role == "expert":
            checks.append({
                "label": f"Anchor semantic >= {PRIMARY_SIMILARITY_FLOOR:.2f}",
                "passed": sem >= PRIMARY_SIMILARITY_FLOOR,
                "value": round(sem, 3),
            })
    if role == "fresher":
        checks.append({
            "label": f"Complementary semantic >= {SUPPORTING_SIMILARITY_FLOOR:.2f}",
            "passed": sem >= SUPPORTING_SIMILARITY_FLOOR,
            "value": round(sem, 3),
        })
        checks.append({
            "label": f"Authority >= {SUPPORTING_AUTHORITY_FLOOR:.2f}",
            "passed": authority >= SUPPORTING_AUTHORITY_FLOOR,
            "value": round(authority, 3),
        })
    if "kg" in (reviewer.get("routing_signals") or []):
        label = "Low-confidence graph evidence" if kg_conf < KG_CONFIDENCE_USE_THRESHOLD else "Graph evidence available"
        if scoring_mode == "semantic_kg":
            label += " (diagnostic only)"
        checks.append({
            "label": label,
            "passed": True,
            "value": round(kg_conf, 3),
        })
    return checks


def _relevance_confidence(reviewer):
    sim = max(0.0, min(1.0, float(reviewer.get("normalized_similarity", reviewer.get("similarity_score", 0.0)) or 0.0)))
    if sim < 0.40:
        return sim * 0.75
    authority = max(0.0, min(1.0, float(reviewer.get("authority_score", 0.0) or 0.0)))
    kg_conf = max(0.0, min(1.0, float(reviewer.get("kg_confidence", 0.0) or 0.0)))
    return max(0.0, min(1.0, 0.60 * sim + 0.20 * authority + 0.20 * kg_conf))


def _relevance_confidence_label(score):
    score = float(score or 0.0)
    if score < 0.40:
        return "limited aggregate confidence"
    if score < 0.65:
        return "moderate semantic alignment"
    return "high"


def _dominant_signal_label(contributions):
    signal_values = {
        "semantic": float(contributions.get("semantic", 0.0) or 0.0),
        "authority": float(contributions.get("authority", 0.0) or 0.0),
        "kg": float(contributions.get("kg", 0.0) or 0.0),
        "llm": float(contributions.get("llm", 0.0) or 0.0),
        "workload": float(contributions.get("workload", 0.0) or 0.0),
    }
    return max(signal_values, key=signal_values.get)


def _kg_confidence_for_candidate(reviewer):
    reviewer_id = str(reviewer.get("reviewer_id"))
    graph_conf = (_KG_CACHE_INFO.get("kg_confidence_by_reviewer") or {}).get(reviewer_id)
    if graph_conf is not None:
        return max(0.0, min(1.0, float(graph_conf or 0.0)))

    kg_score = max(0.0, min(1.0, float(reviewer.get("kg_score", 0.0) or 0.0)))
    if _KG_CACHE_INFO.get("fallback"):
        return min(0.49, kg_score)
    return 0.0


def _standardize_scores(values):
    if not values:
        return []
    vals = [float(v or 0.0) for v in values]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance)
    if std <= 1e-9:
        return [0.5 for _ in vals]
    z_scores = [(v - mean) / std for v in vals]
    return [_sigmoid(z) for z in z_scores]


def _minmax_scores(values):
    if not values:
        return []
    vals = [float(v or 0.0) for v in values]
    low = min(vals)
    high = max(vals)
    denom = high - low
    if denom <= 1e-8:
        return [0.5 for _ in vals]
    return [(v - low) / (denom + 1e-8) for v in vals]


def _cap_assignment_score(score):
    return max(0.0, min(1.0, float(score or 0.0)))


def _legacy_role_score_bonus(role):
    if role == "expert":
        return 0.55
    if role == "moderate":
        return 0.25
    return 0.0


def _display_signals(base_signals, reviewer):
    contributions = (reviewer.get("score_debug") or {}).get("contributions") or {}
    if contributions:
        threshold = 0.005
        active = []
        if "semantic" in base_signals and float(contributions.get("semantic", 0.0) or 0.0) > threshold:
            active.append("semantic")
        if "authority" in base_signals and float(contributions.get("authority", 0.0) or 0.0) > threshold:
            active.append("authority")
        if "kg" in base_signals and float(contributions.get("kg", 0.0) or 0.0) > threshold:
            active.append("kg")
        if "llm" in base_signals and float(contributions.get("llm", 0.0) or 0.0) > threshold:
            active.append("expertise")
        if "workload" in base_signals and float(contributions.get("workload", 0.0) or 0.0) > threshold:
            active.append("workload")
        if "keyword" in base_signals:
            active.append("keyword")
        return active or ["semantic"]

    signals = []
    if "semantic" in base_signals:
        signals.append("semantic")
    if "authority" in base_signals:
        signals.append("authority")
    if "kg" in base_signals:
        signals.append("kg")
    if "llm" in base_signals:
        signals.append("expertise")
    if "workload" in base_signals:
        signals.append("workload")
    if "keyword" in base_signals:
        signals.append("keyword")
    return signals or ["semantic"]


def should_use_cross_encoder(candidates):
    if not (USE_CROSS and CROSS_TOP_K > 0):
        return False
    if not CROSS_ONLY_UNCERTAIN:
        return True
    if len(candidates) < 6:
        return True

    ranked = sorted(candidates, key=lambda row: row.get("similarity_score", 0.0), reverse=True)
    top1 = float(ranked[0].get("similarity_score", 0.0) or 0.0)
    top6 = float(ranked[5].get("similarity_score", 0.0) or 0.0)
    return (top1 - top6) < CROSS_MARGIN_THRESHOLD


def rerank_with_cross_encoder(paper_text, candidates, paper_id=None):
    if not USE_CROSS:
        return candidates

    model = _get_cross_model()
    if model is None:
        return candidates

    if not should_use_cross_encoder(candidates):
        return candidates

    filtered_candidates = [
        reviewer for reviewer in candidates
        if reviewer.get("similarity_score", 0.0) >= CROSS_SCORE_THRESHOLD
    ]
    if not filtered_candidates:
        return candidates

    pairs = []
    missing = []
    score_values = []
    for reviewer in filtered_candidates:
        reviewer_text = reviewer.get("reviewer_text", "")
        key = cross_cache_key(paper_id, reviewer.get("reviewer_id"), paper_text, reviewer_text)
        cached = get_cached_cross_score(key)
        if cached is None:
            pairs.append((paper_text, reviewer_text))
            missing.append((reviewer, key))
            score_values.append(None)
        else:
            score_values.append(float(cached))

    cross_start = time.time() if CROSS_DEBUG_TIMING else None
    if pairs:
        try:
            scores = model.predict(
                pairs,
                batch_size=CROSS_BATCH_SIZE,
                show_progress_bar=False,
            )
        except TypeError:
            scores = model.predict(pairs, batch_size=CROSS_BATCH_SIZE)
        except Exception:
            return candidates

        score_iter = iter(float(score) for score in scores)
        for idx, score in enumerate(score_values):
            if score is not None:
                continue
            value = next(score_iter)
            reviewer, key = missing.pop(0)
            set_cached_cross_score(key, value)
            score_values[idx] = value
        _save_cross_score_cache()

    if cross_start is not None:
        print("Cross time:", round(time.time() - cross_start, 2), "sec")

    score_values = [float(score) for score in score_values if score is not None]
    low = min(score_values) if score_values else 0.0
    high = max(score_values) if score_values else 1.0
    for reviewer, score in zip(filtered_candidates, score_values):
        ce_norm = (score - low) / (high - low) if high > low else 1.0
        sem = float(reviewer.get("similarity_score", 0.0) or 0.0)
        reviewer["cross_score_raw"] = float(score)
        reviewer["cross_score"] = float(ce_norm)
        reviewer["raw_final_score"] = 0.7 * ce_norm + 0.3 * sem
        reviewer["final_score"] = reviewer["raw_final_score"]
        reviewer["rank_signal"] = reviewer["final_score"]

    candidates.sort(
        key=lambda x: x.get("final_score", x.get("similarity_score", 0.0)),
        reverse=True,
    )
    return candidates


def _reviewer_profile_chunks(reviewer, max_publications=None):
    if os.getenv("REVIEWER_PROFILE_CHUNKED", "true").lower() not in ("1", "true", "yes"):
        return [build_reviewer_text(reviewer)]

    if max_publications is None:
        max_publications = int(os.getenv("REVIEWER_PROFILE_PUBLICATIONS", "12"))
    keywords = _reviewer_keywords(reviewer, limit=20)
    venues = _reviewer_top_venues(reviewer, limit=5)
    base = " ".join(
        part
        for part in [
            _to_text(reviewer.get("area", "") or reviewer.get("areas", "")),
            _to_text(
                reviewer.get("research_interests", "")
                or reviewer.get("interests", "")
                or reviewer.get("expertise", "")
                or reviewer.get("topics", "")
            ),
            _to_text(keywords),
            _to_text(venues),
        ]
        if part
    ).strip()
    chunks = [base] if base else []

    for pub in _representative_publications(reviewer, limit=max_publications):
        text = " ".join(
            part
            for part in [
                _to_text(pub.get("title", "")),
                _to_text(pub.get("abstract", "")),
            ]
            if part
        ).strip()
        if text:
            chunks.append(text)

    return chunks or [build_reviewer_text(reviewer)]


def _mean_pool_embeddings(embeddings):
    if np is None or embeddings is None or len(embeddings) == 0:
        return None
    pooled = np.mean(np.asarray(embeddings, dtype=float), axis=0)
    norm = np.linalg.norm(pooled)
    if norm:
        pooled = pooled / norm
    return pooled


def _encode_reviewer_profiles(reviewers, model=None):
    model = bi_model if model is None else model
    if model is None:
        return None

    reviewer_chunks = [_reviewer_profile_chunks(reviewer) for reviewer in reviewers]
    flat_chunks = [_semantic_document_text(chunk) for chunks in reviewer_chunks for chunk in chunks]
    if not flat_chunks:
        return None

    encoded = model.encode(
        flat_chunks,
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = []
    offset = 0
    for chunks in reviewer_chunks:
        chunk_embeddings = encoded[offset : offset + len(chunks)]
        offset += len(chunks)
        embeddings.append(_mean_pool_embeddings(chunk_embeddings))

    return np.asarray(embeddings, dtype=float)


def prepare_reviewers_for_ranking(reviewers):
    global _REVIEWER_EMBEDDINGS
    global _REVIEWER_INDEX
    global _SECONDARY_REVIEWER_EMBEDDINGS
    global _SECONDARY_REVIEWER_INDEX
    global _REVIEWER_CACHE_INFO

    if not reviewers:
        return reviewers

    for reviewer in reviewers:
        reviewer_text = reviewer.get("reviewer_text") or build_reviewer_text(reviewer)
        reviewer["reviewer_text"] = reviewer_text
        reviewer["_reviewer_embedding"] = None
        reviewer["_secondary_reviewer_embedding"] = None
        reviewer["_reviewer_keyword_set"] = _keyword_set_from_text(
            " ".join(
                part for part in [
                    reviewer_text,
                    _to_text(reviewer.get("keywords", [])),
                ] if part
            )
        )
        reviewer["_reviewer_token_set"] = _lexical_token_set(
            " ".join(
                part for part in [
                    reviewer_text,
                    _to_text(reviewer.get("keywords", [])),
                ] if part
            )
        )

    reviewer_texts = [reviewer.get("reviewer_text", "") for reviewer in reviewers]
    cache_signature = _reviewer_cache_signature(reviewer_texts)
    cache_path = _reviewer_embedding_cache_path()
    meta_path = _reviewer_embedding_meta_path()
    secondary_cache_path = _secondary_reviewer_embedding_cache_path()
    secondary_meta_path = _secondary_reviewer_embedding_meta_path()

    embeddings = None
    secondary_embeddings = None
    loaded_from_disk = False
    secondary_loaded_from_disk = False

    if np is not None and os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cached_embeddings = np.load(cache_path)
            if (
                meta.get("signature") == cache_signature
                and meta.get("count") == len(reviewers)
                and meta.get("dim") == MODEL_EMBEDDING_DIM
                and len(cached_embeddings) == len(reviewers)
            ):
                embeddings = cached_embeddings
                loaded_from_disk = True
        except Exception:
            embeddings = None

    if embeddings is None and SENTS_AVAILABLE and bi_model is not None and np is not None:
        os.makedirs(_cache_dir(), exist_ok=True)
        embeddings = _encode_reviewer_profiles(reviewers)
        try:
            np.save(cache_path, embeddings)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "signature": cache_signature,
                        "count": len(reviewers),
                        "dim": MODEL_EMBEDDING_DIM,
                        "model": _ACTIVE_EMBEDDING_MODEL,
                        "profile_version": "instruction_profile_v9_scientific_profile",
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    if (
        np is not None
        and secondary_bi_model is not None
        and os.path.exists(secondary_cache_path)
        and os.path.exists(secondary_meta_path)
    ):
        try:
            with open(secondary_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cached_embeddings = np.load(secondary_cache_path)
            if (
                meta.get("signature") == cache_signature
                and meta.get("count") == len(reviewers)
                and meta.get("dim") == SECONDARY_EMBEDDING_DIM
                and len(cached_embeddings) == len(reviewers)
            ):
                secondary_embeddings = cached_embeddings
                secondary_loaded_from_disk = True
        except Exception:
            secondary_embeddings = None

    if secondary_embeddings is None and SENTS_AVAILABLE and secondary_bi_model is not None and np is not None:
        os.makedirs(_cache_dir(), exist_ok=True)
        secondary_embeddings = _encode_reviewer_profiles(reviewers, model=secondary_bi_model)
        try:
            np.save(secondary_cache_path, secondary_embeddings)
            with open(secondary_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "signature": cache_signature,
                        "count": len(reviewers),
                        "dim": SECONDARY_EMBEDDING_DIM,
                        "model": _ACTIVE_SECONDARY_EMBEDDING_MODEL,
                        "profile_version": "instruction_profile_v9_scientific_profile",
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    _REVIEWER_EMBEDDINGS = embeddings
    _REVIEWER_INDEX = None
    _SECONDARY_REVIEWER_EMBEDDINGS = secondary_embeddings
    _SECONDARY_REVIEWER_INDEX = None
    index_status = "unavailable"
    secondary_index_status = "unavailable"
    if embeddings is not None and np is not None:
        matrix = np.asarray(embeddings, dtype="float32")
        if FAISS_AVAILABLE:
            try:
                _REVIEWER_INDEX = faiss.IndexFlatIP(matrix.shape[1])
                _REVIEWER_INDEX.add(matrix)
                index_status = "faiss"
            except Exception:
                _REVIEWER_INDEX = None
                index_status = "numpy"
        else:
            index_status = "numpy"

    if secondary_embeddings is not None and np is not None:
        matrix = np.asarray(secondary_embeddings, dtype="float32")
        if FAISS_AVAILABLE:
            try:
                _SECONDARY_REVIEWER_INDEX = faiss.IndexFlatIP(matrix.shape[1])
                _SECONDARY_REVIEWER_INDEX.add(matrix)
                secondary_index_status = "faiss"
            except Exception:
                _SECONDARY_REVIEWER_INDEX = None
                secondary_index_status = "numpy"
        else:
            secondary_index_status = "numpy"

    _REVIEWER_CACHE_INFO = {
        "status": "loaded" if loaded_from_disk else ("built" if embeddings is not None else "unavailable"),
        "path": cache_path,
        "count": len(reviewers),
        "model": _ACTIVE_EMBEDDING_MODEL,
        "secondary_status": "loaded" if secondary_loaded_from_disk else ("built" if secondary_embeddings is not None else "unavailable"),
        "secondary_path": secondary_cache_path,
        "secondary_model": _ACTIVE_SECONDARY_EMBEDDING_MODEL,
        "secondary_weight": _SECONDARY_WEIGHT if secondary_embeddings is not None else 0.0,
        "profile_version": "instruction_profile_v9_scientific_profile",
        "index": index_status,
        "secondary_index": secondary_index_status,
        "retrieval_depth": TOP_K,
        "rerank_depth": CROSS_TOP_K if USE_CROSS else 0,
    }

    if embeddings is not None:
        for reviewer, emb in zip(reviewers, embeddings):
            reviewer["_reviewer_embedding"] = emb
            reviewer_text = reviewer.get("reviewer_text", "")
            if reviewer_text:
                _TEXT_EMBED_CACHE[reviewer_text] = emb

    if secondary_embeddings is not None:
        for reviewer, emb in zip(reviewers, secondary_embeddings):
            reviewer["_secondary_reviewer_embedding"] = emb
            reviewer_text = reviewer.get("reviewer_text", "")
            if reviewer_text:
                _SECONDARY_TEXT_EMBED_CACHE[reviewer_text] = emb

    return reviewers


def get_reviewer_cache_info():
    return dict(_REVIEWER_CACHE_INFO)


def _search_reviewer_matrix(query_embedding, matrix, index, top_k):
    if query_embedding is None or matrix is None or np is None:
        return [], None

    query = np.asarray([query_embedding], dtype="float32")
    if index is not None:
        try:
            scores, indices = index.search(query, top_k)
            return [int(idx) for idx in indices[0] if int(idx) >= 0], np.asarray(scores[0], dtype=float)
        except Exception:
            pass

    matrix = np.asarray(matrix, dtype="float32")
    scores = np.dot(matrix, np.asarray(query_embedding, dtype="float32"))
    if top_k < len(scores):
        candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
    else:
        candidate_idx = np.argsort(-scores)
    return [int(idx) for idx in candidate_idx], np.asarray(scores[candidate_idx], dtype=float)


def _retrieve_reviewer_candidates(paper_text, reviewers, candidate_k):
    if not paper_text or not reviewers:
        return [], None

    top_k = min(candidate_k or len(reviewers), len(reviewers))
    retrieval_k = min(
        len(reviewers),
        max(
            top_k,
            int(os.getenv("HYBRID_RETRIEVAL_POOL", str(max(150, top_k * 2)))),
        ),
    )
    paper_emb = _embed_paper_text(paper_text)
    if paper_emb is None or _REVIEWER_EMBEDDINGS is None or np is None:
        return list(range(len(reviewers))), None

    primary_indices, primary_scores = _search_reviewer_matrix(
        paper_emb,
        _REVIEWER_EMBEDDINGS,
        _REVIEWER_INDEX,
        retrieval_k,
    )
    primary_score_map = {
        idx: float(score)
        for idx, score in zip(primary_indices, primary_scores if primary_scores is not None else [])
    }

    secondary_score_map = {}
    secondary_indices = []
    secondary_emb = _embed_secondary_paper_text(paper_text)
    if secondary_emb is not None and _SECONDARY_REVIEWER_EMBEDDINGS is not None:
        secondary_indices, secondary_scores = _search_reviewer_matrix(
            secondary_emb,
            _SECONDARY_REVIEWER_EMBEDDINGS,
            _SECONDARY_REVIEWER_INDEX,
            retrieval_k,
        )
        secondary_score_map = {
            idx: float(score)
            for idx, score in zip(secondary_indices, secondary_scores if secondary_scores is not None else [])
        }

    lexical_indices = []
    if os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower() in ("1", "true", "yes"):
        paper_tokens = _lexical_token_set(paper_text)
        paper_keywords = _keyword_set_from_text(paper_text)
        lexical_scores = [
            (idx, _lexical_overlap_score(paper_tokens, paper_keywords, reviewer))
            for idx, reviewer in enumerate(reviewers)
        ]
        lexical_scores.sort(key=lambda item: item[1], reverse=True)
        lexical_k = min(
            len(reviewers),
            int(os.getenv("LEXICAL_RETRIEVAL_K", str(max(75, top_k)))),
        )
        lexical_indices = [idx for idx, score in lexical_scores[:lexical_k] if score > 0.0]

    candidate_pool = list(dict.fromkeys(primary_indices + secondary_indices + lexical_indices))
    if not candidate_pool:
        return [], None

    fused_scores = {}
    for idx in candidate_pool:
        primary = primary_score_map.get(idx)
        if primary is None:
            reviewer_emb = reviewers[idx].get("_reviewer_embedding")
            primary = _cosine_from_embeddings(paper_emb, reviewer_emb)
        secondary = secondary_score_map.get(idx)
        if secondary is None and secondary_emb is not None:
            secondary = _cosine_from_embeddings(secondary_emb, reviewers[idx].get("_secondary_reviewer_embedding"))
        if secondary is None:
            secondary = primary
        lexical = 0.0
        if os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower() in ("1", "true", "yes"):
            lexical = _lexical_overlap_score(_lexical_token_set(paper_text), _keyword_set_from_text(paper_text), reviewers[idx])
        fused_scores[idx] = (
            0.82 * ((_PRIMARY_WEIGHT * float(primary or 0.0)) + (_SECONDARY_WEIGHT * float(secondary or 0.0)))
            + 0.18 * float(lexical or 0.0)
        )

    ranked = sorted(candidate_pool, key=lambda idx: fused_scores.get(idx, 0.0), reverse=True)[:top_k]
    return ranked, np.asarray([fused_scores[idx] for idx in ranked], dtype=float)


def get_kg_cache_info():
    return dict(_KG_CACHE_INFO)


def _kg_node_id(prefix, value):
    text = kw.normalize(_to_text(value)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return f"{prefix}_{text[:80]}" if text else ""


def _publication_text(publication):
    return " ".join(
        part
        for part in [
            _to_text(publication.get("title", "")),
            _to_text(publication.get("abstract", "")),
            _to_text(publication.get("keywords", [])),
        ]
        if part
    ).strip()


def _topic_terms_for_text(text, limit=8):
    return enrich_keywords(clean_keywords(extract_keyphrases(text or "")))[:limit]


def _add_topic_edges(graph, source_node, text, source_emb=None, threshold=0.35):
    added = 0
    for topic in _topic_terms_for_text(text, limit=8):
        topic_node = _kg_node_id("t", topic)
        if not topic_node:
            continue
        if source_emb is not None:
            topic_emb = _embed_text(topic)
            if topic_emb is not None and _cosine_from_embeddings(source_emb, topic_emb) < threshold:
                continue
        graph.add_node(topic_node, node_type="topic", label=topic)
        graph.add_edge(source_node, topic_node, edge_type="has_topic")
        added += 1
    return added


def _build_reviewer_kg_graph(paper_id, paper_text, candidates, limit=50):
    if not NETWORKX_AVAILABLE:
        return None, {}

    graph = nx.Graph()
    paper_node = f"p_{paper_id}"
    graph.add_node(paper_node, node_type="paper", label=str(paper_id))
    paper_emb = _embed_paper_text(paper_text)
    _add_topic_edges(graph, paper_node, paper_text, source_emb=paper_emb, threshold=0.20)

    publication_nodes = []
    kg_candidates = list(candidates[:limit])

    for reviewer in kg_candidates:
        reviewer_id = str(reviewer.get("reviewer_id", "")).strip()
        if not reviewer_id:
            continue
        reviewer_node = f"r_{reviewer_id}"
        reviewer_text = reviewer.get("reviewer_text") or build_reviewer_text(reviewer)
        reviewer_emb = _embed_text(reviewer_text)
        graph.add_node(reviewer_node, node_type="reviewer", label=reviewer_id)
        _add_topic_edges(graph, reviewer_node, reviewer_text, source_emb=reviewer_emb, threshold=0.25)

        for topic in _reviewer_keywords(reviewer, limit=10):
            topic_node = _kg_node_id("t", topic)
            if topic_node:
                graph.add_node(topic_node, node_type="topic", label=topic)
                graph.add_edge(reviewer_node, topic_node, edge_type="works_on")

        for venue in _reviewer_top_venues(reviewer, limit=2):
            venue_node = _kg_node_id("v", venue)
            if venue_node:
                graph.add_node(venue_node, node_type="venue", label=venue)
                graph.add_edge(reviewer_node, venue_node, edge_type="publishes_in")

        for idx, publication in enumerate(_representative_publications(reviewer, limit=5)):
            pub_text = _publication_text(publication)
            if not pub_text:
                continue
            pub_title = _to_text(publication.get("title", "")) or f"{reviewer_id}_{idx}"
            pub_node = _kg_node_id("pub", f"{reviewer_id}_{pub_title}")
            if not pub_node:
                continue
            pub_emb = _embed_text(pub_text)
            graph.add_node(pub_node, node_type="paper", label=pub_title)
            graph.add_edge(reviewer_node, pub_node, edge_type="authored")
            _add_topic_edges(graph, pub_node, pub_text, source_emb=pub_emb, threshold=0.20)

            venue = _to_text(publication.get("venue", "") or publication.get("conference", ""))
            venue_node = _kg_node_id("v", venue)
            if venue_node:
                graph.add_node(venue_node, node_type="venue", label=venue)
                graph.add_edge(pub_node, venue_node, edge_type="published_in")

            publication_nodes.append((pub_node, pub_emb, pub_text))

    for pub_node, pub_emb, pub_text in publication_nodes:
        sim = _cosine_from_embeddings(paper_emb, pub_emb) if paper_emb is not None and pub_emb is not None else semantic_overlap_score(paper_text, pub_text)
        if sim >= 0.80:
            graph.add_edge(paper_node, pub_node, edge_type="similar_paper", weight=float(sim))

    for i, (node_a, emb_a, text_a) in enumerate(publication_nodes):
        for node_b, emb_b, text_b in publication_nodes[i + 1:]:
            sim = _cosine_from_embeddings(emb_a, emb_b) if emb_a is not None and emb_b is not None else semantic_overlap_score(text_a, text_b)
            if sim >= 0.80:
                graph.add_edge(node_a, node_b, edge_type="similar_paper", weight=float(sim))

    return graph, {str(reviewer.get("reviewer_id")): f"r_{reviewer.get('reviewer_id')}" for reviewer in kg_candidates}


def _kg_path_count(graph, paper_node, reviewer_node, cutoff=3, limit=100):
    if graph is None or paper_node not in graph or reviewer_node not in graph or not NETWORKX_AVAILABLE:
        return 0
    count = 0
    try:
        if graph.has_edge(paper_node, reviewer_node):
            count += 1
        count += len(list(nx.common_neighbors(graph, paper_node, reviewer_node)))
        for _path in nx.all_simple_paths(graph, paper_node, reviewer_node, cutoff=cutoff):
            count += 1
            if count >= limit:
                return limit
    except Exception:
        return count
    return min(count, limit)


def _kg_confidence_from_graph(graph, paper_node, reviewer_node, max_degree=1):
    if graph is None or paper_node not in graph or reviewer_node not in graph:
        return 0.0
    num_paths = _kg_path_count(graph, paper_node, reviewer_node)
    confidence = math.log1p(num_paths) / 5.0
    return max(0.0, min(1.0, confidence))


def _generate_candidate_kg_scores(paper_id, paper_text, candidates, limit=None):
    global _KG_CACHE_INFO

    limit = max(1, min(int(limit or KG_TOP_K), len(candidates))) if candidates else 0
    kg_candidates = candidates[:limit]

    candidate_fingerprint = [
        [
            str(reviewer.get("reviewer_id", "")),
            str(reviewer.get("updated_at", "")),
            len(reviewer.get("publications", []) or []),
        ]
        for reviewer in kg_candidates
    ]
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "paper_id": str(paper_id),
                "paper_text": paper_text,
                "candidates": candidate_fingerprint,
                "schema": "typed_reviewer_paper_topic_graph_v1",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cached = _KG_RESULT_CACHE.get(cache_key)
    if cached:
        _KG_CACHE_INFO = dict(cached.get("info", {}))
        _KG_CACHE_INFO["cache_hit"] = True
        return dict(cached.get("scores", {}))

    persistent_graph = _get_persistent_kg()
    if persistent_graph is not None and graph_overlap_score is not None:
        graph_scores = {}
        graph_confidences = {}
        for reviewer in kg_candidates:
            reviewer_id = str(reviewer.get("reviewer_id", ""))
            score, confidence = graph_overlap_score(persistent_graph, paper_id, reviewer_id)
            graph_scores[reviewer_id] = float(score or 0.0)
            graph_confidences[reviewer_id] = float(confidence or 0.0)
        if any(value > 0.0 for value in graph_scores.values()):
            _KG_CACHE_INFO = {
                "status": "persistent_graph",
                "schema": _PERSISTENT_KG_META.get("schema", "persistent_reviewer_kg_v1"),
                "nodes": _PERSISTENT_KG_META.get("nodes", persistent_graph.number_of_nodes()),
                "edges": _PERSISTENT_KG_META.get("edges", persistent_graph.number_of_edges()),
                "candidate_limit": limit,
                "kg_confidence_by_reviewer": graph_confidences,
                "cache_hit": False,
            }
            _KG_RESULT_CACHE[cache_key] = {
                "scores": dict(graph_scores),
                "info": dict(_KG_CACHE_INFO),
            }
            return graph_scores

    if not USE_NODE2VEC_KG:
        _KG_CACHE_INFO = {
            "status": "fast_proxy",
            "candidate_limit": limit,
            "schema": "citation_topic_proxy_v1",
            "cache_hit": False,
        }
        return {}

    if not (
        KG_NODE2VEC_AVAILABLE
        and generate_embeddings is not None
        and NETWORKX_AVAILABLE
    ):
        _KG_CACHE_INFO = {"status": "unavailable"}
        return {}

    try:
        graph, reviewer_nodes = _build_reviewer_kg_graph(paper_id, paper_text, kg_candidates, limit=limit)
        embeddings = generate_embeddings(graph)
    except Exception as exc:
        _KG_CACHE_INFO = {"status": "unavailable", "reason": str(exc)}
        return {}

    paper_node = f"p_{paper_id}"
    max_degree = max((degree for _node, degree in graph.degree()), default=1)

    def _score_reviewer_kg(item):
        reviewer_id, reviewer_node = item
        if paper_node in embeddings and reviewer_node in embeddings:
            score = compute_node2vec_kg_similarity(paper_id, reviewer_id, embeddings)
        else:
            score = 0.0
        confidence = _kg_confidence_from_graph(graph, paper_node, reviewer_node, max_degree=max_degree)
        return reviewer_id, score, confidence

    reviewer_items = list(reviewer_nodes.items())
    if JOBLIB_AVAILABLE and len(reviewer_items) > 4:
        scored_items = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_score_reviewer_kg)(item) for item in reviewer_items
        )
    else:
        scored_items = [_score_reviewer_kg(item) for item in reviewer_items]

    scores = {reviewer_id: score for reviewer_id, score, _confidence in scored_items}
    confidences = {reviewer_id: confidence for reviewer_id, _score, confidence in scored_items}

    _KG_CACHE_INFO = {
        "status": "built" if embeddings else "empty",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "embeddings": len(embeddings),
        "candidate_limit": limit,
        "schema": "typed_reviewer_paper_topic_graph_v1",
        "max_degree": max_degree,
        "kg_confidence_by_reviewer": confidences,
        "cache_hit": False,
    }
    _KG_RESULT_CACHE[cache_key] = {
        "scores": dict(scores),
        "info": dict(_KG_CACHE_INFO),
    }

    return scores


def _paper_reference_keys(paper):
    keys = set()
    if not paper:
        return keys
    for field in ("references", "reference_titles", "citations", "cited_papers"):
        values = paper.get(field, []) or []
        if isinstance(values, str):
            values = [values]
        for item in values:
            text = item.get("title", "") if isinstance(item, dict) else item
            text = kw.normalize(_to_text(text)).lower().strip()
            if text:
                keys.add(text)
    return keys


def citation_overlap(paper, reviewer):
    """Fast KG fallback: overlap paper citation/keyword concepts with reviewer work."""
    paper_text = build_paper_text(paper)
    paper_terms = set(enrich_keywords(clean_keywords(extract_keyphrases(paper_text))))
    paper_terms |= _paper_reference_keys(paper)
    if not paper_terms:
        return 0.0

    reviewer_terms = set(_reviewer_keywords(reviewer, limit=20))
    for pub in _representative_publications(reviewer, limit=20):
        reviewer_terms.add(_to_text(pub.get("title", "")))
        reviewer_terms.update(clean_keywords(extract_keyphrases(_to_text(pub.get("abstract", "")))))

    reviewer_terms = {kw.normalize(_to_text(term)).lower().strip() for term in reviewer_terms if _to_text(term).strip()}
    if not reviewer_terms:
        return 0.0

    overlap = 0
    for p_term in paper_terms:
        p_norm = kw.normalize(_to_text(p_term)).lower().strip()
        if not p_norm:
            continue
        if any(p_norm == r_term or p_norm in r_term or r_term in p_norm for r_term in reviewer_terms):
            overlap += 1
    return min(1.0, overlap / max(3, min(len(paper_terms), 10)))


def sparse_topic_kg_proxy(paper, reviewer):
    """Low-cost KG proxy when graph edges are sparse: paper/topic to reviewer/topic overlap."""
    paper_text = build_paper_text(paper)
    paper_terms = {
        kw.normalize(_to_text(term)).lower().strip()
        for term in enrich_keywords(clean_keywords(extract_keyphrases(paper_text)))
        if _to_text(term).strip()
    }
    reviewer_terms = {
        kw.normalize(_to_text(term)).lower().strip()
        for term in _reviewer_keywords(reviewer, limit=20)
        if _to_text(term).strip()
    }
    for pub in _representative_publications(reviewer, limit=8):
        pub_text = " ".join(
            part for part in [
                _to_text(pub.get("title", "")),
                _to_text(pub.get("abstract", "")),
                _to_text(pub.get("keywords", "")),
            ] if part
        )
        reviewer_terms.update(
            kw.normalize(_to_text(term)).lower().strip()
            for term in enrich_keywords(clean_keywords(extract_keyphrases(pub_text)))
            if _to_text(term).strip()
        )

    paper_terms = {term for term in paper_terms if term}
    reviewer_terms = {term for term in reviewer_terms if term}
    if not paper_terms or not reviewer_terms:
        return 0.0

    overlap = 0
    for p_term in paper_terms:
        if any(p_term == r_term or p_term in r_term or r_term in p_term for r_term in reviewer_terms):
            overlap += 1
    return min(0.75, overlap / max(3, min(len(paper_terms), 12)))


def _normalize_candidate_kg_scores(paper, candidates, raw_scores):
    scores = {
        str(reviewer.get("reviewer_id")): max(0.0, float(raw_scores.get(str(reviewer.get("reviewer_id")), 0.0) or 0.0))
        for reviewer in candidates
    }
    if any(value > 0.0 for value in scores.values()):
        min_kg = min(scores.values())
        max_kg = max(scores.values())
        denom = max_kg - min_kg
        normalized = {
            rid: ((value - min_kg) / denom) if denom > 1e-8 else 0.5
            for rid, value in scores.items()
        }
        _KG_CACHE_INFO.update({
            "fallback": "",
            "normalization": "minmax",
            "min_raw_kg": round(min_kg, 6),
            "max_raw_kg": round(max_kg, 6),
        })
        return normalized

    fallback = {
        str(reviewer.get("reviewer_id")): citation_overlap(paper, reviewer)
        for reviewer in candidates
    }
    if not any(value > 0.0 for value in fallback.values()):
        fallback = {
            str(reviewer.get("reviewer_id")): sparse_topic_kg_proxy(paper, reviewer)
            for reviewer in candidates
        }
    if not any(value > 0.0 for value in fallback.values()):
        _KG_CACHE_INFO.update({"fallback": "", "normalization": "no_kg_evidence", "max_raw_kg": 0.0})
        return {rid: 0.0 for rid in fallback}

    min_kg = min(fallback.values())
    max_kg = max(fallback.values())
    denom = max_kg - min_kg
    fallback_cap = min(0.85, max_kg)
    normalized = {
        rid: (((value - min_kg) / denom) if denom > 1e-8 else 0.5) * fallback_cap
        for rid, value in fallback.items()
    }
    _KG_CACHE_INFO.update({
        "fallback": "citation_overlap_or_sparse_topic_proxy",
        "normalization": "minmax_with_evidence_cap",
        "min_raw_kg": round(min_kg, 6),
        "max_raw_kg": round(max_kg, 6),
        "fallback_cap": round(fallback_cap, 6),
    })
    confidence_by_reviewer = dict(_KG_CACHE_INFO.get("kg_confidence_by_reviewer") or {})
    for rid, value in normalized.items():
        confidence_by_reviewer[rid] = max(
            float(confidence_by_reviewer.get(rid, 0.0) or 0.0),
            min(0.35, 0.20 + float(value or 0.0) * 0.20),
        )
    _KG_CACHE_INFO["kg_confidence_by_reviewer"] = confidence_by_reviewer
    return normalized


def _debug_routing(signals, candidates):
    if not MCP_DEBUG:
        return

    print("Signals used:", signals)
    for reviewer in candidates[:3]:
        print(
            reviewer.get("reviewer_id"),
            reviewer.get("similarity_score", 0.0),
            reviewer.get("kg_score", 0.0),
            reviewer.get("final_score", 0.0),
        )


def rank_reviewers_for_paper(paper, reviewers, reviewer_usage=None, candidate_k=TOP_K, diversity_penalty=0.0, scoring_mode=None):
    if not paper or not reviewers:
        return []

    scoring_mode = scoring_mode or RANKING_SCORING_MODE
    paper_id = str(paper.get("paper_id", ""))
    paper_text = build_paper_text(paper)
    signals = ["semantic"]
    if scoring_mode == "full":
        signals.append("authority")
        signals.append("kg")
        signals.append("llm")
        signals.append("workload")
    elif scoring_mode == "semantic_kg":
        signals.append("kg")
    elif scoring_mode == "semantic_llm":
        signals.append("llm")
    elif scoring_mode == "semantic_kg_llm":
        signals.append("kg")
        signals.append("llm")
    paper_profile = generate_manuscript_profile(paper) if paper else {}
    paper_keywords = _keyword_set_from_text(
        " ".join(
            part for part in [
                paper_text,
                _to_text(paper.get("keywords", [])),
            ] if part
        )
    )
    candidate_indices, retrieved_scores = _retrieve_reviewer_candidates(paper_text, reviewers, candidate_k)
    score_iter = [] if retrieved_scores is None else retrieved_scores
    retrieved_score_map = {idx: float(score) for idx, score in zip(candidate_indices, score_iter)}
    paper_emb = _embed_paper_text(paper_text) if retrieved_scores is None and paper_text else None
    ranked_reviewers = []

    for idx in candidate_indices:
        reviewer = reviewers[idx]
        reviewer_id = str(reviewer.get("reviewer_id", ""))
        reviewer_text = reviewer.get("reviewer_text") or build_reviewer_text(reviewer)
        if retrieved_scores is not None:
            sim = retrieved_score_map.get(idx, 0.0)
        else:
            reviewer_emb = reviewer.get("_reviewer_embedding")
            sim = _cosine_from_embeddings(paper_emb, reviewer_emb) if paper_emb is not None and reviewer_emb is not None else 0.0
        reviewer_keywords = reviewer.get("_reviewer_keyword_set") or set()

        ranked_reviewers.append({
            **reviewer,
            "reviewer_text": reviewer_text,
            "similarity_score": sim,
            "semantic_score": sim,
            "kg_score": 0.0,
            "keyword_overlap": len(paper_keywords & reviewer_keywords),
        })

    ranked_reviewers.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    candidates = ranked_reviewers
    kg_scores = {}
    if "kg" in signals:
        kg_limit = min(KG_TOP_K, len(candidates))
        kg_scores = _generate_candidate_kg_scores(paper_id, paper_text, candidates, limit=kg_limit)
        kg_scores = _normalize_candidate_kg_scores(paper, candidates[:kg_limit], kg_scores)
    for reviewer in candidates:
        reviewer["kg_score"] = float(kg_scores.get(str(reviewer.get("reviewer_id")), 0.0) or 0.0)

    candidate_stats = compute_stats(candidates)
    normalized_sims = _minmax_scores([r.get("similarity_score", 0.0) for r in candidates])
    for llm_rank_idx, (reviewer, normalized_similarity) in enumerate(zip(candidates, normalized_sims), start=1):
        authority = compute_authority(reviewer, candidate_stats)
        keyword_overlap = min(int(reviewer.get("keyword_overlap", 0) or 0), 3)
        reviewer["normalized_similarity"] = normalized_similarity
        reviewer["authority_score"] = authority
        reviewer["role"] = classify_reviewer(reviewer, candidate_stats)
        use_llm_suitability = "llm" in signals and llm_rank_idx <= max(1, int(LLM_SUITABILITY_TOP))
        reviewer_profile = generate_reviewer_profile(reviewer) if use_llm_suitability else {}
        llm_reasoning = _llm_suitability_reasoning(paper_profile, reviewer_profile) if use_llm_suitability else {
            "final_suitability": 0.0,
            "overlap_terms": [],
            "source": "outside-llm-rerank-window" if "llm" in signals else "disabled",
        }
        llm_match = float(llm_reasoning.get("final_suitability", 0.0) or 0.0)
        llm_overlap = llm_reasoning.get("overlap_terms", [])
        reviewer["manuscript_profile"] = paper_profile
        reviewer["reviewer_profile"] = reviewer_profile
        reviewer["llm_suitability_reasoning"] = llm_reasoning
        reviewer["llm_suitability_score"] = llm_match
        reviewer["llm_match_score"] = llm_match
        reviewer["llm_profile_overlap"] = llm_overlap
        reviewer["llm_rerank_rank"] = llm_rank_idx
        reviewer["llm_rerank_window"] = max(1, int(LLM_SUITABILITY_TOP))
        reviewer["keyword_overlap_capped"] = keyword_overlap
        reviewer["cross_score"] = float(reviewer.get("cross_score", 0.0) or 0.0)
        reviewer["rank_signal"] = normalized_similarity
        reviewer["routing_signals"] = signals
        reviewer["kg_confidence"] = _kg_confidence_for_candidate(reviewer)
        reviewer["relevance_confidence"] = _relevance_confidence(reviewer)
        reviewer["coi"] = _coi_status(paper, reviewer)
        reviewer["semantic_threshold_failed"] = (
            scoring_mode == "full"
            and normalized_similarity < SEMANTIC_REJECTION_THRESHOLD
        )

        score = _hybrid_score(
            normalized_similarity,
            authority,
            reviewer.get("kg_score", 0.0),
            mode=scoring_mode,
            kg_confidence=reviewer["kg_confidence"],
            role=reviewer["role"],
            llm_match=reviewer["llm_match_score"],
        )
        contributions = _score_contributions(
            normalized_similarity,
            authority,
            reviewer.get("kg_score", 0.0),
            scoring_mode,
            reviewer["kg_confidence"],
            reviewer["role"],
            reviewer["llm_match_score"],
        )
        reviewer["score_debug"] = {
            "semantic_raw": float(reviewer.get("similarity_score", 0.0) or 0.0),
            "semantic_norm": normalized_similarity,
            "authority": authority,
            "kg": float(reviewer.get("kg_score", 0.0) or 0.0),
            "llm_match": reviewer["llm_match_score"],
            "llm_suitability_score": reviewer["llm_suitability_score"],
            "llm_suitability_reasoning": reviewer["llm_suitability_reasoning"],
            "llm_profile_overlap": reviewer["llm_profile_overlap"],
            "manuscript_profile_source": paper_profile.get("source", ""),
            "reviewer_profile_source": reviewer_profile.get("source", ""),
            "kg_confidence": reviewer["kg_confidence"],
            "kg_weight_reduced": "kg" in signals and contributions["weights"].get("kg", 0.0) < contributions["weights"].get("base_kg", 0.15),
            "kg_capped": contributions.get("kg_capped", False),
            "kg_disabled": contributions.get("kg_disabled", False),
            "kg_effective": contributions.get("kg_effective", 0.0),
            "semantic_rejected": contributions.get("semantic_rejected", False),
            "relevance_confidence": reviewer["relevance_confidence"],
            "relevance_confidence_label": _relevance_confidence_label(reviewer["relevance_confidence"]),
            "dominant_signal": _dominant_signal_label(contributions),
            "contributions": contributions,
            "final_before_policy": score,
        }
        score = _apply_reviewer_score_policy(reviewer, score)
        reviewer["score_debug"]["final"] = score
        reviewer["rejection_reasons"] = _rejection_reasons(reviewer, scoring_mode)
        if reviewer["coi"].get("flagged"):
            reviewer["rejection_reasons"].append(
                "conflict of interest: " + ", ".join(reviewer["coi"].get("reasons") or ["COI rule"])
            )
        reviewer["threshold_checks"] = _threshold_checks(reviewer, scoring_mode)

        reviewer["raw_final_score"] = max(0.0, score)
        reviewer["final_score"] = reviewer["raw_final_score"]
        if reviewer["coi"].get("flagged"):
            reviewer["final_score"] = float("-inf")
        reviewer["routing_signals"] = _display_signals(signals, reviewer)

    if USE_RRF_FUSION and scoring_mode == "full":
        _apply_rrf_fusion(candidates)

    if USE_CROSS and CROSS_TOP_K > 0:
        semantic_limit = min(SEMANTIC_RERANK_TOP, len(candidates))
        cross_limit = min(CROSS_TOP_K, semantic_limit)
        semantic_head = candidates[:semantic_limit]
        reranked_head = rerank_with_cross_encoder(paper_text, semantic_head[:cross_limit], paper_id=paper_id)
        candidates = reranked_head + semantic_head[cross_limit:] + candidates[semantic_limit:]

    if reviewer_usage and diversity_penalty:
        for reviewer in candidates:
            load = reviewer_usage.get(str(reviewer.get("reviewer_id")), 0)
            capacity = reviewer_capacity_status(reviewer, load).get("capacity", 1)
            availability = max(0.0, 1.0 - (float(load or 0) / max(1.0, float(capacity or 1))))
            if reviewer_capacity_status(reviewer, load).get("hard_excluded"):
                reviewer["final_score"] = float("-inf")
            else:
                reviewer["availability_score"] = availability
                reviewer["final_score"] = max(0.0, reviewer["final_score"] * availability)

    if USE_CROSS and CROSS_TOP_K > 0:
        head = candidates[: min(CROSS_TOP_K, len(candidates))]
        tail = candidates[min(CROSS_TOP_K, len(candidates)) :]
        head.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        candidates = head + tail
    else:
        candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    _debug_routing(signals, candidates)
    return candidates


def compute_stats(reviewers):
    h_vals = [get_h_index(r) for r in reviewers]
    c_vals = [get_citations(r) for r in reviewers]
    b_vals = [safe_get(r, "behavior_score") for r in reviewers]
    positive_h = [h for h in h_vals if h > 0]
    positive_c = [c for c in c_vals if c > 0]

    return {
        "h_min": min(h_vals) if h_vals else 0,
        "h_max": max(h_vals) if h_vals else 0,
        "h_p25": percentile(h_vals, 0.25) if h_vals else 0,
        "h_p75": percentile(h_vals, 0.75) if h_vals else 0,
        "c_min": min(c_vals) if c_vals else 0,
        "c_max": max(c_vals) if c_vals else 0,
        "b_min": min(b_vals) if b_vals else 0,
        "b_max": max(b_vals) if b_vals else 0,
        "h_avg": (sum(positive_h) / len(positive_h)) if positive_h else 0,
        "c_avg": (sum(positive_c) / len(positive_c)) if positive_c else 0,
    }


def compute_authority(r, stats):
    h = get_h_index(r)
    c = get_citations(r)
    if not _has_metric_value(r, ("h_index", "h_index_proxy", "hindex", "hIndex")):
        h = stats.get("h_avg", h)
        r["h_index"] = round(h, 1)
        r["metadata_fallback"] = True
    if not _has_metric_value(r, ("citations", "total_citations", "citations_total", "numCitations")):
        c = stats.get("c_avg", c)
        r["citations"] = int(round(c))
        r["metadata_fallback"] = True

    h_n = normalize(h, stats["h_min"], stats["h_max"]) if stats["h_max"] != stats["h_min"] else 0.0
    c_n = log_normalize(c, stats["c_min"], stats["c_max"]) if stats["c_max"] != stats["c_min"] else 0.0

    return min(0.80, 0.7 * h_n + 0.3 * c_n)


def compute_topic_overlap(paper, reviewer):
    if not paper:
        return 0.0, [], 0.0

    paper_text = build_paper_text(paper)
    reviewer_text = build_reviewer_text(reviewer)

    paper_kw = clean_keywords(extract_keyphrases(paper_text))
    reviewer_kw = clean_keywords(extract_keyphrases(reviewer_text))

    paper_kw = enrich_keywords(paper_kw)
    reviewer_kw = enrich_keywords(reviewer_kw)

    lex_overlap = compute_overlap(paper_kw, reviewer_kw)

    if not paper_kw:
        return 0.0, [], 0.0

    if lex_overlap:
        score = len(lex_overlap) / max(len(paper_kw), 1)
        return score, lex_overlap[:3], 0.0

    if reviewer.get("_prefetched_similarity"):
        sem_score = float(reviewer.get("semantic_score", reviewer.get("similarity_score", 0.0)) or 0.0)
        return sem_score * 0.5 if sem_score > SEM_THRESHOLD else 0.0, paper_kw[:3], sem_score
    else:
        try:
            sem_score = semantic_overlap_score(paper_text, reviewer_text)
        except Exception:
            sem_score = 0.0

    try:
        tfidf_sim, terms = semantic_overlap(paper_text, reviewer_text, top_k=5)
    except Exception:
        terms = extract_semantic_terms(paper_text, reviewer_text)

    cleaned_terms = clean_keywords(terms) if terms else []

    ordered = sorted(cleaned_terms, key=lambda x: (-len(x.split()), -len(x)))
    filtered = []
    seen = set()
    for t in ordered:
        toks = t.split()
        if any(tok in seen for tok in toks):
            continue
        filtered.append(t)
        for tok in toks:
            seen.add(tok)

    filtered = filtered[:3]

    if sem_score > SEM_THRESHOLD:
        return sem_score * 0.5, filtered, sem_score
    else:
        return 0.0, filtered, sem_score


def word_match(a, b):
    if a == b:
        return True

    pattern = r'\b' + re.escape(a) + r'\b'
    if re.search(pattern, b):
        return True

    if a.split()[0] == b.split()[0]:
        return True

    return False


def compute_overlap(paper_kw, reviewer_kw):
    """Return list of overlapping terms using word-boundary and phrase heuristics."""
    overlap = []

    for p in paper_kw:
        for r in reviewer_kw:

            if not p or len(p) < 4:
                continue

            if word_match(p, r):
                overlap.append(p)

    return list(set(overlap))


def format_term_list(terms):
    """Nicely format a list of terms: 'a', 'a and b', or 'a, b, and c'."""
    if not terms:
        return ""
    terms = list(terms)
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return f"{', '.join(terms[:-1])}, and {terms[-1]}"


def deduplicate_concepts(concepts):
    """Normalize and remove exact-duplicate concept strings while preserving order."""
    seen = set()
    cleaned = []

    def _norm_keys(token):
        t = token.strip()
        try:
            base = kw.normalize(t)
        except Exception:
            base = t.lower()
        keys = [base]
        if base.endswith("ies") and len(base) > 5:
            keys.append(base[:-3] + "y")
        if base.endswith("s") and len(base) > 4 and not (base.endswith("ss") or base.endswith("sis") or base.endswith("us") or base.endswith("is")):
            keys.append(base[:-1])
        return keys

    for c in concepts or []:
        keys = _norm_keys(c)
        if any(k in seen for k in keys):
            continue
        for k in keys:
            seen.add(k)
        cleaned.append(c)
    return cleaned


def select_strong_concepts(keywords):
    """Prefer multi-word phrase concepts; fallback to top tokens if none found."""
    if not keywords:
        return []
    phrases = [k for k in keywords if len(k) > 4 and " " in k]
    if phrases:
        return phrases[:3]
    return keywords[:3]


def generate_domain_phrase(concepts):
    """Map detected concepts to a clean, publication-quality phrase."""
    if not concepts:
        return "the research domain"
    low = [c.lower() for c in concepts]

    if any("program synthesis" in c or "code generation" in c or "program generation" in c for c in low):
        return "program synthesis and code generation"

    if any("transformer" in c or "attention" in c for c in low):
        if any("decod" in c or "generation" in c for c in low):
            return "transformer-based modeling and decoding"
        return "transformer-based modeling"

    if any("decod" in c or "generat" in c or "inference" in c for c in low):
        return "generation and decoding strategies"

    if any("optim" in c or "efficiency" in c or "performance" in c for c in low):
        return "optimization and efficiency"

    return ", ".join(concepts[:2])


def intersect_concepts(core_concepts, reviewer_terms):
    """Return concepts from core_concepts that appear in reviewer_terms (preserving core order).

    If none match, return the first two core_concepts as a sensible fallback.
    """
    if not core_concepts:
        return select_strong_concepts(reviewer_terms)[:2] or []

    results = []
    for c in core_concepts:
        c_norm = kw.normalize(c).lower()
        for r in reviewer_terms or []:
            r_norm = kw.normalize(r).lower()
            if c_norm == r_norm or c_norm in r_norm or r_norm in c_norm:
                results.append(c)
                break

    if not results:
        results = core_concepts[:2]

    return deduplicate_concepts(results)


def enrich_concepts(reviewer_concepts, paper_core):
    """Inject top paper core concepts into reviewer concepts to ensure shared core.

    Keeps order and deduplicates while preferring the paper core terms.
    """
    paper_core = paper_core or []
    reviewer_concepts = reviewer_concepts or []
    combined = list(dict.fromkeys(paper_core[:2] + reviewer_concepts))
    return combined[:3]


def semantic_overlap(paper_text, reviewer_text, top_k=5):
    """Compute TF-IDF cosine similarity and return top contributing terms for the paper."""
    if not SKLEARN_AVAILABLE:
        return 0.0, []

    docs = [paper_text, reviewer_text]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        vec = vectorizer.fit_transform(docs)
        sim = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])

        feature_names = vectorizer.get_feature_names_out()
        tfidf_arr = vec.toarray()
        if tfidf_arr.shape[1] > 0:
            idx = tfidf_arr[0].argsort()[::-1][:top_k]
            terms = [feature_names[i] for i in idx]
        else:
            terms = []

        return sim, terms
    except Exception:
        return 0.0, []


def semantic_overlap_score(paper_text, reviewer_text):
    """Semantic similarity using sentence-transformers embeddings (MiniLM).

    Falls back to TF-IDF cosine if embeddings are not available.
    """
    if SENTS_AVAILABLE and bi_model is not None:
        try:
            a = _embed_paper_text(paper_text)
            b = _embed_reviewer_text(reviewer_text)
            if a is None or b is None:
                return 0.0
            if SKLEARN_AVAILABLE:
                return float(cosine_similarity([a], [b])[0][0])
            else:
                import numpy as _np
                denom = (_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-12
                return float(_np.dot(a, b) / denom)
        except Exception:
            return 0.0

    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf = vectorizer.fit_transform([paper_text, reviewer_text])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(score)
        except Exception:
            return 0.0

    return 0.0


def extract_semantic_terms(paper_text, reviewer_text):
    """Extract simple shared lexical terms between paper and reviewer texts.

    Keep only useful terms (length > 4) and return up to 3.
    """
    paper_words = set(paper_text.lower().split())
    reviewer_words = set(reviewer_text.lower().split())

    common = paper_words & reviewer_words

    common = [w for w in common if len(w) > 4]

    return list(common)[:3]


def percentile(values, pct):
    values = sorted(float(v or 0.0) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[int(rank)]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def _reviewer_similarity(a, b):
    if a.get("_prefetched_similarity") or b.get("_prefetched_similarity"):
        a_terms = set(str(t).lower() for t in a.get("overlap_terms", []) if t)
        b_terms = set(str(t).lower() for t in b.get("overlap_terms", []) if t)
        union = a_terms | b_terms
        if union:
            return len(a_terms & b_terms) / len(union)
        return 0.0

    a_emb = a.get("_reviewer_embedding")
    b_emb = b.get("_reviewer_embedding")
    if a_emb is None:
        a_emb = _embed_text(a.get("reviewer_text") or build_reviewer_text(a))
    if b_emb is None:
        b_emb = _embed_text(b.get("reviewer_text") or build_reviewer_text(b))
    emb_sim = _cosine_from_embeddings(a_emb, b_emb)
    if emb_sim > 0:
        return max(0.0, min(1.0, emb_sim))

    a_terms = set(str(t).lower() for t in a.get("overlap_terms", []) if t)
    b_terms = set(str(t).lower() for t in b.get("overlap_terms", []) if t)
    union = a_terms | b_terms
    if not union:
        return 0.0
    return len(a_terms & b_terms) / len(union)


def _diversity_penalty(candidate, selected):
    if not selected:
        return 0.0
    return max(_reviewer_similarity(candidate, reviewer) for reviewer in selected)


def _coverage_terms(reviewer):
    terms = []
    terms.extend(reviewer.get("explanation_concepts", []) or [])
    terms.extend(reviewer.get("overlap_terms", []) or [])
    terms.extend(_reviewer_keywords(reviewer, limit=8) or [])
    return {
        kw.normalize(_to_text(term)).lower().strip()
        for term in terms
        if _to_text(term).strip()
    }


def _coverage_gain(candidate, selected):
    candidate_terms = _coverage_terms(candidate)
    if not candidate_terms:
        return 0.0
    selected_terms = set()
    for reviewer in selected:
        selected_terms.update(_coverage_terms(reviewer))
    if not selected_terms:
        return 1.0
    return len(candidate_terms - selected_terms) / max(len(candidate_terms), 1)


def _apply_diversity_adjustment(candidate, selected):
    penalty = _diversity_penalty(candidate, selected)
    coverage_gain = _coverage_gain(candidate, selected)
    base_score = float(candidate.get("base_final_score", candidate.get("final_score", 0.0)) or 0.0)
    diversity_gain = max(0.0, min(1.0, 1.0 - penalty))
    adjusted_base = _cap_assignment_score(base_score - 0.30 * penalty)
    candidate["diversity_penalty"] = penalty
    candidate["diversity_gain"] = max(0.0, min(1.0, diversity_gain))
    candidate["coverage_gain"] = max(0.0, min(1.0, coverage_gain))
    candidate["base_final_score"] = adjusted_base
    candidate["final_score"] = adjusted_base
    candidate["diversity_adjusted_score"] = candidate["final_score"]
    if candidate.get("score_debug"):
        candidate["score_debug"]["diversity_penalty"] = penalty
        candidate["score_debug"]["coverage_gain"] = candidate["coverage_gain"]
        candidate["score_debug"]["final_after_diversity"] = adjusted_base
        candidate["score_debug"]["final"] = adjusted_base
    return candidate


def _constraint_summary(role, reviewer):
    if not reviewer:
        return ""
    role = str(role).lower()
    if role == "expert":
        return "Anchor semantic floor + authority" if reviewer.get("constraint_satisfied") else "Best available anchor fit"
    if role == "moderate":
        return "Above-median similarity" if reviewer.get("constraint_satisfied") else "Best available balanced match"
    if role == "fresher":
        return "Complementary topical coverage with semantic floor" if reviewer.get("constraint_satisfied") else "Best available complementary coverage"
    return "Role constraint checked"


def classify_reviewer(r, stats=None):
    h = get_h_index(r)

    if stats and stats.get("h_p75") is not None and stats.get("h_p25") is not None:
        if h >= float(stats.get("h_p75") or 0):
            return "expert"
        if h <= float(stats.get("h_p25") or 0):
            return "fresher"
        return "moderate"

    if h >= 8:
        return "expert"
    elif h >= 4:
        return "moderate"
    else:
        return "fresher"


def reviewer_capacity(reviewer, stats=None):
    role = str(reviewer.get("role") or classify_reviewer(reviewer, stats)).lower()
    authority = float(reviewer.get("authority_score", 0.0) or 0.0)

    if role == "expert":
        return SENIOR_REVIEWER_CAPACITY if authority >= 0.65 else SENIOR_AUTHORITY_CAPACITY
    if role == "moderate":
        return MID_REVIEWER_CAPACITY
    if role == "fresher":
        return JUNIOR_REVIEWER_CAPACITY
    return DEFAULT_REVIEWER_CAPACITY


def reviewer_capacity_status(reviewer, current_load=0, stats=None):
    capacity = max(1, int(reviewer_capacity(reviewer, stats)))
    load = max(0, int(current_load or 0))
    ratio = load / capacity
    if load >= capacity:
        status = "at_capacity"
    elif ratio >= NEAR_CAPACITY_RATIO:
        status = "near_capacity"
    else:
        status = "available"
    return {
        "capacity": capacity,
        "current_load": load,
        "load_ratio": ratio,
        "capacity_status": status,
        "hard_excluded": bool(ENABLE_HARD_CAPACITY_EXCLUSION and load >= capacity),
    }


def select_role_balanced_reviewers(reviewers, stats, top_n=3):
    valid = [
        r for r in reviewers
        if r.get("final_score") != float("-inf")
        and not r.get("semantic_threshold_failed")
        and float(r.get("normalized_similarity", 0.0) or 0.0) >= PANEL_MIN_SIMILARITY
        and float(r.get("final_score", 0.0) or 0.0) >= PANEL_MIN_FINAL_SCORE
    ]
    valid.sort(key=lambda x: x.get("normalized_similarity", x.get("similarity_score", 0.0)), reverse=True)
    if not valid:
        return []

    semantic_window = valid[: min(max(12, top_n * 6), len(valid))]
    similarity_median = percentile([r.get("normalized_similarity", r.get("similarity_score", 0.0)) for r in valid], 0.50)
    authority_median = percentile([r.get("authority_score", 0.0) for r in valid], 0.50)
    primary_similarity_floor = max(PRIMARY_SIMILARITY_FLOOR, similarity_median)

    for reviewer in valid:
        reviewer["role"] = classify_reviewer(reviewer, stats)

    primary_pool = [
        r for r in semantic_window
        if float(r.get("normalized_similarity", 0.0) or 0.0) >= primary_similarity_floor
    ]
    if not primary_pool:
        primary_pool = [
            r for r in semantic_window
            if float(r.get("normalized_similarity", 0.0) or 0.0) >= PRIMARY_SIMILARITY_FLOOR
        ]
    if not primary_pool:
        primary_pool = semantic_window[: min(3, len(semantic_window))]
    expert_pool = [
        r for r in primary_pool
        if r.get("role") == "expert"
    ]
    expert = max(
        expert_pool or primary_pool,
        key=lambda r: (
            float(r.get("authority_score", 0.0) or 0.0),
            float(r.get("normalized_similarity", 0.0) or 0.0),
        ),
        default=None,
    )
    if expert:
        expert["role"] = "expert"
        expert["constraint_satisfied"] = (
            float(expert.get("normalized_similarity", 0.0) or 0.0) >= primary_similarity_floor
            and bool(expert_pool and expert in expert_pool)
        )
        expert["constraint_label"] = _constraint_summary("expert", expert)
        expert["primary_similarity_floor"] = primary_similarity_floor

    remaining = [r for r in valid if not expert or r.get("reviewer_id") != expert.get("reviewer_id")]
    moderate_pool = [
        r for r in remaining[: min(max(15, top_n * 6), len(remaining))]
        if float(r.get("normalized_similarity", 0.0) or 0.0) >= similarity_median
        and (
            r.get("role") == "moderate"
            or float(r.get("authority_score", 0.0) or 0.0) >= max(0.25, authority_median)
        )
    ]
    moderate_fallback = [
        r for r in remaining
        if r.get("role") in ("moderate", "expert")
        or float(r.get("authority_score", 0.0) or 0.0) >= max(0.25, authority_median)
    ]
    moderate_candidates = list(moderate_pool or moderate_fallback or remaining)
    for candidate in moderate_candidates:
        selected_anchor = [expert] if expert else []
        redundancy = _diversity_penalty(candidate, selected_anchor)
        coverage_gain = _coverage_gain(candidate, selected_anchor)
        candidate["diversity_penalty"] = max(0.0, min(1.0, redundancy))
        candidate["diversity_gain"] = max(0.0, min(1.0, 1.0 - redundancy))
        candidate["coverage_gain"] = max(0.0, min(1.0, coverage_gain))
        relevance = (
            0.68 * float(candidate.get("normalized_similarity", 0.0) or 0.0)
            + 0.32 * float(candidate.get("authority_score", 0.0) or 0.0)
        )
        candidate["panel_marginal_score"] = (
            0.64 * relevance
            + 0.28 * candidate["coverage_gain"]
            - 0.12 * candidate["diversity_penalty"]
        )
        if candidate.get("score_debug"):
            candidate["score_debug"]["diversity_penalty"] = candidate["diversity_penalty"]
            candidate["score_debug"]["coverage_gain"] = candidate["coverage_gain"]
            candidate["score_debug"]["mmr_score"] = candidate["panel_marginal_score"]
    moderate = max(
        moderate_candidates,
        key=lambda r: (
            float(r.get("panel_marginal_score", 0.0) or 0.0),
            float(r.get("normalized_similarity", 0.0) or 0.0),
        ),
        default=None,
    )
    if moderate:
        moderate["role"] = "moderate"
        moderate["constraint_satisfied"] = bool(moderate_pool and moderate in moderate_pool)
        moderate["constraint_label"] = _constraint_summary("moderate", moderate)

    selected_head = [r for r in [expert, moderate] if r]
    used_ids = {str(r.get("reviewer_id")) for r in selected_head}
    fresher_pool = [
        r for r in valid
        if str(r.get("reviewer_id")) not in used_ids
        and (
            r.get("role") == "fresher"
            or float(r.get("authority_score", 0.0) or 0.0) <= authority_median
        )
        and float(r.get("normalized_similarity", 0.0) or 0.0) >= max(SUPPORTING_SIMILARITY_FLOOR, similarity_median * 0.75)
        and float(r.get("authority_score", 0.0) or 0.0) >= SUPPORTING_AUTHORITY_FLOOR
    ]
    fresher_fallback = [
        r for r in remaining
        if float(r.get("normalized_similarity", 0.0) or 0.0) >= SUPPORTING_SIMILARITY_FLOOR
        and float(r.get("authority_score", 0.0) or 0.0) >= SUPPORTING_AUTHORITY_FLOOR
    ]
    fresher_ranked = [_apply_diversity_adjustment(candidate, selected_head) for candidate in (fresher_pool or fresher_fallback)]
    mmr_lambda = 0.64
    for candidate in fresher_ranked:
        relevance = (
            0.70 * float(candidate.get("normalized_similarity", 0.0) or 0.0)
            + 0.30 * float(candidate.get("authority_score", 0.0) or 0.0)
        )
        redundancy = float(candidate.get("diversity_penalty", 0.0) or 0.0)
        candidate["mmr_score"] = (
            mmr_lambda * relevance
            - (1.0 - mmr_lambda) * redundancy
            + 0.22 * float(candidate.get("coverage_gain", 0.0) or 0.0)
        )
        candidate["fresher_selection_score"] = candidate["mmr_score"]
    fresher_ranked.sort(key=lambda x: x.get("fresher_selection_score", 0.0), reverse=True)

    fresher = next((r for r in fresher_ranked if str(r.get("reviewer_id")) not in used_ids), None)
    if fresher:
        fresher["role"] = "fresher"
        fresher["constraint_satisfied"] = (
            float(fresher.get("diversity_gain", 0.0) or 0.0) >= 0.20
            and float(fresher.get("normalized_similarity", 0.0) or 0.0) >= SUPPORTING_SIMILARITY_FLOOR
            and float(fresher.get("authority_score", 0.0) or 0.0) >= SUPPORTING_AUTHORITY_FLOOR
        )
        fresher["constraint_label"] = "Complementary coverage reliability constraint" if fresher.get("constraint_satisfied") else _constraint_summary("fresher", fresher)
        if fresher.get("score_debug"):
            fresher["score_debug"]["mmr_score"] = fresher.get("mmr_score", 0.0)

    selected = [x for x in [expert, moderate, fresher] if x]
    if len(selected) < top_n:
        for candidate in valid:
            if candidate not in selected:
                if len(selected) >= 2 and (
                    float(candidate.get("normalized_similarity", 0.0) or 0.0) < SUPPORTING_SIMILARITY_FLOOR
                    or float(candidate.get("authority_score", 0.0) or 0.0) < SUPPORTING_AUTHORITY_FLOOR
                ):
                    candidate.setdefault("rejection_reasons", []).append(
                        "selected as fallback to complete requested panel size"
                    )
                    candidate["constraint_satisfied"] = False
                    candidate["constraint_label"] = "Fallback panel completion"
                selected.append(candidate)
            if len(selected) >= top_n:
                break

    if len(selected) < top_n:
        selected_ids = {str(r.get("reviewer_id")) for r in selected}
        fallback_pool = [
            r for r in reviewers
            if str(r.get("reviewer_id")) not in selected_ids
            and r.get("final_score") != float("-inf")
        ]
        fallback_pool.sort(
            key=lambda r: (
                float(r.get("normalized_similarity", r.get("similarity_score", 0.0)) or 0.0),
                float(r.get("final_score", 0.0) or 0.0),
                float(r.get("authority_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for candidate in fallback_pool:
            candidate.setdefault("rejection_reasons", []).append(
                "selected as fallback to complete requested panel size"
            )
            candidate["constraint_satisfied"] = False
            candidate["constraint_label"] = "Fallback panel completion"
            selected.append(candidate)
            if len(selected) >= top_n:
                break

    return selected


def build_diagnostic_context(reviewers):
    valid = [
        r for r in reviewers
        if isinstance(r.get("final_score", 0.0), (int, float))
        and r.get("final_score", 0.0) != float("-inf")
    ]
    sims = [float(r.get("similarity_score", 0.0) or 0.0) for r in valid]
    auths = [float(r.get("authority_score", 0.0) or 0.0) for r in valid]
    kgs = [float(r.get("kg_score", 0.0) or 0.0) for r in valid]

    return {
        "avg_similarity": (sum(sims) / len(sims)) if sims else 0.0,
        "top_similarity": max(sims) if sims else 0.0,
        "avg_authority": (sum(auths) / len(auths)) if auths else 0.0,
        "avg_kg": (sum(kgs) / len(kgs)) if kgs else 0.0,
    }


def apply_role_ordered_scores(selected):
    previous_score = None

    for reviewer in selected:
        role = reviewer.get("role", classify_reviewer(reviewer))
        base_score = float(reviewer.get("base_final_score", reviewer.get("raw_final_score", 0.0)) or 0.0)
        role_bias = 0.0
        role_adjusted = _cap_assignment_score(base_score)

        if previous_score is not None and role_adjusted >= previous_score:
            role_adjusted = _cap_assignment_score(previous_score - 0.03)

        reviewer["role_bias"] = role_bias
        reviewer["base_final_score"] = base_score
        reviewer["assignment_score"] = role_adjusted
        reviewer["final_score"] = role_adjusted
        previous_score = role_adjusted

    return selected


def build_weakness(r, context=None):
    context = context or {}
    role = r.get("role", classify_reviewer(r))
    authority = float(r.get("authority_score", 0) or 0)
    similarity = float(r.get("similarity_score", 0) or 0)
    kg_score = float(r.get("kg_score", 0) or 0)
    avg_authority = float(context.get("avg_authority", 0.0) or 0.0)

    if role == "expert":
        if similarity < 0.40:
            return "Low topic similarity"
        if authority < max(0.45, avg_authority * 0.80):
            return "Lower academic authority for expert role"

    elif role == "moderate":
        if 0.30 <= similarity <= 0.50:
            return "Moderate alignment"
        if similarity < 0.30 and (not USE_KG or kg_score < 0.15):
            return "Limited topic alignment"
        if authority < 0.25:
            return "Lower academic authority"

    else:
        if similarity < 0.25 and (not USE_KG or kg_score < 0.15):
            return "Exploratory coverage"
        if get_citations(r) < 20:
            return "Exploratory coverage with limited citation evidence"

    if role == "expert":
        return "Strong authority profile with sufficient topic alignment"
    if role == "moderate":
        return "Balanced topic alignment with sufficient authority"
    return "Relevant topical exposure with acceptable reliability evidence"


def build_comparison_reason(selected, all_reviewers):
    comparison_map = {}

    for r in selected:
        better_than = []

        for other in all_reviewers:
            if other["reviewer_id"] == r["reviewer_id"]:
                continue

            if r["final_score"] > other["final_score"]:
                better_than.append(other)

        comparison_map[r["reviewer_id"]] = better_than[:2]

    return comparison_map


def build_fast_reason(r):
    overlap = r.get("overlap_terms", [])[:3]

    if overlap:
        overlap_text = ", ".join(overlap)
        topic_part = f"topic alignment ({overlap_text})"
    else:
        topic_part = "general domain relevance (no strong keyword overlap)"

    sim = round(r.get("similarity_score", 0), 2)
    auth = round(r.get("authority_score", 0), 2)
    role = r.get("role", "")

    if role == "expert":
        return (
            f"Strong reviewer profile with {topic_part}, "
            f"supported by strong academic authority ({auth}) and similarity ({sim})."
        )

    elif role == "moderate":
        return (
            f"Good domain match with {topic_part} and balanced authority ({auth}) "
            f"and competitive similarity ({sim})."
        )

    else:
        return (
            f"Relevant {topic_part} with acceptable similarity ({sim}), "
            f"with complementary coverage and authority evidence ({auth})."
        )


def add_comparison(selected, all_reviewers):

    for r in selected:
        better_than = [
            o for o in all_reviewers
            if o["reviewer_id"] != r["reviewer_id"]
            and r["final_score"] > o["final_score"]
        ]

        if better_than:
            r["reason"] += f" Ranked higher than {len(better_than)} candidates due to stronger combination of domain relevance and academic authority."


def assign_reviewers(reviewers, paper=None, ground_truth=None, scoring_mode="full", reviewer_usage=None):

    if not reviewers:
        return []

    stats = compute_stats(reviewers)

    for r in reviewers:
        r["authority_score"] = compute_authority(r, stats)

    paper_core_concepts = []
    paper_text = build_paper_text(paper) if paper else ""
    paper_id = str(paper.get("paper_id")) if paper and paper.get("paper_id") is not None else ""
    signal_cache_key = hashlib.sha256(
        json.dumps(
            {
                "paper_id": paper_id,
                "paper_text": paper_text,
                "signal_version": "instruction_minmax_role_first_v2",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    signals = ["semantic"]
    if scoring_mode == "full":
        signals.append("authority")
        signals.append("kg")
        signals.append("llm")
        signals.append("workload")
    elif scoring_mode == "semantic_kg":
        signals.append("kg")
    elif scoring_mode == "semantic_llm":
        signals.append("llm")
    elif scoring_mode == "semantic_kg_llm":
        signals.append("kg")
        signals.append("llm")
    paper_profile = generate_manuscript_profile(paper) if paper else {}
    if paper:
        pk = clean_keywords(extract_keyphrases(paper_text))
        pk = enrich_keywords(pk)
        paper_core_concepts = select_strong_concepts(pk)
        paper_core_concepts = deduplicate_concepts(paper_core_concepts)
        if not paper_core_concepts:
            paper_core_concepts = (pk or [])[:2]

    for r in reviewers:
        if (
            r.get("_paper_signal_cache_key") == signal_cache_key
            and "similarity_score" in r
            and "semantic_score" in r
            and "overlap_terms" in r
        ):
            continue

        reviewer_text = build_reviewer_text(r)
        r["reviewer_text"] = reviewer_text
        if r.get("_prefetched_similarity") and "similarity_score" in r:
            live_similarity = float(r.get("similarity_score", 0.0) or 0.0)
        else:
            live_similarity = semantic_overlap_score(paper_text, reviewer_text) if paper_text and reviewer_text else 0.0
        r["similarity_score"] = live_similarity

        overlap_score, overlap_terms, sem_score = compute_topic_overlap(paper, r)

        r["overlap_score"] = overlap_score
        r["semantic_score"] = live_similarity if live_similarity else sem_score
        r["overlap_terms"] = [
            t for t in overlap_terms
            if t not in GENERIC_TERMS and len(t) > 3
        ]

        r["overlap_terms"] = deduplicate_concepts(r.get("overlap_terms", []))

        norms = {t.strip().lower() for t in r["overlap_terms"]}
        if "conversational" in norms:
            new_terms = []
            for t in r["overlap_terms"]:
                if t.strip().lower() == "conversational":
                    if "dialog systems" not in norms:
                        new_terms.append("dialog systems")
                        norms.add("dialog systems")
                    if "conversational interfaces" not in norms:
                        new_terms.append("conversational interfaces")
                        norms.add("conversational interfaces")
                else:
                    new_terms.append(t)
            r["overlap_terms"] = deduplicate_concepts(new_terms)

        reviewer_terms = r.get("overlap_terms", [])
        reviewer_concepts = intersect_concepts(paper_core_concepts, reviewer_terms)
        r["explanation_concepts"] = reviewer_concepts
        r["_paper_signal_cache_key"] = signal_cache_key

    sorted_by_sim = sorted(
        reviewers,
        key=lambda x: x.get("similarity_score", 0),
        reverse=True,
    )
    for idx, reviewer in enumerate(sorted_by_sim, start=1):
        reviewer["initial_semantic_rank"] = idx

    candidate_limit = int(os.getenv("ASSIGNMENT_CANDIDATE_LIMIT", str(ASSIGNMENT_CANDIDATE_LIMIT)))
    candidates = sorted_by_sim[:candidate_limit] if candidate_limit > 0 else sorted_by_sim
    candidate_ids = {str(r.get("reviewer_id")) for r in candidates}

    kg_scores = {}
    if paper and "kg" in signals:
        kg_limit = min(KG_TOP_K, len(candidates))
        kg_scores = _generate_candidate_kg_scores(paper_id, paper_text, candidates, limit=kg_limit)
        kg_scores = _normalize_candidate_kg_scores(paper, candidates[:kg_limit], kg_scores)
    for r in reviewers:
        r["kg_score"] = float(kg_scores.get(str(r.get("reviewer_id")), 0.0) or 0.0)

    gt_ids = None
    if ground_truth and paper_id:
        gt_ids = ground_truth.get(paper_id)
    if gt_ids is None and paper:
        gt_ids = paper.get("ground_truth_reviewer_ids")
    if gt_ids is None and paper:
        gt_ids = paper.get("reviewer_ids")
    if gt_ids:
        top_100 = [str(r["reviewer_id"]) for r in sorted_by_sim[:100]]
        gt = [str(rid) for rid in gt_ids]
        overlap = set(top_100) & set(gt)
        print("Overlap:", overlap)

    normalized_sims = _minmax_scores([r.get("similarity_score", 0) for r in candidates])

    for llm_rank_idx, (r, sim) in enumerate(zip(candidates, normalized_sims), start=1):
        authority = compute_authority(r, stats)

        kg_score = float(r.get("kg_score", 0.0) or 0.0)
        r["routing_signals"] = signals
        r["role"] = classify_reviewer(r, stats)
        use_llm_suitability = "llm" in signals and llm_rank_idx <= max(1, int(LLM_SUITABILITY_TOP))
        reviewer_profile = generate_reviewer_profile(r) if use_llm_suitability else {}
        llm_reasoning = _llm_suitability_reasoning(paper_profile, reviewer_profile) if use_llm_suitability else {
            "final_suitability": 0.0,
            "overlap_terms": [],
            "source": "outside-llm-rerank-window" if "llm" in signals else "disabled",
        }
        llm_match = float(llm_reasoning.get("final_suitability", 0.0) or 0.0)
        llm_overlap = llm_reasoning.get("overlap_terms", [])
        r["manuscript_profile"] = paper_profile
        r["reviewer_profile"] = reviewer_profile
        r["llm_suitability_reasoning"] = llm_reasoning
        r["llm_suitability_score"] = llm_match
        r["llm_match_score"] = llm_match
        r["llm_profile_overlap"] = llm_overlap
        r["llm_rerank_rank"] = llm_rank_idx
        r["llm_rerank_window"] = max(1, int(LLM_SUITABILITY_TOP))
        r["normalized_similarity"] = max(0.0, min(1.0, float(sim or 0.0)))
        r["authority_score"] = max(0.0, min(1.0, float(authority or 0.0)))
        r["kg_score"] = max(0.0, min(1.0, float(kg_score or 0.0)))
        r["scoring_mode"] = scoring_mode
        r["kg_confidence"] = _kg_confidence_for_candidate(r)
        r["relevance_confidence"] = _relevance_confidence(r)
        r["coi"] = _coi_status(paper, r)
        workload_score = 1.0
        if reviewer_usage:
            load = int((reviewer_usage or {}).get(str(r.get("reviewer_id")), 0) or 0)
            capacity = reviewer_capacity_status(r, load, stats)
            workload_score = max(0.0, 1.0 - float(capacity.get("load_ratio", 0.0) or 0.0))
            r["current_load"] = load
            r["availability_score"] = workload_score
            r["workload_capacity_status"] = capacity
        r["semantic_threshold_failed"] = (
            scoring_mode == "full"
            and r["normalized_similarity"] < SEMANTIC_REJECTION_THRESHOLD
        )

        score = _hybrid_score(
            r["normalized_similarity"],
            r["authority_score"],
            r["kg_score"],
            mode=scoring_mode,
            kg_confidence=r["kg_confidence"],
            role=r["role"],
            llm_match=r["llm_match_score"],
            workload_score=workload_score,
        )
        contributions = _score_contributions(
            r["normalized_similarity"],
            r["authority_score"],
            r["kg_score"],
            scoring_mode,
            r["kg_confidence"],
            r["role"],
            r["llm_match_score"],
            workload_score,
        )
        r["score_debug"] = {
            "semantic_raw": float(r.get("similarity_score", 0.0) or 0.0),
            "semantic_norm": r["normalized_similarity"],
            "authority": r["authority_score"],
            "kg": r["kg_score"],
            "llm_match": r["llm_match_score"],
            "llm_suitability_score": r["llm_suitability_score"],
            "llm_suitability_reasoning": r["llm_suitability_reasoning"],
            "llm_profile_overlap": r["llm_profile_overlap"],
            "workload_score": workload_score,
            "manuscript_profile_source": paper_profile.get("source", ""),
            "reviewer_profile_source": reviewer_profile.get("source", ""),
            "kg_confidence": r["kg_confidence"],
            "kg_weight_reduced": "kg" in signals and contributions["weights"].get("kg", 0.0) < contributions["weights"].get("base_kg", 0.15),
            "kg_capped": contributions.get("kg_capped", False),
            "kg_disabled": contributions.get("kg_disabled", False),
            "kg_effective": contributions.get("kg_effective", 0.0),
            "semantic_rejected": contributions.get("semantic_rejected", False),
            "relevance_confidence": r["relevance_confidence"],
            "relevance_confidence_label": _relevance_confidence_label(r["relevance_confidence"]),
            "dominant_signal": _dominant_signal_label(contributions),
            "contributions": contributions,
            "final_before_policy": score,
        }
        score = _apply_reviewer_score_policy(r, score)
        r["score_debug"]["final"] = score
        r["rejection_reasons"] = _rejection_reasons(r, scoring_mode)
        if r["coi"].get("flagged"):
            r["rejection_reasons"].append(
                "conflict of interest: " + ", ".join(r["coi"].get("reasons") or ["COI rule"])
            )
        r["threshold_checks"] = _threshold_checks(r, scoring_mode)
        r["base_final_score"] = max(0.0, score)
        r["role_bias"] = 0.0
        r["raw_final_score"] = _cap_assignment_score(score)
        r["final_score"] = r["raw_final_score"]
        if reviewer_usage:
            capacity = r.get("workload_capacity_status") or reviewer_capacity_status(r, r.get("current_load", 0), stats)
            if capacity.get("hard_excluded"):
                r["rejection_reasons"].append("reviewer at workload capacity")
                r["final_score"] = float("-inf")
        if r["coi"].get("flagged"):
            r["final_score"] = float("-inf")
        r["routing_signals"] = _display_signals(signals, r)

    if USE_RRF_FUSION and scoring_mode == "full":
        _apply_rrf_fusion(candidates)

    for r in reviewers:
        if candidate_limit > 0 and str(r.get("reviewer_id")) not in candidate_ids:
            r["final_score"] = float("-inf")

    reviewers.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    total_ranked_candidates = max(1, len([r for r in reviewers if r.get("final_score") != float("-inf")]))
    for idx, reviewer in enumerate(reviewers, start=1):
        if reviewer.get("final_score") == float("-inf"):
            continue
        reviewer["final_rank"] = idx
        reviewer["rank_percentile"] = idx / total_ranked_candidates
        reviewer["rank_margin"] = reviewer["rank_percentile"]
    _debug_routing(signals, reviewers)
    selected = select_role_balanced_reviewers(reviewers, stats, top_n=3)
    for idx, reviewer in enumerate(selected, start=1):
        reviewer["final_rank"] = idx
        reviewer["rank_margin"] = reviewer.get("rank_percentile", idx / max(1, len(selected)))
    diagnostic_context = build_diagnostic_context(reviewers)

    comparison_map = build_comparison_reason(selected, reviewers)

    for r in selected:
        r["comparison"] = comparison_map.get(r["reviewer_id"], [])

    ENABLE_LLM = os.getenv("ENABLE_LLM_ENHANCEMENT", "false").lower() in ("1", "true", "yes")

    for r in selected:
        sim = r.get("similarity_score", 0)
        auth = r.get("authority_score", 0)
        final = r.get("final_score", 0)

        paper_title = paper.get("title", "") if paper else ""
        top_pub = ""
        if r.get("publications"):
            top_pub = r.get("publications", [{}])[0].get("title", "") or ""

        h_index = get_h_index(r)
        citations = get_citations(r)
        overlap_terms = r.get("overlap_terms", [])[:5]
        sem_score = r.get("semantic_score", 0)

        concepts = r.get("explanation_concepts") or select_strong_concepts(overlap_terms)
        concepts = deduplicate_concepts(concepts)
        domain_phrase = generate_domain_phrase(concepts)

        role = r.get("role", classify_reviewer(r, stats))

        anchor = concepts[0] if concepts else "the research domain"
        top_two = ", ".join(concepts[:2]) if concepts else domain_phrase

        rank_position = sum(1 for other in reviewers if final > other.get("final_score", 0))

        if role == "expert":
            reason = (
                f"Selected as primary because the authority evidence is strong "
                f"(h-index={h_index}, citations={citations}) and the profile connects to {domain_phrase}. "
            )
            if anchor != "the research domain":
                reason += f"The strongest topical anchor is {anchor}. "
            if diagnostic_context.get("top_similarity", 0.0) > sim:
                reason += "A higher raw semantic match exists, but this slot prioritizes a senior reviewer who can lead the assessment. "
            reason += f"The combined evidence placed this reviewer above {rank_position} candidates for the primary role."

        elif role == "moderate":
            reason = (
                f"Selected as secondary for a balanced profile: semantic fit around {domain_phrase} "
                f"with usable authority evidence (h-index={h_index}). "
                f"The role favors a reliable technical match after the primary authority slot is filled. "
                f"This reviewer adds methodological coverage without duplicating the primary reviewer too closely."
            )

        else:
            reason = (
                f"Selected as supporting reviewer because the semantic evidence is relevant to {domain_phrase} "
                f"even though seniority is limited (h-index={h_index}). "
                "This reviewer maximizes complementary topical coverage absent from the primary reviewers. "
                "The final score is lower than the similarity score because this role applies a reliability penalty for limited authority. "
                f"The reviewer is useful for complementary coverage, with senior reviewers carrying the main decision weight."
            )

        comparison = r.get("comparison") or []
        if comparison:
            strongest = max(
                comparison,
                key=lambda other: float(other.get("normalized_similarity", other.get("similarity_score", 0.0)) or 0.0),
            )
            reason += (
                f" Selected over reviewer {strongest.get('reviewer_id')} due to stronger role fit "
                f"and final calibrated score ({final:.2f} vs {float(strongest.get('final_score', 0.0) or 0.0):.2f})."
            )

        if top_pub:
            example = (top_pub.split(":")[0] or top_pub)[:50]
            reason += f" Example work: '{example}...'"

        r["reason"] = reason

        r["weakness"] = build_weakness(r, diagnostic_context)

        if paper and ENABLE_LLM:
            def _enhance(p, rv, s, a, f):
                try:
                    llm_reason = generate_reviewer_reason(p, rv, s, a, f)
                    if llm_reason:
                        rv["reason"] = llm_reason
                except Exception as e:
                    print("âš ï¸ LLM enhancement failed:", e)

            t = threading.Thread(target=_enhance, args=(paper, r, sim, auth, final), daemon=True)
            t.start()


    return selected

if __name__ == "__main__":

    reviewers = [
        {
            "reviewer_id": "A",
            "h_index": 9,
            "citations": 383,
            "behavior_score": 49,
            "similarity_score": 0.85,
            "publications": [{"title": "Knowledge Graph Systems"}]
        },
        {
            "reviewer_id": "B",
            "h_index": 6,
            "citations": 200,
            "behavior_score": 40,
            "similarity_score": 0.78,
            "publications": [{"title": "Recommender Systems"}]
        },
        {
            "reviewer_id": "C",
            "h_index": 2,
            "citations": 50,
            "behavior_score": 20,
            "similarity_score": 0.60,
            "publications": [{"title": "Data Mining Techniques"}]
        },
    ]

    paper = {
        "title": "Knowledge Graph based Recommendation System",
        "abstract": "This paper focuses on semantic recommendation using knowledge graphs."
    }

    result = assign_reviewers(reviewers, paper)

    print("\nFINAL SELECTION:\n")
    for r in result:
        print(
            r["role"],
            r["reviewer_id"],
            "Panel-adjusted score:", round(r["final_score"], 3),
            "|", r["reason"]
        )
