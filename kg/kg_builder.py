import hashlib
import json
import os
import pickle
import re

import networkx as nx


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

PERSISTENT_KG_PATH = os.path.join(OUTPUT_DIR, "persistent_reviewer_kg.pkl")
PERSISTENT_KG_META_PATH = os.path.join(OUTPUT_DIR, "persistent_reviewer_kg_meta.json")


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _stable_text(value):
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _slug(value, limit=96):
    text = _stable_text(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:limit]


def _node(prefix, value):
    slug = _slug(value)
    return f"{prefix}_{slug}" if slug else ""


def _extract_entities(value):
    if isinstance(value, dict):
        labels = value.get("labels")
        if labels is None:
            labels = value.get("all_labels")
        if labels is None and isinstance(value.get("weights"), dict):
            labels = list(value["weights"].keys())
        value = labels or []

    if isinstance(value, str):
        value = [value]

    entities = []
    seen = set()
    for entity in value or []:
        normalized = _stable_text(entity).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        entities.append(normalized)
    return entities


def _topic_values(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("weights"), dict):
            return list(payload["weights"].keys())
        return payload.get("labels") or payload.get("all_labels") or []
    return payload or []


def _publication_key(reviewer_id, publication):
    title = _stable_text(publication.get("title", "")).strip()
    if not title:
        return ""
    digest = hashlib.sha1(f"{reviewer_id}:{title}".encode("utf-8")).hexdigest()[:12]
    return f"pub_{digest}"


def _add_edge(graph, source, target, edge_type, weight=1.0, **attrs):
    if not source or not target:
        return
    if graph.has_edge(source, target):
        graph[source][target]["weight"] = max(float(graph[source][target].get("weight", 0.0)), float(weight))
        edge_types = set(graph[source][target].get("edge_types", []))
        edge_types.add(edge_type)
        graph[source][target]["edge_types"] = sorted(edge_types)
        return
    graph.add_edge(source, target, edge_type=edge_type, edge_types=[edge_type], weight=float(weight), **attrs)


def build_graph(papers, reviewers):
    """Backward-compatible lightweight entity graph used by older callers."""
    graph = nx.Graph()

    for pid, ents in (papers or {}).items():
        paper_node = f"p_{pid}"
        graph.add_node(paper_node, node_type="paper", label=str(pid))
        for entity in _extract_entities(ents):
            entity_node = _node("e", entity)
            graph.add_node(entity_node, node_type="entity", label=entity)
            _add_edge(graph, paper_node, entity_node, "has_entity")

    for rid, ents in (reviewers or {}).items():
        reviewer_node = f"r_{rid}"
        graph.add_node(reviewer_node, node_type="reviewer", label=str(rid))
        for entity in _extract_entities(ents):
            entity_node = _node("e", entity)
            graph.add_node(entity_node, node_type="entity", label=entity)
            _add_edge(graph, reviewer_node, entity_node, "has_entity")

    return graph


def _add_topics(graph, owner_node, topics, source):
    for topic in _extract_entities(topics):
        topic_node = _node("t", topic)
        if not topic_node:
            continue
        graph.add_node(topic_node, node_type="topic", label=topic)
        _add_edge(graph, owner_node, topic_node, "has_topic", weight=1.0, source=source)


def _add_authors(graph, paper_node, authors):
    for author in authors or []:
        author_node = _node("a", author)
        if not author_node:
            continue
        graph.add_node(author_node, node_type="author", label=_stable_text(author))
        _add_edge(graph, paper_node, author_node, "authored_by", weight=1.0)


def _add_reviewer_publications(graph, reviewer):
    reviewer_id = str(reviewer.get("reviewer_id", "")).strip()
    reviewer_node = f"r_{reviewer_id}"
    graph.add_node(
        reviewer_node,
        node_type="reviewer",
        label=reviewer_id,
        h_index=reviewer.get("h_index") or reviewer.get("h_index_proxy") or reviewer.get("hIndex") or 0,
        citations=reviewer.get("citations") or reviewer.get("total_citations") or reviewer.get("numCitations") or 0,
        behavior_score=reviewer.get("behavior_score", 0.0),
    )

    affiliation = reviewer.get("affiliation") or reviewer.get("institution") or reviewer.get("organization")
    affiliation_node = _node("i", affiliation)
    if affiliation_node:
        graph.add_node(affiliation_node, node_type="institution", label=_stable_text(affiliation))
        _add_edge(graph, reviewer_node, affiliation_node, "affiliated_with", weight=1.0)

    for keyword in reviewer.get("keywords", []) or []:
        _add_topics(graph, reviewer_node, [keyword], "reviewer_keyword")

    for publication in reviewer.get("publications", []) or []:
        pub_node = _publication_key(reviewer_id, publication)
        if not pub_node:
            continue
        graph.add_node(
            pub_node,
            node_type="publication",
            label=_stable_text(publication.get("title", "")),
            citations=publication.get("citations", 0),
        )
        _add_edge(graph, reviewer_node, pub_node, "authored", weight=1.0)

        venue = publication.get("venue") or publication.get("conference")
        venue_node = _node("v", venue)
        if venue_node:
            graph.add_node(venue_node, node_type="venue", label=_stable_text(venue))
            _add_edge(graph, pub_node, venue_node, "published_in", weight=0.8)
            _add_edge(graph, reviewer_node, venue_node, "publishes_in", weight=0.5)

        title = _stable_text(publication.get("title", ""))
        abstract = _stable_text(publication.get("abstract", ""))
        for phrase in _simple_keyphrases(f"{title} {abstract}", limit=8):
            _add_topics(graph, pub_node, [phrase], "publication_text")
            _add_topics(graph, reviewer_node, [phrase], "publication_profile")


def _simple_keyphrases(text, limit=10):
    stop = {
        "this", "that", "with", "from", "using", "based", "paper", "approach",
        "system", "systems", "method", "methods", "model", "models", "data",
        "result", "results", "study", "work", "propose", "present",
    }
    words = [
        word for word in re.findall(r"[a-z][a-z0-9]{3,}", _stable_text(text).lower())
        if word not in stop
    ]
    phrases = []
    seen = set()
    for size in (3, 2, 1):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[index:index + size])
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def build_persistent_graph(manuscripts=None, reviewers=None, output_dir=OUTPUT_DIR):
    manuscripts = manuscripts if manuscripts is not None else (_load_json(os.path.join(DATA_DIR, "manuscripts.json"), []) or [])
    reviewers = reviewers if reviewers is not None else (_load_json(os.path.join(DATA_DIR, "reviewers.json"), []) or [])

    gliner_ms = _load_json(os.path.join(output_dir, "gliner_manuscript_labels.json"), {}) or {}
    gliner_rev = _load_json(os.path.join(output_dir, "gliner_reviewer_labels.json"), {}) or {}
    cso_ms = _load_json(os.path.join(output_dir, "cso_manuscript_topics.json"), {}) or {}
    cso_rev = _load_json(os.path.join(output_dir, "cso_reviewer_topics.json"), {}) or {}

    graph = nx.Graph()

    for paper in manuscripts:
        paper_id = str(paper.get("paper_id", "")).strip()
        if not paper_id:
            continue
        paper_node = f"p_{paper_id}"
        graph.add_node(
            paper_node,
            node_type="manuscript",
            label=_stable_text(paper.get("title", "")),
            conference=_stable_text(paper.get("conference", "")),
        )
        _add_authors(graph, paper_node, paper.get("authors", []))
        _add_topics(graph, paper_node, _topic_values(gliner_ms.get(paper_id)), "gliner")
        _add_topics(graph, paper_node, _topic_values(cso_ms.get(paper_id)), "cso")
        _add_topics(graph, paper_node, _simple_keyphrases(f"{paper.get('title', '')} {paper.get('abstract', '')}", limit=10), "text")

    for reviewer in reviewers:
        reviewer_id = str(reviewer.get("reviewer_id", "")).strip()
        if not reviewer_id:
            continue
        reviewer_node = f"r_{reviewer_id}"
        _add_reviewer_publications(graph, reviewer)
        _add_topics(graph, reviewer_node, _topic_values(gliner_rev.get(reviewer_id)), "gliner")
        _add_topics(graph, reviewer_node, _topic_values(cso_rev.get(reviewer_id)), "cso")

    return graph


def graph_signature(manuscripts=None, reviewers=None, output_dir=OUTPUT_DIR):
    payload = {
        "schema": "persistent_reviewer_kg_v1",
        "manuscripts": len(manuscripts or _load_json(os.path.join(DATA_DIR, "manuscripts.json"), []) or []),
        "reviewers": len(reviewers or _load_json(os.path.join(DATA_DIR, "reviewers.json"), []) or []),
    }
    for name in (
        "gliner_manuscript_labels.json",
        "gliner_reviewer_labels.json",
        "cso_manuscript_topics.json",
        "cso_reviewer_topics.json",
    ):
        path = os.path.join(output_dir, name)
        payload[name] = os.path.getmtime(path) if os.path.exists(path) else 0
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_or_build_persistent_graph(manuscripts=None, reviewers=None, force=False, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    signature = graph_signature(manuscripts, reviewers, output_dir)
    if not force and os.path.exists(PERSISTENT_KG_PATH) and os.path.exists(PERSISTENT_KG_META_PATH):
        meta = _load_json(PERSISTENT_KG_META_PATH, {}) or {}
        if meta.get("signature") == signature:
            try:
                with open(PERSISTENT_KG_PATH, "rb") as handle:
                    return pickle.load(handle), meta
            except Exception:
                pass

    graph = build_persistent_graph(manuscripts, reviewers, output_dir)
    meta = {
        "signature": signature,
        "schema": "persistent_reviewer_kg_v1",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(sorted(_count_node_types(graph).items())),
    }
    with open(PERSISTENT_KG_PATH, "wb") as handle:
        pickle.dump(graph, handle)
    with open(PERSISTENT_KG_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return graph, meta


def _count_node_types(graph):
    counts = {}
    for _node, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        counts[node_type] = counts.get(node_type, 0) + 1
    return counts


def graph_overlap_score(graph, paper_id, reviewer_id):
    paper_node = f"p_{paper_id}"
    reviewer_node = f"r_{reviewer_id}"
    if graph is None or paper_node not in graph or reviewer_node not in graph:
        return 0.0, 0.0

    paper_neighbors = set(graph.neighbors(paper_node))
    reviewer_neighbors = set(graph.neighbors(reviewer_node))
    direct_overlap = paper_neighbors & reviewer_neighbors

    two_hop = set()
    for neighbor in reviewer_neighbors:
        two_hop.update(graph.neighbors(neighbor))
    bridge_overlap = paper_neighbors & two_hop

    overlap_score = (len(direct_overlap) + 0.35 * len(bridge_overlap)) / max(3.0, len(paper_neighbors))
    confidence = min(1.0, (len(direct_overlap) + len(bridge_overlap)) / 8.0)
    return min(1.0, overlap_score), confidence
