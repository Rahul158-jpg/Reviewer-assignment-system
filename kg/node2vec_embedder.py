import hashlib
import os
import pickle

try:
    from node2vec import Node2Vec
except Exception:
    Node2Vec = None

try:
    import numpy as np
except Exception:
    np = None

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
STABLE_CACHE_PATH = os.path.join(CACHE_DIR, "kg_emb.pkl")


def _graph_cache_path(G, dimensions, walk_length, num_walks):
    node_parts = [
        f"{str(node)}:{attrs.get('node_type', '')}"
        for node, attrs in G.nodes(data=True)
    ]
    edge_parts = [
        " -- ".join(sorted([str(source), str(target)]))
        for source, target in G.edges()
    ]
    payload = "\n".join(
        [
            "schema=typed_feature_fallback_v2",
            f"dimensions={dimensions}",
            f"walk_length={walk_length}",
            f"num_walks={num_walks}",
            *sorted(node_parts),
            *sorted(edge_parts),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"kg_embeddings_{digest}.pkl")


def _fallback_graph_embeddings(G, dimensions):
    if np is None:
        return {}

    feature_nodes = [
        str(node)
        for node, attrs in G.nodes(data=True)
        if attrs.get("node_type") in {"entity", "topic", "venue", "author"}
    ]
    if not feature_nodes:
        return {}

    feature_index = {node: idx for idx, node in enumerate(feature_nodes)}
    raw = {}

    for node in G.nodes():
        node_key = str(node)
        vector = np.zeros(len(feature_nodes), dtype=float)

        if node_key in feature_index:
            vector[feature_index[node_key]] = 1.0

        for neighbor in G.neighbors(node):
            neighbor_key = str(neighbor)
            if neighbor_key in feature_index:
                vector[feature_index[neighbor_key]] = 1.0
            for second_hop in G.neighbors(neighbor):
                second_key = str(second_hop)
                if second_key in feature_index:
                    vector[feature_index[second_key]] = max(vector[feature_index[second_key]], 0.5)

        raw[node_key] = vector

    if len(feature_nodes) > dimensions:
        rng = np.random.default_rng(42)
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(dimensions),
            size=(len(feature_nodes), dimensions),
        )
        raw = {node: vector @ projection for node, vector in raw.items()}

    embeddings = {}
    for node, vector in raw.items():
        norm = np.linalg.norm(vector)
        embeddings[node] = vector / norm if norm else vector

    return embeddings


def generate_embeddings(G, dimensions=64, walk_length=10, num_walks=50, stable_cache=False):
    if G is None or G.number_of_nodes() == 0:
        return {}

    cache_path = STABLE_CACHE_PATH if stable_cache else _graph_cache_path(G, dimensions, walk_length, num_walks)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    if Node2Vec is None:
        embeddings = _fallback_graph_embeddings(G, dimensions)
        if embeddings:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
        return embeddings

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1,
        quiet=True,
    )
    model = node2vec.fit(window=5, min_count=1)

    embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings
