import re
import json
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

CSO_TOPICS = {
    # ── Natural Language Processing ──
    "natural language processing", "nlp", "text mining",
    "information extraction", "named entity recognition",
    "relation extraction", "text classification",
    "sentiment analysis", "machine translation",
    "question answering", "text summarization",
    "language model", "language models",
    "large language model", "large language models",
    "transformer", "transformers", "bert", "gpt",
    "word embedding", "word embeddings", "word2vec",
    "pre-trained model", "pretrained model",
    "fine-tuning", "fine tuning", "prompt learning",
    "prompt engineering", "few-shot learning",
    "zero-shot learning", "transfer learning",
    "sequence labeling", "part-of-speech tagging",
    "dependency parsing", "coreference resolution",
    "text generation", "natural language generation",
    "natural language understanding",
    "reading comprehension", "textual entailment",
    "semantic similarity", "semantic parsing",
    "information retrieval", "document retrieval",
    "topic modeling", "latent dirichlet allocation",
    "lda", "word sense disambiguation",
    "speech recognition", "dialogue systems",
    "chatbot", "conversational ai",
    "machine comprehension", "abstractive summarization",
    "extractive summarization", "document classification",
    "text representation", "lexical analysis",

    # ── Knowledge Graphs and Semantic Web ──
    "knowledge graph", "knowledge graphs",
    "knowledge base", "knowledge bases",
    "semantic web", "linked data", "ontology",
    "ontologies", "rdf", "owl", "sparql",
    "knowledge representation", "knowledge engineering",
    "entity linking", "entity recognition",
    "knowledge graph embedding",
    "knowledge graph completion",
    "link prediction", "triple classification",
    "ontology alignment", "schema matching",
    "dbpedia", "wikidata", "freebase", "yago",
    "semantic search", "semantic annotation",
    "open information extraction", "openie",
    "relation learning", "knowledge acquisition",
    "fact extraction", "graph neural network",
    "graph neural networks", "gnn",
    "heterogeneous graph", "knowledge-intensive",
    "entity disambiguation", "entity resolution",
    "web semantics", "data integration",
    "conceptual modeling", "taxonomies",
    "controlled vocabulary", "thesaurus",

    # ── Machine Learning ──
    "machine learning", "deep learning",
    "neural network", "neural networks",
    "convolutional neural network", "cnn",
    "recurrent neural network", "rnn", "lstm",
    "attention mechanism", "self-attention",
    "multi-head attention", "encoder decoder",
    "generative adversarial network", "gan",
    "variational autoencoder", "vae",
    "reinforcement learning", "supervised learning",
    "unsupervised learning", "semi-supervised",
    "federated learning", "continual learning",
    "meta-learning", "active learning",
    "data augmentation", "regularization",
    "dropout", "batch normalization",
    "gradient descent", "backpropagation",
    "hyperparameter tuning", "model compression",
    "knowledge distillation", "pruning",
    "quantization", "neural architecture search",
    "representation learning", "embedding",
    "embeddings", "feature extraction",
    "feature engineering", "dimensionality reduction",
    "principal component analysis", "pca",
    "clustering", "classification", "regression",
    "random forest", "support vector machine", "svm",
    "gradient boosting", "xgboost",
    "bayesian network", "bayesian inference",
    "probabilistic model", "generative model",

    # ── Information Systems and Databases ──
    "information system", "information systems",
    "database", "relational database",
    "nosql", "graph database",
    "data management", "data quality",
    "data integration", "data warehouse",
    "data lake", "data pipeline",
    "query processing", "query optimization",
    "indexing", "data retrieval",
    "recommendation system", "recommender system",
    "collaborative filtering", "content-based filtering",
    "semantic annotation", "metadata",
    "data schema", "data model",

    # ── Computer Vision ──
    "computer vision", "image classification",
    "object detection", "image segmentation",
    "image recognition", "visual question answering",
    "image captioning", "scene understanding",
    "video analysis", "face recognition",
    "optical character recognition", "ocr",

    # ── Web and Social Media ──
    "social network", "social media",
    "web mining", "web scraping",
    "crowdsourcing", "community detection",
    "fake news detection", "misinformation",
    "hate speech", "opinion mining",
    "event detection", "trend analysis",
    "twitter", "social network analysis",

    # ── Peer Review and Scholarly ──
    "peer review", "reviewer assignment",
    "scholarly communication", "citation network",
    "citation analysis", "bibliometrics",
    "academic paper", "research paper",
    "conference paper", "journal article",
    "co-authorship", "research community",
    "expertise matching", "reviewer recommendation",

    # ── Multi-Agent Systems and AI ──
    "multi-agent system", "multi-agent systems",
    "autonomous agent", "intelligent agent",
    "model context protocol", "mcp",
    "agent coordination", "task allocation",
    "artificial intelligence", "ai",
    "expert system", "decision support",
    "intelligent system", "cognitive computing",

    # ── Ethics and Fairness ──
    "fairness", "bias", "explainability",
    "interpretability", "transparency",
    "privacy", "ethics", "responsible ai",
    "algorithmic fairness", "debiasing",
    "conflict of interest",

    # ── Programming and Software ──
    "programming language", "software engineering",
    "code generation", "program synthesis",
    "static analysis", "bug detection",
    "software testing", "code completion",
    "compiler", "parallel computing",
    "distributed systems", "cloud computing",
    "microservices", "api",

    # ── Data Science ──
    "data science", "big data", "analytics",
    "data visualization", "statistical analysis",
    "predictive modeling", "time series",
    "anomaly detection", "pattern recognition",

    # ── Evaluation and Metrics ──
    "evaluation", "benchmark",
    "precision", "recall", "f1 score",
    "accuracy", "mean reciprocal rank", "mrr",
    "mean average precision", "map",
    "ndcg", "rouge", "bleu",
}

_SORTED_TOPICS = sorted(CSO_TOPICS, key=len, reverse=True)
_PATTERNS = [
    (topic, re.compile(
        r'\b' + re.escape(topic) + r'\b',
        re.IGNORECASE
    ))
    for topic in _SORTED_TOPICS
]


# ─────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────

def get_cso_topics(abstract_text: str) -> list[str]:
    """
    Extract CSO topics from abstract text using
    keyword matching against the CSO vocabulary.

    Replicates the output of the original
    cso_classifier package without the dependency.

    Args:
        abstract_text: raw abstract string

    Returns:
        list of matched CSO topic strings
    """
    if not abstract_text or not abstract_text.strip():
        return []

    matched = []
    seen    = set()
    text    = abstract_text.lower()

    for topic, pattern in _PATTERNS:
        if topic in seen:
            continue
        if pattern.search(text):
            matched.append(topic)
            seen.add(topic)

    return matched


def classify_abstract(abstract_text: str) -> dict:
    """
    Full classification result mimicking the
    original cso_classifier.run() output format.

    Args:
        abstract_text: raw abstract string

    Returns:
        dict with 'syntactic', 'semantic', 'enhanced' keys
    """
    topics = get_cso_topics(abstract_text)
    return {
        "syntactic": topics,
        "semantic":  topics,
        "enhanced":  topics,
    }


def classify_manuscripts(
    manuscripts: list[dict],
    text_field:  str = "abstract",
) -> dict:
    """
    Classify all manuscripts and return topic lists.

    Args:
        manuscripts: list of manuscript dicts
                     (from manuscripts.json)
        text_field:  field to classify ('abstract'
                     or 'title')

    Returns:
        dict { paper_id: [topic, ...] }
    """
    results = {}
    for m in manuscripts:
        pid   = m.get("paper_id", "")
        text  = m.get(text_field, "") or ""
        combined = m.get("title", "") + " " + text
        results[pid] = get_cso_topics(combined)

    return results


def classify_reviewers(
    reviewers: list[dict],
) -> dict:
    """
    Classify all reviewer publication abstracts
    and return aggregated topic lists.

    Args:
        reviewers: list of reviewer dicts
                   (from reviewers.json)

    Returns:
        dict { reviewer_id: [topic, ...] }
    """
    results = {}
    for r in reviewers:
        rid    = r.get("reviewer_id", "")
        topics = set()

        for pub in r.get("publications", []):
            title    = pub.get("title",    "") or ""
            abstract = pub.get("abstract", "") or ""
            text     = title + " " + abstract
            for t in get_cso_topics(text):
                topics.add(t)

        results[rid] = list(topics)

    return results


# ─────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────

if __name__ == "__main__":

    import json, os

    DATA_DIR   = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data")
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MS_PATH  = os.path.join(DATA_DIR, "manuscripts.json")
    REV_PATH = os.path.join(DATA_DIR, "reviewers.json")

    print("=" * 52)
    print("  CSO Classifier — Reviewer Recommendation")
    print("  (No external package required)")
    print("=" * 52)

    # ── Classify manuscripts ──────────────
    if os.path.exists(MS_PATH):
        print(f"\n  Loading manuscripts...")
        with open(MS_PATH, encoding="utf-8") as f:
            manuscripts = json.load(f)

        print(f"  Classifying {len(manuscripts)} papers...")
        ms_topics = classify_manuscripts(manuscripts)

        out_path = os.path.join(
            OUTPUT_DIR, "cso_manuscript_topics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ms_topics, f, indent=2)

        # Show sample
        sample_id = list(ms_topics.keys())[0]
        print(f"\n  Sample — paper {sample_id}:")
        print(f"  Topics: {ms_topics[sample_id][:8]}")
        print(f"\n  ✅ Manuscript topics saved to:")
        print(f"     {out_path}")
    else:
        print(f"  ⚠️  manuscripts.json not found at {MS_PATH}")

    # ── Classify reviewers ────────────────
    if os.path.exists(REV_PATH):
        print(f"\n  Loading reviewers...")
        with open(REV_PATH, encoding="utf-8") as f:
            reviewers = json.load(f)

        print(f"  Classifying {len(reviewers)} reviewers...")
        rev_topics = classify_reviewers(reviewers)

        out_path = os.path.join(
            OUTPUT_DIR, "cso_reviewer_topics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rev_topics, f, indent=2)

        # Show sample
        sample_id = list(rev_topics.keys())[0]
        print(f"\n  Sample — reviewer {sample_id}:")
        print(f"  Topics: {rev_topics[sample_id][:8]}")
        print(f"\n  ✅ Reviewer topics saved to:")
        print(f"     {out_path}")
    else:
        print(f"  ⚠️  reviewers.json not found at {REV_PATH}")

    print("\n" + "=" * 52)
    print("  CSO classification complete.")
    print("  Output feeds into kg_builder.py")
    print("=" * 52)