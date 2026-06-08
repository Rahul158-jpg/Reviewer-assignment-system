# A Hybrid Knowledge Graph Based Model for Intelligent Reviewer Assignment

This repository contains the code, datasets, experiments, and evaluation system developed for the research paper:

**"A Multi-Agent LLM and Knowledge Graph system for Intelligent Reviewer Assignment in Scholarly Publishing"**

---

## 🧠 Overview

Reviewer assignment is a critical task in scholarly publishing that directly impacts review quality, fairness, and publication outcomes. Traditional reviewer recommendation systems primarily rely on semantic similarity between manuscripts and reviewer profiles, often ignoring reviewer expertise reliability, workload balance, conflict of interest detection, and manuscript quality factors.

This project proposes a **Multi-Agent LLM and Knowledge Graph Framework** that combines semantic retrieval, graph-based expertise modeling, reviewer suitability estimation, fairness-aware ranking, and conflict detection into a unified reviewer assignment pipeline.

The framework integrates:

* Semantic similarity using Sentence Transformers
* Knowledge Graph construction from reviewer publications
* Reviewer expertise profiling through Large Language Models
* Manuscript quality assessment
* Reviewer reliability estimation
* Conflict-of-interest detection
* Fairness aware workload balancing
* Multi agent decision aggregation

The system aims to improve reviewer recommendation quality while maintaining transparency, fairness, and scalability for modern scholarly publishing environments.

---

# 📁 Repository Structure

```text
📦Reviewer-Assignment-System/
│
├── data/
│   ├── manuscripts.csv
│   ├── reviewers.csv
│   ├── ground_truth.csv
│   └── conflict_labels.csv
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── text_cleaning.py
│   │   ├── topic_extraction.py
│   │   └── embedding_generation.py
│   │
│   ├── knowledge_graph/
│   │   ├── kg_builder.py
│   │   ├── graph_features.py
│   │   └── node_embeddings.py
│   │
│   ├── semantic_matching/
│   │   ├── semantic_similarity.py
│   │   └── candidate_retrieval.py
│   │
│   ├── llm_agents/
│   │   ├── expertise_agent.py
│   │   ├── suitability_agent.py
│   │   ├── reliability_agent.py
│   │   ├── quality_agent.py
│   │   └── decision_agent.py
│   │
│   ├── fairness/
│   │   ├── workload_balancer.py
│   │   └── fairness_metrics.py
│   │
│   ├── conflict_detection/
│   │   ├── conflict_detector.py
│   │   └── conflict_evaluation.py
│   │
│   ├── ranking/
│   │   ├── hybrid_ranking.py
│   │   └── rank_aggregation.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       ├── ablation_study.py
│       └── metrics.py
│
├── results/
│   ├── evaluation_summary.md
│   ├── ablation_results.csv
│   ├── fairness_values.csv
│   ├── conflict_detection_evaluation.csv
│   └── reviewer_rankings.csv
│
├── requirements.txt
├── README.md
└── main.py
```

---

# 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/reviewer-assignment-system.git

cd reviewer-assignment-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📚 Technologies Used

* Python 3.10+
* Sentence Transformers
* Hugging Face Transformers
* NetworkX
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Knowledge Graph Techniques
* Large Language Models (LLMs)

---

# ⚙️ Methodology

## 1️⃣ Semantic Matching

The system generates dense embeddings for manuscripts and reviewer profiles using transformer-based language models.

Features:

* Semantic similarity score
* Topic overlap
* Candidate reviewer retrieval

---

## 2️⃣ Knowledge Graph Construction

Reviewer expertise is represented through a scholarly knowledge graph constructed from publication metadata.

Graph features include:

* Expertise entities
* Research topics
* Author topic relationships
* Reviewer topic connectivity

Generated features:

* KG Score
* KG Confidence
* KG Effective Score
* Graph-based ranking signals

---

## 3️⃣ LLM-Based Reviewer Expertise Analysis

Specialized LLM agents analyze reviewer profiles and manuscript content to estimate:

* Domain expertise
* Topic suitability
* Reviewer reliability
* Expertise confidence

The agents provide structured reviewer suitability scores that complement semantic and graph-based signals.

---

## 4️⃣ Manuscript Quality Assessment

A dedicated quality assessment module evaluates:

* Research clarity
* Technical depth
* Novelty indicators
* Methodological completeness

Quality scores are incorporated into the reviewer assignment process.

---

## 5️⃣ Conflict of Interest Detection

The framework automatically detects potential conflicts based on:

* Institutional overlap
* Research collaboration patterns
* Author-reviewer relationships
* Historical publication connections

Conflict candidates are filtered before final ranking.

---

## 6️⃣ Fairness-Aware Ranking

To prevent reviewer overload, the system incorporates workload balancing.

Fairness considerations:

* Current reviewer workload
* Assignment distribution
* Reviewer availability
* Sustainable review allocation

---

## 7️⃣ Multi-Agent Decision Aggregation

Outputs from semantic retrieval, knowledge graphs, and LLM agents are combined using a weighted ranking system to generate final reviewer recommendations.

---

# 📊 Evaluation Metrics

The framework is evaluated using:

### Ranking Metrics

* Mean Average Precision (MAP)
* Mean Reciprocal Rank (MRR)
* Precision@K
* Recall@K
* NDCG

### Fairness Metrics

* Assignment Distribution
* Workload Variance
* Fairness Index

### Conflict Detection Metrics

* Precision
* Recall
* F1-Score

---

# 🧪 Experimental Results

## Ablation Study

| Configuration                   | MAP    |
| ------------------------------- | ------ |
| Semantic Only                   | 0.0624 |
| Semantic + KG                   | 0.0625 |
| Semantic + LLM Suitability      | 0.0605 |
| Semantic + KG + LLM Suitability | 0.0600 |
| Full Framework                  | 0.0613 |

---

## Feature Importance

| Feature                  | Contribution (%) |
| ------------------------ | ---------------- |
| KG Effective Score       | 30.7             |
| KG Confidence            | 20.5             |
| KG Rank Reciprocal       | 15.4             |
| KG Score                 | 13.8             |
| Semantic Similarity      | 9.9              |
| Semantic Rank Reciprocal | 9.2              |
| Topic Overlap            | 7.6              |

---

## Key Findings

* Knowledge Graph features dominate ranking decisions.
* Semantic similarity alone is insufficient for reviewer assignment.
* Reviewer expertise modeling improves interpretability.
* Conflict detection increases assignment reliability.

---

# 🚀 Running the System

Run the complete reviewer assignment pipeline:

```bash
python main.py
```

Run evaluation:

```bash
python evaluate.py
```

Run ablation study:

```bash
python ablation_study.py
```

Generate fairness analysis:

```bash
python fairness_metrics.py
```

Evaluate conflict detection:

```bash
python conflict_evaluation.py
```

---

# 📈 Outputs

The system generates:

* Ranked reviewer recommendations
* Reviewer suitability scores
* Knowledge graph features
* Conflict detection reports
* Fairness statistics
* Evaluation summaries

Results are stored in the `results/` directory.

---

# 🎯 Contributions

* Multi-agent reviewer assignment architecture
* Knowledge graph based expertise modeling
* LLM driven reviewer suitability estimation
* Conflict-of-interest detection framework
* Fairness aware reviewer allocation
* Comprehensive ablation and evaluation analysis
