# AHA-Bench: Ad Hoc Dataset Retrieval Benchmark

AHA-Bench is a large-scale benchmark for **ad hoc dataset retrieval in natural language**, grounded in real-world open data from the European Data Portal (data.europa.eu).  
It is designed to evaluate retrieval systems in **exploratory dataset discovery scenarios**, where users express information needs in free-form natural language rather than keyword queries.

---

## 🔍 What Does the Benchmark Look Like?

### Dataset Collection and Topic Distribution

AHA-Bench contains **7,000 real-world datasets** sampled from data.europa.eu across **14 topical domains**.  
To avoid over-representation of dominant categories, datasets are sampled proportionally by topic.

#### Topic Distribution

| Topic | # Datasets | % |
|------|-----------:|---:|
| Agriculture, fisheries, forestry and food | 1,414 | 20.2 |
| Justice, legal system and public safety | 1,272 | 18.2 |
| Environment | 896 | 12.8 |
| Government and public sector | 773 | 11.0 |
| Economy and finance | 542 | 7.7 |
| Science and technology | 483 | 6.9 |
| Population and society | 440 | 6.3 |
| Regions and cities | 414 | 5.9 |
| Transport | 341 | 4.9 |
| Education, culture and sport | 190 | 2.7 |
| Health | 135 | 1.9 |
| Energy | 53 | 0.8 |
| Provisional data | 41 | 0.6 |
| International issues | 6 | 0.1 |
| **Total** | **7,000** | **100** |

<p align="center">
  <img src="figures/topic_distribution.png" width="600">
</p>

*Figure 1: Topic distribution of datasets in AHA-Bench.*

---

## 🧠 Query Styles in AHA-Bench

To reflect realistic dataset discovery behavior, queries are generated using an LLM and grounded in a **query intent taxonomy** derived from prior studies of conversational dataset search.

Each dataset is associated with two evaluation queries:

### Query Types

- **Described Queries**  
  Explicitly describe the desired dataset in natural language.  
  *Example:*  
  > “Find a dataset about air pollution levels across European cities.”

- **Implied Queries**  
  Express a problem-oriented information need that presupposes relevant datasets.  
  *Example:*  
  > “How has air quality changed in major European cities over time?”

Dataset-request queries (e.g., queries that explicitly name a dataset) are generated as auxiliary artifacts but **excluded from retrieval evaluation**.

### Query Statistics

| Statistic | Value |
|---------|------:|
| Total number of queries | 20,594 |
| Described queries (evaluation) | 6,858 |
| Implied queries (evaluation) | 6,864 |
| Dataset-request queries (auxiliary) | 6,872 |
| Avg. described query length | 20.19 words |
| Avg. implied query length | 19.87 words |
| Avg. request query length | 12.09 words |

<p align="center">
  <img src="figures/query_style_distribution.png" width="600">
</p>

*Figure 2: Distribution of query types in AHA-Bench.*

---

## 🎯 Relevance Structure

Relevance labels are **graded (0–4)** and constructed via a hybrid dense-retrieval and LLM-based annotation pipeline.

Key characteristics:

- **99.26%** of queries have at least one highly relevant dataset (score ≥ 3)
- Most queries admit **multiple relevant datasets**, reflecting exploratory search scenarios

| # Highly Relevant Datasets (≥3) | # Queries | % Queries |
|--------------------------------|----------:|----------:|
| 0 | 102 | 0.74 |
| 1 | 919 | 6.70 |
| 2 | 4,187 | 30.52 |
| 3 | 4,057 | 29.57 |
| 4 | 1,746 | 12.73 |
| 5 | 2,710 | 19.75 |

<p align="center">
  <img src="figures/relevance_multiplicity.png" width="600">
</p>

*Figure 3: Relevance multiplicity per query.*

---

## 📊 Benchmark Performance

We evaluate three classes of systems:

1. **Classical retrievers** (BM25, dense embeddings)
2. **Retrieval-free LLMs** (query-only)
3. **Retrieval-augmented LLM systems (RAG)**

Metrics reported: **P@3, R@3, MRR, nDCG@3**

### Main Results

| Model | P@3 | R@3 | MRR | nDCG@3 |
|------|----:|----:|----:|------:|
| **Classical Retrievers** |||||
| BM25 | 0.362 | 0.293 | 0.469 | 0.541 |
| all-MiniLM-L6-v2 | 0.313 | 0.252 | 0.423 | 0.480 |
| Qwen3 embeddings | 0.362 | 0.293 | 0.471 | 0.538 |
| **E5-large-v2** | **0.448** | **0.356** | **0.553** | **0.633** |
| **Retrieval-free LLM baselines (query-only)** |||||
| *Open-weight models* |||||
| LLaMA3.1-8B | 0.001 | 0.001 | 0.004 | 0.004 |
| LLaMA3.1-70B | 0.004 | 0.003 | 0.009 | 0.009 |
| Qwen2.5-7B | 0.001 | 0.001 | 0.003 | 0.003 |
| Qwen2.5-72B | 0.002 | 0.002 | 0.005 | 0.005 |
| Qwen3-80B | 0.001 | 0.001 | 0.002 | 0.002 |
| *Closed-source models* |||||
| GPT-4.1 | 0.002 | 0.001 | 0.005 | 0.005 |
| GPT-4.1 mini | 0.000 | 0.000 | 0.001 | 0.001 |
| GPT-4.1 nano | 0.001 | 0.001 | 0.002 | 0.002 |
| GPT-5.2 | 0.003 | 0.003 | 0.007 | 0.008 |
| **Retrieval-augmented systems** |||||
| RAG (BM25) + LLaMA3.1-70B | 0.389 | 0.312 | 0.718 | 0.723 |
| **RAG (E5-large-v2) + LLaMA3.1-70B** | **0.482** | **0.381** | **0.817** | **0.824** |

<p align="center">
  <img src="figures/model_performance.png" width="700">
</p>

*Figure 4: Retrieval performance across model classes.*

**Key takeaway:**  
- Query-only LLMs perform near zero  
- Dense retrieval substantially outperforms BM25  
- Retrieval-augmented LLMs achieve the strongest performance, but the task remains far from saturated

---

## 📦 What’s Included in This Repository

- Dataset metadata (7,000 datasets)
- Natural-language queries (described & implied)
- Graded relevance labels (0–4)
- Baseline retrieval and evaluation code
- Scripts for reproducing benchmark analysis and plots

---


