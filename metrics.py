# ==========================================================
# EUROPA RETRIEVAL EVALUATION SCRIPT (NO pandas / NO sklearn)
# ==========================================================

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

TOP_K = 3

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())

# --------------------------------------------------
# LOAD GROUND TRUTH (JSONL)
# relevance = llm_relevance_score >= 2
# --------------------------------------------------
def load_ground_truth(path):
    gt = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj['query']
            rel = [
                normalize(r['title'])
                for r in obj['final_results']
                if r.get('llm_relevance_score', 0) >= 2
            ]
            gt[q] = rel
    return gt

# --------------------------------------------------
# PARSERS
# --------------------------------------------------
def parse_ranked_list_listformat(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for obj in data:
        q = obj['User_Query']
        ranked = sorted(obj['Assistant'], key=lambda x: x['rank'])
        res[q] = [normalize(d['dataset_title']) for d in ranked]
    return res


def parse_ranked_dictformat(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for obj in data:
        q = obj['User_Query']
        ranked = sorted(
            obj['Assistant'].items(),
            key=lambda x: int(x[0].split('_')[-1])
        )
        res[q] = [normalize(v) for _, v in ranked]
    return res


def parse_llm_text(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for obj in data:
        q = obj['User_Query']
        text = obj['Assistant']
        titles = re.findall(r"Dataset \d+: (.*?) \(\d+\)", text)
        res[q] = [normalize(t) for t in titles]
    return res

# --------------------------------------------------
# METRICS
# --------------------------------------------------
def precision_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt)) / k if pred else 0.0


def recall_at_k(pred, gt, k):
    return len(set(pred[:k]) & set(gt)) / len(gt) if gt else 0.0


def mrr(pred, gt):
    for i, p in enumerate(pred, 1):
        if p in gt:
            return 1.0 / i
    return 0.0


def dcg(rels):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))


def ndcg_at_k(pred, gt, k):
    rels = [1 if p in gt else 0 for p in pred[:k]]
    if not any(rels):
        return 0.0
    ideal = sorted(rels, reverse=True)
    return dcg(rels) / dcg(ideal)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
GROUND_TRUTH = load_ground_truth('evaluation_results.jsonl')

MODELS = {
    'BM25': parse_ranked_list_listformat('bm25_retrieval_results.json'),
    'E5': parse_ranked_list_listformat('E5_without_sentenceTransformer.json'),
    'Qwen-Embedding': parse_ranked_dictformat('qwen+cosinesimilarity.json'),
    'MiniLM': parse_llm_text('all-MiniLM-L6-v2_cosine_similarity.json'),
    'LLaMA-8B': parse_llm_text('llm_answers_by_topics_llama.json'),
    'Qwen-8B': parse_llm_text('llm_answers_by_topics_Qwen3-8B.json')
}

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
results = []

for model, outputs in MODELS.items():
    P, R, M, N = [], [], [], []
    for q, gt in GROUND_TRUTH.items():
        pred = outputs.get(q, [])
        P.append(precision_at_k(pred, gt, TOP_K))
        R.append(recall_at_k(pred, gt, TOP_K))
        M.append(mrr(pred, gt))
        N.append(ndcg_at_k(pred, gt, TOP_K))

    results.append({
        "Model": model,
        f"Precision@{TOP_K}": np.mean(P),
        f"Recall@{TOP_K}": np.mean(R),
        "MRR": np.mean(M),
        f"nDCG@{TOP_K}": np.mean(N)
    })

# --------------------------------------------------
# SAVE CSV (NO pandas)
# --------------------------------------------------
csv_file = "europa_retrieval_metrics.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n=== Retrieval Metrics ===")
for r in results:
    print(r)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
labels = [r["Model"] for r in results]
metrics = [k for k in results[0] if k != "Model"]

x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(10,6))
for i, metric in enumerate(metrics):
    plt.bar(x + i*width, [r[metric] for r in results], width, label=metric)

plt.xticks(x + width, labels, rotation=30)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Europa Dataset Retrieval Evaluation")
plt.legend()
plt.tight_layout()
plt.savefig("europa_retrieval_metrics.png", dpi=300)
plt.show()
