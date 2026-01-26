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
# PARSERS (FIXED)
# --------------------------------------------------
def parse_ranked_list_listformat(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for obj in data:
        q = obj['User_Query']
        assistant = obj.get('Assistant', [])

        # Ensure we can handle list of dicts OR list of strings
        if isinstance(assistant, list) and all(isinstance(x, dict) and 'dataset_title' in x for x in assistant):
            ranked = sorted(assistant, key=lambda x: x.get("rank", 0))
            res[q] = [normalize(d['dataset_title']) for d in ranked]
        else:
            res[q] = [normalize(str(d)) for d in assistant]
    return res


def parse_ranked_dictformat(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for obj in data:
        q = obj['User_Query']
        assistant = obj.get('Assistant', {})

        if isinstance(assistant, dict):
            try:
                ranked = sorted(
                    assistant.items(),
                    key=lambda x: int(re.search(r'(\d+)$', x[0]).group(1))
                )
            except:
                ranked = assistant.items()
            res[q] = [normalize(v) for _, v in ranked]
        else:
            # fallback if not a dict
            res[q] = [normalize(str(v)) for v in assistant]
    return res


def parse_llm_text(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for obj in data:
        q = obj.get("User_Query") or obj.get("query")
        assistant = obj.get("Assistant", [])

        titles = []

        # --------------------------------------------------
        # CASE 1: Assistant is STRING (LLMs)
        # --------------------------------------------------
        if isinstance(assistant, str):
            # Match Dataset 1: Title (optional score)
            titles = re.findall(r"Dataset \d+:\s*(.*?)(?:\s*\(\d+\))?(?:$|\n)", assistant, re.MULTILINE)
            if not titles:
                # fallback: split by line breaks or Dataset patterns
                lines = re.split(r'\n|Dataset \d+:', assistant)
                titles = [normalize(l) for l in lines if l.strip()]

        # --------------------------------------------------
        # CASE 2: Assistant is LIST (BM25, E5)
        # --------------------------------------------------
        elif isinstance(assistant, list):
            if all(isinstance(x, dict) and 'dataset_title' in x for x in assistant):
                ranked = sorted(assistant, key=lambda d: d.get("rank", 0))
                titles = [normalize(x["dataset_title"]) for x in ranked]
            else:
                titles = [normalize(str(x)) for x in assistant]

        # --------------------------------------------------
        # CASE 3: Assistant is DICT (Qwen cosine)
        # --------------------------------------------------
        elif isinstance(assistant, dict):
            try:
                ranked = sorted(
                    assistant.items(),
                    key=lambda x: int(re.search(r'(\d+)$', x[0]).group(1))
                )
            except:
                ranked = assistant.items()
            titles = [normalize(v) for _, v in ranked]

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
GROUND_TRUTH = load_ground_truth('evaluation_results.json')

MODELS = {
    'BM25':parse_llm_text('BM25.json'),
    'E5': parse_llm_text('E5.json'),
    'Qwen+cosine_similarity': parse_llm_text('qwen+cosinesimilarity.json'),
    'all-minilm-l6-v2': parse_llm_text('all-MiniLM-L6-v2_cosine_similarity.json'),
    'LLaMA-8B': parse_llm_text('Llama_8B.json'),
    'LLaMA-70B': parse_llm_text('Llama_70B.json'),
    'Qwen-7B': parse_llm_text('Qwen2.5-7B-Instruct-Turbo.json'),
    'Qwen-72B': parse_llm_text('Qwen_72B.json'),
    'Qwen-80B': parse_llm_text('Qwen3-Next-80B-A3B-Instruct.json'),
    'RAG_BM25+Llama-70B': parse_llm_text('bm25_llama70b_rag_results.json'),
    'RAG_E5+Llama-70B': parse_llm_text('e5_llama70b_rag_results.json'),
    'GPT-4.1': parse_llm_text('llm_answers_by_topic_gpt4.1.json'),
    'GPT-4.1_mini': parse_llm_text('llm_answers_by_topic_gpt4.1-mini.json'),
    'GPT-4.1_nano': parse_llm_text('llm_answers_by_topic_gpt_4.1_nano.json'),
    'GPT-5.2': parse_llm_text('llm_answers_by_topic_gpt_5.2.json'),

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
# --------------------------------------------------
# PLOT (AUTO-SCALED FOR SMALL VALUES)
# --------------------------------------------------
labels = [r["Model"] for r in results]
metrics = [k for k in results[0] if k != "Model"]

x = np.arange(len(labels))
width = 0.2

plt.figure(figsize=(10,6))

# Plot bars
for i, metric in enumerate(metrics):
    plt.bar(
        x + i * width,
        [r[metric] for r in results],
        width,
        label=metric
    )

# ---- AUTO Y-SCALE BASED ON DATA ----
all_scores = [r[m] for r in results for m in metrics]
y_max = max(all_scores)

plt.ylim(0, y_max * 1.3)  # headroom so bars are not clipped

# ---- AXES & LABELS ----
plt.xticks(x + width, labels, rotation=30)
plt.ylabel("Score")
plt.title("Europa Dataset Retrieval Evaluation")

# ---- FORMAT Y TICKS (IMPORTANT) ----
plt.ticklabel_format(axis="y", style="plain")
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.4f}")
)

# ---- OPTIONAL: VALUE LABELS (HIGHLY RECOMMENDED) ----
for i, metric in enumerate(metrics):
    for j, r in enumerate(results):
        value = r[metric]
        plt.text(
            j + i * width,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

plt.legend()
plt.tight_layout()
plt.savefig("europa_retrieval_metrics.png", dpi=300)
plt.show()

#-----
#HEATMAP
#-----


models = [r["Model"] for r in results]
metrics = [k for k in results[0] if k != "Model"]

data = np.array([[r[m] for m in metrics] for r in results])

plt.figure(figsize=(10, 6))
plt.imshow(data, aspect="auto")

plt.colorbar(label="Score")

plt.xticks(range(len(metrics)), metrics, rotation=30)
plt.yticks(range(len(models)), models)

plt.title("Retrieval Performance Heatmap")

# value labels
for i in range(len(models)):
    for j in range(len(metrics)):
        plt.text(j, i, f"{data[i, j]:.3f}",
                 ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("europa_heatmap.png", dpi=300)
plt.show()
