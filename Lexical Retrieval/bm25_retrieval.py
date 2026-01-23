import json
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# -------------------------------------------------
# File paths
# -------------------------------------------------
DATASETS_FILE = "datasets_landing_metadata.json"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "bm25_retrieval_results.json"

# -------------------------------------------------
# Lightweight tokenizer (NO nltk)
# -------------------------------------------------
def tokenize(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

# -------------------------------------------------
# Load dataset metadata
# -------------------------------------------------
with open(DATASETS_FILE, "r", encoding="utf-8") as f:
    datasets_by_topic = json.load(f)

corpus_tokens = []
dataset_titles = []

for topic, datasets in datasets_by_topic.items():
    for ds in datasets:
        title = ds.get("title", "")
        description = ds.get("description", "")
        combined = f"{title} {description}"
        corpus_tokens.append(tokenize(combined))
        dataset_titles.append(title)

print(f"✅ Loaded {len(dataset_titles)} datasets")

# -------------------------------------------------
# Build BM25 index
# -------------------------------------------------
bm25 = BM25Okapi(corpus_tokens)

# -------------------------------------------------
# Load queries
# -------------------------------------------------
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# -------------------------------------------------
# Retrieval
# -------------------------------------------------
results = []

for topic, queries in tqdm(queries_by_topic.items(), desc="Processing topics"):
    for query in queries:
        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:3]

        assistant = []
        for rank, idx in enumerate(reversed(top_indices)):
            assistant.append({
                "dataset_title": dataset_titles[idx],
                "rank": rank  # 0 = least relevant, 2 = most relevant
            })

        results.append({
            "User_Query": query,
            "Assistant": assistant
        })

# -------------------------------------------------
# Save output
# -------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ BM25 lexical retrieval complete")
print(f"📁 Output saved to: {OUTPUT_FILE}")
