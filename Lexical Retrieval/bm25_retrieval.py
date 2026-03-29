import json
import re
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# -------------------------------------------------
# File paths
# -------------------------------------------------
DATASETS_FILE = "datasets_metadata.jsonl"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "bm25_retrieval_results.json"

DATASET_CHECKPOINT = "checkpoint_datasets.json"
RESULTS_CHECKPOINT = "checkpoint_results.jsonl"

# -------------------------------------------------
# Tokenizer
# -------------------------------------------------
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

def tokenize(text):
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())

# -------------------------------------------------
# LOAD DATASETS
# -------------------------------------------------
corpus_tokens = []
dataset_titles = []

start_line = 0

if os.path.exists(DATASET_CHECKPOINT):
    with open(DATASET_CHECKPOINT, "r") as f:
        start_line = json.load(f).get("last_line", 0)

print(f"📥 Resuming dataset load from line {start_line}...")

with open(DATASETS_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f)):
        if i < start_line:
            continue

        try:
            ds = json.loads(line)
            title = ds.get("title", "")
            description = ds.get("description", "")

            combined = f"{title} {description[:300]}"
            tokens = tokenize(combined)

            corpus_tokens.append(tokens)
            dataset_titles.append(title)

        except Exception:
            continue

        # Save checkpoint every 10k lines
        if i % 10000 == 0:
            with open(DATASET_CHECKPOINT, "w") as ckpt:
                json.dump({"last_line": i}, ckpt)

print(f"✅ Loaded {len(dataset_titles)} datasets")

# -------------------------------------------------
# Build BM25
# -------------------------------------------------
print("⚙️ Building BM25 index...")
bm25 = BM25Okapi(corpus_tokens)

# -------------------------------------------------
# LOAD QUERIES
# -------------------------------------------------
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# -------------------------------------------------
# LOAD COMPLETED RESULTS 
# -------------------------------------------------
processed_queries = set()

if os.path.exists(RESULTS_CHECKPOINT):
    print("📂 Loading previous results checkpoint...")
    with open(RESULTS_CHECKPOINT, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            processed_queries.add(obj["User_Query"])

# -------------------------------------------------
# Retrieval
# -------------------------------------------------
results = []

for topic, queries in tqdm(queries_by_topic.items(), desc="Processing topics"):
    for query in queries:

        if query in processed_queries:
            continue  

        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Faster top-k
        import numpy as np
        scores = np.array(scores)
        top_indices = np.argpartition(scores, -3)[-3:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        assistant = []
        for rank, idx in enumerate(reversed(top_indices)):
            assistant.append({
                "dataset_title": dataset_titles[idx],
                "rank": rank
            })

        result = {
            "User_Query": query,
            "Assistant": assistant
        }

        results.append(result)

     
        with open(RESULTS_CHECKPOINT, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

# -------------------------------------------------
# Final merge output
# -------------------------------------------------
print("💾 Merging final output...")

final_results = []

with open(RESULTS_CHECKPOINT, "r", encoding="utf-8") as f:
    for line in f:
        final_results.append(json.loads(line))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print("\n✅ BM25 retrieval complete")
print(f"📁 Output saved to: {OUTPUT_FILE}")
