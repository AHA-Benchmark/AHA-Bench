import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
E5_MODEL_NAME = "intfloat/e5-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METADATA_FILE = "datasets_landing_metadata.json"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "dense_retrieval_e5_top3.json"

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("🔄 Loading E5 model...")
model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
print(f"✅ Model loaded on {DEVICE}")

# -------------------------------------------------
# LOAD DATASETS
# -------------------------------------------------
print("📂 Loading dataset metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_by_topic = json.load(f)

datasets = []
dataset_titles = []

for topic, items in metadata_by_topic.items():
    for item in items:
        title = item.get("title", "").strip()
        if title:
            datasets.append(title)
            dataset_titles.append(title)

print(f"📊 Total datasets indexed: {len(dataset_titles)}")

# -------------------------------------------------
# EMBED DATASETS (ONCE)
# -------------------------------------------------
print("🔢 Encoding dataset titles...")
dataset_embeddings = model.encode(
    ["passage: " + title for title in dataset_titles],
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# -------------------------------------------------
# LOAD QUERIES
# -------------------------------------------------
print("📂 Loading queries...")
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

all_queries = []
for topic, queries in queries_by_topic.items():
    for q in queries:
        if q.strip():
            all_queries.append(q.strip())

print(f"📊 Total queries to process: {len(all_queries)}")

# -------------------------------------------------
# RETRIEVAL
# -------------------------------------------------
results = []

print("🔍 Performing dense retrieval...")
for query in tqdm(all_queries, desc="Retrieving"):
    # Encode query (E5 format)
    query_embedding = model.encode(
        "query: " + query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    # Cosine similarity
    scores = cosine_similarity(query_embedding, dataset_embeddings)[0]

    # Top 3 indices (highest similarity)
    top_indices = np.argsort(scores)[-3:][::-1]

    assistant_output = []
    for rank, idx in enumerate(top_indices[::-1]):
        assistant_output.append({
            "dataset_title": dataset_titles[idx],
            "rank": rank  # 0 = least relevant, 2 = most relevant
        })

    results.append({
        "User_Query": query,
        "Assistant": assistant_output
    })

# -------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Dense retrieval complete!")
print(f"📁 Output saved to: {OUTPUT_FILE}")
