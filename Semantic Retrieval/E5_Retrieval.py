import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
E5_MODEL = "intfloat/e5-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LENGTH = 512
OUTPUT_FILE = "e5_dense_retrieval_results.json"
CHECKPOINT_FILE = "retrieval_checkpoint.json"

DATASET_FILE = "datasets_metadata.jsonl"

TOP_K = 3

# -------------------------------------------------
# LOAD QUERIES
# -------------------------------------------------
with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# Flatten queries for easier resume
all_queries = []
for topic, queries in queries_by_topic.items():
    for query in queries:
        all_queries.append(query)

# -------------------------------------------------
# LOAD DATASETS (STREAM JSONL)
# -------------------------------------------------
all_datasets = []
dataset_texts = []

print("Loading datasets...")

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        ds = json.loads(line)

        dataset_text = (
            f"Title: {ds.get('title','')} "
            f"Description: {ds.get('description','')} "
            f"Publisher: {ds.get('publisher','')}"
        )

        all_datasets.append({
            "title": ds.get("title", "")
        })

        dataset_texts.append(f"passage: {dataset_text}")

print(f"Loaded {len(dataset_texts)} datasets")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(E5_MODEL)
model = AutoModel.from_pretrained(E5_MODEL).to(DEVICE)
model.eval()

# -------------------------------------------------
# EMBEDDING FUNCTION
# -------------------------------------------------
def embed_texts(texts):

    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):

        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():

            outputs = model(**inputs)

            emb = outputs.last_hidden_state.mean(dim=1)

            emb = F.normalize(emb, p=2, dim=1)

        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)

# -------------------------------------------------
# EMBED DATASETS
# -------------------------------------------------
print("Embedding dataset metadata...")

dataset_embeddings = []

for i in tqdm(range(0, len(dataset_texts), 10000)):

    batch = dataset_texts[i:i+10000]

    emb = embed_texts(batch)

    dataset_embeddings.append(emb)

dataset_embeddings = torch.cat(dataset_embeddings, dim=0)

print("Dataset embeddings shape:", dataset_embeddings.shape)

# -------------------------------------------------
# BUILD FAISS INDEX (ONLY CHANGE IN RETRIEVAL PIPELINE)
# -------------------------------------------------
print("Building FAISS index...")

dim = dataset_embeddings.shape[1]

# cosine similarity = inner product (since normalized)
index = faiss.IndexFlatIP(dim)

# convert to numpy float32
dataset_embeddings_np = dataset_embeddings.numpy().astype("float32")

index.add(dataset_embeddings_np)

print("FAISS index built with", index.ntotal, "vectors")

# -------------------------------------------------
# LOAD CHECKPOINT IF EXISTS
# -------------------------------------------------
if os.path.exists(CHECKPOINT_FILE):
    print("Loading checkpoint...")
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
        results = checkpoint_data["results"]
        start_idx = checkpoint_data["last_index"] + 1
else:
    results = []
    start_idx = 0

print(f"Resuming from query index: {start_idx}")

# -------------------------------------------------
# RETRIEVAL USING FAISS
# -------------------------------------------------
print("Running retrieval...")

for i in tqdm(range(start_idx, len(all_queries)), desc="Queries"):

    query = all_queries[i]

    query_input = f"query: {query}"

    query_embedding = embed_texts([query_input])

    query_np = query_embedding.numpy().astype("float32")

    # FAISS search
    scores, indices = index.search(query_np, TOP_K)

    ranked_results = []

    for rank, idx in enumerate(indices[0]):

        ranked_results.append({
            "rank": rank,
            "dataset_title": all_datasets[idx]["title"]
        })

    results.append({
        "User_Query": query,
        "Assistant": ranked_results
    })

    # -------------------------------------------------
    # SAVE CHECKPOINT AFTER EACH QUERY
    # -------------------------------------------------
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "last_index": i,
            "results": results
        }, f, indent=2, ensure_ascii=False)

# -------------------------------------------------
# SAVE FINAL OUTPUT
# -------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# remove checkpoint after success
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print("\nDense retrieval completed (FAISS)")
print(f"Output saved to: {OUTPUT_FILE}")
