import json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import faiss
import os

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
OUTPUT_FILE = "dense_retrieval_top3.json"
CHECKPOINT_FILE = "retrieval_checkpoint.json"

DATASET_FILE = "datasets_metadata.jsonl"

BATCH_SIZE = 256

# ---------------------------
# Load queries
# ---------------------------
with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# Flatten queries with topic tracking
all_queries = []
for topic, queries in queries_by_topic.items():
    for query in queries:
        all_queries.append((topic, query))

# ---------------------------
# Load datasets (JSONL stream)
# ---------------------------
print("Loading datasets...")

dataset_titles = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        ds = json.loads(line)
        title = ds.get("title")

        if title is None:
            title = ""

        dataset_titles.append(str(title))

print(f"Loaded {len(dataset_titles)} datasets")

# ---------------------------
# Initialize model
# ---------------------------
print(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# ---------------------------
# Compute dataset embeddings (batched)
# ---------------------------
print("Encoding dataset titles...")

dataset_embeddings_list = []

for i in tqdm(range(0, len(dataset_titles), BATCH_SIZE)):
    
    batch = dataset_titles[i:i+BATCH_SIZE]
    
    emb = model.encode(
        batch,
        convert_to_tensor=True,
        show_progress_bar=False
    ).cpu()
    
    dataset_embeddings_list.append(emb)

dataset_embeddings = torch.cat(dataset_embeddings_list, dim=0)

print("Dataset embeddings shape:", dataset_embeddings.shape)

# ---------------------------
# Normalize embeddings (for cosine similarity via FAISS)
# ---------------------------
dataset_embeddings = torch.nn.functional.normalize(dataset_embeddings, p=2, dim=1)

# Convert to numpy for FAISS
dataset_embeddings_np = dataset_embeddings.numpy().astype("float32")

# ---------------------------
# Build FAISS index (INNER PRODUCT = cosine since normalized)
# ---------------------------
print("Building FAISS index...")

dimension = dataset_embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(dataset_embeddings_np)

print(f"FAISS index built with {index.ntotal} vectors")

# ---------------------------
# Load checkpoint if exists
# ---------------------------
results = []
start_idx = 0

if os.path.exists(CHECKPOINT_FILE):
    print("Loading checkpoint...")
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
        results = checkpoint["results"]
        start_idx = checkpoint["last_index"] + 1
    print(f"Resuming from query index {start_idx}")

# ---------------------------
# Retrieval (FAISS)
# ---------------------------
print("Running retrieval...")

for i in tqdm(range(start_idx, len(all_queries))):

    topic, query = all_queries[i]

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    ).cpu()

    # Normalize query
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

    query_np = query_embedding.numpy().astype("float32").reshape(1, -1)

    # FAISS search
    scores, indices = index.search(query_np, TOP_K)

    indices = indices[0]
    scores = scores[0]

    # ranking logic unchanged
    top_datasets = []
    for rank, idx in enumerate(reversed(indices)):
        top_datasets.append(f"Dataset {rank+1}: {dataset_titles[idx]} ({rank})")

    results.append({
        "User_Query": query,
        "Assistant": " ".join(top_datasets)
    })

    # ---------------------------
    # Save checkpoint every query
    # ---------------------------
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "last_index": i,
            "results": results
        }, f, ensure_ascii=False, indent=2)

# ---------------------------
# Save final output
# ---------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Remove checkpoint after success
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"✅ Dense retrieval complete. Results saved to {OUTPUT_FILE}")
