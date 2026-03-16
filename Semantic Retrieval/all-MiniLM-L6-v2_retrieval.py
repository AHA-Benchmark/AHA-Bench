import json
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
OUTPUT_FILE = "dense_retrieval_top3.json"

DATASET_FILE = "datasets_metadata.jsonl"

BATCH_SIZE = 256
SIMILARITY_CHUNK = 50000   # compare 50k vectors at a time

# ---------------------------
# Load queries
# ---------------------------
with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

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
# Retrieval
# ---------------------------
results = []

print("Running retrieval...")

for topic, queries in queries_by_topic.items():

    for query in queries:

        query_embedding = model.encode(
            query,
            convert_to_tensor=True
        ).cpu()

        best_scores = []
        best_indices = []

        # chunk similarity to avoid large matrix
        for start in range(0, len(dataset_embeddings), SIMILARITY_CHUNK):

            end = start + SIMILARITY_CHUNK
            chunk = dataset_embeddings[start:end]

            cosine_scores = util.cos_sim(query_embedding, chunk)[0]

            top_results = torch.topk(cosine_scores, k=TOP_K)

            for score, idx in zip(top_results.values, top_results.indices):
                best_scores.append(score.item())
                best_indices.append(start + idx.item())

        # global top3
        combined = list(zip(best_scores, best_indices))
        combined = sorted(combined, key=lambda x: x[0], reverse=True)[:TOP_K]

        indices = [x[1] for x in combined]

        # ranking logic unchanged
        top_datasets = []
        for rank, idx in enumerate(reversed(indices)):
            top_datasets.append(f"Dataset {rank+1}: {dataset_titles[idx]} ({rank})")

        results.append({
            "User_Query": query,
            "Assistant": " ".join(top_datasets)
        })

# ---------------------------
# Save output
# ---------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Dense retrieval complete. Results saved to {OUTPUT_FILE}")
