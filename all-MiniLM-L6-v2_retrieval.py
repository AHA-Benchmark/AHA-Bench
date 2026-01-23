import json
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # number of datasets to retrieve
OUTPUT_FILE = "dense_retrieval_top3.json"

# ---------------------------
# Load data
# ---------------------------
with open("datasets_landing_metadata.json", "r", encoding="utf-8") as f:
    datasets_by_topic = json.load(f)

with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# Flatten datasets into a single list with title & topic
all_datasets = []
for topic, datasets in datasets_by_topic.items():
    for ds in datasets:
        all_datasets.append({"title": ds['title'], "topic": topic})

# ---------------------------
# Initialize model
# ---------------------------
print(f"Loading sentence-transformers model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# Precompute embeddings for all datasets
dataset_titles = [d['title'] for d in all_datasets]
dataset_embeddings = model.encode(dataset_titles, convert_to_tensor=True, show_progress_bar=True)

# ---------------------------
# Retrieval function
# ---------------------------
results = []

for topic, queries in queries_by_topic.items():
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)
        # Compute cosine similarity
        cosine_scores = util.cos_sim(query_embedding, dataset_embeddings)[0]
        # Get top-K indices
        top_results = torch.topk(cosine_scores, k=TOP_K)
        indices = top_results.indices.tolist()
        scores = top_results.values.tolist()

        # Assign rank: 2 = most relevant, 0 = least relevant in top 3
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
