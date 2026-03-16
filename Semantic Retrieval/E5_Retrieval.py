import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
E5_MODEL = "intfloat/e5-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LENGTH = 512
OUTPUT_FILE = "e5_dense_retrieval_results.json"

DATASET_FILE = "datasets_metadata.jsonl"

SIMILARITY_CHUNK = 50000   # compare 50k vectors at a time

# -------------------------------------------------
# LOAD QUERIES
# -------------------------------------------------
with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

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
# EMBED DATASETS (BATCHED)
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
# RETRIEVAL
# -------------------------------------------------
results = []

print("Running retrieval...")

for topic, queries in tqdm(queries_by_topic.items(), desc="Topics"):

    for query in queries:

        query_input = f"query: {query}"

        query_embedding = embed_texts([query_input])

        # chunked cosine similarity
        best_scores = []
        best_indices = []

        for start in range(0, len(dataset_embeddings), SIMILARITY_CHUNK):

            end = start + SIMILARITY_CHUNK

            chunk = dataset_embeddings[start:end]

            scores = torch.matmul(query_embedding, chunk.T).squeeze(0)

            top_k = torch.topk(scores, k=3)

            for score, idx in zip(top_k.values, top_k.indices):

                best_scores.append(score.item())
                best_indices.append(start + idx.item())

        # global top3
        top3 = sorted(
            zip(best_scores, best_indices),
            key=lambda x: x[0],
            reverse=True
        )[:3]

        ranked_results = []

        for rank, (_, idx) in enumerate(reversed(top3)):

            ranked_results.append({
                "rank": rank,
                "dataset_title": all_datasets[idx]["title"]
            })

        results.append({
            "User_Query": query,
            "Assistant": ranked_results
        })

# -------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nDense retrieval completed")
print(f"Output saved to: {OUTPUT_FILE}")
