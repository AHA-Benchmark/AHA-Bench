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

# -------------------------------------------------
# LOAD INPUT FILES
# -------------------------------------------------
with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

with open("datasets_landing_metadata.json", "r", encoding="utf-8") as f:
    datasets_by_topic = json.load(f)

# -------------------------------------------------
# FLATTEN DATASETS (USE FULL METADATA)
# -------------------------------------------------
all_datasets = []

for topic, datasets in datasets_by_topic.items():
    for ds in datasets:
        dataset_text = (
            f"Title: {ds.get('title','')} "
            f"Description: {ds.get('description','')} "
            f"Categories: {' '.join(ds.get('categories', []))} "
            f"Publisher: {ds.get('publisher','')} "
            f"Topic: {topic}"
        )

        all_datasets.append({
            "title": ds.get("title", ""),
            "text": f"passage: {dataset_text}"
        })

print(f"Loaded {len(all_datasets)} datasets")

# -------------------------------------------------
# LOAD E5 MODEL
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(E5_MODEL)
model = AutoModel.from_pretrained(E5_MODEL).to(DEVICE)
model.eval()

# -------------------------------------------------
# EMBEDDING FUNCTION (MEAN POOLING)
# -------------------------------------------------
def embed_texts(texts):
    all_embeddings = []

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
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

# -------------------------------------------------
# EMBED DATASETS
# -------------------------------------------------
print("Embedding dataset metadata...")
dataset_embeddings = embed_texts([ds["text"] for ds in all_datasets])

# -------------------------------------------------
# RETRIEVAL
# -------------------------------------------------
results = []

print("Running E5 + cosine retrieval...")
for topic, queries in tqdm(queries_by_topic.items(), desc="Topics"):
    for query in queries:
        query_input = f"query: {query}"
        query_embedding = embed_texts([query_input])

        # Cosine similarity
        scores = torch.matmul(query_embedding, dataset_embeddings.T).squeeze(0)

        top_k = torch.topk(scores, k=3)

        # Rank: 2 = most relevant, 0 = least relevant
        ranked_results = []
        for rank, idx in enumerate(reversed(top_k.indices.tolist())):
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

print("\n✅ Dense retrieval completed")
print(f"📁 Output saved to: {OUTPUT_FILE}")
