import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
HF_CACHE_DIR = "/nas/netstore/ldv/ge83qiw/hf_cache"

DATASETS_FILE = "datasets_landing_metadata.json"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "qwen_dense_retrieval_top3.json"

MAX_LENGTH = 512
BATCH_SIZE = 16

# -------------------------------------------------
# Load model & tokenizer
# -------------------------------------------------
print("🔄 Loading Qwen embedding model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, cache_dir=HF_CACHE_DIR
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    cache_dir=HF_CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

device = next(model.parameters()).device
print(f"✅ Model loaded on {device}")

# -------------------------------------------------
# Mean pooling (E5-style)
# -------------------------------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )

# -------------------------------------------------
# Encode texts
# -------------------------------------------------
def encode_texts(texts):
    embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            output = model(**inputs)
            emb = mean_pooling(output, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)

# -------------------------------------------------
# Load datasets
# -------------------------------------------------
with open(DATASETS_FILE, "r", encoding="utf-8") as f:
    datasets_by_topic = json.load(f)

datasets = []
dataset_titles = []

for topic, items in datasets_by_topic.items():
    for ds in items:
        text = f"{ds.get('title', '')}. {ds.get('description', '')}"
        datasets.append(text)
        dataset_titles.append(ds.get("title", "Unknown"))

print(f"📦 Total datasets: {len(datasets)}")

# -------------------------------------------------
# Encode datasets once
# -------------------------------------------------
dataset_embeddings = encode_texts(datasets)
dataset_embeddings = dataset_embeddings.numpy()

# -------------------------------------------------
# Load queries
# -------------------------------------------------
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# Flatten queries
queries = []
for topic, qs in queries_by_topic.items():
    for q in qs:
        queries.append(q)

print(f"🔍 Total queries: {len(queries)}")

# -------------------------------------------------
# Encode queries
# -------------------------------------------------
query_embeddings = encode_texts(queries).numpy()

# -------------------------------------------------
# Cosine similarity retrieval
# -------------------------------------------------
results = []

for q_idx, query in enumerate(tqdm(queries, desc="Retrieving")):
    q_emb = query_embeddings[q_idx]

    scores = np.dot(dataset_embeddings, q_emb)
    top_indices = np.argsort(scores)[-3:][::-1]

    assistant_output = {
        "Dataset 1": dataset_titles[top_indices[0]],
        "Dataset 2": dataset_titles[top_indices[1]],
        "Dataset 3": dataset_titles[top_indices[2]],
    }

    ranked_output = {
        "User_Query": query,
        "Assistant": {
            "rank_2": assistant_output["Dataset 1"],
            "rank_1": assistant_output["Dataset 2"],
            "rank_0": assistant_output["Dataset 3"],
        }
    }

    results.append(ranked_output)

# -------------------------------------------------
# Save output
# -------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Dense retrieval complete!")
print(f"📁 Results saved to: {OUTPUT_FILE}")
