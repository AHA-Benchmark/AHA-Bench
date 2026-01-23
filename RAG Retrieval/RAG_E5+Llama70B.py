import os
import json
import time
import faiss
import torch
from tqdm import tqdm
from datetime import datetime
from together import Together
from sentence_transformers import SentenceTransformer

# =====================================================
# 1️⃣ Together API Key
# =====================================================
os.environ["TOGETHER_API_KEY"] = #API_key
client = Together()

LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EMBED_MODEL = "intfloat/e5-base-v2"

# =====================================================
# 2️⃣ Files & parameters
# =====================================================
DATASETS_FILE = "datasets_landing_metadata.json"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "e5_llama70b_rag_results.json"
LOG_FILE = f"e5_llama70b_rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

TOP_K_RETRIEVAL = 5
SLEEP_TIME = 0.2

# =====================================================
# 3️⃣ Load E5 embedding model
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)
print("✅ E5 embedding model loaded")

# =====================================================
# 4️⃣ Load dataset metadata
# =====================================================
with open(DATASETS_FILE, "r", encoding="utf-8") as f:
    datasets_by_topic = json.load(f)

dataset_records = []
corpus_texts = []

for topic, datasets in datasets_by_topic.items():
    for ds in datasets:
        title = ds.get("title", "").strip()
        desc = ds.get("description", "").strip()
        text = f"{title}. {desc}"
        dataset_records.append({
            "title": title,
            "description": desc
        })
        corpus_texts.append(f"passage: {text}")

print(f"✅ Loaded {len(dataset_records)} datasets")

# =====================================================
# 5️⃣ Build FAISS index (E5)
# =====================================================
print("🔄 Encoding dataset embeddings...")
corpus_embeddings = embedder.encode(
    corpus_texts,
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_embeddings)

print("✅ FAISS index built")

# =====================================================
# 6️⃣ Load queries (flattened)
# =====================================================
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

flat_queries = []
for topic, queries in queries_by_topic.items():
    for q in queries:
        flat_queries.append({
            "topic": topic,
            "query": q
        })

print(f"✅ Loaded {len(flat_queries)} queries")

# =====================================================
# 7️⃣ Resume mechanism
# =====================================================
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    completed = {(r["topic"], r["User_Query"]) for r in results}
    print(f"🔁 Resuming — {len(completed)} queries already completed")
else:
    results = []
    completed = set()

# =====================================================
# 8️⃣ RAG Prompt (STRICT)
# =====================================================
RAG_PROMPT = """
You are a dataset recommender for the EU Open Data Portal (data.europa.eu).

IMPORTANT RULES:
- You MUST use ONLY the datasets listed below
- You MUST NOT invent, rename, or modify dataset titles
- You MUST output ONLY dataset titles with relevance scores

Retrieved Datasets:
{retrieved_datasets}

Task:
Rank up to 3 datasets by relevance to the user query.

Scoring:
2 = most relevant
1 = relevant
0 = weakly relevant

Output format (STRICT):
Dataset 1: <TITLE> (score)
Dataset 2: <TITLE> (score)
Dataset 3: <TITLE> (score)

User Query:
{query}

Answer:
"""

# =====================================================
# 9️⃣ Helper: format retrieved datasets
# =====================================================
def format_retrieved(datasets):
    blocks = []
    for i, ds in enumerate(datasets, 1):
        blocks.append(
            f"{i}. Title: {ds['title']}\n   Description: {ds['description']}"
        )
    return "\n\n".join(blocks)

# =====================================================
# 🔟 LLaMA generation
# =====================================================
def generate_rag_answer(query, retrieved):
    prompt = RAG_PROMPT.format(
        retrieved_datasets=format_retrieved(retrieved),
        query=query
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return ""

# =====================================================
# 1️⃣1️⃣ Main RAG loop (E5 + LLaMA)
# =====================================================
print("🧠 Running E5 + LLaMA-70B RAG...")

with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"E5 + LLaMA-70B RAG Log — {datetime.now()}\n")
    log.write("=" * 70 + "\n")

    for item in tqdm(flat_queries, desc="Processing Queries"):
        topic = item["topic"]
        query = item["query"]

        if (topic, query) in completed:
            continue

        # ---- Dense Retrieval (E5) ----
        query_embedding = embedder.encode(
            f"query: {query}",
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(
            query_embedding.reshape(1, -1),
            TOP_K_RETRIEVAL
        )

        retrieved = [dataset_records[i] for i in indices[0]]

        # ---- Generation ----
        answer = generate_rag_answer(query, retrieved)

        record = {
            "topic": topic,
            "User_Query": query,
            "Retrieved_Datasets": retrieved,
            "Assistant": answer
        }

        results.append(record)
        completed.add((topic, query))

        # ---- Logging ----
        log.write(f"Topic: {topic}\n")
        log.write(f"Query: {query}\n")
        log.write(f"Answer:\n{answer}\n")
        log.write("-" * 50 + "\n")
        log.flush()

        # ---- Incremental save ----
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_TIME)

print("\n✅ E5 + LLaMA-70B RAG complete")
print(f"📁 Results saved to: {OUTPUT_FILE}")
print(f"📝 Log file: {LOG_FILE}")
