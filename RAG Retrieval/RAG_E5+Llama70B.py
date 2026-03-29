import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from together import Together
from sentence_transformers import SentenceTransformer

# =====================================================
# 1️⃣ Together API Key
# =====================================================
os.environ["TOGETHER_API_KEY"] = #API Key
client = Together()

LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EMBED_MODEL = "intfloat/e5-base-v2"

# =====================================================
# 2️⃣ Files & parameters
# =====================================================
DATASETS_FILE = "datasets_metadata.jsonl"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "e5_llama70b_rag_results.json"
LOG_FILE = f"e5_llama70b_rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

TOP_K_RETRIEVAL = 5
SLEEP_TIME = 0.2
EMBED_BATCH_SIZE = 512

FAISS_INDEX_FILE = "datasets_faiss.index"
EMBEDDINGS_FILE = "dataset_embeddings.npy"

# =====================================================
# 3️⃣ Load E5 embedding model
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=device)
print("✅ E5 embedding model loaded")

# =====================================================
# 4️⃣ Load dataset metadata from JSONL
# =====================================================
dataset_records = []
corpus_texts = []

print("📂 Loading datasets from JSONL...")

with open(DATASETS_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):

        if not line.strip():
            continue

        item = json.loads(line)

        title = (item.get("title") or "").strip()
        desc = (item.get("description") or "").strip()

        if not title and not desc:
            continue
            
        dataset_records.append({
            "title": title,
            "description": desc
        })

        corpus_texts.append(f"passage: {title}. {desc}")

print(f"✅ Loaded {len(dataset_records)} datasets")

# =====================================================
# 5️⃣ Build / Load FAISS index
# =====================================================
dim = embedder.get_sentence_embedding_dimension()

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
    print("⚡ Loading FAISS index and embeddings from disk...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    all_embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
else:
    print("⚙️ Building FAISS index (first run)...")
    index = faiss.IndexFlatIP(dim)
    all_embeddings = []

    for i in tqdm(range(0, len(corpus_texts), EMBED_BATCH_SIZE)):
        batch = corpus_texts[i:i+EMBED_BATCH_SIZE]

        emb = embedder.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        index.add(emb)
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)

    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(EMBEDDINGS_FILE, all_embeddings)
    print("✅ FAISS index built and saved to disk")

# =====================================================
# 6️⃣ Load queries
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
# 7️⃣ Resume mechanism (Smart Resume)
# =====================================================
results = []
completed = set()
incomplete_queries = {}

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    for r in results:
        key = (r["topic"], r["User_Query"])
        assistant = r.get("Assistant", "").strip()
        if assistant == "":
            incomplete_queries[key] = r
        else:
            completed.add(key)

    print(f"🔁 Completed queries: {len(completed)}")
    print(f"⚠️ Incomplete queries to retry: {len(incomplete_queries)}")
else:
    print("🆕 No previous output file — starting fresh")

# =====================================================
# 8️⃣ RAG Prompt
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
# 🔟 LLaMA generation with 402-safe handling
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
        err_str = str(e)
        if "credit_limit" in err_str or "402" in err_str:
            print("❌ TogetherAI credits exhausted. Stopping safely.")
            return None  # stop the main loop
        else:
            print(f"⚠️ LLM error: {e}")
            return ""

# =====================================================
# 1️⃣1️⃣ Main RAG loop
# =====================================================
print("🧠 Running E5 + LLaMA-70B RAG...")

with open(LOG_FILE, "a", encoding="utf-8") as log:

    log.write(f"E5 + LLaMA-70B RAG Log — {datetime.now()}\n")
    log.write("=" * 70 + "\n")

    for item in tqdm(flat_queries, desc="Processing Queries"):

        topic = item["topic"]
        query = item["query"]
        key = (topic, query)

        if key in completed:
            continue

        # ---- Query embedding ----
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

        if answer is None:
            print(f"⏸️ Stopping loop at query: {query}")
            break  # stop safely if credits exhausted

        record = {
            "topic": topic,
            "User_Query": query,
            "Retrieved_Datasets": retrieved,
            "Assistant": answer
        }

        # ---- Update results incrementally ----
        if key in incomplete_queries:
            for r in results:
                if r["topic"] == topic and r["User_Query"] == query:
                    r["Assistant"] = answer
                    r["Retrieved_Datasets"] = retrieved
                    break
        else:
            results.append(record)

        completed.add(key)

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
