import os
import json
import re
import time
from tqdm import tqdm
from datetime import datetime
from rank_bm25 import BM25Okapi
from together import Together

# =====================================================
# 1️⃣ Together API Key
# =====================================================
os.environ["TOGETHER_API_KEY"] = #APIKey
client = Together()

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# =====================================================
# 2️⃣ File paths
# =====================================================
DATASETS_FILE = "datasets_landing_metadata.json"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "bm25_llama70b_rag_results.json"
LOG_FILE = f"bm25_llama70b_rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

TOP_K_RETRIEVAL = 5   # BM25 top-k
SLEEP_TIME = 0.2     # rate limiting

# =====================================================
# 3️⃣ Tokenizer for BM25
# =====================================================
def tokenize(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

# =====================================================
# 4️⃣ Load dataset metadata
# =====================================================
with open(DATASETS_FILE, "r", encoding="utf-8") as f:
    datasets_by_topic = json.load(f)

corpus_tokens = []
dataset_records = []

for topic, datasets in datasets_by_topic.items():
    for ds in datasets:
        title = ds.get("title", "").strip()
        desc = ds.get("description", "").strip()
        combined = f"{title} {desc}"
        corpus_tokens.append(tokenize(combined))
        dataset_records.append({
            "title": title,
            "description": desc
        })

print(f"✅ Loaded {len(dataset_records)} datasets")

# =====================================================
# 5️⃣ Build BM25 index
# =====================================================
bm25 = BM25Okapi(corpus_tokens)

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
    print(f"🔁 Resuming run — {len(completed)} queries already done")
else:
    results = []
    completed = set()

# =====================================================
# 8️⃣ RAG Prompt Template (STRICT)
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
# 🔟 LLaMA RAG generation
# =====================================================
def generate_rag_answer(query, retrieved):
    prompt = RAG_PROMPT.format(
        retrieved_datasets=format_retrieved(retrieved),
        query=query
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
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
# 1️⃣1️⃣ Main RAG Loop (CRASH SAFE)
# =====================================================
print("🧠 Running BM25 + LLaMA-70B RAG...")

with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"BM25 + LLaMA-70B RAG Log — {datetime.now()}\n")
    log.write("=" * 70 + "\n")

    for item in tqdm(flat_queries, desc="Processing Queries"):
        topic = item["topic"]
        query = item["query"]

        if (topic, query) in completed:
            continue

        # ---- BM25 Retrieval ----
        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:TOP_K_RETRIEVAL]

        retrieved = [dataset_records[i] for i in top_indices]

        # ---- LLM Generation ----
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

        # ---- Incremental save (CRITICAL) ----
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_TIME)

print("\n✅ RAG run complete")
print(f"📁 Results saved to: {OUTPUT_FILE}")
print(f"📝 Log file: {LOG_FILE}")
