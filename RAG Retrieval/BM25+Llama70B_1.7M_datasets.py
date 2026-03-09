import os
import json
import time
from tqdm import tqdm
from datetime import datetime
from elasticsearch import Elasticsearch, helpers
from together import Together
import hashlib
import shutil
from functools import lru_cache

# =====================================================
# 1️⃣ Together API Key
# =====================================================
os.environ["TOGETHER_API_KEY"] = #APIKEY
client = Together()
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# =====================================================
# 2️⃣ File paths
# =====================================================
DATASETS_FILE = "datasets_metadata.jsonl"
QUERIES_FILE = "queries_by_topic.json"
OUTPUT_FILE = "es_llama_rag_results.json"
LOG_FILE = f"es_llama_rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

TOP_K_RETRIEVAL = 5
SLEEP_TIME = 0.2
SAVE_INTERVAL = 10  # Save every 10 queries

# =====================================================
# 3️⃣ Elasticsearch setup
# =====================================================
ES_USER = "elastic"
ES_PASSWORD = #PASSWORD
ES_TIMEOUT = 30
ES_MAX_RETRIES = 3
INDEX_NAME = "datasets_index"

def get_es_client(max_retries=5, wait_time=5):
    """Return an Elasticsearch client with automatic retry if connection fails"""
    for attempt in range(max_retries):
        try:
            es = Elasticsearch(
                "https://127.0.0.1:9200",
                basic_auth=(ES_USER, ES_PASSWORD),
                verify_certs=False,
                max_retries=ES_MAX_RETRIES,
                retry_on_timeout=True,
                request_timeout=ES_TIMEOUT
            )
            if es.ping():
                print("✅ Connected to Elasticsearch")
                return es
            else:
                raise ConnectionError("Elasticsearch ping failed")
        except Exception as e:
            print(f"⚠️ Elasticsearch connection failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(wait_time * (attempt + 1))
    raise ConnectionError("❌ Could not connect to Elasticsearch after multiple attempts")

es = get_es_client()

# =====================================================
# 4️⃣ Indexing datasets (if index empty)
# =====================================================
def index_datasets():
    """Index datasets with auto-reconnect and only if index empty"""
    try:
        count = es.count(index=INDEX_NAME)['count']
        if count > 0:
            print(f"✅ Index already contains {count} documents, skipping indexing")
            return
    except:
        pass

    if not os.path.exists(OUTPUT_FILE):
        # Delete old index if exists
        if es.indices.exists(index=INDEX_NAME):
            print(f"Deleting old index: {INDEX_NAME}")
            es.indices.delete(index=INDEX_NAME)
        # Create new index
        es.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "refresh_interval": "30s"
                },
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "boost": 2},
                        "description": {"type": "text"},
                        "original_id": {"type": "keyword"}
                    }
                }
            }
        )

    print("Indexing datasets to Elasticsearch...")
    actions = []

    with open(DATASETS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                original_id = doc.get("id", "")
                safe_id = (hashlib.sha256(original_id.encode('utf-8')).hexdigest()
                           if original_id and len(original_id.encode('utf-8')) > 500
                           else original_id)

                action = {
                    "_index": INDEX_NAME,
                    "_source": {
                        "title": doc.get("title", ""),
                        "description": doc.get("description", ""),
                        "original_id": original_id
                    }
                }
                if safe_id:
                    action["_id"] = safe_id
                actions.append(action)

                # Bulk indexing in batches
                if len(actions) >= 5000:
                    try:
                        helpers.bulk(es, actions, stats_only=True, raise_on_error=False)
                    except:
                        print("⚠️ Bulk indexing failed, reconnecting ES...")
                        es = get_es_client()
                        helpers.bulk(es, actions, stats_only=True, raise_on_error=False)
                    actions = []
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    if actions:
        try:
            helpers.bulk(es, actions, stats_only=True, raise_on_error=False)
        except:
            es = get_es_client()
            helpers.bulk(es, actions, stats_only=True, raise_on_error=False)

index_datasets()

# =====================================================
# 5️⃣ Load queries
# =====================================================
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

flat_queries = [{"topic": t, "query": q} for t, qs in queries_by_topic.items() for q in qs]
print(f"✅ Loaded {len(flat_queries)} queries")

# =====================================================
# 6️⃣ Resume mechanism
# =====================================================
def safe_load_results(output_file):
    if not os.path.exists(output_file):
        return [], set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed = {(r["topic"], r["User_Query"]) for r in results}
        print(f"🔁 Resuming from {output_file} — {len(completed)} queries already done")
        return results, completed
    except Exception as e:
        print(f"⚠️ Could not load results: {e}")
        return [], set()

def safe_save_results(results, output_file):
    try:
        temp_file = output_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, output_file)
        return True
    except Exception as e:
        print(f"⚠️ Save error: {e}")
        return False

results, completed = safe_load_results(OUTPUT_FILE)

# =====================================================
# 7️⃣ RAG prompt
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

def format_retrieved(datasets):
    return "\n\n".join([f"{i+1}. Title: {d['title']}\n   Description: {d['description']}"
                        for i, d in enumerate(datasets)])

# =====================================================
# 8️⃣ LLaMA RAG generation
# =====================================================
def generate_rag_answer(query, retrieved, retries=3):
    prompt = RAG_PROMPT.format(retrieved_datasets=format_retrieved(retrieved), query=query)
    for attempt in range(retries):
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
            if attempt == retries - 1:
                print(f"❌ LLM error after {retries} attempts: {e}")
            else:
                print(f"⚠️ LLM error, retrying: {e}")
                time.sleep(2 ** attempt)
    return ""

# =====================================================
# 9️⃣ Elasticsearch search with caching
# =====================================================
@lru_cache(maxsize=100)
def cached_search(query_hash, query):
    return search_with_retry(query, use_cache=False)

def search_with_retry(query, max_retries=3, use_cache=True):
    if use_cache:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        try:
            return cached_search(query_hash, query)
        except:
            cached_search.cache_clear()
    for attempt in range(max_retries):
        try:
            res = es.search(
                index=INDEX_NAME,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "description"],
                            "type": "best_fields",
                            "operator": "or",
                            "tie_breaker": 0.3
                        }
                    },
                    "size": TOP_K_RETRIEVAL
                },
                request_timeout=ES_TIMEOUT
            )
            return res
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Search failed after {max_retries} attempts: {e}")
                return None
            wait = 2 ** attempt
            print(f"⚠️ Search error, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)

# =====================================================
# 🔟 Main RAG loop
# =====================================================
print("🧠 Running Elasticsearch + LLaMA RAG...")
print(f"📊 Total queries: {len(flat_queries)}, Completed: {len(completed)}, Remaining: {len(flat_queries) - len(completed)}")

with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"Elasticsearch + LLaMA RAG Log — {datetime.now()}\n")
    log.write("=" * 70 + "\n")
    log.write(f"Resuming with {len(completed)} completed queries\n\n")

    last_save_count = len(results)

    for i, item in enumerate(tqdm(flat_queries, desc="Processing Queries")):
        topic = item["topic"]
        query = item["query"]

        if (topic, query) in completed:
            continue

        # Ensure ES is connected
        if not es.ping():
            print("⚠️ Lost Elasticsearch connection, reconnecting...")
            es = get_es_client()

        # Search
        res = search_with_retry(query)
        if res is None:
            results.append({
                "topic": topic,
                "User_Query": query,
                "Retrieved_Datasets": [],
                "Assistant": "ERROR: Search failed"
            })
            completed.add((topic, query))
            log.write(f"❌ FAILED - Topic: {topic}\nQuery: {query}\nError: Search failed\n{'-'*50}\n")
            log.flush()
            continue

        retrieved = [{"title": hit["_source"]["title"], "description": hit["_source"]["description"]}
                     for hit in res["hits"]["hits"]]

        answer = generate_rag_answer(query, retrieved)

        record = {
            "topic": topic,
            "User_Query": query,
            "Retrieved_Datasets": retrieved,
            "Assistant": answer
        }

        results.append(record)
        completed.add((topic, query))

        log.write(f"Topic: {topic}\nQuery: {query}\nAnswer:\n{answer}\n{'-'*50}\n")
        log.flush()

        if len(results) - last_save_count >= SAVE_INTERVAL:
            if safe_save_results(results, OUTPUT_FILE):
                print(f"✅ Progress saved: {len(results)} records")
                last_save_count = len(results)

        time.sleep(SLEEP_TIME)

# Final save
print("💾 Performing final save...")
if safe_save_results(results, OUTPUT_FILE):
    print(f"✅ Final results saved to: {OUTPUT_FILE}")
else:
    print("❌ CRITICAL: Could not save final results!")

print(f"\n✅ RAG run complete with {len(results)} queries processed")
print(f"📁 Results saved to: {OUTPUT_FILE}")
print(f"📝 Log file: {LOG_FILE}")
