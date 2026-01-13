import json
import random
from math import floor
from collections import defaultdict, Counter

QUERIES_BY_TOPIC_FILE = "queries_by_topic.json"
QUERIES_WITH_RESULTS_FILE = "groundtruth_faiss+LLM.json"
OUTPUT_FILE = "sampled_1k.json"

TARGET_SAMPLE_SIZE = 1000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --------------------------------
# 1. Load queries by topic
# --------------------------------
with open(QUERIES_BY_TOPIC_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# Build query -> topic map
query_to_topic = {}
for topic, queries in queries_by_topic.items():
    for q in queries:
        query_to_topic[q.strip()] = topic

# --------------------------------
# 2. Load full query → dataset file
# --------------------------------
query_data = {}
with open(QUERIES_WITH_RESULTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        q = obj["query"].strip()

        if q in query_to_topic:
            query_data[q] = {
                "query": q,
                "topic": query_to_topic[q],
                "datasets": [r["title"] for r in obj.get("final_results", [])]
            }

# --------------------------------
# 3. Group by topic
# --------------------------------
by_topic = defaultdict(list)
for item in query_data.values():
    by_topic[item["topic"]].append(item)

topic_counts = {t: len(v) for t, v in by_topic.items()}
total_queries = sum(topic_counts.values())

# --------------------------------
# 4. Proportional sample sizes
# --------------------------------
topic_sample_sizes = {}
for topic, count in topic_counts.items():
    proportion = count / total_queries
    topic_sample_sizes[topic] = max(1, floor(proportion * TARGET_SAMPLE_SIZE))

# Fix rounding overflow
while sum(topic_sample_sizes.values()) > TARGET_SAMPLE_SIZE:
    largest = max(topic_sample_sizes, key=topic_sample_sizes.get)
    topic_sample_sizes[largest] -= 1

# --------------------------------
# 5. Sample
# --------------------------------
sampled = []
for topic, size in topic_sample_sizes.items():
    sampled.extend(
        random.sample(by_topic[topic], min(size, len(by_topic[topic])))
    )

random.shuffle(sampled)

# --------------------------------
# 6. Save output (topic INCLUDED)
# --------------------------------
final_output = [
    {
        "query": item["query"],
        "topic": item["topic"],
        "datasets": item["datasets"]
    }
    for item in sampled
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

# --------------------------------
# 7. Sanity check
# --------------------------------
print(f"Saved {len(final_output)} queries to {OUTPUT_FILE}")
print("Topic distribution:")
print(Counter([item["topic"] for item in sampled]))
