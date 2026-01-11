import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Helpers
# -----------------------------
def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

# -----------------------------
# Load ground truth
# -----------------------------
with open("datasets_with_conversational_queries.json", "r", encoding="utf-8") as f:
    query_data = json.load(f)

query_to_gt = {}
for item in query_data:
    gt_title = normalize(item["title"])

    if item.get("described_dataset_query"):
        query_to_gt[normalize(item["described_dataset_query"])] = {
            "gt_title": gt_title,
            "type": "described"
        }

    if item.get("implied_dataset_query"):
        query_to_gt[normalize(item["implied_dataset_query"])] = {
            "gt_title": gt_title,
            "type": "implied"
        }

# -----------------------------
# Load retrieval results
# -----------------------------
results = []
with open("groundtruth_faiss+LLM.json", "r", encoding="utf-8") as f:
    for line in f:
        results.append(json.loads(line))

# -----------------------------
# Compute Exact Match @5
# -----------------------------
stats = defaultdict(lambda: {
    "total": 0,
    "exact_match_at_5": 0
})

per_query = []

for res in results:
    q = normalize(res["query"])
    if q not in query_to_gt:
        continue

    gt = query_to_gt[q]["gt_title"]
    qtype = query_to_gt[q]["type"]

    retrieved_titles = [
        normalize(r["title"]) for r in res.get("final_results", [])
    ]

    stats[qtype]["total"] += 1
    match = gt in retrieved_titles

    if match:
        stats[qtype]["exact_match_at_5"] += 1

    per_query.append({
        "query": res["query"],
        "query_type": qtype,
        "ground_truth_title": gt,
        "exact_match_at_5": match
    })

# -----------------------------
# Save metrics
# -----------------------------
final_metrics = {}
for qtype, s in stats.items():
    final_metrics[qtype] = {
        "total_queries": s["total"],
        "exact_match_at_5": s["exact_match_at_5"],
        "exact_match_rate": (
            s["exact_match_at_5"] / s["total"] if s["total"] > 0 else 0
        )
    }

with open("exact_match_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)

with open("per_query_exact_match.json", "w", encoding="utf-8") as f:
    json.dump(per_query, f, indent=2)

# -----------------------------
# Bar chart
# -----------------------------
labels = list(final_metrics.keys())
values = [final_metrics[k]["exact_match_rate"] for k in labels]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Exact Title Match @5")
plt.ylim(0, 1)
plt.title("Exact Title Match @5 by Query Type")
plt.savefig("exact_match_bar_chart.png")
plt.close()

# -----------------------------
# Pie chart
# -----------------------------
total = sum(s["total"] for s in stats.values())
matched = sum(s["exact_match_at_5"] for s in stats.values())

plt.figure()
plt.pie(
    [matched, total - matched],
    labels=["Exact Match", "No Match"],
    autopct="%1.1f%%"
)
plt.title("Overall Exact Title Match @5")
plt.savefig("exact_match_pie_chart.png")
plt.close()

print("Evaluation complete.")
print("Saved:")
print("- exact_match_metrics.json")
print("- per_query_exact_match.json")
print("- exact_match_bar_chart.png")
print("- exact_match_pie_chart.png")
