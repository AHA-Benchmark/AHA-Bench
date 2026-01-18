import json
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load JSONL file
# -----------------------------
rows = []
with open("GPT4mini_jude_results.jsonl", "r") as f:
    for line in f:
        item = json.loads(line.strip())
        query = item["query"]
        for j in item["judgments"]:
            rows.append({
                "query": query,
                "title": j["title"],
                "relevant": j["relevant"]
            })

df = pd.DataFrame(rows)

# -----------------------------
# Compute per-query metrics
# -----------------------------
summary = df.groupby("query")["relevant"].agg(relevant_in_top5="sum").reset_index()
summary["precision_at_5"] = summary["relevant_in_top5"] / 5

# -----------------------------
# Save per-query results to CSV (no printing)
# -----------------------------
summary.to_csv("per_query_metrics.csv", index=False)

# -----------------------------
# Global metrics
# -----------------------------
avg_p_at_5 = summary["precision_at_5"].mean()
success_at_5 = (summary["relevant_in_top5"] > 0).mean()

# Save global metrics to a separate CSV
global_metrics = pd.DataFrame({
    "Average_Precision_at_5": [avg_p_at_5],
    "Success_at_5": [success_at_5]
})
global_metrics.to_csv("global_metrics.csv", index=False)

# -----------------------------
# Count queries by number of relevant datasets
# -----------------------------
counts = summary["relevant_in_top5"].value_counts().sort_index()
for i in range(6):
    if i not in counts:
        counts[i] = 0
counts = counts.sort_index()

# Save count distribution to CSV
counts_df = counts.reset_index()
counts_df.columns = ["Number_of_relevant_datasets", "Number_of_queries"]
counts_df.to_csv("relevant_counts_distribution.csv", index=False)

# -----------------------------
# Bar plot: Distribution of queries by relevant count
# -----------------------------
plt.figure(figsize=(8, 6))
plt.bar(counts.index.astype(str), counts.values, color='skyblue')
plt.xlabel("Number of relevant datasets in top 5")
plt.ylabel("Number of queries")
plt.title("Distribution of queries by number of relevant datasets")
for i, v in enumerate(counts.values):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.tight_layout()
plt.savefig("queries_by_relevant_count.png", dpi=300)
plt.close()  # Close plot instead of showing it
