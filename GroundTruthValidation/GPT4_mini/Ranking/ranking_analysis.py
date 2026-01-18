import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # nicer plot style

# -----------------------------
# Load JSONL file
# -----------------------------
rows = []
with open("gpt_validation_results.jsonl", "r") as f:
    for line in f:
        item = json.loads(line.strip())
        query = item["query"]
        for j in item["gpt_scores"]:
            rows.append({
                "query": query,
                "title": j["title"],
                "score": j["score"]
            })

df = pd.DataFrame(rows)

# -----------------------------
# Compute per-query metrics
# -----------------------------
summary = df.groupby("query")["score"].agg(
    max_score="max",
    avg_score="mean",
    top3_avg=lambda x: x.nlargest(3).mean(),
    high_score_count=lambda x: (x >= 3).sum()  # count of datasets with score >= 3
).reset_index()

# Save per-query summary
summary.to_csv("gpt_score_summary.csv", index=False)

# -----------------------------
# 1️⃣ Distribution of all scores
# -----------------------------
plt.figure(figsize=(8,6))
sns.histplot(df["score"], bins=range(0,6), kde=False, color="skyblue", edgecolor="black")
plt.xlabel("GPT Score")
plt.ylabel("Number of dataset-query pairs")
plt.title("Distribution of GPT Scores Across All Datasets")
plt.xticks(range(0,6))
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 2️⃣ High-scoring datasets per query
# -----------------------------
counts = summary["high_score_count"].value_counts().sort_index()

plt.figure(figsize=(8,6))
sns.barplot(x=counts.index.astype(str), y=counts.values, color="salmon")
plt.xlabel("Number of high-scoring datasets (score >= 3) per query")
plt.ylabel("Number of queries")
plt.title("Queries by High-Scoring Dataset Count")
for i, v in enumerate(counts.values):
    plt.text(i, v + 0.2, str(v), ha='center')
plt.tight_layout()
plt.savefig("high_score_counts.png", dpi=300)
plt.close()

# -----------------------------
# 3️⃣ Average vs Max Score per Query
# -----------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=summary, 
    x="avg_score", 
    y="max_score", 
    size="high_score_count", 
    hue="high_score_count", 
    palette="viridis", 
    sizes=(50,200),
    alpha=0.7,
    edgecolor="k"
)
plt.xlabel("Average Score per Query")
plt.ylabel("Max Score per Query")
plt.title("Average vs Max GPT Score per Query")
plt.legend(title="High-score count (score >= 3)", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("avg_vs_max_score.png", dpi=300)
plt.close()

# -----------------------------
# 4️⃣ Optional: top 3 average per query (compact view)
# -----------------------------
plt.figure(figsize=(10,6))
summary_sorted = summary.sort_values("top3_avg", ascending=False).head(20)  # top 20 queries for clarity
sns.barplot(
    x="top3_avg",
    y="query",
    data=summary_sorted,
    palette="coolwarm"
)
plt.xlabel("Top-3 Average Score")
plt.ylabel("Query")
plt.title("Top 20 Queries by Top-3 Average GPT Score")
plt.tight_layout()
plt.savefig("top3_avg_scores.png", dpi=300)
plt.close()
