import os
import json
import time
from tqdm import tqdm
from datetime import datetime
from together import Together

# =====================================================
# 1️⃣ Set Together API key
# =====================================================
os.environ["TOGETHER_API_KEY"] = #API Key

# Initialize Together client
client = Together()
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# =====================================================
# 2️⃣ Load queries
# =====================================================
QUERIES_FILE = "queries_by_topic.json"

with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

print(f"Loaded queries for {len(queries_by_topic)} topics")

flat_queries = []
for topic, queries in queries_by_topic.items():
    for q in queries:
        flat_queries.append({"topic": topic, "query": q})

print(f"Total queries to process: {len(flat_queries)}")

# =====================================================
# 3️⃣ Few-shot prompt template
# =====================================================
FEW_SHOT_PROMPT = """
You are a dataset recommender for the EU Open Data Portal (data.europa.eu).

Rules:
- You must recommend only datasets that exist on the EU Open Data Portal
- Output ONLY dataset titles, ranked by relevance
- Use relevance scores (2 = most relevant, 1 = relevant, 0 = weakly relevant)
- Output at most 3 dataset titles
- Do NOT include descriptions, URLs, explanations, or extra text

Examples:

User: Can you find me a dataset with the average property prices in London?    
Assistant: "Sure, below are the dataset which include average property prices of London : 
Dataset 1: Average House Prices by Borough, Ward, MSOA & LSOA (2)
Dataset 2: Ratio of House Prices to Earnings, Borough (2)
Dataset 3: UK House Price Index (1) 

User: Can I access data on food waste collection volumes across London boroughs?    
Assistant: "Sure below i have listed datasets on food waste collection volumes across London boroughs
Dataset 1: Household Waste Recycling Rates, Borough (2)
Dataset 2: Waste Reduction and Recycling Plans (RRPs) (1) 
Dataset 3: Ealing Reduction and Recycling Plan (1)
                        
User: Are there publicly available datasets tracking accessibility in UK public transport systems?    
Assistant: "Yes, below listed are the publicly available datasets tracking accessibility in UK public transport systems
Dataset 1: Accessibility of London Underground Stations(2)
Dataset 2: Public Transport Journeys by Type of Transport (1)
Dataset 3:Origin and destination of public transport journeys(1)

User: {query}
Assistant:
"""

# =====================================================
# 4️⃣ Answer generation function
# =====================================================
def generate_answer(query, max_tokens=200, temperature=0.4, top_p=0.9):
    """
    Generate an answer using Together v2 SDK.
    """
    prompt = FEW_SHOT_PROMPT.format(query=query)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error generating answer for query '{query}': {e}")
        return ""

# =====================================================
# 5️⃣ Output files
# =====================================================
OUTPUT_FILE = "llm_answers_by_topic_together.json"
LOG_FILE = f"llm_answers_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    completed_queries = {(r["topic"], r["User_Query"]) for r in results}
    print(f"Resuming from previous run. Already completed {len(completed_queries)} queries.")
else:
    results = []
    completed_queries = set()

print("🧠 Generating answers...")

with open(LOG_FILE, "a", encoding="utf-8") as log:
    if os.path.getsize(LOG_FILE) == 0:
        log.write(f"LLM Answering Log - {datetime.now()}\n")
        log.write(f"Model: {MODEL_NAME}\n")
        log.write("=" * 60 + "\n")

    for item in tqdm(flat_queries, desc="Processing Queries"):
        topic = item["topic"]
        query = item["query"]

        if (topic, query) in completed_queries:
            continue

        answer = generate_answer(query)
        results.append({
            "topic": topic,
            "User_Query": query,
            "Assistant": answer
        })

        # Logging
        log.write(f"Topic: {topic}\n")
        log.write(f"User_Query: {query}\n")
        log.write(f"Assistant: {answer}\n")
        log.write("-" * 40 + "\n")
        log.flush()

        # Save results incrementally
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.1)

print("\n✅ Answer generation complete!")
print(f"📁 Saved answers to: {OUTPUT_FILE}")
print(f"📝 Log file: {LOG_FILE}")
