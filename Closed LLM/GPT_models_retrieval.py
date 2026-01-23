import os
import json
import time
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

# =====================================================
# 1️⃣ Set OpenAI API key
# =====================================================
os.environ["OPENAI_API_KEY"] = #API KEY

# Initialize OpenAI client
client = OpenAI()

#MODEL_NAME = "gpt-4.1-mini"
#MODEL_NAME = "gpt-4.1"
#MODEL_NAME = "gpt-4.1-nano"
MODEL_NAME = "gpt-5.2"

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
- Recommend only datasets that exist on the EU Open Data Portal
- Output ONLY dataset titles, ranked by relevance
- Use relevance scores (2 = most relevant, 1 = relevant, 0 = weakly relevant)
- Output at most 3 dataset titles
- Do NOT include descriptions, URLs, explanations, or extra text

Examples:

User: Can you find me a dataset with the average property prices in London?    
Assistant: 
Dataset 1: Average House Prices by Borough, Ward, MSOA & LSOA (2)
Dataset 2: Ratio of House Prices to Earnings, Borough (2)
Dataset 3: UK House Price Index (1)

User: Can I access data on food waste collection volumes across London boroughs?    
Assistant: 
Dataset 1: Household Waste Recycling Rates, Borough (2)
Dataset 2: Waste Reduction and Recycling Plans (RRPs) (1)
Dataset 3: Ealing Reduction and Recycling Plan (1)

User: Are there publicly available datasets tracking accessibility in UK public transport systems?    
Assistant: 
Dataset 1: Accessibility of London Underground Stations (2)
Dataset 2: Public Transport Journeys by Type of Transport (1)
Dataset 3: Origin and destination of public transport journeys (1)

Now answer the following query:
{query}
"""

# =====================================================
# 4️⃣ Helper function: extract text from Responses API
# =====================================================
def extract_text_from_response(response):
    """
    Extract textual output from a Responses API response object.
    """
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    if hasattr(response, "output") and response.output:
        texts = []
        for item in response.output:
            if "content" in item:
                for c in item["content"]:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
        return "\n".join(texts) if texts else None

    return None

# =====================================================
# 5️⃣ Generate answer function
# =====================================================
def generate_answer(query, max_tokens=200):
    formatted_prompt = FEW_SHOT_PROMPT.format(query=query)
    
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": formatted_prompt}],
            max_output_tokens=max_tokens,
            store=False
        )
        answer_text = extract_text_from_response(response)
        if answer_text:
            return answer_text.strip()
        return "ERROR: No text returned"
    
    except Exception as e:
        print(f"⚠️ Error generating answer for query '{query[:50]}...': {type(e).__name__}: {str(e)[:100]}")
        time.sleep(2)
        return f"ERROR: {type(e).__name__}"

# =====================================================
# 6️⃣ Test API connection
# =====================================================
def test_api_connection():
    print("Testing API connection...")
    try:
        test_response = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": "Say 'API test successful' in one sentence."}],
            max_output_tokens=50,
            store=False
        )
        result = extract_text_from_response(test_response)
        print(f"✅ API Test successful: {result}")
        return True
    except Exception as e:
        print(f"❌ API Test failed: {type(e).__name__}: {str(e)}")
        return False

# =====================================================
# 7️⃣ Output files
# =====================================================
OUTPUT_FILE = "llm_answers_by_topic_gpt_5_nano.json"
LOG_FILE = f"llm_answers_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# =====================================================
# 8️⃣ Main execution
# =====================================================
if __name__ == "__main__":
    if not test_api_connection():
        print("Exiting due to API connection failure.")
        exit(1)
    
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

            # Save incrementally
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            time.sleep(0.1)

    print("\n✅ Answer generation complete!")
    print(f"📁 Saved answers to: {OUTPUT_FILE}")
    print(f"📝 Log file: {LOG_FILE}")
