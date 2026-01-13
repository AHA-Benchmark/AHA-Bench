import json
import time
import openai
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
openai.api_key = # API Key
INPUT_FILE = "sampled_1k.json" 
OUTPUT_FILE = "gpt_validation_results.jsonl"
MODEL = "gpt-4.1-mini"
BATCH_DELAY = 0.5  
# ==============================
# LOAD INPUT
# ==============================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==============================
# BUILD PROMPT
# ==============================
def build_prompt(query_obj):
    query_text = query_obj["query"]
    candidates = query_obj["datasets"]

    dataset_block = ""
    for ds in candidates:
        dataset_block += f"- {ds}\n"

    prompt = f"""
You are a dataset relevance judge.

QUERY:
{query_text}

TASK:
For each of the 5 datasets below, assign a relevance score using the following scale:
4 = Highly relevant – directly answers the query
3 = Strongly related
2 = Somewhat related
1 = Weakly related
0 = Not relevant

INSTRUCTIONS:
- Return only a JSON array of 5 objects, each object must have:
  "title": <dataset title>
  "score": <0-4>
- Use the exact dataset titles shown above
- Do NOT include explanations

DATASETS:
{dataset_block}

RESPONSE (JSON ONLY):
"""
    return prompt

# ==============================
# RUN GPT-4.1-mini
# ==============================
with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
    for query_obj in tqdm(data, desc="Processing queries"):
        prompt = build_prompt(query_obj)
        success = False
        retries = 0

        while not success and retries < 3:
            try:
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that scores datasets for relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=512
                )

                text = response.choices[0].message.content.strip()

                # Parse JSON (handle ```json ``` if GPT wraps it)
                try:
                    gpt_scores = json.loads(text)
                except json.JSONDecodeError:
                    text_clean = text.replace("```json", "").replace("```", "").strip()
                    gpt_scores = json.loads(text_clean)

                # Save the results
                out_obj = {
                    "query": query_obj["query"],
                    "gpt_scores": gpt_scores
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()
                success = True

            except Exception as e:
                retries += 1
                print(f"Error on query: {query_obj['query'][:50]}... Retrying ({retries}) Error: {e}")
                time.sleep(2 * retries)

        time.sleep(BATCH_DELAY)

print(f"Finished GPT validation. Results saved to {OUTPUT_FILE}")
