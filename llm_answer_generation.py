import json
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForVision2Seq

# =====================================================
# Load queries
# =====================================================
QUERIES_FILE = "queries_by_topic.json"

with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

print(f"Loaded queries for {len(queries_by_topic)} topics")

# =====================================================
# Load Qwen3-VL-8B-Instruct (TEXT-ONLY MODE)
# =====================================================
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

print(f"🔄 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🔄 Loading model: {MODEL_NAME}")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
print("✅ Model loaded successfully")

# =====================================================
# Few-shot prompt template
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
# Answer generation function
# =====================================================
def generate_answer(query, max_new_tokens=200):
    prompt = FEW_SHOT_PROMPT.format(query=query)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    if prompt in decoded:
        decoded = decoded.split(prompt)[-1].strip()

    # Keep only first paragraph
    answer = decoded.split("\n\n")[0].strip()

    return answer

# =====================================================
# Main answering loop
# =====================================================
results = []
log_file = f"llm_answers_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

print("🧠 Generating answers...")

with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"LLM Answering Log - {datetime.now()}\n")
    log.write(f"Model: {MODEL_NAME}\n")
    log.write("=" * 60 + "\n")

    for topic, queries in tqdm(queries_by_topic.items(), desc="Topics"):
        for query in queries:
            try:
                answer = generate_answer(query)

                results.append({
                    "User_Query": query,
                    "Assistant": answer
                })

                log.write(f"User_Query: {query}\n")
                log.write(f"Assistant: {answer}\n")
                log.write("-" * 40 + "\n")

            except Exception as e:
                results.append({
                    "User_Query": query,
                    "Assistant": ""
                })

                log.write(f"User_Query: {query}\n")
                log.write(f"ERROR: {str(e)}\n")
                log.write("-" * 40 + "\n")

# =====================================================
# Save results
# =====================================================
OUTPUT_FILE = "llm_answers_by_topic.json"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Answer generation complete!")
print(f"📁 Saved answers to: {OUTPUT_FILE}")
print(f"📝 Log file: {log_file}")
