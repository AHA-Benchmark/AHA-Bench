import json
import torch
import re
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoTokenizer
import os

# -------------------------------------------------
# Cache configuration - MUST BE AT TOP
# -------------------------------------------------
HF_CACHE_DIR = "/nas/netstore/ldv/ge83qiw/hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = HF_CACHE_DIR

# -------------------------------------------------
# Load metadata
# -------------------------------------------------
with open("datasets_landing_metadata.json", "r", encoding="utf-8") as f:
    all_metadata_by_topic = json.load(f)

all_metadata = []
for topic, datasets in all_metadata_by_topic.items():
    for ds in datasets:
        ds["topic"] = topic
        all_metadata.append(ds)

print(f"Total datasets to process: {len(all_metadata)}")

# -------------------------------------------------
# Model & tokenizer setup
# -------------------------------------------------
model_name = "Qwen/Qwen3-VL-8B-Instruct"

print(f"🔄 Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"🔄 Loading model from {model_name}...")
print(f"📁 Using cache: {HF_CACHE_DIR}")
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    cache_dir=HF_CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print(f"✅ Model loaded on device: {model.device}")

# -------------------------------------------------
# Monitoring setup
# -------------------------------------------------
log_file = f"query_generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_result(result, index, total):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Result {index}/{total} ---\n")
        f.write(f"ID: {result['id']}\n")
        f.write(f"Title: {result['title']}\n")
        f.write(f"Dataset Request Query: {result.get('dataset_request_query', '')}\n")
        f.write(f"Described Dataset Query: {result.get('described_dataset_query', '')}\n")
        f.write(f"Implied Dataset Query: {result.get('implied_dataset_query', '')}\n")
        f.write("---\n")

# -------------------------------------------------
# Query generation functions
# -------------------------------------------------
def _generate_single_query(prompt, max_new_tokens=100):
    """Generate a single query with one retry if empty"""
    
    def _generate_once():
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1500
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.split(prompt)[-1].strip() if prompt in text else text

        query = response.split('\n')[0].strip()
        query = re.sub(r"^[\"\']|[\"\']$", "", query)
        
        # Clean up any remaining instruction text
        query = re.sub(r'(Query:|Response:|Answer:).*$', '', query, flags=re.IGNORECASE).strip()
        
        if query and not query.endswith("?"):
            query = query.rstrip('.') + "?"
        return query if query and len(query) >= 10 else ""

    # First attempt
    query = _generate_once()
    
    # Retry once if empty
    if not query:
        query = _generate_once()
    
    return query

def generate_dataset_request_query(meta, max_new_tokens=100):
    title = meta.get("title", "")
    description = meta.get("description", "")
    publisher = meta.get("publisher", "")

    if not title:
        return ""

    prompt = f"""
Generate one natural-sounding user search query where someone explicitly requests a specific dataset by title. The query should:
- Mention the dataset title in the question
- Sound like a real person searching for data
- Be limited to 100 characters
- Use natural language, not technical jargon

Example: "Where can I find the air quality monitoring dataset for London?"

Title: {title}
Description: {description}
Publisher: {publisher or "Unknown"}

Natural search query:
"""
    return _generate_single_query(prompt, max_new_tokens)

def generate_described_dataset_query(meta, max_new_tokens=100):
    title = meta.get("title", "")
    description = meta.get("description", "")
    publisher = meta.get("publisher", "")

    if not description:
        return ""

    prompt = f"""
Generate one natural-sounding user search query where someone describes the kind of data they need and includes the word "dataset". The query should:
- Describe the type of data or information needed
- Include the word "dataset" or "data" explicitly 
- Sound like a real researcher or analyst
- Be limited to 150 characters

Example: "I'm looking for a dataset about bicycle sharing programs in European cities"

Context information:
Title: {title}
Description: {description}
Publisher: {publisher or "Unknown"}

Natural search query:
"""
    return _generate_single_query(prompt, max_new_tokens)

def generate_implied_dataset_query(meta, max_new_tokens=100):
    title = meta.get("title", "")
    description = meta.get("description", "")
    publisher = meta.get("publisher", "")

    if not description:
        return ""

    prompt = f"""
Generate one natural-sounding user search query where someone asks a question that implies they need dataset information, but does NOT use the word "dataset" or "data". The query should:
- Sound like a curious researcher or analyst exploring a topic
- NOT use the words "data" or "dataset"
- Be limited to 200 characters
- Be conversational and natural

Example: "What are the most popular tourist attractions in Paris based on visitor numbers?"

Context information:
Title: {title}
Description: {description}
Publisher: {publisher or "Unknown"}

Natural search query:
"""
    return _generate_single_query(prompt, max_new_tokens)

# -------------------------------------------------
# Main generation function
# -------------------------------------------------
def generate_all_queries(meta):
    title = meta.get("title", "")
    
    if not title and not meta.get("description", ""):
        return {
            "id": meta["id"],
            "title": title,
            "topic": meta.get("topic", "Unknown"),
            "dataset_request_query": "",
            "described_dataset_query": "",
            "implied_dataset_query": ""
        }

    dataset_request = generate_dataset_request_query(meta)
    described_dataset = generate_described_dataset_query(meta)
    implied_dataset = generate_implied_dataset_query(meta)

    return {
        "id": meta["id"],
        "title": title,
        "topic": meta.get("topic", "Unknown"),
        "dataset_request_query": dataset_request,
        "described_dataset_query": described_dataset,
        "implied_dataset_query": implied_dataset
    }

# -------------------------------------------------
# Generation loop with monitoring
# -------------------------------------------------
results = []
queries_by_topic = {}
print(f"Generating queries for {len(all_metadata)} datasets...")
print(f"📝 Logging to: {log_file}")

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Query Generation Log - Started at {datetime.now()}\n")
    f.write(f"Total datasets: {len(all_metadata)}\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Cache directory: {HF_CACHE_DIR}\n")
    f.write("=" * 50 + "\n")

for i, meta in enumerate(tqdm(all_metadata, desc="Processing datasets")):
    result = generate_all_queries(meta)
    results.append(result)
    
    topic = result.get("topic", "Unknown")
    if topic not in queries_by_topic:
        queries_by_topic[topic] = []
    
    if result["described_dataset_query"]:
        queries_by_topic[topic].append(result["described_dataset_query"])
    if result["implied_dataset_query"]:
        queries_by_topic[topic].append(result["implied_dataset_query"])
    
    log_result(result, i, len(all_metadata))
    
    if i % 20 == 0:
        print(f"\n📊 Progress: {i}/{len(all_metadata)}")
        print(f"Latest: {result['title'][:60]}...")
        print(f"Request: {result['dataset_request_query']}")
        print(f"Described: {result['described_dataset_query']}")
        print(f"Implied: {result['implied_dataset_query']}")

# -------------------------------------------------
# Save results
# -------------------------------------------------
with open("datasets_with_conversational_queries_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open("queries_by_topic.json", "w", encoding="utf-8") as f:
    json.dump(queries_by_topic, f, indent=2, ensure_ascii=False)

# Statistics
valid_requests = [r for r in results if r["dataset_request_query"]]
valid_described = [r for r in results if r["described_dataset_query"]]
valid_implied = [r for r in results if r["implied_dataset_query"]]

print(f"\n✅ Generation complete!")
print(f"📁 Full queries saved to: datasets_with_conversational_queries_v2.json")
print(f"📁 Topic-based queries saved to: queries_by_topic.json")
print(f"📊 Generated {len(valid_requests)} dataset request queries")
print(f"📊 Generated {len(valid_described)} described dataset queries") 
print(f"📊 Generated {len(valid_implied)} implied dataset queries")
print(f"📝 Detailed log available at: {log_file}")
