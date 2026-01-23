import requests
import json
import time
from tqdm import tqdm
import os

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
TOPIC_LIMITS = {
    "Agriculture, fisheries, forestry and food": 1414,
    "Economy and finance": 542,
    "Education, culture and sport": 190,
    "Energy": 53,
    "Environment": 896,
    "Government and public sector": 773,
    "Health": 135,
    "International issues": 6,
    "Justice, legal system and public safety": 1272,
    "Population and society": 440,
    "Provisional data": 41,
    "Regions and cities": 414,
    "Science and technology": 483,
    "Transport": 341
}

PAGE_SIZE = 100               # API max items per request (safe default)
DELAY_BETWEEN_REQUESTS = 1    # polite delay
OUTPUT_FILE = "datasets_landing_metadata.json"
API_URL = "https://data.europa.eu/api/hub/search/search"


# ---------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------
def fetch_page(query, page, limit):
    """Fetch a single page of search results."""
    params = {
        "q": query,
        "datasetOnly": "true",
        "page": page,
        "limit": limit
    }
    try:
        response = requests.get(API_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("result", {}).get("results", [])
    except Exception as e:
        print(f"⚠️ Error fetching page {page} for '{query}': {e}")
        return []


def extract_field(data_dict, default="Not available"):
    if isinstance(data_dict, dict):
        return data_dict.get("en") or next(iter(data_dict.values()), default)
    return data_dict or default


def get_publisher(publisher_data):
    if isinstance(publisher_data, dict):
        name_data = publisher_data.get("name", {})
        return extract_field(name_data)
    return publisher_data or "Not available"


def process_dataset(item):
    return {
        "id": item.get("id"),
        "title": extract_field(item.get("title", {})),
        "description": extract_field(item.get("description", {})),
        "categories": [
            cat.get("label", {}).get("en", "Unknown")
            for cat in item.get("categories", [])
        ],
        "publisher": get_publisher(item.get("publisher")),
        "modified": item.get("modified", "Not available"),
        "url": f"https://data.europa.eu/data/datasets/{item.get('id')}?locale=en"
    }


def save_to_json(data, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------
# MAIN LOGIC
# ---------------------------------------------
if __name__ == "__main__":
    print("⏳ Starting metadata collection from data.europa.eu ...")

    datasets_by_topic = {}

    for topic, target_count in tqdm(TOPIC_LIMITS.items(), desc="Collecting datasets"):
        collected = []
        page = 0

        while len(collected) < target_count:
            items = fetch_page(topic, page, PAGE_SIZE)
            if not items:  # no more results returned
                break

            collected.extend(items)
            page += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)

        # Trim to exact target count
        collected = collected[:target_count]

        # Process datasets
        datasets_by_topic[topic] = [process_dataset(item) for item in collected]

    save_to_json(datasets_by_topic, OUTPUT_FILE)
    print(f"✅ Metadata collection complete! Saved → {OUTPUT_FILE}")
