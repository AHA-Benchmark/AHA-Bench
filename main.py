import subprocess
import os
from huggingface_hub import login

# Authentication


# Cache setup - Set this at the VERY BEGINNING
HF_CACHE_DIR = "/nas/netstore/ldv/ge83qiw/hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TORCH_HOME"] = os.path.join(HF_CACHE_DIR, "torch")

# Create cache directory if it doesn't exist
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

def main():
    print("🚀 Starting automated pipeline...")
    print(f"📁 Using cache directory: {HF_CACHE_DIR}")
    
    # Step 1: Run metadata extraction
    print("⏳ Step 1: Running metadata extraction...")
    subprocess.run(["python", "fetching_metadata.py"], check=True)
    
    # Step 2: Run query generation
    print("⏳ Step 2: Running query generator...")
    subprocess.run(["python", "query_generator.py"], check=True)
    
    # Step 3: Run FAISS search and LLM relevance scoring
    print("⏳ Step 3: Running FAISS search and LLM relevance scoring...")
    subprocess.run(["python", "faiss_scoring.py"], check=True)
    
    # Step 4: Compare results
    print("⏳ Step 4: Comparing FAISS vs LLM results...")
    subprocess.run(["python", "compare_results.py"], check=True)
    
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
