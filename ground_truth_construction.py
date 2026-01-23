import json
import faiss
import numpy as np
import torch
import logging
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ============================================================
# Logger setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Load data
# ============================================================
with open("queries_by_topic.json", "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

with open("datasets_landing_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Normalize topic names
def norm(t):
    return str(t).strip().lower().replace(" ", "_")

queries_by_topic = {norm(k): v for k, v in queries_by_topic.items()}
metadata = {norm(k): v for k, v in metadata.items()}

# ============================================================
# Helper: Load already processed queries for resumption
# ============================================================
def load_processed_queries(path="evaluation_results.jsonl"):
    processed = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj.get("query")
                    if q:
                        processed.add(q)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return processed

# ============================================================
# Embedding Model
# ============================================================
class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def mean_pool(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

    def embed(self, texts, batch_size=16):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                emb = self.mean_pool(out.last_hidden_state, inputs["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

# ============================================================
# FAISS Index per topic
# ============================================================
class TopicIndex:
    def __init__(self, dim=768):
        self.indices = {}
        self.meta = {}
        self.dim = dim

    def add(self, topic, embeddings, meta):
        index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self.indices[topic] = index
        self.meta[topic] = meta

    def search(self, topic, query_emb, k=20):
        if topic not in self.indices:
            return []

        index = self.indices[topic]
        faiss.normalize_L2(query_emb)
        sims, idxs = index.search(query_emb.astype("float32"), k)

        results = []
        for sim, i in zip(sims[0], idxs[0]):
            ds = self.meta[topic][i]
            results.append({
                "dataset_index": int(i),
                "title": ds.get("title", ""),
                "description": ds.get("description", ""),
                "publisher": ds.get("publisher", ""),
                "url": ds.get("url", ""),
                "faiss_similarity": float(sim)
            })
        return results

# ============================================================
# LLM Reranker
# ============================================================
class LLMReranker:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        logger.info(f"Loading reranker model: {model_name}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def extract_json_from_text(self, text):
        """Extract JSON array robustly from LLM response"""
        json_patterns = [
            r'\[\s*\{.*?\}\s*\]',  # Array of objects
            r'\[.*\]',              # Any array
        ]
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    json_str = re.sub(r',\s*\]', ']', json_str)
                    json_str = re.sub(r',\s*\}', '}', json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(text)
        except:
            pass
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx+1])
            except:
                pass
        logger.warning(f"Could not extract JSON from: {text[:500]}...")
        return []

    def safe_convert_index(self, idx):
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, str):
            numbers = re.findall(r'\d+', idx)
            if numbers:
                return int(numbers[0])
        return -1

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []

        candidate_block = ""
        for i, ds in enumerate(candidates):
            candidate_block += f"[{i}] Title: {ds['title']}\n"
            candidate_block += f"Description: {ds['description'][:300]}\n"
            candidate_block += f"Publisher: {ds.get('publisher', 'N/A')}\n---\n"

        prompt = f"""You are a dataset relevance assessor. Rank these datasets by relevance to the query.

QUERY: {query}

CRITERIA:
- Score 4: Directly matches query topic and content
- Score 3: Strongly related
- Score 2: Somewhat related
- Score 1: Weakly related
- Score 0: Not relevant

INSTRUCTIONS:
- Return ONLY a JSON array with top {top_k} most relevant datasets
- Each item must have: "dataset_index" and "score" (0-4)
- Use the exact index numbers from the candidate list

CANDIDATES:
{candidate_block}

RESPONSE (JSON only):"""

        try:
            inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tok.pad_token_id
                )
            response_text = self.tok.decode(outputs[0], skip_special_tokens=True)
            if prompt in response_text:
                response_text = response_text.split(prompt)[-1].strip()
            logger.info(f"LLM Response: {response_text[:500]}...")
            scores_data = self.extract_json_from_text(response_text)
            if not scores_data:
                return self.fallback_ranking(candidates, top_k)
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return self.fallback_ranking(candidates, top_k)

        # Process rankings
        valid_scores = []
        for item in scores_data:
            if isinstance(item, dict):
                idx = self.safe_convert_index(item.get("dataset_index"))
                score = item.get("score", 0)
                if 0 <= idx < len(candidates) and isinstance(score, (int, float)):
                    valid_scores.append((idx, score))

        # Remove duplicates keeping highest score
        score_dict = {}
        for idx, score in valid_scores:
            if idx not in score_dict or score > score_dict[idx]:
                score_dict[idx] = score

        ranked = []
        for idx, score in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            ds = candidates[idx].copy()
            ds["llm_relevance_score"] = int(score)
            ranked.append(ds)

        return ranked

    def fallback_ranking(self, candidates, top_k):
        ranked = sorted(candidates, key=lambda x: x["faiss_similarity"], reverse=True)[:top_k]
        for ds in ranked:
            ds["llm_relevance_score"] = int(4 * ds["faiss_similarity"])  # Scale 0-4
        return ranked

# ============================================================
# Retrieval Pipeline
# ============================================================
class RetrievalPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.faiss = TopicIndex()
        self.reranker = LLMReranker()
        self.build()

    def build(self):
        logger.info("Building FAISS indices...")
        for topic, ds_list in tqdm(metadata.items(), desc="Building FAISS indices"):
            texts = [f"{d.get('title','')}. {d.get('description','')}" for d in ds_list]
            emb = self.embedder.embed(texts)
            self.faiss.add(topic, emb, ds_list)
        logger.info("FAISS indices built.")

    def process(self, query):
        q_emb = self.embedder.embed([query])

        # Retrieve top candidates from all topics
        candidates = []
        for topic in self.faiss.indices:
            candidates += self.faiss.search(topic, q_emb, k=10)

        # Remove duplicates by URL
        seen_urls = set()
        unique_candidates = []
        for cand in sorted(candidates, key=lambda x: x["faiss_similarity"], reverse=True):
            if cand["url"] not in seen_urls:
                seen_urls.add(cand["url"])
                unique_candidates.append(cand)
        candidates = unique_candidates[:15]  # take top 15 for reranking

        final = self.reranker.rerank(query, candidates, top_k=5)
        return {
            "query": query,
            "final_results": [
                {
                    "title": ds.get("title", ""),
                    "llm_relevance_score": ds.get("llm_relevance_score", 0),
                    "faiss_similarity": ds.get("faiss_similarity", 0)
                } for ds in final
            ]
        }

    def run_all(self):
        all_queries = [(topic, q) for topic, qs in queries_by_topic.items() for q in qs]
        processed_queries = load_processed_queries("evaluation_results.jsonl")
        logger.info(f"Resuming: {len(processed_queries)} queries already done.")

        with open("evaluation_results.jsonl", "a", encoding="utf-8") as f:
            for topic, query in tqdm(all_queries, desc="Processing all queries", unit="query"):
                if query in processed_queries:
                    continue
                try:
                    result = self.process(query)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    logger.info(f"Saved result for query: {query}")
                except Exception as e:
                    logger.error(f"Failed to process query '{query}': {e}")
                    empty_result = {
                        "query": query,
                        "final_results": []
                    }
                    f.write(json.dumps(empty_result, ensure_ascii=False) + "\n")
                    f.flush()

        logger.info("Finished streaming results to evaluation_results.jsonl")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info("Starting retrieval pipeline...")
    pipeline = RetrievalPipeline()
    pipeline.run_all()
    logger.info("Done.")
