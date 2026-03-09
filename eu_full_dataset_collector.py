from rdflib import ConjunctiveGraph, URIRef
import zipfile
import json
from tqdm import tqdm

ZIP_PATH = "data_europa_eu_dump.zip"
OUTPUT_FILE = "datasets_metadata.jsonl"

# RDF predicates
DCAT_DATASET = URIRef("http://www.w3.org/ns/dcat#Dataset")
DCT_TITLE = URIRef("http://purl.org/dc/terms/title")
DCT_DESCRIPTION = URIRef("http://purl.org/dc/terms/description")
DCT_PUBLISHER = URIRef("http://purl.org/dc/terms/publisher")


def safe_literal(obj):
    try:
        return str(obj)
    except Exception:
        return None


def extract_literal(graph, subject, predicate):
    for obj in graph.objects(subject, predicate):
        return safe_literal(obj)
    return None


def process_trig(trig_data, outfile):
    g = ConjunctiveGraph()
    try:
        g.parse(data=trig_data, format="trig")
    except Exception as e:
        print(f"⚠️ Failed to parse TRIG: {e}")
        return

    # iterate over all graphs (default + named)
    for context in g.contexts():
        for dataset in context.subjects(predicate=None, object=DCAT_DATASET):
            try:
                record = {
                    "id": str(dataset),
                    "title": extract_literal(context, dataset, DCT_TITLE),
                    "description": extract_literal(context, dataset, DCT_DESCRIPTION),
                    "publisher": extract_literal(context, dataset, DCT_PUBLISHER)
                }
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")
                outfile.flush()
            except Exception:
                continue


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        trig_files = [f for f in z.namelist() if f.endswith(".trig")]

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            for file in tqdm(trig_files):
                try:
                    with z.open(file) as f:
                        trig_data = f.read().decode("utf-8", errors="ignore")
                    process_trig(trig_data, out)
                except Exception as e:
                    print(f"⚠️ Skipping {file}: {e}")
                    continue

    print("✅ Extraction complete")


if __name__ == "__main__":
    main()
