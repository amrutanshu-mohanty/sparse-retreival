import argparse
import json
import os
import time
import subprocess
from pathlib import Path

# Setup JAVA_HOME since it's downloaded locally
os.environ["JAVA_HOME"] = str(Path.cwd() / "jdk-21.0.2")

import ir_datasets
from tqdm import tqdm


def prepare_dataset(dataset_name: str, output_dir: Path):
    """
    Loads dataset via ir_datasets, extracts docs, formats as JSONL for Pyserini,
    and returns corpus size and number of queries.
    """
    print(f"Loading dataset: {dataset_name} using ir_datasets...")
    
    # ir_datasets expects formats like 'beir/scifact/test' to get queries, 
    # but docs can be accessed from the base or split.
    # We will load the test split to get queries, and docs are corpus-wide.
    dataset_id = f"beir/{dataset_name}/test"
    
    try:
        dataset = ir_datasets.load(dataset_id)
    except KeyError:
        print(f"Warning: {dataset_id} not found, falling back to base 'beir/{dataset_name}'")
        dataset_id = f"beir/{dataset_name}"
        dataset = ir_datasets.load(dataset_id)
    
    docs_output_file = output_dir / "corpus.jsonl"
    
    corpus_size = 0
    
    print(f"Writing documents to {docs_output_file}...")
    with open(docs_output_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(dataset.docs_iter(), desc="Processing Docs"):
            # BEIR datasets typically have doc_id, text, title
            # Pyserini expects 'id' and 'contents'
            title = getattr(doc, 'title', '')
            text = getattr(doc, 'text', '')
            
            # Combine title and text (standard approach for BEIR in Pyserini)
            if title and text:
                contents = f"{title}\n{text}"
            else:
                contents = title or text
                
            pyserini_doc = {
                "id": doc.doc_id,
                "contents": contents
            }
            f.write(json.dumps(pyserini_doc) + '\n')
            corpus_size += 1
            
    num_queries = 0
    print("Counting queries...")
    if dataset.has_queries():
        for _ in dataset.queries_iter():
            num_queries += 1
    else:
        print("No queries found in this split.")
        
    return corpus_size, num_queries


def build_index(input_dir: Path, index_dir: Path):
    """
    Runs Pyserini Lucene indexer as a subprocess.
    """
    print(f"Building index from {input_dir} to {index_dir}...")
    
    # Construct pyserini index command
    # We need storePositions, storeDocvectors, storeRaw for Part 4a feedback
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(input_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8", # Adjust threads based on CPU cores
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw"
    ]
    
    start_time = time.time()
    
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    build_time = time.time() - start_time
    
    if process.returncode != 0:
        print(f"Error during indexing!\n{process.stdout}")
        raise RuntimeError("Indexing failed.")
    else:
        print("Indexing completed successfully.")
        
    return build_time


def get_dir_size(path: Path):
    """Returns size of a directory in bytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def format_size(size_bytes: float):
    """Formats bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0


def demonstrate_index_reader(index_dir: Path):
    """Demonstrates how to pull stats required for Part 4a."""
    print("\n--- Demonstrating IndexReader ---")
    try:
        from pyserini.index.lucene import IndexReader
        index_reader = IndexReader(str(index_dir))
        stats = index_reader.stats()
        print(f"Total documents in index: {stats['documents']}")
        print(f"Total terms in index: {stats.get('total_terms', stats.get('unique_terms', 'N/A'))}")
        
        print("IndexReader initialized successfully. You can use it to fetch doc vectors and term frequencies:")
        print("e.g. doc_vector = index_reader.get_document_vector('doc_id')")
    except ImportError:
         print("Pyserini not installed or Java not found. Skipping IndexReader demo.")
    except Exception as e:
        print(f"Could not load IndexReader (is Java environment setup correctly?): {e}")


def main():
    parser = argparse.ArgumentParser(description="Build Pyserini indexes for BEIR datasets.")
    parser.add_argument("--datasets", nargs='+', default=['scifact'], 
                        help="List of datasets to process (e.g., scifact fever hotpotqa)")
    args = parser.parse_args()

    base_dir = Path.cwd()
    data_dir = base_dir / "data"
    indexes_dir = base_dir / "indexes"
    
    data_dir.mkdir(exist_ok=True)
    indexes_dir.mkdir(exist_ok=True)
    
    report_lines = [
        "Dataset Report Deliverables",
        "==========================="
    ]
    
    for ds_name in args.datasets:
        print(f"\n========== Processing {ds_name.upper()} ==========")
        
        ds_data_dir = data_dir / ds_name
        ds_data_dir.mkdir(exist_ok=True)
        ds_index_dir = indexes_dir / ds_name
        
        # 1. Prepare Data
        corpus_size, num_queries = prepare_dataset(ds_name, ds_data_dir)
        
        # 2. Build Index
        build_time = build_index(ds_data_dir, ds_index_dir)
        
        # 3. Calculate Index Size
        index_size_bytes = get_dir_size(ds_index_dir)
        index_size_str = format_size(index_size_bytes)
        
        # 4. Record Results
        report = (
            f"Dataset: {ds_name}\n"
            f" - Corpus Size: {corpus_size:,} documents\n"
            f" - Number of Queries (test split): {num_queries:,}\n"
            f" - Index Build Time: {build_time:.2f} seconds\n"
            f" - On-Disk Index Size: {index_size_str}\n"
        )
        print(f"\n{report}")
        report_lines.append(report)
        
        # 5. IndexReader Demo
        demonstrate_index_reader(ds_index_dir)

    # Write report to file
    with open("report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("\nReport written to report.txt")


if __name__ == "__main__":
    main()
