import argparse
import json
import os
import sys
from pathlib import Path

# Setup Java environment to avoid Pyserini crashes
os.environ["_JAVA_OPTIONS"] = "-Xmx1g"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Ensure JAVA_HOME matches your OS configuration
if sys.platform != "win32":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

from pyserini.search.lucene import LuceneSearcher
import re

def verify_doc(dataset, doc_id, terms):
    index_path = Path("indexes") / dataset
    if not index_path.exists():
        print(f"Index not found at {index_path}")
        return
        
    searcher = LuceneSearcher(str(index_path))
    raw_doc = searcher.doc(doc_id)
    
    if not raw_doc:
        print(f"Doc ID {doc_id} not found in {dataset} index.")
        return
        
    doc_json = json.loads(raw_doc.raw())
    full_text = doc_json.get("contents", "")
    
    print(f"\n==================================================")
    print(f" Dataset: {dataset} | Doc ID: {doc_id}")
    print(f" Total document length: {len(full_text)} characters")
    print(f"==================================================")
    
    print("\n[Term Presence Check]")
    for term in terms:
        # Case insensitive exact word match
        matches = list(re.finditer(rf'\b{re.escape(term)}\b', full_text, flags=re.IGNORECASE))
        if matches:
            print(f" ✅ '{term}': FOUND ({len(matches)} occurrences)")
            first_match = matches[0]
            start = max(0, first_match.start() - 40)
            end = min(len(full_text), first_match.end() + 40)
            context = full_text[start:end].replace('\n', ' ')
            print(f"    Context: \"...{context}...\"")
        else:
            print(f" ❌ '{term}': NOT FOUND")
            
    print("\n[Full Text Preview (First 800 chars)]")
    print("-" * 50)
    print(full_text[:800])
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify query terms against a full gold document.")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name")
    parser.add_argument("--doc_id", type=str, default="195689316", help="Gold document ID")
    parser.add_argument("--terms", nargs="+", default=["Obesity", "decreases", "life", "quality"], 
                        help="Words to check for in the document")
    
    args = parser.parse_args()
    verify_doc(args.dataset, args.doc_id, args.terms)
