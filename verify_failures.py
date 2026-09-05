import argparse
import json
import os
import sys
import re
from pathlib import Path

# Setup Java environment to avoid Pyserini crashes
os.environ.setdefault("OPENAI_API_KEY", "dummy")
if "JAVA_HOME" not in os.environ:
    workspace_dir = Path(__file__).resolve().parent
    jdk_dirs = list(workspace_dir.glob("jdk-21*"))
    if jdk_dirs:
        os.environ["JAVA_HOME"] = str(jdk_dirs[0])
    elif Path("/usr/lib/jvm/java-21-openjdk-amd64").exists():
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

if "JAVA_HOME" in os.environ:
    java_bin = os.path.join(os.environ["JAVA_HOME"], "bin")
    if java_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{java_bin}{os.pathsep}{os.environ.get('PATH', '')}"

import ir_datasets
from pyserini.search.lucene import LuceneSearcher


def get_dataset_split(dataset_name: str):
    """Returns standard test/dev split for the dataset."""
    if dataset_name == "msmarco":
        return "beir/msmarco/dev"
    return f"beir/{dataset_name}/test"


def check_term_presence(full_text: str, terms: list):
    """Checks and prints context for each term in the document."""
    print("\n[Term Presence Check in Document]")
    for term in terms:
        clean_term = term.strip(" ,.:;\"'!?()[]{}")
        if not clean_term:
            continue
        matches = list(re.finditer(rf'\b{re.escape(clean_term)}\b', full_text, flags=re.IGNORECASE))
        if matches:
            print(f"  ✅ '{clean_term}': FOUND ({len(matches)} occurrences)")
            first_match = matches[0]
            start = max(0, first_match.start() - 40)
            end = min(len(full_text), first_match.end() + 40)
            context = full_text[start:end].replace('\n', ' ')
            print(f"     Context: \"...{context}...\"")
        else:
            print(f"  ❌ '{clean_term}': NOT FOUND")


def print_single_doc(searcher, dataset: str, doc_id: str, terms: list, preview_chars: int = 800):
    """Fetches and displays a document from the Lucene index."""
    raw_doc = searcher.doc(doc_id)
    if not raw_doc:
        print(f"\n⚠️ Doc ID '{doc_id}' not found in {dataset} index.")
        return

    full_text = ""
    try:
        doc_json = json.loads(raw_doc.raw())
        full_text = doc_json.get("contents", doc_json.get("text", ""))
    except Exception:
        pass
    if not full_text:
        full_text = raw_doc.contents() or ""

    print(f"\n" + "-" * 60)
    print(f" Document ID: {doc_id}")
    print(f" Total document length: {len(full_text)} characters")
    print("-" * 60)

    if terms:
        check_term_presence(full_text, terms)

    print(f"\n[Full Text Preview (First {preview_chars} chars)]")
    print("-" * 50)
    print(full_text[:preview_chars])
    if len(full_text) > preview_chars:
        print("... [truncated]")
    print("-" * 50)


def verify(dataset_name: str, query_id: str = None, doc_id: str = None, terms: list = None, preview_chars: int = 800):
    index_path = Path("indexes") / dataset_name
    if not index_path.exists():
        print(f"Index not found at {index_path}. Make sure to build the index first.")
        return

    searcher = LuceneSearcher(str(index_path))

    # Case 1: Query ID is provided -> Look up Query Text and Gold Docs
    if query_id:
        split_id = get_dataset_split(dataset_name)
        print(f"\nLoading {split_id} metadata...")
        try:
            ds = ir_datasets.load(split_id)
        except KeyError:
            ds = ir_datasets.load(f"beir/{dataset_name}")

        # Find query text
        query_text = None
        for q in ds.queries_iter():
            if str(q.query_id) == str(query_id):
                query_text = q.text
                break

        if not query_text:
            print(f"❌ Query ID '{query_id}' not found in {split_id}.")
            return

        # Find gold docs
        gold_doc_ids = []
        for qrel in ds.qrels_iter():
            if str(qrel.query_id) == str(query_id) and qrel.relevance > 0:
                gold_doc_ids.append((str(qrel.doc_id), qrel.relevance))

        print("\n" + "=" * 70)
        print(f" DATASET: {dataset_name.upper()} | QUERY ID: {query_id}")
        print(f" QUERY TEXT: \"{query_text}\"")
        print(f" GOLD RELEVANT DOCS FOUND: {len(gold_doc_ids)}")
        for g_id, rel in gold_doc_ids:
            print(f"   -> Doc ID: {g_id} (Relevance: {rel})")
        print("=" * 70)

        # Default terms to query words if not explicitly passed
        if not terms:
            terms = [w for w in re.findall(r'\b\w+\b', query_text) if len(w) > 2]

        if doc_id:
            # Show specific doc
            print_single_doc(searcher, dataset_name, doc_id, terms, preview_chars)
        elif gold_doc_ids:
            # Show all gold docs for this query
            for g_id, rel in gold_doc_ids:
                print_single_doc(searcher, dataset_name, g_id, terms, preview_chars)
        else:
            print(f"No positive gold documents found in qrels for Query ID '{query_id}'.")

    # Case 2: Only Doc ID is provided
    elif doc_id:
        print("\n" + "=" * 70)
        print(f" DATASET: {dataset_name.upper()} | DOC ID: {doc_id}")
        print("=" * 70)
        if not terms:
            terms = []
        print_single_doc(searcher, dataset_name, doc_id, terms, preview_chars)

    else:
        print("Please provide either --query_id or --doc_id.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify queries and gold documents against Lucene indexes.")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name (scifact, fever, hotpotqa, msmarco)")
    parser.add_argument("--query_id", "--qid", type=str, default=None, help="Query ID to look up")
    parser.add_argument("--doc_id", "--did", type=str, default=None, help="Gold document ID")
    parser.add_argument("--terms", nargs="+", default=None, help="Specific words/terms to check in the document")
    parser.add_argument("--preview", type=int, default=800, help="Number of preview characters to print (default: 800)")

    args = parser.parse_args()
    verify(
        dataset_name=args.dataset,
        query_id=args.query_id,
        doc_id=args.doc_id,
        terms=args.terms,
        preview_chars=args.preview
    )
