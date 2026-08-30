import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import re
import json
import random
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup JAVA_HOME and PATH dynamically for Windows before importing pyserini/jnius
def setup_java():
    os.environ["_JAVA_OPTIONS"] = "-Xmx1g"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if "JAVA_HOME" not in os.environ:
        workspace_dir = Path(__file__).resolve().parent
        jdk_dirs = list(workspace_dir.glob("jdk-21*"))
        if jdk_dirs:
            os.environ["JAVA_HOME"] = str(jdk_dirs[0])
            
    if "JAVA_HOME" in os.environ:
        java_home = os.environ["JAVA_HOME"]
        if sys.platform == "win32":
            bin_path = os.path.join(java_home, "bin")
            server_path = os.path.join(java_home, "bin", "server")
            paths = os.environ.get("PATH", "").split(os.pathsep)
            if bin_path not in paths:
                paths.insert(0, bin_path)
            if server_path not in paths:
                paths.insert(0, server_path)
            os.environ["PATH"] = os.pathsep.join(paths)

setup_java()
os.environ["OPENAI_API_KEY"] = "dummy"

from pyserini.search.lucene import LuceneSearcher
import ir_datasets

# Standard English Stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could",
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes",
    "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im",
    "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt",
    "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
    "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent",
    "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys",
    "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
    "yourselves"
}

def tokenize(text: str, remove_stopwords: bool = False):
    """Lowercases, removes punctuation, and tokenizes text."""
    if not text:
        return set()
    tokens = re.findall(r'\b\w+\b', text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return set(tokens)

def compute_jaccard(set1: set, set2: set):
    """Computes Jaccard coefficient between two token sets."""
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return len(set1.intersection(set2)) / union

def run_analysis(dataset_name: str, index_dir: Path, k1: float, b: float, sample_size: int = 500):
    print(f"\n=======================================================")
    print(f"       Running Vocabulary Mismatch Analysis: {dataset_name.upper()}")
    print(f"=======================================================")
    
    # 1. Load dataset test split queries and qrels
    dataset_id = f"beir/{dataset_name}/test"
    print(f"Loading queries and qrels for {dataset_id}...")
    ds = ir_datasets.load(dataset_id)
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    
    qrels = {}
    for qrel in ds.qrels_iter():
        if qrel.relevance > 0:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = set()
            qrels[qrel.query_id].add(qrel.doc_id)
            
    print(f"Loaded {len(queries)} queries and {len(qrels)} queries with relevant documents.")
    
    # Keep only queries that have at least one gold document in qrels
    valid_qids = [qid for qid in queries.keys() if qid in qrels]
    print(f"Valid queries with relevance judgements: {len(valid_qids)}")
    
    # Initialize searcher
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)
    
    results = []
    
    # Perform retrieval and document analysis
    print("Running retrieval and computing Jaccard overlaps...")
    for qid in tqdm(valid_qids, desc="Processing queries"):
        qtext = queries[qid]
        gold_doc_ids = qrels[qid]
        
        # Search top 50 documents
        hits = searcher.search(qtext, k=50)
        retrieved_ids = [hit.docid for hit in hits]
        
        # Parse the gold document(s) text
        gold_docs_text = {}
        for g_id in gold_doc_ids:
            try:
                raw_doc = searcher.doc(g_id)
                if raw_doc:
                    doc_json = json.loads(raw_doc.raw())
                    gold_docs_text[g_id] = doc_json.get("contents", "")
            except Exception as e:
                pass
                
        # If we failed to load any gold doc text, skip this query
        if not gold_docs_text:
            continue
            
        for g_id, g_text in gold_docs_text.items():
            # Determine success at k=10, 20, 50
            success_10 = g_id in retrieved_ids[:10]
            success_20 = g_id in retrieved_ids[:20]
            success_50 = g_id in retrieved_ids[:50]
            
            # Tokenize and compute Jaccard
            q_tokens_all = tokenize(qtext, remove_stopwords=False)
            d_tokens_all = tokenize(g_text, remove_stopwords=False)
            jaccard_all = compute_jaccard(q_tokens_all, d_tokens_all)
            
            q_tokens_no_stop = tokenize(qtext, remove_stopwords=True)
            d_tokens_no_stop = tokenize(g_text, remove_stopwords=True)
            jaccard_no_stop = compute_jaccard(q_tokens_no_stop, d_tokens_no_stop)
            
            results.append({
                "qid": qid,
                "query_text": qtext,
                "doc_id": g_id,
                "doc_text": g_text,
                "success_10": success_10,
                "success_20": success_20,
                "success_50": success_50,
                "jaccard_all": jaccard_all,
                "jaccard_no_stop": jaccard_no_stop,
                "rank": retrieved_ids.index(g_id) + 1 if g_id in retrieved_ids else -1
            })
            
    searcher.close()
    
    # 2. Statistics and comparison
    # Draw a random subset of 500 query-document pairs (using fixed seed 42)
    random.seed(42)
    subset_results = random.sample(results, min(sample_size, len(results))) if len(results) > sample_size else results
    
    # Helper to compute and format stats
    def get_stats(data_list, jaccard_key):
        success_overlaps_10 = [r[jaccard_key] for r in data_list if r["success_10"]]
        failure_overlaps_10 = [r[jaccard_key] for r in data_list if not r["success_10"]]
        
        success_overlaps_50 = [r[jaccard_key] for r in data_list if r["success_50"]]
        failure_overlaps_50 = [r[jaccard_key] for r in data_list if not r["success_50"]]
        
        return {
            "success_count_10": len(success_overlaps_10),
            "failure_count_10": len(failure_overlaps_10),
            "success_mean_10": np.mean(success_overlaps_10) if success_overlaps_10 else 0.0,
            "failure_mean_10": np.mean(failure_overlaps_10) if failure_overlaps_10 else 0.0,
            "success_median_10": np.median(success_overlaps_10) if success_overlaps_10 else 0.0,
            "failure_median_10": np.median(failure_overlaps_10) if failure_overlaps_10 else 0.0,
            "success_std_10": np.std(success_overlaps_10) if success_overlaps_10 else 0.0,
            "failure_std_10": np.std(failure_overlaps_10) if failure_overlaps_10 else 0.0,
            
            "success_count_50": len(success_overlaps_50),
            "failure_count_50": len(failure_overlaps_50),
            "success_mean_50": np.mean(success_overlaps_50) if success_overlaps_50 else 0.0,
            "failure_mean_50": np.mean(failure_overlaps_50) if failure_overlaps_50 else 0.0,
            "success_median_50": np.median(success_overlaps_50) if success_overlaps_50 else 0.0,
            "failure_median_50": np.median(failure_overlaps_50) if failure_overlaps_50 else 0.0,
            "success_std_50": np.std(success_overlaps_50) if success_overlaps_50 else 0.0,
            "failure_std_50": np.std(failure_overlaps_50) if failure_overlaps_50 else 0.0,
        }
        
    stats_full_all = get_stats(results, "jaccard_all")
    stats_full_no_stop = get_stats(results, "jaccard_no_stop")
    stats_sub_all = get_stats(subset_results, "jaccard_all")
    stats_sub_no_stop = get_stats(subset_results, "jaccard_no_stop")
    
    print("\n--- Summary Statistics (Full set vs Sampled subset) ---")
    print(f"{'Group':<30} | {'Full Count':<10} | {'Full Mean':<10} | {'Full Med':<10} | {'Sub Count':<10} | {'Sub Mean':<10} | {'Sub Med':<10}")
    print("-" * 105)
    
    # At k=10
    print(f"{'Success@10 (All Tokens)':<30} | {stats_full_all['success_count_10']:<10d} | {stats_full_all['success_mean_10']:<10.4f} | {stats_full_all['success_median_10']:<10.4f} | {stats_sub_all['success_count_10']:<10d} | {stats_sub_all['success_mean_10']:<10.4f} | {stats_sub_all['success_median_10']:<10.4f}")
    print(f"{'Failure@10 (All Tokens)':<30} | {stats_full_all['failure_count_10']:<10d} | {stats_full_all['failure_mean_10']:<10.4f} | {stats_full_all['failure_median_10']:<10.4f} | {stats_sub_all['failure_count_10']:<10d} | {stats_sub_all['failure_mean_10']:<10.4f} | {stats_sub_all['failure_median_10']:<10.4f}")
    print(f"{'Success@10 (No Stopwords)':<30} | {stats_full_no_stop['success_count_10']:<10d} | {stats_full_no_stop['success_mean_10']:<10.4f} | {stats_full_no_stop['success_median_10']:<10.4f} | {stats_sub_no_stop['success_count_10']:<10d} | {stats_sub_no_stop['success_mean_10']:<10.4f} | {stats_sub_no_stop['success_median_10']:<10.4f}")
    print(f"{'Failure@10 (No Stopwords)':<30} | {stats_full_no_stop['failure_count_10']:<10d} | {stats_full_no_stop['failure_mean_10']:<10.4f} | {stats_full_no_stop['failure_median_10']:<10.4f} | {stats_sub_no_stop['failure_count_10']:<10d} | {stats_sub_no_stop['failure_mean_10']:<10.4f} | {stats_sub_no_stop['failure_median_10']:<10.4f}")
    
    # At k=50
    print(f"{'Success@50 (No Stopwords)':<30} | {stats_full_no_stop['success_count_50']:<10d} | {stats_full_no_stop['success_mean_50']:<10.4f} | {stats_full_no_stop['success_median_50']:<10.4f} | {stats_sub_no_stop['success_count_50']:<10d} | {stats_sub_no_stop['success_mean_50']:<10.4f} | {stats_sub_no_stop['success_median_50']:<10.4f}")
    print(f"{'Failure@50 (No Stopwords)':<30} | {stats_full_no_stop['failure_count_50']:<10d} | {stats_full_no_stop['failure_mean_50']:<10.4f} | {stats_full_no_stop['failure_median_50']:<10.4f} | {stats_sub_no_stop['failure_count_50']:<10d} | {stats_sub_no_stop['failure_mean_50']:<10.4f} | {stats_sub_no_stop['failure_median_50']:<10.4f}")

    # 3. Generate and save plots
    os.makedirs("part3_plots", exist_ok=True)
    
    def plot_dist(data_list, title_prefix, filename_suffix, jaccard_key):
        success_overlaps = [r[jaccard_key] for r in data_list if r["success_10"]]
        failure_overlaps = [r[jaccard_key] for r in data_list if not r["success_10"]]
        
        plt.figure(figsize=(10, 6))
        plt.hist(success_overlaps, bins=25, alpha=0.5, label=f"Success (N={len(success_overlaps)})", color="green", density=True)
        plt.hist(failure_overlaps, bins=25, alpha=0.5, label=f"Failure (N={len(failure_overlaps)})", color="red", density=True)
        plt.xlabel("Jaccard Overlap")
        plt.ylabel("Density")
        plt.title(f"{title_prefix} - Jaccard Overlap Distribution (Success vs Failure at k=10)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plot_path = Path("part3_plots") / f"{dataset_name}_{filename_suffix}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
    plot_dist(results, "Full Test Set (All Tokens)", "full_all", "jaccard_all")
    plot_dist(results, "Full Test Set (Stopwords Removed)", "full_no_stop", "jaccard_no_stop")
    plot_dist(subset_results, "Subset (Stopwords Removed)", "sub_no_stop", "jaccard_no_stop")

    # 4. Extract some failure examples
    failures = [r for r in results if not r["success_10"]]
    failures = sorted(failures, key=lambda x: x["jaccard_no_stop"])
    
    print(f"\n--- Concrete Failure Examples for {dataset_name.upper()} ---")
    examples = []
    for i, f in enumerate(failures[:min(10, len(failures))]):
        examples.append({
            "qid": f["qid"],
            "query": f["query_text"],
            "doc_id": f["doc_id"],
            "doc_text": f["doc_text"],
            "jaccard_all": f["jaccard_all"],
            "jaccard_no_stop": f["jaccard_no_stop"],
            "rank": f["rank"]
        })
        print(f"\nExample {i+1}:")
        print(f" Query ID: {f['qid']}")
        print(f" Query: {f['query_text']}")
        print(f" Gold Doc ID: {f['doc_id']}")
        print(f" Gold Doc Text: {f['doc_text'][:300]}...")
        print(f" Jaccard Overlap (All): {f['jaccard_all']:.4f}")
        print(f" Jaccard Overlap (No Stopwords): {f['jaccard_no_stop']:.4f}")
        print(f" BM25 Retrieval Rank: {f['rank']}")
        
    # Collect failed query IDs for each threshold
    failed_qids_10 = sorted(list({r["qid"] for r in results if not r["success_10"]}))
    failed_qids_20 = sorted(list({r["qid"] for r in results if not r["success_20"]}))
    failed_qids_50 = sorted(list({r["qid"] for r in results if not r["success_50"]}))
    
    print(f"\n--- Failed Query IDs at k=10 (Total: {len(failed_qids_10)}) for {dataset_name.upper()} ---")
    print(failed_qids_10)
    print(f"\n--- Failed Query IDs at k=20 (Total: {len(failed_qids_20)}) for {dataset_name.upper()} ---")
    print(failed_qids_20)
    print(f"\n--- Failed Query IDs at k=50 (Total: {len(failed_qids_50)}) for {dataset_name.upper()} ---")
    print(failed_qids_50)
        
    return {
        "dataset": dataset_name,
        "stats_full_all": stats_full_all,
        "stats_full_no_stop": stats_full_no_stop,
        "stats_sub_all": stats_sub_all,
        "stats_sub_no_stop": stats_sub_no_stop,
        "examples": examples,
        "failed_qids_10": failed_qids_10,
        "failed_qids_20": failed_qids_20,
        "failed_qids_50": failed_qids_50
    }

def main():
    base_dir = Path.cwd()
    indexes_dir = base_dir / "indexes"
    
    # Tuned BM25 configurations from part2_results.txt
    configs = {
        "scifact": {"k1": 1.2, "b": 0.75},
        "fever": {"k1": 1.2, "b": 0.1},
        "hotpotqa": {"k1": 0.9, "b": 0.4}
    }
    
    all_dataset_analysis = {}
    for ds in ["scifact", "fever", "hotpotqa"]:
        ds_index_dir = indexes_dir / ds
        if ds_index_dir.exists():
            cfg = configs[ds]
            res = run_analysis(ds, ds_index_dir, cfg["k1"], cfg["b"])
            all_dataset_analysis[ds] = res
        else:
            print(f"Warning: Index directory {ds_index_dir} does not exist. Skipping.")

    # Write summary text file report
    report_file = base_dir / "part3_analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Part 3: Vocabulary Mismatch Analysis Report\n")
        f.write("============================================\n\n")
        for ds, data in all_dataset_analysis.items():
            f.write(f"Dataset: {ds.upper()}\n")
            f.write("-" * (9 + len(ds)) + "\n")
            
            s_full = data["stats_full_no_stop"]
            s_sub = data["stats_sub_no_stop"]
            f.write(f"Success/Failure Statistics (Stopwords Removed):\n")
            f.write(f" - Full set size: {s_full['success_count_10'] + s_full['failure_count_10']} pairs\n")
            f.write(f" - Subset size: {s_sub['success_count_10'] + s_sub['failure_count_10']} pairs\n")
            f.write(f" - Success mean Jaccard overlap (k=10): {s_full['success_mean_10']:.4f} (full) vs {s_sub['success_mean_10']:.4f} (subset)\n")
            f.write(f" - Failure mean Jaccard overlap (k=10): {s_full['failure_mean_10']:.4f} (full) vs {s_sub['failure_mean_10']:.4f} (subset)\n")
            f.write(f" - Success median Jaccard overlap (k=10): {s_full['success_median_10']:.4f} (full) vs {s_sub['success_median_10']:.4f} (subset)\n")
            f.write(f" - Failure median Jaccard overlap (k=10): {s_full['failure_median_10']:.4f} (full) vs {s_sub['failure_median_10']:.4f} (subset)\n")
            
            f.write(f"\nConcrete Failure Cases (Top 5):\n")
            for i, ex in enumerate(data["examples"][:5]):
                f.write(f" Example {i+1}:\n")
                f.write(f"  Query ID: {ex['qid']}\n")
                f.write(f"  Query: {ex['query']}\n")
                f.write(f"  Doc ID: {ex['doc_id']}\n")
                f.write(f"  Doc Jaccard (No Stopwords): {ex['jaccard_no_stop']:.4f}\n")
                f.write(f"  BM25 Rank: {ex['rank']}\n")
                f.write(f"  Doc Text snippet: {ex['doc_text'][:200]}...\n\n")
            
            f.write(f"\nFailed Query IDs at k=10 (Total: {len(data['failed_qids_10'])}):\n")
            f.write(f"{data['failed_qids_10']}\n")
            f.write(f"\nFailed Query IDs at k=20 (Total: {len(data['failed_qids_20'])}):\n")
            f.write(f"{data['failed_qids_20']}\n")
            f.write(f"\nFailed Query IDs at k=50 (Total: {len(data['failed_qids_50'])}):\n")
            f.write(f"{data['failed_qids_50']}\n")
            
            f.write("\n" + "="*80 + "\n\n")

    print(f"\nFull report written to {report_file}")

if __name__ == "__main__":
    main()

