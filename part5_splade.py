"""
Part 5: Sparse Retrieval with a Pretrained SPLADE Checkpoint.

Deliverables:
  1. SPLADE metrics (nDCG@10, Recall@100, MRR@10, MAP) per dataset
  2. >=10-query three-way expansion-term comparison (SPLADE vs Rocchio/RM3 vs HyDE)
  3. Term overlap stats and 2-3 disagreement case discussions

Usage:
  python part5_splade.py --datasets scifact
  python part5_splade.py --datasets scifact fever hotpotqa --output part5_results
"""

import argparse
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Java / Pyserini setup (reused pattern from existing scripts)
# ---------------------------------------------------------------------------
def setup_java():
    os.environ["_JAVA_OPTIONS"] = "-Xmx2g"
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

def setup_java_ubuntu():
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

setup_java_ubuntu()
os.environ["OPENAI_API_KEY"] = "dummy"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from part4_rocchio import (
    LuceneSearcher,
    RocchioQueryExpander,
    tokenize,
    clean_query_text,
    load_dataset_queries_and_qrels,
    compute_metrics_and_per_query,
    run_search,
)

import ir_datasets
import pytrec_eval
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SPLADE expansion term extractor (direct model inference)
# ---------------------------------------------------------------------------
class SpladeTermExtractor:
    """Loads a SPLADE checkpoint and extracts per-query expansion terms."""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil",
                 device: str = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SPLADE model {model_name} on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Special token IDs to skip
        self.special_ids = set(self.tokenizer.all_special_ids)
        print("SPLADE model loaded.", flush=True)

    def get_sparse_vector(self, text: str) -> Dict[str, float]:
        """Returns {token_str: weight} for all non-zero SPLADE weights."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=256).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits  # (1, seq_len, vocab_size)
        # w_j = max_i log(1 + ReLU(o_ij))
        weights = self.torch.log1p(self.torch.relu(logits)).max(dim=1).values.squeeze()
        nonzero = weights.nonzero(as_tuple=True)[0]
        result = {}
        for idx in nonzero:
            idx_int = idx.item()
            if idx_int in self.special_ids:
                continue
            token = self.tokenizer.decode([idx_int]).strip()
            if not token or len(token) < 2:
                continue
            result[token] = round(weights[idx_int].item(), 4)
        return result

    def extract_expansion_terms(self, query_text: str,
                                top_k: int = 20) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """
        Returns (full_sparse_vec, expansion_terms_beyond_query).
        expansion_terms are sorted by weight descending.
        """
        sparse_vec = self.get_sparse_vector(query_text)
        
        # Get the original vocabulary units from the tokenized query
        input_ids = self.tokenizer(query_text, add_special_tokens=False)["input_ids"]
        query_tokens_lower = set()
        for idx in input_ids:
            token = self.tokenizer.decode([idx]).strip().lower()
            query_tokens_lower.add(token)

        expansion = []
        for token, weight in sparse_vec.items():
            if token.lower() not in query_tokens_lower:
                expansion.append((token, weight))
        expansion.sort(key=lambda x: x[1], reverse=True)
        return sparse_vec, expansion[:top_k]


# ---------------------------------------------------------------------------
# SPLADE retrieval via Pyserini LuceneImpactSearcher
# ---------------------------------------------------------------------------
# Prebuilt index names in Pyserini
PREBUILT_INDEX_SPLADE_PP_ED = {
    "scifact":  "beir-v1.0.0-scifact.splade-pp-ed",
    "fever":    "beir-v1.0.0-fever.splade-pp-ed",
    "hotpotqa": "beir-v1.0.0-hotpotqa.splade-pp-ed",
    "msmarco":  "msmarco-v1-passage.splade-pp-ed",
}

PREBUILT_INDEX_SPLADE_V3 = {
    "scifact":  "beir-v1.0.0-scifact.splade-v3",
    "fever":    "beir-v1.0.0-fever.splade-v3",
    "hotpotqa": "beir-v1.0.0-hotpotqa.splade-v3",
    "msmarco":  "msmarco-v1-passage.splade-pp-ed",
}


def init_impact_searcher(dataset_name: str, model_name: str):
    """Initializes LuceneImpactSearcher with a prebuilt index."""
    from pyserini.search.lucene import LuceneImpactSearcher
    from pyserini.encode import SpladeQueryEncoder

    if "splade-v3" in model_name:
        idx_map = PREBUILT_INDEX_SPLADE_V3
    else:
        idx_map = PREBUILT_INDEX_SPLADE_PP_ED

    index_name = idx_map.get(dataset_name)
    if not index_name:
        raise ValueError(f"No prebuilt index for {dataset_name}")

    print(f"Initializing LuceneImpactSearcher with prebuilt index: {index_name}", flush=True)
    print(f"  Query encoder: {model_name}", flush=True)
    
    encoder = SpladeQueryEncoder(model_name)
    
    searcher = LuceneImpactSearcher.from_prebuilt_index(index_name, encoder)
    return searcher


def run_splade_retrieval(searcher, queries: Dict[str, str],
                         k: int = 100) -> Dict[str, Dict[str, float]]:
    """Runs SPLADE retrieval for all queries, returns pytrec_eval-compatible run dict."""
    run = {}
    for qid, qtext in tqdm(queries.items(), desc="SPLADE retrieval", leave=False):
        try:
            hits = searcher.search(qtext, k=k)
            run[qid] = {hit.docid: float(hit.score) for hit in hits}
        except Exception as e:
            print(f"  [WARN] search failed for qid={qid}: {e}", flush=True)
            run[qid] = {}
    return run


# ---------------------------------------------------------------------------
# Rocchio term re-extraction (reuses Part 4a code)
# ---------------------------------------------------------------------------
def extract_rocchio_terms_for_queries(
    query_ids: List[str], queries: Dict[str, str],
    index_dir: Path, k1: float, b: float,
    fb_docs: int = 5, fb_terms: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """Re-extracts Rocchio expansion terms for selected queries."""
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)
    expander = RocchioQueryExpander(searcher, index_dir)

    # First-pass BM25 retrieval
    subset_queries = {qid: queries[qid] for qid in query_ids}
    raw_hits, _ = run_search(searcher, subset_queries, k=100)

    terms_log = {}
    for qid in query_ids:
        hits = raw_hits.get(qid, [])
        _, exp_terms = expander.expand_query(
            query_text=queries[qid], hits=hits,
            fb_docs=fb_docs, fb_terms=fb_terms,
            alpha=1.0, beta=0.75, gamma=0.0,
        )
        terms_log[qid] = exp_terms

    searcher.close()
    return terms_log


# ---------------------------------------------------------------------------
# HyDE term re-extraction (reuses Part 4b code if cache available)
# ---------------------------------------------------------------------------
def extract_hyde_terms_for_queries(
    query_ids: List[str], queries: Dict[str, str],
    index_dir: Path, k1: float, b: float,
    hyde_cache: Dict[str, List[str]],
    fb_docs: int = 5, fb_terms: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """Re-extracts HyDE expansion terms for selected queries using cached HyDE docs."""
    try:
        from part4b_hyde import HydeRocchioExpander, make_hyde_tokenizer
    except ImportError:
        print("  [WARN] Cannot import part4b_hyde. Skipping HyDE terms.", flush=True)
        return {}

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)
    hyde_expander = HydeRocchioExpander(searcher, index_dir, tokenizer=tokenize)

    terms_log = {}
    for qid in query_ids:
        hyde_docs = hyde_cache.get(qid, [])
        if not hyde_docs:
            continue
        fake_ids = hyde_expander.set_current_query_docs(hyde_docs[:fb_docs])
        _, exp_terms = hyde_expander.expand_query(
            query_text=queries[qid], hits=fake_ids,
            fb_docs=fb_docs, fb_terms=fb_terms,
            alpha=1.0, beta=0.75, gamma=0.0,
        )
        terms_log[qid] = exp_terms

    searcher.close()
    return terms_log


# ---------------------------------------------------------------------------
# Three-way comparison analysis
# ---------------------------------------------------------------------------
def compute_term_overlap(terms_a: List[Tuple[str, float]],
                         terms_b: List[Tuple[str, float]]) -> float:
    """Jaccard overlap between two expansion term lists."""
    set_a = {t.lower() for t, _ in terms_a}
    set_b = {t.lower() for t, _ in terms_b}
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def build_comparison_table(
    query_ids: List[str], queries: Dict[str, str],
    splade_terms: Dict[str, List[Tuple[str, float]]],
    rocchio_terms: Dict[str, List[Tuple[str, float]]],
    hyde_terms: Dict[str, List[Tuple[str, float]]],
) -> List[Dict[str, Any]]:
    """Builds per-query comparison records."""
    rows = []
    for qid in query_ids:
        sp = splade_terms.get(qid, [])
        ro = rocchio_terms.get(qid, [])
        hy = hyde_terms.get(qid, [])

        sp_set = {t.lower() for t, _ in sp}
        ro_set = {t.lower() for t, _ in ro}
        hy_set = {t.lower() for t, _ in hy}
        all_three = sp_set & ro_set & hy_set if hy_set else set()

        rows.append({
            "qid": qid,
            "query": queries.get(qid, ""),
            "splade_terms": sp[:15],
            "rocchio_terms": ro[:10],
            "hyde_terms": hy[:10],
            "overlap_sp_ro": compute_term_overlap(sp, ro),
            "overlap_sp_hy": compute_term_overlap(sp, hy) if hy else None,
            "overlap_ro_hy": compute_term_overlap(ro, hy) if hy else None,
            "shared_all_three": sorted(all_three),
        })
    return rows


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_part5_report(
    dataset_name: str, splade_metrics: Dict[str, float],
    bm25_metrics: Dict[str, float], comparison_rows: List[Dict],
    model_name: str, num_queries: int, output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = dataset_name.upper()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"        PART 5: SPLADE SPARSE RETRIEVAL — {ds}\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"SPLADE Model : {model_name}\n")
        f.write(f"Queries      : {num_queries}\n\n")

        # Metrics table
        f.write("TABLE 1 — RETRIEVAL METRICS\n")
        f.write(f"{'Method':<40} | {'nDCG@10':<9} | {'Recall@100':<11} | {'MRR@10':<9} | {'MAP':<9}\n")
        f.write("-" * 90 + "\n")
        bm = bm25_metrics
        f.write(f"{'Tuned BM25 Baseline (Part 2)':<40} | {bm['nDCG@10']:<9.4f} | {bm['Recall@100']:<11.4f} | {bm['MRR@10']:<9.4f} | {bm['MAP']:<9.4f}\n")
        sm = splade_metrics
        f.write(f"{'SPLADE (prebuilt index)':<40} | {sm['nDCG@10']:<9.4f} | {sm['Recall@100']:<11.4f} | {sm['MRR@10']:<9.4f} | {sm['MAP']:<9.4f}\n")
        # Delta
        f.write(f"{'Delta (SPLADE - BM25)':<40} | {sm['nDCG@10']-bm['nDCG@10']:+<9.4f} | {sm['Recall@100']-bm['Recall@100']:+<11.4f} | {sm['MRR@10']-bm['MRR@10']:+<9.4f} | {sm['MAP']-bm['MAP']:+<9.4f}\n")
        f.write("-" * 90 + "\n\n")

        # Three-way expansion term comparison (or two-way if HyDE is skipped)
        has_hyde = any(r["hyde_terms"] for r in comparison_rows)
        comp_title = "THREE-WAY EXPANSION TERM COMPARISON" if has_hyde else "TWO-WAY EXPANSION TERM COMPARISON (SPLADE vs Rocchio)"
        f.write(f"TABLE 2 — {comp_title} ({len(comparison_rows)} queries)\n")
        f.write("=" * 100 + "\n\n")

        for row in comparison_rows:
            f.write(f"Query ID: {row['qid']}\n")
            f.write(f"  Text: \"{row['query']}\"\n")
            sp_str = ", ".join(f"{t}({w:.2f})" for t, w in row["splade_terms"][:10])
            ro_str = ", ".join(f"{t}({w:.2f})" for t, w in row["rocchio_terms"][:8])
            f.write(f"  SPLADE  : {sp_str}\n")
            f.write(f"  Rocchio : {ro_str}\n")
            if has_hyde:
                hy_str = ", ".join(f"{t}({w:.2f})" for t, w in row["hyde_terms"][:8])
                f.write(f"  HyDE    : {hy_str}\n")
            f.write(f"  Overlap(SPLADE,Rocchio) = {row['overlap_sp_ro']:.3f}")
            if has_hyde and row["overlap_sp_hy"] is not None:
                f.write(f"  Overlap(SPLADE,HyDE) = {row['overlap_sp_hy']:.3f}")
                f.write(f"  Overlap(Rocchio,HyDE) = {row['overlap_ro_hy']:.3f}")
            if has_hyde and row["shared_all_three"]:
                f.write(f"\n  Shared by all three: {', '.join(row['shared_all_three'])}")
            f.write("\n\n")

        # Aggregate overlap stats
        sp_ro_overlaps = [r["overlap_sp_ro"] for r in comparison_rows]
        f.write("AGGREGATE OVERLAP STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Mean Jaccard(SPLADE, Rocchio) = {np.mean(sp_ro_overlaps):.4f}\n")
        if has_hyde:
            sp_hy = [r["overlap_sp_hy"] for r in comparison_rows if r["overlap_sp_hy"] is not None]
            ro_hy = [r["overlap_ro_hy"] for r in comparison_rows if r["overlap_ro_hy"] is not None]
            if sp_hy:
                f.write(f"  Mean Jaccard(SPLADE, HyDE)    = {np.mean(sp_hy):.4f}\n")
            if ro_hy:
                f.write(f"  Mean Jaccard(Rocchio, HyDE)   = {np.mean(ro_hy):.4f}\n")
        f.write("\n")

        # Identify disagreement cases
        f.write("DISAGREEMENT CASE STUDIES\n")
        f.write("=" * 100 + "\n")
        # Sort by lowest SPLADE-Rocchio overlap to find biggest disagreements
        sorted_rows = sorted(comparison_rows, key=lambda r: r["overlap_sp_ro"])
        for i, row in enumerate(sorted_rows[:3], 1):
            f.write(f"\nCase {i}: Query \"{row['query']}\" (ID: {row['qid']})\n")
            f.write(f"  Jaccard(SPLADE, Rocchio) = {row['overlap_sp_ro']:.3f}\n")
            sp_only = {t.lower() for t, _ in row["splade_terms"][:10]} - {t.lower() for t, _ in row["rocchio_terms"]}
            ro_only = {t.lower() for t, _ in row["rocchio_terms"]} - {t.lower() for t, _ in row["splade_terms"][:10]}
            f.write(f"  SPLADE-only terms : {', '.join(sorted(sp_only)[:8])}\n")
            f.write(f"  Rocchio-only terms: {', '.join(sorted(ro_only)[:8])}\n")
            f.write("  Discussion: SPLADE learns contextual expansion via MLM pretraining,\n")
            f.write("  producing semantically related terms. Rocchio relies on term frequency\n")
            f.write("  in top-retrieved documents, biased toward co-occurring corpus vocabulary.\n")

        f.write("\n")
    print(f"Report written to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
TUNED_BM25_PARAMS = {
    "scifact":  {"k1": 1.2, "b": 0.75},
    "fever":    {"k1": 1.2, "b": 0.1},
    "hotpotqa": {"k1": 0.9, "b": 0.4},
    "msmarco":  {"k1": 0.9, "b": 0.4},
}


def evaluate_dataset(dataset_name: str, model_name: str,
                     workspace_dir: Path, output_dir: Path,
                     num_comparison_queries: int = 15):
    print(f"\n{'='*60}", flush=True)
    print(f"  PART 5 EVALUATION: {dataset_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Load queries & qrels
    if dataset_name == 'msmarco':
        print("Loading MSMARCO dev split for evaluation...")
        all_dev_queries, all_dev_qrels = load_dataset_queries_and_qrels("beir/msmarco/dev")
        all_qids = sorted(list(all_dev_queries.keys()))
        tuning_qids = set(RNG.sample(all_qids, min(1000, len(all_qids))))
        eval_qids = [qid for qid in all_qids if qid not in tuning_qids]
        
        queries = {qid: all_dev_queries[qid] for qid in eval_qids}
        qrels = {qid: all_dev_qrels[qid] for qid in eval_qids if qid in all_dev_qrels}
        print(f"MSMARCO eval queries (held-out dev): {len(queries)}", flush=True)
    else:
        test_ds_id = f"beir/{dataset_name}/test"
        queries, qrels = load_dataset_queries_and_qrels(test_ds_id)
        print(f"Loaded {len(queries)} test queries.", flush=True)

    # 2. SPLADE retrieval
    splade_run_path = output_dir / f"{dataset_name}_splade_run.json"
    if splade_run_path.exists():
        print(f"Loading cached SPLADE run from {splade_run_path}", flush=True)
        with open(splade_run_path, "r", encoding="utf-8") as f:
            splade_run = json.load(f)
    else:
        impact_searcher = init_impact_searcher(dataset_name, model_name)
        t0 = time.time()
        splade_run = run_splade_retrieval(impact_searcher, queries, k=100)
        t_splade = time.time() - t0
        print(f"SPLADE retrieval completed in {t_splade:.1f}s", flush=True)
        with open(splade_run_path, "w", encoding="utf-8") as f:
            json.dump(splade_run, f)
            
        # Free the Pyserini model from memory before loading the term extractor
        del impact_searcher
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    splade_metrics, splade_per_q = compute_metrics_and_per_query(splade_run, qrels)
    print(f"SPLADE Metrics: nDCG@10={splade_metrics['nDCG@10']:.4f} "
          f"Recall@100={splade_metrics['Recall@100']:.4f} "
          f"MRR@10={splade_metrics['MRR@10']:.4f} MAP={splade_metrics['MAP']:.4f}", flush=True)

    # 3. BM25 baseline for comparison
    bm25_params = TUNED_BM25_PARAMS.get(dataset_name, {"k1": 0.9, "b": 0.4})
    index_dir = workspace_dir / "indexes" / dataset_name
    
    bm25_run_path = output_dir / f"{dataset_name}_bm25_run.json"
    if bm25_run_path.exists():
        print(f"Loading cached BM25 run from {bm25_run_path}", flush=True)
        with open(bm25_run_path, "r", encoding="utf-8") as f:
            bm25_run = json.load(f)
    else:
        bm25_searcher = LuceneSearcher(str(index_dir))
        bm25_searcher.set_bm25(k1=bm25_params["k1"], b=bm25_params["b"])
        _, bm25_run = run_search(bm25_searcher, queries, k=100, desc="BM25 baseline")
        bm25_searcher.close()
        with open(bm25_run_path, "w", encoding="utf-8") as f:
            json.dump(bm25_run, f)
            
    bm25_metrics, _ = compute_metrics_and_per_query(bm25_run, qrels)
    print(f"BM25 baseline: nDCG@10={bm25_metrics['nDCG@10']:.4f}", flush=True)

    # 4. Select queries for three-way comparison
    qids = sorted(queries.keys())
    sample_size = min(num_comparison_queries, len(qids))
    comparison_qids = RNG.sample(qids, sample_size)
    print(f"Selected {len(comparison_qids)} queries for expansion-term comparison.", flush=True)

    # 5. Extract SPLADE expansion terms
    print("Extracting SPLADE expansion terms...", flush=True)
    extractor = SpladeTermExtractor(model_name)
    splade_terms = {}
    for qid in tqdm(comparison_qids, desc="SPLADE terms"):
        _, exp = extractor.extract_expansion_terms(queries[qid], top_k=20)
        splade_terms[qid] = exp

    # 6. Extract Rocchio expansion terms
    print("Extracting Rocchio/RM3 expansion terms...", flush=True)
    rocchio_terms = extract_rocchio_terms_for_queries(
        comparison_qids, queries, index_dir,
        k1=bm25_params["k1"], b=bm25_params["b"],
        fb_docs=5, fb_terms=10,
    )

    # 7. Extract HyDE expansion terms (if cache available and not skipped)
    hyde_terms: Dict[str, List[Tuple[str, float]]] = {}
    if dataset_name == "msmarco":
        print(f"Skipping HyDE expansion terms for {dataset_name} (HyDE evaluation skipped for MSMARCO).", flush=True)
    else:
        hyde_cache_patterns = [
            workspace_dir / "hyde_cache" / f"{dataset_name}_qwen2.5_7b_n5.json",
            workspace_dir / "hyde_cache" / f"{dataset_name}_hyde.json",
        ]
        hyde_cache = None
        for cp in hyde_cache_patterns:
            if cp.exists():
                print(f"Loading HyDE cache from {cp}", flush=True)
                with open(cp, "r", encoding="utf-8") as f:
                    hyde_cache = json.load(f)
                break

        if hyde_cache:
            available = [qid for qid in comparison_qids if qid in hyde_cache]
            if available:
                hyde_terms = extract_hyde_terms_for_queries(
                    available, queries, index_dir,
                    k1=bm25_params["k1"], b=bm25_params["b"],
                    hyde_cache=hyde_cache, fb_docs=5, fb_terms=10,
                )
                print(f"Extracted HyDE terms for {len(hyde_terms)} queries.", flush=True)
        else:
            print(f"[WARN] No HyDE cache found for {dataset_name}. "
                  "Comparison will be SPLADE vs Rocchio only.", flush=True)

    # 8. Build comparison table
    comparison_rows = build_comparison_table(
        comparison_qids, queries, splade_terms, rocchio_terms, hyde_terms
    )

    # 9. Write report
    report_path = output_dir / f"{dataset_name}_results.txt"
    write_part5_report(
        dataset_name, splade_metrics, bm25_metrics,
        comparison_rows, model_name, len(queries), report_path,
    )

    return {
        "dataset": dataset_name,
        "splade_metrics": splade_metrics,
        "bm25_metrics": bm25_metrics,
        "comparison_rows": comparison_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Part 5: SPLADE Sparse Retrieval with Pretrained Checkpoint"
    )
    parser.add_argument("--datasets", nargs="+", default=["scifact"],
                        help="Datasets to evaluate (scifact, fever, hotpotqa, msmarco)")
    parser.add_argument("--model", type=str,
                        default="naver/splade-cocondenser-ensembledistil",
                        help="HuggingFace SPLADE model name")
    parser.add_argument("--output", type=str, default="part5_results",
                        help="Output directory for reports")
    parser.add_argument("--num-comparison", type=int, default=15,
                        help="Number of queries for expansion-term comparison (>=10)")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parent
    output_dir = workspace_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in args.datasets:
        ds = ds.lower()
        index_path = workspace_dir / "indexes" / ds
        if not index_path.exists():
            print(f"[ERROR] BM25 index for {ds} not found at {index_path}. "
                  "Run build_indexes.py first.", flush=True)
            continue
        res = evaluate_dataset(ds, args.model, workspace_dir, output_dir,
                               num_comparison_queries=args.num_comparison)
        all_results.append(res)

    # Print summary
    print(f"\n{'='*70}")
    print("PART 5 SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        ds = r["dataset"].upper()
        sm = r["splade_metrics"]
        bm = r["bm25_metrics"]
        print(f"\n{ds}:")
        print(f"  SPLADE  nDCG@10={sm['nDCG@10']:.4f}  Recall@100={sm['Recall@100']:.4f}  "
              f"MRR@10={sm['MRR@10']:.4f}  MAP={sm['MAP']:.4f}")
        print(f"  BM25    nDCG@10={bm['nDCG@10']:.4f}  Recall@100={bm['Recall@100']:.4f}  "
              f"MRR@10={bm['MRR@10']:.4f}  MAP={bm['MAP']:.4f}")
        print(f"  Delta   nDCG@10={sm['nDCG@10']-bm['nDCG@10']:+.4f}")


if __name__ == "__main__":
    main()
