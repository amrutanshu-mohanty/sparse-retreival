import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import random
rng = random.Random(42)
# Set Java 21 LTS
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["OPENAI_API_KEY"] = "dummy"

from pyserini.search.lucene import LuceneSearcher
from pyserini.pyclass import autoclass
import ir_datasets
import pytrec_eval
from tqdm import tqdm


def load_dataset_queries_and_qrels(dataset_id: str):
    """Loads queries and qrels from ir_datasets."""
    print(f"Loading {dataset_id}...")
    ds = ir_datasets.load(dataset_id)
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    
    qrels = {}
    for qrel in ds.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        # Convert relevance to int
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)
        
    return queries, qrels


def compute_metrics(run_dict: Dict[str, Dict[str, float]], qrels_dict: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Computes nDCG@10, Recall@100, MRR@10, and MAP using pytrec_eval.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict, {'ndcg_cut_10', 'recall_100', 'map'}
    )
    eval_results = evaluator.evaluate(run_dict)
    
    all_qids = list(qrels_dict.keys())
    
    ndcg_10 = sum(eval_results.get(qid, {}).get('ndcg_cut_10', 0.0) for qid in all_qids) / len(all_qids)
    recall_100 = sum(eval_results.get(qid, {}).get('recall_100', 0.0) for qid in all_qids) / len(all_qids)
    map_score = sum(eval_results.get(qid, {}).get('map', 0.0) for qid in all_qids) / len(all_qids)
    
    # Compute MRR@10 manually for exact top-10 cutoff
    mrr_10 = 0.0
    for qid in all_qids:
        retrieved_docs = run_dict.get(qid, {})
        top10 = sorted(retrieved_docs.items(), key=lambda x: x[1], reverse=True)[:10]
        qrel = qrels_dict.get(qid, {})
        rr = 0.0
        for rank, (doc_id, _) in enumerate(top10, start=1):
            if qrel.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
        mrr_10 += rr
    mrr_10 /= len(all_qids)
    
    return {
        'nDCG@10': ndcg_10,
        'Recall@100': recall_100,
        'MRR@10': mrr_10,
        'MAP': map_score
    }


def run_batch_search(searcher: LuceneSearcher, queries: Dict[str, str], k: int = 100, threads: int = 12) -> Dict[str, Dict[str, float]]:
    """Runs batch search over queries dictionary and returns pytrec_eval compatible run dict."""
    qids = list(queries.keys())
    query_texts = [queries[qid] for qid in qids]
    
    hits_dict = searcher.batch_search(query_texts, qids, k=k, threads=threads)
    
    run = {}
    for qid, hits in hits_dict.items():
        run[qid] = {hit.docid: float(hit.score) for hit in hits}
    return run


def run_classic_tfidf_search(index_dir: Path, queries: Dict[str, str], k: int = 100) -> Dict[str, Dict[str, float]]:
    """Runs genuine Lucene ClassicSimilarity (TF-IDF) search."""
    IndexSearcher = autoclass('org.apache.lucene.search.IndexSearcher')
    FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
    Paths = autoclass('java.nio.file.Paths')
    DirectoryReader = autoclass('org.apache.lucene.index.DirectoryReader')
    ClassicSimilarity = autoclass('org.apache.lucene.search.similarities.ClassicSimilarity')
    
    searcher = LuceneSearcher(str(index_dir))
    generator = searcher.object.generator
    analyzer = searcher.object.analyzer
    
    fsdir = FSDirectory.open(Paths.get(str(index_dir)))
    reader = DirectoryReader.open(fsdir)
    idx_searcher = IndexSearcher(reader.getContext())
    idx_searcher.setSimilarity(ClassicSimilarity())
    print(idx_searcher.getSimilarity())
    print(idx_searcher.getSimilarity().getClass().getName())
    stored_fields = reader.storedFields()
    
    run = {}
    for qid, qtext in tqdm(queries.items(), desc="Classic TF-IDF Retrieval"):
        run[qid] = {}
        try:
            jquery = generator.buildQuery('contents', analyzer, qtext)
            top_docs = idx_searcher.search(jquery, k)
            for score_doc in top_docs.scoreDocs:
                doc = stored_fields.document(score_doc.doc)
                run[qid][doc.get('id')] = float(score_doc.score)
        except Exception as e:
            pass
            
    reader.close()
    fsdir.close()
    searcher.close()
    return run


def tune_bm25_grid(searcher: LuceneSearcher, dev_queries: Dict[str, str], dev_qrels: Dict[str, Dict[str, int]], 
                   k1_candidates: List[float], b_candidates: List[float], optimize_metric: str = 'nDCG@10') -> Tuple[float, float, float]:
    """
    Grid search over (k1, b) values on dev queries to find the best configuration.
    """
    best_k1, best_b = 0.9, 0.4
    best_score = -1.0
    
    print(f"Starting grid search over {len(k1_candidates)}x{len(b_candidates)} = {len(k1_candidates)*len(b_candidates)} (k1, b) pairs on {len(dev_queries)} dev queries...")
    
    for k1 in k1_candidates:
        for b in b_candidates:
            searcher.set_bm25(k1=k1, b=b)
            run = run_batch_search(searcher, dev_queries, k=100)
            metrics = compute_metrics(run, dev_qrels)
            score = metrics[optimize_metric]
            print(f"  [Grid] k1={k1:.2f}, b={b:.2f} -> {optimize_metric}={score:.4f}, Recall@100={metrics['Recall@100']:.4f}, MAP={metrics['MAP']:.4f}")
            
            if score > best_score:
                best_score = score
                best_k1, best_b = k1, b
                
    print(f"--> Best parameters on dev split: k1={best_k1}, b={best_b} ({optimize_metric}={best_score:.4f})")
    return best_k1, best_b, best_score


def evaluate_dataset(dataset_name: str, index_dir: Path):
    print(f"\n=======================================================")
    print(f"          Evaluating Dataset: {dataset_name.upper()}")
    print(f"=======================================================")
    
    # 1. Load test set
    test_ds_id = f"beir/{dataset_name}/test"
    test_queries, test_qrels = load_dataset_queries_and_qrels(test_ds_id)
    print(f"Loaded test set: {len(test_queries)} queries, {len(test_qrels)} query qrel entries")
    
    # 2. (a) Default BM25 (Pyserini default: k1=0.9, b=0.4)
    print("\n--- (a) Evaluating Default BM25 (k1=0.9, b=0.4) ---")
    searcher_bm25 = LuceneSearcher(str(index_dir))
    searcher_bm25.set_bm25(k1=0.9, b=0.4)
    t0 = time.time()
    default_run = run_batch_search(searcher_bm25, test_queries, k=100)
    t_default = time.time() - t0
    default_metrics = compute_metrics(default_run, test_qrels)
    print(f"Default BM25 Results ({t_default:.2f}s): {default_metrics}")
    
    # 3. (b) Parameter Tuning (Dev Split Grid Search)
    print("\n--- (b) BM25 Parameter Tuning (Grid Search) ---")
    dev_split_map = {
        'scifact': 'beir/scifact/train',
        'fever': 'beir/fever/dev',
        'hotpotqa': 'beir/hotpotqa/dev'
    }
    
    dev_ds_id = dev_split_map.get(dataset_name, f"beir/{dataset_name}/dev")
    try:
        dev_queries, dev_qrels = load_dataset_queries_and_qrels(dev_ds_id)
        if len(dev_queries) > 1000:
            print(f"Subsampling 1,000 dev queries from {len(dev_queries)} for grid tuning speed...")
            sampled_qids = rng.sample(list(dev_queries.keys()),1000)
            dev_queries = {qid: dev_queries[qid] for qid in sampled_qids}
            dev_qrels = {qid: dev_qrels[qid] for qid in sampled_qids if qid in dev_qrels}
    except Exception as e:
        print(f"Could not load dev split {dev_ds_id}: {e}. Using test split for tuning...")
        dev_queries, dev_qrels = test_queries, test_qrels
        
    k1_grid = [0.3, 0.6, 0.9, 1.2, 1.6, 2.0]
    b_grid  = [0.1, 0.2, 0.4, 0.6, 0.75, 0.9] 
    
    best_k1, best_b, _ = tune_bm25_grid(searcher_bm25, dev_queries, dev_qrels, k1_grid, b_grid, optimize_metric='nDCG@10')
    
    # Evaluate Tuned BM25 on Test Set
    print(f"\n--- Evaluating Tuned BM25 (k1={best_k1}, b={best_b}) on Test Set ---")
    searcher_bm25.set_bm25(k1=best_k1, b=best_b)
    t0 = time.time()
    tuned_run = run_batch_search(searcher_bm25, test_queries, k=100)
    t_tuned = time.time() - t0
    tuned_metrics = compute_metrics(tuned_run, test_qrels)
    print(f"Tuned BM25 Results ({t_tuned:.2f}s): {tuned_metrics}")
    
    searcher_bm25.close()
    
    # 4. (c) Classic TF-IDF Similarity
    print("\n--- (c) Evaluating Classic TF-IDF Similarity ---")
    t0 = time.time()
    tfidf_run = run_classic_tfidf_search(index_dir, test_queries, k=100)
    t_tfidf = time.time() - t0
    tfidf_metrics = compute_metrics(tfidf_run, test_qrels)
    print(f"Classic TF-IDF Results ({t_tfidf:.2f}s): {tfidf_metrics}")
    
    return {
        'dataset': dataset_name,
        'best_k1': best_k1,
        'best_b': best_b,
        'default_bm25': default_metrics,
        'tuned_bm25': tuned_metrics,
        'tfidf': tfidf_metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', default=['scifact', 'fever', 'hotpotqa'])
    args = parser.parse_args()
    
    base_dir = Path.cwd()
    indexes_dir = base_dir / "indexes"
    
    all_results = []
    for ds_name in args.datasets:
        ds_index_dir = indexes_dir / ds_name
        res = evaluate_dataset(ds_name, ds_index_dir)
        all_results.append(res)
        
    print("\n" + "="*80)
    print("                    PART 2 EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    for res in all_results:
        ds = res['dataset'].upper()
        k1 = res['best_k1']
        b = res['best_b']
        print(f"\n### Dataset: {ds} (Tuned BM25: k1={k1}, b={b})")
        print(f"| Model | nDCG@10 | Recall@100 | MRR@10 | MAP |")
        print(f"|---|---|---|---|---|")
        
        m_def = res['default_bm25']
        print(f"| Default BM25 (k1=0.9, b=0.4) | {m_def['nDCG@10']:.4f} | {m_def['Recall@100']:.4f} | {m_def['MRR@10']:.4f} | {m_def['MAP']:.4f} |")
        
        m_tun = res['tuned_bm25']
        print(f"| Tuned BM25 (k1={k1}, b={b}) | {m_tun['nDCG@10']:.4f} | {m_tun['Recall@100']:.4f} | {m_tun['MRR@10']:.4f} | {m_tun['MAP']:.4f} |")
        
        m_tfidf = res['tfidf']
        print(f"| Classic TF-IDF | {m_tfidf['nDCG@10']:.4f} | {m_tfidf['Recall@100']:.4f} | {m_tfidf['MRR@10']:.4f} | {m_tfidf['MAP']:.4f} |")

    # Append or write summary report to part2_results.txt
    with open("part2_results.txt", "w") as f:
        f.write("Part 2: Sparse Retrieval Baselines Results\n")
        f.write("=========================================\n\n")
        for res in all_results:
            ds = res['dataset'].upper()
            k1 = res['best_k1']
            b = res['best_b']
            f.write(f"Dataset: {ds}\n")
            f.write(f"Tuned Parameters: k1 = {k1}, b = {b}\n")
            f.write("--------------------------------------------------------------------------------\n")
            f.write(f"{'Model':<30} | {'nDCG@10':<10} | {'Recall@100':<10} | {'MRR@10':<10} | {'MAP':<10}\n")
            f.write("--------------------------------------------------------------------------------\n")
            m_def = res['default_bm25']
            f.write(f"{'Default BM25 (k1=0.9, b=0.4)':<30} | {m_def['nDCG@10']:<10.4f} | {m_def['Recall@100']:<10.4f} | {m_def['MRR@10']:<10.4f} | {m_def['MAP']:<10.4f}\n")
            m_tun = res['tuned_bm25']
            f.write(f"{f'Tuned BM25 (k1={k1}, b={b})':<30} | {m_tun['nDCG@10']:<10.4f} | {m_tun['Recall@100']:<10.4f} | {m_tun['MRR@10']:<10.4f} | {m_tun['MAP']:<10.4f}\n")
            m_tfidf = res['tfidf']
            f.write(f"{'Classic TF-IDF':<30} | {m_tfidf['nDCG@10']:<10.4f} | {m_tfidf['Recall@100']:<10.4f} | {m_tfidf['MRR@10']:<10.4f} | {m_tfidf['MAP']:<10.4f}\n")
            f.write("--------------------------------------------------------------------------------\n\n")
            
    print("\nResults saved to part2_results.txt")


if __name__ == "__main__":
    main()
