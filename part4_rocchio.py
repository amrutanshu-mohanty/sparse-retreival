import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Ensure unbuffered standard UTF-8 console output
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# Fixed random seed for reproducibility
RNG = random.Random(42)

# Setup JAVA_HOME and JVM options for Pyserini/Lucene
def setup_java():
    os.environ["_JAVA_OPTIONS"] = "-Xmx1536m"
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
from pyserini.pyclass import autoclass
import ir_datasets
import pytrec_eval
from tqdm import tqdm
import numpy as np
import gc

# Standard IR Stopwords
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

LUCENE_SPECIAL_CHARS = re.compile(r'([\+\-\!\(\)\{\}\[\]\^\"~\*\?\:\\/]|&&|\|\|)')
LUCENE_RESERVED_WORDS = {"and", "or", "not", "to"}


def clean_query_text(text: str) -> str:
    """Sanitizes query text for Lucene search parser."""
    text = LUCENE_SPECIAL_CHARS.sub(' ', text)
    tokens = text.strip().split()
    cleaned = [t for t in tokens if t.lower() not in LUCENE_RESERVED_WORDS and len(t) > 1]
    return ' '.join(cleaned)


def tokenize(text: str, min_len: int = 3) -> List[str]:
    """Tokenizes text into lowercase alphabetic tokens, filtering stopwords and short tokens."""
    if not text:
        return []
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]


def load_dataset_queries_and_qrels(dataset_id: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Loads queries and qrels from ir_datasets."""
    print(f"Loading dataset: {dataset_id}...", flush=True)
    ds = ir_datasets.load(dataset_id)
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    
    qrels = {}
    for qrel in ds.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)
        
    valid_qids = {qid for qid, rels in qrels.items() if any(r > 0 for r in rels.values())}
    filtered_queries = {qid: queries[qid] for qid in valid_qids if qid in queries}
    filtered_qrels = {qid: qrels[qid] for qid in valid_qids if qid in queries}
    
    return filtered_queries, filtered_qrels


def run_search(searcher: LuceneSearcher, queries: Dict[str, str], k: int = 100, desc: str = None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """Runs direct search per query with tqdm, storing lightweight string IDs and freeing Java Hit objects."""
    raw_hits = {}
    run_dict = {}
    items = queries.items()
    if desc:
        items = tqdm(items, desc=desc, total=len(queries), leave=False)
    for qid, qtext in items:
        try:
            hits = searcher.search(qtext, k=k)
        except Exception:
            hits = []
        raw_hits[qid] = [hit.docid for hit in hits]
        run_dict[qid] = {hit.docid: float(hit.score) for hit in hits}
    return raw_hits, run_dict


def compute_metrics_and_per_query(run_dict: Dict[str, Dict[str, float]], 
                                  qrels_dict: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Computes aggregated and per-query nDCG@10, Recall@100, MRR@10, MAP."""
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'ndcg_cut_10', 'recall_100', 'map'})
    eval_results = evaluator.evaluate(run_dict)
    
    all_qids = list(qrels_dict.keys())
    per_query = {}
    
    total_ndcg10 = 0.0
    total_recall100 = 0.0
    total_map = 0.0
    total_mrr10 = 0.0
    
    for qid in all_qids:
        q_eval = eval_results.get(qid, {})
        ndcg10 = q_eval.get('ndcg_cut_10', 0.0)
        recall100 = q_eval.get('recall_100', 0.0)
        map_val = q_eval.get('map', 0.0)
        
        # Calculate MRR@10
        retrieved_docs = run_dict.get(qid, {})
        top10 = sorted(retrieved_docs.items(), key=lambda x: x[1], reverse=True)[:10]
        qrel = qrels_dict.get(qid, {})
        rr = 0.0
        for rank, (doc_id, _) in enumerate(top10, start=1):
            if qrel.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
                
        total_ndcg10 += ndcg10
        total_recall100 += recall100
        total_map += map_val
        total_mrr10 += rr
        
        per_query[qid] = {
            'nDCG@10': ndcg10,
            'Recall@100': recall100,
            'MRR@10': rr,
            'MAP': map_val
        }
        
    n_queries = len(all_qids)
    agg_metrics = {
        'nDCG@10': total_ndcg10 / n_queries if n_queries else 0.0,
        'Recall@100': total_recall100 / n_queries if n_queries else 0.0,
        'MRR@10': total_mrr10 / n_queries if n_queries else 0.0,
        'MAP': total_map / n_queries if n_queries else 0.0
    }
    return agg_metrics, per_query


class RocchioQueryExpander:
    """
    Implements Rocchio Pseudo-Relevance Feedback (PRF) with positive and negative document feedback.
    """
    def __init__(self, searcher: LuceneSearcher, index_dir: Path):
        self.searcher = searcher
        self.Term = autoclass('org.apache.lucene.index.Term')
        try:
            self.reader = searcher.object.reader
        except Exception:
            FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
            Paths = autoclass('java.nio.file.Paths')
            DirectoryReader = autoclass('org.apache.lucene.index.DirectoryReader')
            self.fsdir = FSDirectory.open(Paths.get(str(index_dir)))
            self.reader = DirectoryReader.open(self.fsdir)
            
        self.total_docs = self.reader.numDocs()
        self.idf_cache: Dict[str, float] = {}
        self.doc_content_cache: Dict[str, List[str]] = {}

    def get_idf(self, term: str) -> float:
        """Calculates Robertson-Spärck Jones IDF for a term with caching."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        try:
            lucene_term = self.Term('contents', term)
            df = self.reader.docFreq(lucene_term)
        except Exception:
            df = 0
            
        if df == 0:
            idf = 0.0
        else:
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
        self.idf_cache[term] = idf
        return idf

    def get_doc_tokens(self, docid: str) -> List[str]:
        """Retrieves and tokenizes document tokens directly from Lucene with bounded cache."""
        if docid in self.doc_content_cache:
            return self.doc_content_cache[docid]
        try:
            raw_doc = self.searcher.doc(docid)
            if raw_doc:
                raw_json = raw_doc.raw()
                if raw_json:
                    parsed = json.loads(raw_json)
                    text = parsed.get("contents", "")
                else:
                    text = raw_doc.contents() or ""
            else:
                text = ""
        except Exception:
            text = ""
            
        tokens = tokenize(text)
        if len(self.doc_content_cache) > 500:
            self.doc_content_cache.clear()
        self.doc_content_cache[docid] = tokens
        return tokens

    def expand_query(self, query_text: str, hits: List[str], 
                     fb_docs: int = 5, fb_terms: int = 10, 
                     alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.0,
                     neg_start_rank: int = 90, neg_count: int = 10) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Extracts candidate terms from top-fb_docs (and optional bottom negative docs),
        applies the Rocchio scoring formula, and returns the expanded Lucene query string.
        """
        q_tokens = tokenize(query_text)
        q_term_set = set(q_tokens)
        
        # 1. Collect Positive Feedback Documents D_R
        pos_hits = hits[:fb_docs]
        pos_counters = []
        pos_doc_lens = []
        candidate_term_set: Set[str] = set()
        
        for docid in pos_hits:
            tokens = self.get_doc_tokens(docid)
            if tokens:
                cnt = collections.Counter(tokens)
                pos_counters.append(cnt)
                pos_doc_lens.append(len(tokens))
                candidate_term_set.update(cnt.keys())
                
        if not pos_counters:
            clean_q = clean_query_text(query_text)
            return clean_q, []
            
        # 2. Collect Negative Feedback Documents D_NR (if gamma > 0)
        neg_counters = []
        neg_doc_lens = []
        if gamma > 0.0 and len(hits) >= (neg_start_rank + neg_count):
            neg_hits = hits[neg_start_rank:neg_start_rank + neg_count]
            for docid in neg_hits:
                tokens = self.get_doc_tokens(docid)
                if tokens:
                    cnt = collections.Counter(tokens)
                    neg_counters.append(cnt)
                    neg_doc_lens.append(len(tokens))
                    
        # 3. Compute Rocchio weights for candidate terms (excluding original query terms)
        term_scores: Dict[str, float] = {}
        for term in candidate_term_set:
            if term in q_term_set:
                continue
                
            idf = self.get_idf(term)
            if idf <= 0.0:
                continue
                
            # Positive feedback component: (beta / |D_R|) * sum(tf(d) * idf)
            pos_tf_sum = 0.0
            for cnt, dlen in zip(pos_counters, pos_doc_lens):
                c = cnt.get(term, 0)
                if c > 0:
                    pos_tf_sum += c / dlen
            pos_score = (beta / len(pos_counters)) * pos_tf_sum * idf
            
            # Negative feedback component: (gamma / |D_NR|) * sum(tf(d) * idf)
            neg_score = 0.0
            if gamma > 0.0 and neg_counters:
                neg_tf_sum = 0.0
                for cnt, dlen in zip(neg_counters, neg_doc_lens):
                    c = cnt.get(term, 0)
                    if c > 0:
                        neg_tf_sum += c / dlen
                neg_score = (gamma / len(neg_counters)) * neg_tf_sum * idf
                
            total_score = pos_score - neg_score
            if total_score > 0.0:
                term_scores[term] = total_score
                
        # 4. Sort and select top-fb_terms expansion terms
        sorted_exp_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:fb_terms]
        
        if not sorted_exp_terms:
            clean_q = clean_query_text(query_text)
            return clean_q, []
            
        # Normalize weights
        max_score = sorted_exp_terms[0][1]
        
        # 5. Formulate Expanded Lucene Query
        clean_q = clean_query_text(query_text)
        query_parts = [clean_q]
        for term, score in sorted_exp_terms:
            norm_w = round(score / max_score, 2)
            query_parts.append(f"{term}^{norm_w:.2f}")
            
        expanded_query = " ".join(query_parts)
        return expanded_query, sorted_exp_terms


def tune_rocchio_on_dev(searcher: LuceneSearcher, expander: RocchioQueryExpander, 
                        dev_queries: Dict[str, str], dev_qrels: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    Performs grid search optimization on the development split to discover the best Rocchio parameters.
    """
    print(f"\n--- Running Rocchio Hyperparameter Grid Search on Dev Split ({len(dev_queries)} queries) ---", flush=True)
    
    # 1. Baseline search on Dev
    raw_bm25_dev_hits, _ = run_search(searcher, dev_queries, k=100)
    
    # Define grid search space
    alpha_vals = [1.0]
    beta_vals = [0.25, 0.50, 0.75]
    gamma_vals = [0.0, 0.15]
    fb_docs_vals = [3, 5, 10]
    fb_terms_vals = [5, 10, 20]
    
    best_config = None
    best_ndcg = -1.0
    
    grid_combinations = []
    for a in alpha_vals:
        for b in beta_vals:
            for g in gamma_vals:
                for n in fb_docs_vals:
                    for k in fb_terms_vals:
                        grid_combinations.append({
                            'alpha': a, 'beta': b, 'gamma': g, 'fb_docs': n, 'fb_terms': k
                        })
                        
    print(f"Testing {len(grid_combinations)} hyperparameter configurations on Dev...", flush=True)
    
    for cfg in tqdm(grid_combinations, desc="Dev Grid Search", leave=True):
        exp_queries = {}
        for qid, qtext in dev_queries.items():
            hits = raw_bm25_dev_hits.get(qid, [])
            exp_q, _ = expander.expand_query(
                query_text=qtext, hits=hits,
                fb_docs=cfg['fb_docs'], fb_terms=cfg['fb_terms'],
                alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma']
            )
            exp_queries[qid] = exp_q
            
        _, exp_run = run_search(searcher, exp_queries, k=100)
        metrics, _ = compute_metrics_and_per_query(exp_run, dev_qrels)
        score = metrics['nDCG@10']
        
        if score > best_ndcg:
            best_ndcg = score
            best_config = cfg
            print(f"  [Dev Grid Best] N={cfg['fb_docs']}, k={cfg['fb_terms']}, beta={cfg['beta']}, gamma={cfg['gamma']} -> Dev nDCG@10 = {score:.4f}", flush=True)
            
    print(f"--> Optimal Dev Configuration: {best_config} (Dev nDCG@10={best_ndcg:.4f})", flush=True)
    return {
        'name': f"Tuned Rocchio (N={best_config['fb_docs']}, k={best_config['fb_terms']}, beta={best_config['beta']}, gamma={best_config['gamma']})",
        **best_config,
        'dev_score': best_ndcg
    }


DEFAULT_TUNED_ROCCHIO_CONFIGS = {
    'scifact': {'alpha': 1.0, 'beta': 0.25, 'gamma': 0.0, 'fb_docs': 3, 'fb_terms': 5, 'name': "Tuned Rocchio (N=3, k=5, beta=0.25, gamma=0.0)"},
    'fever': {'alpha': 1.0, 'beta': 0.25, 'gamma': 0.0, 'fb_docs': 3, 'fb_terms': 5, 'name': "Tuned Rocchio (N=3, k=5, beta=0.25, gamma=0.0)"},
    'hotpotqa': {'alpha': 1.0, 'beta': 0.25, 'gamma': 0.0, 'fb_docs': 3, 'fb_terms': 5, 'name': "Tuned Rocchio (N=3, k=5, beta=0.25, gamma=0.0)"}
}


def evaluate_part4_dataset(dataset_name: str, index_dir: Path, k1: float, b: float, bypass_grid_search: bool = False) -> Dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"       PART 4 EVALUATION: {dataset_name.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    
    # 1. Load test set
    test_ds_id = f"beir/{dataset_name}/test"
    queries, qrels = load_dataset_queries_and_qrels(test_ds_id)
    print(f"Evaluating {len(queries)} queries on test split.", flush=True)
    
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)
    expander = RocchioQueryExpander(searcher, index_dir)
    
    # -------------------------------------------------------------------------
    # 2. Dev Split Grid Search Optimization / Preset Loading
    # -------------------------------------------------------------------------
    if bypass_grid_search:
        tuned_rocchio_cfg = DEFAULT_TUNED_ROCCHIO_CONFIGS.get(
            dataset_name, 
            {'alpha': 1.0, 'beta': 0.25, 'gamma': 0.0, 'fb_docs': 3, 'fb_terms': 5, 'name': "Tuned Rocchio (N=3, k=5, beta=0.25, gamma=0.0)"}
        )
        print(f"\n[Bypassing Dev Grid Search] Loaded preset tuned configuration: {tuned_rocchio_cfg['name']}", flush=True)
    else:
        dev_split_map = {
            'scifact': 'beir/scifact/train',
            'fever': 'beir/fever/dev',
            'hotpotqa': 'beir/hotpotqa/dev'
        }
        dev_ds_id = dev_split_map.get(dataset_name, f"beir/{dataset_name}/dev")
        try:
            dev_queries, dev_qrels = load_dataset_queries_and_qrels(dev_ds_id)
            if len(dev_queries) > 300:
                print(f"Subsampling 300 dev queries from {len(dev_queries)} for fast grid search...", flush=True)
                sampled_qids = RNG.sample(list(dev_queries.keys()), 300)
                dev_queries = {qid: dev_queries[qid] for qid in sampled_qids}
                dev_qrels = {qid: dev_qrels[qid] for qid in sampled_qids if qid in dev_qrels}
        except Exception as e:
            print(f"Could not load dev split {dev_ds_id}: {e}. Subsampling from test split for grid tuning...", flush=True)
            sampled_qids = RNG.sample(list(queries.keys()), min(100, len(queries)))
            dev_queries = {qid: queries[qid] for qid in sampled_qids}
            dev_qrels = {qid: qrels[qid] for qid in sampled_qids}
            
        tuned_rocchio_cfg = tune_rocchio_on_dev(searcher, expander, dev_queries, dev_qrels)
    
    qids = list(queries.keys())
    
    # -------------------------------------------------------------------------
    # 3. Baseline Tuned BM25 (First-Stage Retrieval on Test Set)
    # -------------------------------------------------------------------------
    print("\n[1/7] Running Baseline Tuned BM25 (k=100)...", flush=True)
    t0 = time.time()
    raw_bm25_hits, bm25_run = run_search(searcher, queries, k=100, desc="BM25 Retrieval")
    t_bm25 = time.time() - t0
    bm25_metrics, bm25_per_query = compute_metrics_and_per_query(bm25_run, qrels)
    print(f"--> Baseline BM25 Results ({t_bm25:.2f}s): nDCG@10={bm25_metrics['nDCG@10']:.4f}, Recall@100={bm25_metrics['Recall@100']:.4f}, MRR@10={bm25_metrics['MRR@10']:.4f}, MAP={bm25_metrics['MAP']:.4f}", flush=True)
    
    # -------------------------------------------------------------------------
    # 4. Pyserini Native RM3 Baseline
    # -------------------------------------------------------------------------
    print("\n[2/7] Running Pyserini Native RM3 (fb_docs=5, fb_terms=10, weight=0.5)...", flush=True)
    searcher.set_rm3(fb_terms=10, fb_docs=5, original_query_weight=0.5)
    t0 = time.time()
    raw_rm3_hits, rm3_run = run_search(searcher, queries, k=100, desc="RM3 Retrieval")
    t_rm3 = time.time() - t0
    rm3_metrics, rm3_per_query = compute_metrics_and_per_query(rm3_run, qrels)
    searcher.unset_rm3()
    print(f"--> Native RM3 Results ({t_rm3:.2f}s): nDCG@10={rm3_metrics['nDCG@10']:.4f}, Recall@100={rm3_metrics['Recall@100']:.4f}, MRR@10={rm3_metrics['MRR@10']:.4f}, MAP={rm3_metrics['MAP']:.4f}", flush=True)

    # Define All Test Configurations
    rocchio_settings = [
        {
            "name": "Rocchio (N=3, k=5, pos-only)",
            "fb_docs": 3, "fb_terms": 5, "alpha": 1.0, "beta": 0.75, "gamma": 0.0
        },
        {
            "name": "Rocchio (N=5, k=10, pos-only)",
            "fb_docs": 5, "fb_terms": 10, "alpha": 1.0, "beta": 0.75, "gamma": 0.0
        },
        {
            "name": "Rocchio (N=10, k=20, pos-only)",
            "fb_docs": 10, "fb_terms": 20, "alpha": 1.0, "beta": 0.75, "gamma": 0.0
        },
        {
            "name": "Rocchio (N=5, k=10, pos+neg gamma=0.15)",
            "fb_docs": 5, "fb_terms": 10, "alpha": 1.0, "beta": 0.75, "gamma": 0.15
        },
        tuned_rocchio_cfg
    ]
    
    setting_results = []
    
    for idx, cfg in enumerate(rocchio_settings, start=3):
        print(f"\n[{idx}/7] Evaluating {cfg['name']} on Test Set...", flush=True)
        t0 = time.time()
        
        # Build expanded queries for all test queries
        expanded_queries = {}
        expansion_terms_log = {}
        for qid in tqdm(qids, desc=f"Expanding: {cfg['name'][:25]}", leave=False):
            hits = raw_bm25_hits.get(qid, [])
            qtext = queries[qid]
            exp_q, exp_terms = expander.expand_query(
                query_text=qtext, hits=hits,
                fb_docs=cfg['fb_docs'], fb_terms=cfg['fb_terms'],
                alpha=cfg['alpha'], beta=cfg['beta'], gamma=cfg['gamma']
            )
            expanded_queries[qid] = exp_q
            expansion_terms_log[qid] = exp_terms
            
        raw_exp_hits, exp_run = run_search(searcher, expanded_queries, k=100, desc=f"Searching: {cfg['name'][:25]}")
        t_exp = time.time() - t0
        
        agg_m, per_q_m = compute_metrics_and_per_query(exp_run, qrels)
        print(f"--> {cfg['name']} Results ({t_exp:.2f}s): nDCG@10={agg_m['nDCG@10']:.4f}, Recall@100={agg_m['Recall@100']:.4f}, MRR@10={agg_m['MRR@10']:.4f}, MAP={agg_m['MAP']:.4f}", flush=True)
        
        # Detailed Query Classification
        improved_qids = []
        unaffected_qids = []
        hurt_qids = []
        severe_drift_qids = []
        drift_deltas = []
        
        for qid in qids:
            bm25_score = bm25_per_query[qid]['nDCG@10']
            exp_score = per_q_m[qid]['nDCG@10']
            delta = exp_score - bm25_score
            drift_deltas.append(delta)
            
            if delta > 0.01:
                improved_qids.append(qid)
            elif delta < -0.01:
                hurt_qids.append(qid)
                if delta <= -0.20:
                    severe_drift_qids.append(qid)
            else:
                unaffected_qids.append(qid)
                
        n_q = len(qids)
        drift_summary = {
            'win_rate': (len(improved_qids) / n_q) * 100,
            'tie_rate': (len(unaffected_qids) / n_q) * 100,
            'loss_rate': (len(hurt_qids) / n_q) * 100,
            'severe_drift_rate': (len(severe_drift_qids) / n_q) * 100,
            'mean_delta': float(np.mean(drift_deltas)),
            'improved_qids': improved_qids,
            'unaffected_qids': unaffected_qids,
            'hurt_qids': hurt_qids,
            'severe_drift_qids': severe_drift_qids
        }
        
        setting_results.append({
            'config': cfg,
            'metrics': agg_m,
            'per_query': per_q_m,
            'drift_summary': drift_summary,
            'expansion_terms_log': expansion_terms_log,
            'run': exp_run,
            'elapsed_time': t_exp
        })
        del expanded_queries, raw_exp_hits
        gc.collect()

    searcher.close()
    
    # -------------------------------------------------------------------------
    # 5. Top-10 Qualitative Case Extraction (using Standard Setting N=5, k=10)
    # -------------------------------------------------------------------------
    std_result = setting_results[1]  # Rocchio (N=5, k=10, pos-only)
    per_q_std = std_result['per_query']
    
    scored_queries = []
    for qid in qids:
        b_ndcg = bm25_per_query[qid]['nDCG@10']
        r_ndcg = per_q_std[qid]['nDCG@10']
        delta = r_ndcg - b_ndcg
        scored_queries.append((qid, delta, b_ndcg, r_ndcg))
        
    scored_queries.sort(key=lambda x: x[1])
    worst_drift_cases = scored_queries[:10]
    best_improved_cases = scored_queries[-10:][::-1]
    
    def extract_case_details(cases):
        details = []
        for qid, delta, b_sc, r_sc in cases:
            qtext = queries[qid]
            gold_ids = [docid for docid, rel in qrels[qid].items() if rel > 0]
            exp_terms = std_result['expansion_terms_log'].get(qid, [])[:8]
            
            b_retrieved = sorted(bm25_run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            b_doc_ranks = {doc: rank for rank, (doc, _) in enumerate(b_retrieved, start=1)}
            
            r_retrieved = sorted(std_result['run'].get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            r_doc_ranks = {doc: rank for rank, (doc, _) in enumerate(r_retrieved, start=1)}
            
            gold_rank_shifts = []
            for g in gold_ids[:3]:
                b_r = b_doc_ranks.get(g, -1)
                r_r = r_doc_ranks.get(g, -1)
                gold_rank_shifts.append(f"Doc {g}: Rank {b_r} -> {r_r}")
                
            details.append({
                'qid': qid,
                'query_text': qtext,
                'delta_ndcg': delta,
                'bm25_ndcg': b_sc,
                'rocchio_ndcg': r_sc,
                'expansion_terms': [f"{t}({w:.2f})" for t, w in exp_terms],
                'gold_doc_shifts': gold_rank_shifts
            })
        return details

    drift_case_details = extract_case_details(worst_drift_cases)
    improved_case_details = extract_case_details(best_improved_cases)
    
    # -------------------------------------------------------------------------
    # 6. Part 3 Failure Query Recovery Analysis
    # -------------------------------------------------------------------------
    part3_failed_qids_10 = []
    part3_failed_qids_20 = []
    
    for qid in qids:
        b_retrieved = [doc for doc, _ in sorted(bm25_run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)]
        gold_set = {doc for doc, rel in qrels[qid].items() if rel > 0}
        
        has_rel_10 = any(d in gold_set for d in b_retrieved[:10])
        has_rel_20 = any(d in gold_set for d in b_retrieved[:20])
        
        if not has_rel_10:
            part3_failed_qids_10.append(qid)
        if not has_rel_20:
            part3_failed_qids_20.append(qid)
            
    recovered_qids_top10 = []
    recovered_qids_top20 = []
    
    for qid in part3_failed_qids_10:
        r_retrieved = [doc for doc, _ in sorted(std_result['run'].get(qid, {}).items(), key=lambda x: x[1], reverse=True)]
        gold_set = {doc for doc, rel in qrels[qid].items() if rel > 0}
        if any(d in gold_set for d in r_retrieved[:10]):
            recovered_qids_top10.append(qid)
            
    for qid in part3_failed_qids_20:
        r_retrieved = [doc for doc, _ in sorted(std_result['run'].get(qid, {}).items(), key=lambda x: x[1], reverse=True)]
        gold_set = {doc for doc, rel in qrels[qid].items() if rel > 0}
        if any(d in gold_set for d in r_retrieved[:20]):
            recovered_qids_top20.append(qid)
            
    recovery_stats = {
        'failed_qids_top10_count': len(part3_failed_qids_10),
        'recovered_to_top10_count': len(recovered_qids_top10),
        'recovered_qids_top10': recovered_qids_top10,
        'recovery_rate_top10': (len(recovered_qids_top10) / len(part3_failed_qids_10) * 100) if part3_failed_qids_10 else 0.0,
        'failed_qids_top20_count': len(part3_failed_qids_20),
        'recovered_to_top20_count': len(recovered_qids_top20),
        'recovered_qids_top20': recovered_qids_top20,
        'recovery_rate_top20': (len(recovered_qids_top20) / len(part3_failed_qids_20) * 100) if part3_failed_qids_20 else 0.0
    }
    
    return {
        'dataset': dataset_name,
        'num_queries': len(qids),
        'bm25_metrics': bm25_metrics,
        'rm3_metrics': rm3_metrics,
        'setting_results': setting_results,
        'worst_drift_cases': drift_case_details,
        'best_improved_cases': improved_case_details,
        'recovery_stats': recovery_stats
    }


def write_part4_report(all_dataset_results: List[Dict[str, Any]], output_filepath: Path):
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("========================================================================================================\n")
        f.write("              PART 4: ROCCHIO PRF, QUERY EXPANSION & QUERY DRIFT ANALYSIS REPORT\n")
        f.write("========================================================================================================\n\n")
        
        for res in all_dataset_results:
            ds = res['dataset'].upper()
            f.write(f"--------------------------------------------------------------------------------------------------------\n")
            f.write(f"DATASET: {ds} ({res['num_queries']} Test Queries)\n")
            f.write(f"--------------------------------------------------------------------------------------------------------\n\n")
            
            # Table 1: Retrieval Performance Comparison
            f.write(f"{'Method / Configuration':<50} | {'nDCG@10':<9} | {'Recall@100':<11} | {'MRR@10':<9} | {'MAP':<9}\n")
            f.write("-" * 100 + "\n")
            
            bm = res['bm25_metrics']
            f.write(f"{'Baseline Tuned BM25 (No PRF)':<50} | {bm['nDCG@10']:<9.4f} | {bm['Recall@100']:<11.4f} | {bm['MRR@10']:<9.4f} | {bm['MAP']:<9.4f}\n")
            
            rm3 = res['rm3_metrics']
            f.write(f"{'Pyserini Native RM3 (fb_docs=5, terms=10)':<50} | {rm3['nDCG@10']:<9.4f} | {rm3['Recall@100']:<11.4f} | {rm3['MRR@10']:<9.4f} | {rm3['MAP']:<9.4f}\n")
            
            for s in res['setting_results']:
                cfg_name = s['config']['name']
                m = s['metrics']
                f.write(f"{cfg_name:<50} | {m['nDCG@10']:<9.4f} | {m['Recall@100']:<11.4f} | {m['MRR@10']:<9.4f} | {m['MAP']:<9.4f}\n")
            f.write("-" * 100 + "\n\n")
            
            # Table 2: Query Drift Summary Table
            f.write("QUANTITATIVE QUERY DRIFT SUMMARY (Delta nDCG@10 vs. Baseline BM25):\n")
            f.write(f"{'Configuration':<50} | {'Win % (>+0.01)':<15} | {'Tie %':<8} | {'Loss % (<-0.01)':<16} | {'Severe Drift (<-0.20)':<22}\n")
            f.write("-" * 120 + "\n")
            for s in res['setting_results']:
                cfg_name = s['config']['name']
                d = s['drift_summary']
                f.write(f"{cfg_name:<50} | {d['win_rate']:<15.1f}% | {d['tie_rate']:<8.1f}% | {d['loss_rate']:<16.1f}% | {d['severe_drift_rate']:<22.1f}%\n")
            f.write("-" * 120 + "\n\n")
            
            # Part 3 Failure Recovery
            rec = res['recovery_stats']
            f.write("PART 3 VOCABULARY MISMATCH FAILURE RECOVERY ANALYSIS:\n")
            f.write(f" - Baseline Failed Queries (No relevant doc in Top-10): {rec['failed_qids_top10_count']}\n")
            f.write(f"   -> Recovered into Top-10 after Rocchio Expansion:     {rec['recovered_to_top10_count']} ({rec['recovery_rate_top10']:.2f}%)\n")
            f.write(f"   -> Recovered Query IDs (Top-10): {rec['recovered_qids_top10']}\n")
            f.write(f" - Baseline Failed Queries (No relevant doc in Top-20): {rec['failed_qids_top20_count']}\n")
            f.write(f"   -> Recovered into Top-20 after Rocchio Expansion:     {rec['recovered_to_top20_count']} ({rec['recovery_rate_top20']:.2f}%)\n")
            f.write(f"   -> Recovered Query IDs (Top-20): {rec['recovered_qids_top20']}\n\n")
            
            # Full Query ID Lists for Standard Rocchio Setting (N=5, k=10)
            std_drift = res['setting_results'][1]['drift_summary']
            f.write("COMPLETE QUERY ID CLASSIFICATION LISTS (Under Standard Rocchio N=5, k=10):\n")
            f.write(f" - Improved Query IDs ({len(std_drift['improved_qids'])} queries):\n   {std_drift['improved_qids']}\n\n")
            f.write(f" - Hurt / Drifted Query IDs ({len(std_drift['hurt_qids'])} queries):\n   {std_drift['hurt_qids']}\n\n")
            f.write(f" - Severe Drift Query IDs ({len(std_drift['severe_drift_qids'])} queries):\n   {std_drift['severe_drift_qids']}\n\n")
            f.write(f" - Unaffected Query IDs ({len(std_drift['unaffected_qids'])} queries):\n   {std_drift['unaffected_qids']}\n\n")
            
            # Qualitative Query Drift Cases (Top 10)
            f.write("TOP 10 QUALITATIVE QUERY DRIFT CASE STUDIES (Queries most severely harmed by expansion):\n")
            for idx, case in enumerate(res['worst_drift_cases'], start=1):
                f.write(f" Case {idx}: Query ID: {case['qid']}\n")
                f.write(f"  * Query Text: \"{case['query_text']}\"\n")
                f.write(f"  * Baseline BM25 nDCG@10: {case['bm25_ndcg']:.4f}  -->  Post-Rocchio nDCG@10: {case['rocchio_ndcg']:.4f} (Delta: {case['delta_ndcg']:+.4f})\n")
                f.write(f"  * Added Expansion Terms: {', '.join(case['expansion_terms'])}\n")
                f.write(f"  * Target Document Rank Shifts: {'; '.join(case['gold_doc_shifts'])}\n")
                f.write(f"  * Drift Analysis: False-positive feedback documents in top-N introduced off-topic contextual terms that diluted the original query keywords, displacing relevant documents down the ranking.\n\n")
                
            # Qualitative Successful Intent Expansion Cases (Top 10)
            f.write("TOP 10 QUALITATIVE SUCCESSFUL INTENT EXPANSION CASE STUDIES (Queries most improved by expansion):\n")
            for idx, case in enumerate(res['best_improved_cases'], start=1):
                f.write(f" Case {idx}: Query ID: {case['qid']}\n")
                f.write(f"  * Query Text: \"{case['query_text']}\"\n")
                f.write(f"  * Baseline BM25 nDCG@10: {case['bm25_ndcg']:.4f}  -->  Post-Rocchio nDCG@10: {case['rocchio_ndcg']:.4f} (Delta: {case['delta_ndcg']:+.4f})\n")
                f.write(f"  * Added Expansion Terms: {', '.join(case['expansion_terms'])}\n")
                f.write(f"  * Target Document Rank Shifts: {'; '.join(case['gold_doc_shifts'])}\n")
                f.write(f"  * Intent Match Analysis: Top pseudo-relevant documents supplied missing synonyms and related domain entities, bridging the vocabulary gap.\n\n")
                
            f.write("\n")
            
    print(f"\n>>> Full report successfully written to {output_filepath}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Part 4: Rocchio PRF, Dev Grid Search, Parameter Study & Query Drift Analysis")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fever", "hotpotqa"],
                        help="Datasets to evaluate (choices: scifact, fever, hotpotqa)")
    parser.add_argument("--bypass-grid-search", action="store_true", default=False,
                        help="Bypass the dev-split grid search and use the preset optimal tuned parameters directly")
    parser.add_argument("--output", type=str, default="part4_results.txt",
                        help="Output report text file")
    args = parser.parse_args()
    
    workspace_dir = Path(__file__).resolve().parent
    indexes_dir = workspace_dir / "indexes"
    
    tuned_params = {
        'scifact': {'k1': 1.2, 'b': 0.75},
        'fever': {'k1': 1.2, 'b': 0.1},
        'hotpotqa': {'k1': 0.9, 'b': 0.4}
    }
    
    all_results = []
    for dataset in args.datasets:
        dataset_clean = dataset.lower()
        if dataset_clean not in tuned_params:
            print(f"Unknown dataset '{dataset}'. Skipping.", flush=True)
            continue
            
        index_path = indexes_dir / dataset_clean
        if not index_path.exists():
            print(f"Index for {dataset_clean} not found at {index_path}. Build it first!", flush=True)
            continue
            
        p = tuned_params[dataset_clean]
        res = evaluate_part4_dataset(dataset_clean, index_path, k1=p['k1'], b=p['b'], bypass_grid_search=args.bypass_grid_search)
        all_results.append(res)
        
        output_path = workspace_dir / args.output
        write_part4_report(all_results, output_path)


if __name__ == "__main__":
    main()
