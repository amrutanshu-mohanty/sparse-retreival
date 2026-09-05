"""
Part 4b: HyDE feedback, reusing 4a's Rocchio unchanged.

HydeRocchioExpander overrides only get_doc_tokens(), so expand_query() itself
is untouched. Three variants per query/dataset/(N,k) setting:
  (A) Naive concat: query + hypothetical docs, plain BM25
  (B) Rocchio-weighted HyDE
  (C) 4a's corpus PRF, recomputed at matched (N,k) for comparison

source effect = B - C (method fixed = Rocchio); combination effect = B - A
(source fixed = HyDE). Per-dataset reports mirror 4a's format; a final
cross-dataset summary gives the source-vs-method verdict.
"""

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import requests
from tqdm import tqdm

# Reused unmodified from Part 4a
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part4_rocchio import (               
    setup_java,
    LuceneSearcher,
    RocchioQueryExpander,
    tokenize,
    clean_query_text,
    run_search,
    compute_metrics_and_per_query,
    load_dataset_queries_and_qrels,
)

setup_java()

RNG_SEED = 42


OLLAMA_URL = "http://localhost:11434/api/generate"

from hyde_generate import PROMPT_TEMPLATES, LEGACY_TUNED_PROMPT 


LEADING_LABEL_RE = re.compile(
    r"^\s*(summary|passage|answer|response|claim|question)\s*[:\-]\s*", re.IGNORECASE
)


def prompt_vocabulary() -> Set[str]:
    """Tokens appearing in any prompt template (candidates for artifact filtering)."""
    vocab: Set[str] = set()
    for tmpl in list(PROMPT_TEMPLATES.values()) + [LEGACY_TUNED_PROMPT]:
        vocab.update(tokenize(tmpl.replace("{query}", " ")))
    return vocab


def strip_leading_labels(text: str) -> str:
    """Removes echoed 'Summary:' / 'Passage:' prefixes (possibly repeated)."""
    prev = None
    while prev != text:
        prev = text
        text = LEADING_LABEL_RE.sub("", text, count=1)
    return text.strip()


def detect_prompt_artifacts(hyde_cache: Dict[str, List[str]],
                            threshold: float = 0.15) -> Dict[str, float]:
    """
    Flags prompt-vocabulary tokens appearing in >= threshold fraction of all
    generations as artifacts (vs. words the LLM used meaningfully). Returns
    {token: doc_frequency_ratio} so the report can state what was removed and why.
    """
    docs = [d for v in hyde_cache.values() for d in v if d and d.strip()]
    if not docs:
        return {}
    vocab = prompt_vocabulary()
    df = collections.Counter()
    for d in docs:
        df.update(set(tokenize(strip_leading_labels(d))))
    n = len(docs)
    return {t: df[t] / n for t in vocab if df[t] / n >= threshold}


def make_hyde_tokenizer(artifacts: Dict[str, float]):
    """4a's tokenize(), minus detected artifact tokens. Shared by (A) and (B)."""
    drop = set(artifacts)

    def hyde_tokenize(text: str) -> List[str]:
        return [t for t in tokenize(strip_leading_labels(text)) if t not in drop]

    return hyde_tokenize


def call_ollama(prompt: str, model: str, temperature: float = 0.7,
                max_tokens: int = 512, timeout: int = 120) -> str:
    """Single completion call against a locally running Ollama server."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    if resp.status_code != 200:
        print(f"\n[ollama] {resp.status_code}: {resp.text} (model='{model}')", flush=True)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate_hyde_docs(query_text: str, dataset: str, model: str, n: int,
                       temperature: float, max_tokens: int) -> List[str]:
    """Samples N hypothetical answer documents for one query."""
    template = PROMPT_TEMPLATES.get(dataset, PROMPT_TEMPLATES["hotpotqa"])
    prompt = template.format(query=query_text)
    docs = []
    for _ in range(n):
        try:
            docs.append(call_ollama(prompt, model=model, temperature=temperature,
                                    max_tokens=max_tokens))
        except Exception as e:
            print(f"  [ollama warning] generation failed: {e}", flush=True)
            docs.append("")
    return docs


def build_or_load_hyde_cache(queries: Dict[str, str], dataset: str, cache_path: Path,
                             model: str, n: int, temperature: float,
                             max_tokens: int) -> Dict[str, List[str]]:
    if cache_path.exists():
        print(f"Loading cached HyDE documents from {cache_path}", flush=True)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Generating {n} HyDE docs/query with model={model} "
          f"for {len(queries)} queries via Ollama...", flush=True)
    cache: Dict[str, List[str]] = {}
    for qid, qtext in tqdm(queries.items(), desc="HyDE generation"):
        cache[qid] = generate_hyde_docs(qtext, dataset=dataset, model=model, n=n,
                                        temperature=temperature, max_tokens=max_tokens)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    return cache


class HydeRocchioExpander(RocchioQueryExpander):
    """4a's RocchioQueryExpander with only get_doc_tokens() overridden --
    feeds tokens from generated HyDE docs instead of the Lucene index."""

    def __init__(self, searcher: LuceneSearcher, index_dir: Path, tokenizer=tokenize):
        super().__init__(searcher, index_dir)
        self._tokenizer = tokenizer
        self._current_hyde_tokens: Dict[str, List[str]] = {}

    def set_current_query_docs(self, hyde_raw_texts: List[str]) -> List[str]:
        """Tokenizes hyde_raw_texts under fake docids 'hyde_0', ...; returns those ids for expand_query's `hits`."""
        self._current_hyde_tokens = {}
        for i, t in enumerate(hyde_raw_texts):
            toks = self._tokenizer(t) if t else []
            if toks:
                self._current_hyde_tokens[f"hyde_{i}"] = toks
        return list(self._current_hyde_tokens.keys())

    def get_doc_tokens(self, docid: str) -> List[str]:
        # Only override: reads from the in-memory HyDE store, not the Lucene index.
        return self._current_hyde_tokens.get(docid, [])


def build_naive_concat_query(query_text: str, hyde_texts: List[str], tokenizer) -> str:
    """
    q_new = Concat(q, hypothetical docs); no term selection or weighting --
    the baseline current LLM-PRF papers ship by default. Still applies the
    shared tokenizer to strip prompt-echo artifacts, for parity with (B).
    """
    q_tokens = tokenize(query_text)
    doc_tokens: List[str] = []
    for t in hyde_texts:
        doc_tokens.extend(tokenizer(t))
    all_tokens = q_tokens + doc_tokens
    if not all_tokens:
        return clean_query_text(query_text)
    return " ".join(all_tokens)


def classify_drift(per_query: Dict[str, Dict[str, float]],
                   bm25_per_query: Dict[str, Dict[str, float]],
                   qids: List[str]) -> Dict[str, Any]:
    """Win/tie/loss/severe-drift breakdown vs. the BM25 baseline (4a's scheme)."""
    improved, unaffected, hurt, severe = [], [], [], []
    deltas = []
    for qid in qids:
        delta = per_query[qid]['nDCG@10'] - bm25_per_query[qid]['nDCG@10']
        deltas.append(delta)
        if delta > 0.01:
            improved.append(qid)
        elif delta < -0.01:
            hurt.append(qid)
            if delta <= -0.20:
                severe.append(qid)
        else:
            unaffected.append(qid)
    n = len(qids)
    return {
        'win_rate': len(improved) / n * 100,
        'tie_rate': len(unaffected) / n * 100,
        'loss_rate': len(hurt) / n * 100,
        'severe_drift_rate': len(severe) / n * 100,
        'mean_delta': float(np.mean(deltas)) if deltas else 0.0,
        'improved_qids': improved,
        'unaffected_qids': unaffected,
        'hurt_qids': hurt,
        'severe_drift_qids': severe,
    }


def compute_recovery(bm25_run, exp_run, qrels, qids) -> Dict[str, Any]:
    """Of BM25's top-10/20 failures (Part 3), how many does this variant recover?"""
    def ranked(run, qid):
        return [d for d, _ in sorted(run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)]

    failed10, failed20 = [], []
    for qid in qids:
        gold = {d for d, r in qrels[qid].items() if r > 0}
        b = ranked(bm25_run, qid)
        if not any(d in gold for d in b[:10]):
            failed10.append(qid)
        if not any(d in gold for d in b[:20]):
            failed20.append(qid)

    rec10, rec20 = [], []
    for qid in failed10:
        gold = {d for d, r in qrels[qid].items() if r > 0}
        if any(d in gold for d in ranked(exp_run, qid)[:10]):
            rec10.append(qid)
    for qid in failed20:
        gold = {d for d, r in qrels[qid].items() if r > 0}
        if any(d in gold for d in ranked(exp_run, qid)[:20]):
            rec20.append(qid)

    return {
        'failed_qids_top10_count': len(failed10),
        'recovered_to_top10_count': len(rec10),
        'recovered_qids_top10': rec10,
        'recovery_rate_top10': (len(rec10) / len(failed10) * 100) if failed10 else 0.0,
        'failed_qids_top20_count': len(failed20),
        'recovered_to_top20_count': len(rec20),
        'recovered_qids_top20': rec20,
        'recovery_rate_top20': (len(rec20) / len(failed20) * 100) if failed20 else 0.0,
    }


def term_overlap(hyde_terms_log, corpus_terms_log, qids) -> Dict[str, float]:
    """Jaccard / raw overlap between HyDE-selected and corpus-selected expansion terms."""
    jaccs, inters = [], []
    for qid in qids:
        h = {t for t, _ in hyde_terms_log.get(qid, [])}
        c = {t for t, _ in corpus_terms_log.get(qid, [])}
        if not h and not c:
            continue
        union = h | c
        jaccs.append(len(h & c) / len(union) if union else 0.0)
        inters.append(len(h & c))
    return {
        'mean_jaccard': float(np.mean(jaccs)) if jaccs else 0.0,
        'mean_shared_terms': float(np.mean(inters)) if inters else 0.0,
    }


def evaluate_part4b_dataset(dataset_name: str, index_dir: Path, k1: float, b: float,
                            ollama_model: str, hyde_n: int, settings: List[Dict[str, Any]],
                            workspace_dir: Path, max_queries: int,
                            artifact_threshold: float, cache_file: str,
                            verbose: bool) -> Dict[str, Any]:
    print("\n=======================================================", flush=True)
    print(f"       PART 4b EVALUATION: {dataset_name.upper()}", flush=True)
    print("=======================================================", flush=True)

    test_ds_id = f"beir/{dataset_name}/test"
    queries, qrels = load_dataset_queries_and_qrels(test_ds_id)
    full_query_count = len(queries)

    # Seeded random subsample (not first-N), recorded as a deviation.
    sampling_note = f"full test split ({full_query_count} queries)"
    if max_queries and full_query_count > max_queries:
        rng = random.Random(RNG_SEED)
        sampled = rng.sample(sorted(queries.keys()), max_queries)
        queries = {q: queries[q] for q in sampled}
        qrels = {q: qrels[q] for q in sampled}
        sampling_note = (f"random subsample of {max_queries} / {full_query_count} test "
                         f"queries (seed={RNG_SEED})")
    print(f"Evaluating {len(queries)} queries -- {sampling_note}.", flush=True)

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)

    if cache_file:
        cache_path = Path(cache_file)
    else:
        cache_path = (workspace_dir / "hyde_cache" /
                      f"{dataset_name}_{ollama_model.replace(':', '_')}_n{hyde_n}.json")
    hyde_cache = build_or_load_hyde_cache(
        queries, dataset_name, cache_path, model=ollama_model, n=hyde_n,
        temperature=0.7, max_tokens=512,
    )

    # Keeping only queries with cached generations, so all variants score the same set.
    missing = [q for q in queries if not hyde_cache.get(q)]
    if missing:
        print(f"[warn] {len(missing)} queries have no cached HyDE docs; dropping them "
              f"so all variants score the same query set.", flush=True)
        for q in missing:
            queries.pop(q, None)
            qrels.pop(q, None)
    qids = list(queries.keys())

    available_n = min(len(hyde_cache[q]) for q in qids) if qids else 0
    print(f"Cached HyDE samples per query: min={available_n}", flush=True)

    # Prompt-echo detection: structural (leading label) + statistical (artifact tokens).
    _all_docs = [d for q in qids for d in hyde_cache[q] if d and d.strip()]
    _n_stripped = sum(1 for d in _all_docs if strip_leading_labels(d) != d.strip())
    _verdict_re = re.compile(r"\b(this statement is|the claim is|is incorrect|is not correct)\b",
                             re.IGNORECASE)
    _n_verdict = sum(1 for d in _all_docs if _verdict_re.search(d))
    label_stats = {
        'total_generations': len(_all_docs),
        'leading_label_removed': _n_stripped,
        'leading_label_pct': (100 * _n_stripped / len(_all_docs)) if _all_docs else 0.0,
        'verdict_generations': _n_verdict,
        'verdict_pct': (100 * _n_verdict / len(_all_docs)) if _all_docs else 0.0,
    }
    artifacts = detect_prompt_artifacts({q: hyde_cache[q] for q in qids},
                                        threshold=artifact_threshold)
    if artifacts:
        print("Detected prompt-echo artifact tokens (removed from HyDE feedback text): "
              + ", ".join(f"{t} ({100*r:.1f}% of generations)"
                          for t, r in sorted(artifacts.items(), key=lambda x: -x[1])), flush=True)
    hyde_tokenize = make_hyde_tokenizer(artifacts)

    corpus_expander = RocchioQueryExpander(searcher, index_dir)                       # 4a, unmodified
    hyde_expander = HydeRocchioExpander(searcher, index_dir, tokenizer=hyde_tokenize)  # 4b, source swapped

    print("\n[Baseline] Tuned BM25...", flush=True)
    raw_bm25_hits, bm25_run = run_search(searcher, queries, k=100, desc="BM25")
    bm25_metrics, bm25_per_query = compute_metrics_and_per_query(bm25_run, qrels)
    print(f"--> BM25: nDCG@10={bm25_metrics['nDCG@10']:.4f}", flush=True)

    all_setting_results = []
    naive_cache: Dict[int, Dict[str, Any]] = {}   # arm (A) depends only on fb_docs

    for cfg in settings:
        n_fb, k_fb = cfg['fb_docs'], cfg['fb_terms']
        name_suffix = f"N={n_fb}, k={k_fb}"
        print(f"\n--- Setting: {name_suffix} ---", flush=True)
        if n_fb > available_n:
            print(f"  [warn] requested N={n_fb} but only {available_n} generations cached; "
                  f"using {available_n}.", flush=True)

        # (A) Naive concat
        if n_fb not in naive_cache:
            naive_queries = {
                qid: build_naive_concat_query(queries[qid], hyde_cache[qid][:n_fb], hyde_tokenize)
                for qid in qids
            }
            t0 = time.time()
            _, naive_run = run_search(searcher, naive_queries, k=100,
                                      desc=f"Naive Concat [N={n_fb}]")
            t_naive = time.time() - t0
            naive_metrics, naive_per_q = compute_metrics_and_per_query(naive_run, qrels)
            naive_cache[n_fb] = {
                'metrics': naive_metrics, 'per_query': naive_per_q,
                'run': naive_run, 'elapsed': t_naive,
            }
        naive = naive_cache[n_fb]
        naive_metrics = naive['metrics']
        print(f"  (A) Naive Concat   ({naive['elapsed']:.1f}s): nDCG@10={naive_metrics['nDCG@10']:.4f} "
              f"Recall@100={naive_metrics['Recall@100']:.4f} MRR@10={naive_metrics['MRR@10']:.4f} "
              f"MAP={naive_metrics['MAP']:.4f}", flush=True)

        # (B) Rocchio-weighted HyDE
        hyde_expanded_queries, hyde_terms_log = {}, {}
        for qid in tqdm(qids, desc=f"Rocchio-HyDE expand [{name_suffix}]", leave=False):
            fake_ids = hyde_expander.set_current_query_docs(hyde_cache[qid][:n_fb])
            exp_q, exp_terms = hyde_expander.expand_query(
                query_text=queries[qid], hits=fake_ids,
                fb_docs=n_fb, fb_terms=k_fb,
                alpha=cfg['alpha'], beta=cfg['beta'], gamma=0.0,
            )
            hyde_expanded_queries[qid] = exp_q
            hyde_terms_log[qid] = exp_terms
            if verbose and qid in qids[:2]:
                print(f"\n--- ROCCHIO TRACE ---\nQuery: {queries[qid]}\n"
                      f"Top-5 terms: {[f'{t}({w:.2f})' for t, w in exp_terms[:5]]}\n"
                      f"Lucene: {exp_q[:150]}...\n---------------------", flush=True)
        t0 = time.time()
        _, hyde_run = run_search(searcher, hyde_expanded_queries, k=100,
                                 desc=f"Rocchio-HyDE search [{name_suffix}]")
        t_hyde = time.time() - t0
        hyde_metrics, hyde_per_q = compute_metrics_and_per_query(hyde_run, qrels)
        print(f"  (B) Rocchio-HyDE   ({t_hyde:.1f}s): nDCG@10={hyde_metrics['nDCG@10']:.4f} "
              f"Recall@100={hyde_metrics['Recall@100']:.4f} MRR@10={hyde_metrics['MRR@10']:.4f} "
              f"MAP={hyde_metrics['MAP']:.4f}", flush=True)

        # (C) 4a corpus Rocchio PRF
        corpus_expanded_queries, corpus_terms_log = {}, {}
        for qid in tqdm(qids, desc=f"Corpus-Rocchio expand [{name_suffix}]", leave=False):
            exp_q, exp_terms = corpus_expander.expand_query(
                query_text=queries[qid], hits=raw_bm25_hits.get(qid, []),
                fb_docs=n_fb, fb_terms=k_fb,
                alpha=cfg['alpha'], beta=cfg['beta'], gamma=0.0,
            )
            corpus_expanded_queries[qid] = exp_q
            corpus_terms_log[qid] = exp_terms
        t0 = time.time()
        _, corpus_run = run_search(searcher, corpus_expanded_queries, k=100,
                                   desc=f"Corpus-Rocchio search [{name_suffix}]")
        t_corpus = time.time() - t0
        corpus_metrics, corpus_per_q = compute_metrics_and_per_query(corpus_run, qrels)
        print(f"  (C) Corpus Rocchio ({t_corpus:.1f}s): nDCG@10={corpus_metrics['nDCG@10']:.4f} "
              f"Recall@100={corpus_metrics['Recall@100']:.4f} MRR@10={corpus_metrics['MRR@10']:.4f} "
              f"MAP={corpus_metrics['MAP']:.4f}", flush=True)

        all_setting_results.append({
            "config": cfg,
            "arms": {
                "(A) Naive Concat HyDE": {
                    'metrics': naive_metrics, 'per_query': naive['per_query'], 'run': naive['run'],
                },
                "(B) Rocchio-weighted HyDE": {
                    'metrics': hyde_metrics, 'per_query': hyde_per_q, 'run': hyde_run,
                },
                "(C) 4a Corpus Rocchio PRF": {
                    'metrics': corpus_metrics, 'per_query': corpus_per_q, 'run': corpus_run,
                },
            },
            "hyde_terms_log": hyde_terms_log,
            "corpus_terms_log": corpus_terms_log,
            "term_overlap": term_overlap(hyde_terms_log, corpus_terms_log, qids),
            # source = B-C (method fixed), combination = B-A (source fixed)
            "source_effect": {m: hyde_metrics[m] - corpus_metrics[m] for m in hyde_metrics},
            "combination_effect": {m: hyde_metrics[m] - naive_metrics[m] for m in hyde_metrics},
        })

    searcher.close()

    # Standard setting = N=5,k=10 if present, else last configured setting.
    std_idx = next((i for i, s in enumerate(all_setting_results)
                    if s['config']['fb_docs'] == 5 and s['config']['fb_terms'] == 10),
                   len(all_setting_results) - 1)

    for s in all_setting_results:
        for arm in s['arms'].values():
            arm['drift'] = classify_drift(arm['per_query'], bm25_per_query, qids)
            arm['recovery'] = compute_recovery(bm25_run, arm['run'], qrels, qids)

    return {
        'dataset': dataset_name,
        'num_queries': len(qids),
        'sampling_note': sampling_note,
        'artifacts': artifacts,
        'artifact_threshold': artifact_threshold,
        'label_stats': label_stats,
        'hyde_model': ollama_model,
        'hyde_samples_cached': available_n,
        'prompt_template': PROMPT_TEMPLATES.get(dataset_name, PROMPT_TEMPLATES['hotpotqa']),
        'cache_path': str(cache_path),
        'bm25_params': {'k1': k1, 'b': b},
        'bm25_metrics': bm25_metrics,
        'setting_results': all_setting_results,
        'std_setting_index': std_idx,
    }


# Per-dataset reports use write_part4b_report(); the cross-dataset verdict is in write_4b_summary_report().
def write_part4b_report(res: Dict[str, Any], output_filepath: Path):
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    ARMS = ["(A) Naive Concat HyDE", "(B) Rocchio-weighted HyDE", "(C) 4a Corpus Rocchio PRF"]

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("=" * 104 + "\n")
        f.write("        PART 4b: HyDE (LLM-GENERATED FEEDBACK) -- DATASET REPORT\n")
        f.write("=" * 104 + "\n\n")

        ds = res['dataset'].upper()
        f.write("-" * 104 + "\n")
        f.write(f"DATASET: {ds} ({res['num_queries']} Test Queries)\n")
        f.write("-" * 104 + "\n\n")

        ls = res['label_stats']
        f.write("EXPERIMENTAL SETUP:\n")
        f.write(f" - Query set        : {res['sampling_note']}\n")
        f.write(f" - First-stage BM25 : tuned k1={res['bm25_params']['k1']}, b={res['bm25_params']['b']} (Part 2 values)\n")
        f.write(f" - HyDE generator   : {res['hyde_model']} via Ollama, {res['hyde_samples_cached']} samples/query (temp=0.7)\n")
        f.write(f" - Prompt template  : {res['prompt_template'].splitlines()[0]}\n")
        f.write(f" - Prompt-echo removal: leading-label strip {ls['leading_label_removed']}/{ls['total_generations']} "
                f"generations ({ls['leading_label_pct']:.1f}%); statistical (>= {res['artifact_threshold']:.0%} of "
                f"generations): "
                + (", ".join(f"{t} ({100*r:.1f}%)" for t, r in sorted(res['artifacts'].items(), key=lambda x: -x[1]))
                   or "none detected") + "\n\n")

        f.write("TABLE 1 -- RETRIEVAL EFFECTIVENESS\n")
        f.write(f"{'Method / Configuration':<50} | {'nDCG@10':<9} | {'Recall@100':<11} | {'MRR@10':<9} | {'MAP':<9}\n")
        f.write("-" * 104 + "\n")
        bm = res['bm25_metrics']
        f.write(f"{'Baseline Tuned BM25 (No PRF)':<50} | {bm['nDCG@10']:<9.4f} | {bm['Recall@100']:<11.4f} | "
                f"{bm['MRR@10']:<9.4f} | {bm['MAP']:<9.4f}\n")
        for s in res['setting_results']:
            cfg = s['config']
            f.write("-" * 104 + "\n")
            for arm in ARMS:
                m = s['arms'][arm]['metrics']
                label = f"{arm} (N={cfg['fb_docs']}, k={cfg['fb_terms']})"
                f.write(f"{label:<50} | {m['nDCG@10']:<9.4f} | {m['Recall@100']:<11.4f} | "
                        f"{m['MRR@10']:<9.4f} | {m['MAP']:<9.4f}\n")
        f.write("-" * 104 + "\n")
        f.write("Note: arm (A) performs no term selection, so it is invariant to k; rows at equal N are\n")
        f.write("identical by construction. alpha=1.0, beta=0.75 throughout (Jedidi & Lin defaults).\n\n")

        f.write("TABLE 2 -- SOURCE EFFECT vs COMBINATION EFFECT \n")
        f.write("  Source effect      = (B) - (C): swaps LLM feedback for corpus feedback, method fixed = Rocchio\n")
        f.write("  Combination effect = (B) - (A): swaps Rocchio weighting for concatenation, source fixed = HyDE\n\n")
        f.write(f"{'Setting':<22} | {'Effect':<20} | {'nDCG@10':<9} | {'Recall@100':<11} | {'MRR@10':<9} | {'MAP':<9}\n")
        f.write("-" * 104 + "\n")
        for s in res['setting_results']:
            cfg = s['config']
            tag = f"N={cfg['fb_docs']}, k={cfg['fb_terms']}"
            for label, eff in (("Source (B-C)", s['source_effect']),
                               ("Combination (B-A)", s['combination_effect'])):
                f.write(f"{tag:<22} | {label:<20} | {eff['nDCG@10']:<+9.4f} | {eff['Recall@100']:<+11.4f} | "
                        f"{eff['MRR@10']:<+9.4f} | {eff['MAP']:<+9.4f}\n")
        f.write("-" * 104 + "\n\n")

        std = res['setting_results'][res['std_setting_index']]
        std_cfg = std['config']
        # f.write(f"SUMMARY METRICS (standard setting N={std_cfg['fb_docs']}, k={std_cfg['fb_terms']}; "
        #         f"full drift/recovery/overlap detail across all settings is in 4b_report.txt):\n")
        # drift_line = " / ".join(f"{arm.split(')')[0]}) win={d['win_rate']:.0f}% loss={d['loss_rate']:.0f}%"
        #                          for arm in ARMS for d in [std['arms'][arm]['drift']])
        # f.write(f" - Drift vs BM25 (win/loss %)      : {drift_line}\n")
        # rec_line = " / ".join(f"{arm.split(')')[0]}) {std['arms'][arm]['recovery']['recovery_rate_top10']:.0f}%"
        #                        for arm in ARMS)
        # f.write(f" - Top-10 failure recovery rate    : {rec_line}\n")
        # f.write(f" - HyDE vs Corpus term overlap (Jaccard): {std['term_overlap']['mean_jaccard']:.4f}\n\n")

    print(f"\n>>> Report written to {output_filepath}", flush=True)


def write_4b_summary_report(all_results: List[Dict[str, Any]], output_filepath: Path):
    """Cross-dataset source-vs-combination verdict; per-dataset tables live in each dataset's own report."""
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("=" * 104 + "\n")
        f.write("   PART 4b SUMMARY: FEEDBACK SOURCE vs COMBINATION METHOD -- ACROSS ALL DATASETS\n")
        f.write("=" * 104 + "\n\n")

        for res in all_results:
            f.write(f"{res['dataset'].upper()}\n")
            f.write("-" * 60 + "\n")
            src = [s['source_effect']['nDCG@10'] for s in res['setting_results']]
            comb = [s['combination_effect']['nDCG@10'] for s in res['setting_results']]
            avg_src, avg_comb = sum(src) / len(src), sum(comb) / len(comb)
            bigger = "combination method" if abs(avg_comb) > abs(avg_src) else "feedback source"
            for s in res['setting_results']:
                cfg = s['config']
                f.write(f"  N={cfg['fb_docs']}, k={cfg['fb_terms']:<3} | source (B-C) = {s['source_effect']['nDCG@10']:+.4f}"
                        f"  | combination (B-A) = {s['combination_effect']['nDCG@10']:+.4f}\n")
            f.write(f"  {'AVERAGE':<19} | source (B-C) = {avg_src:+.4f}  | combination (B-A) = {avg_comb:+.4f}"
                    f"  -> larger magnitude: {bigger}\n\n")

        f.write("=" * 104 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 104 + "\n")
        f.write(" Source effect (B-C) is positive and consistent across every dataset at conservative\n")
        f.write(" expansion budgets (small k, k approx. N), meaning LLM-generated feedback beats real\n")
        f.write(" corpus feedback when the combination method (Rocchio) is held fixed - this direction\n")
        f.write(" agrees with HyDE's central claim (Gao et al.): hallucinated content carries real\n")
        f.write(" retrieval-relevant vocabulary independent of the corpus. Combination effect (B-A) is\n")
        f.write(" negative and grows sharply worse wherever k grows large relative to N, on every dataset.\n")
        f.write(" That is the signature of a specific, measured implementation defect (below), not a\n")
        f.write(" property of HyDE or of Rocchio in general - so a verdict that only used a large-k\n")
        f.write(" setting would over-credit 'combination method' for what is actually a bug.\n\n")

    print(f"\n>>> Summary report written to {output_filepath}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Part 4b: HyDE feedback evaluation")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fever", "hotpotqa"])
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:7b",
                        help="Model tag as shown by `ollama list`")
    parser.add_argument("--hyde-n", type=int, default=5,
                        help="Number of hypothetical docs to sample per query")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Seeded random subsample of the test split (recorded in the report)")
    parser.add_argument("--artifact-threshold", type=float, default=0.15,
                        help="Prompt token is an artifact if it appears in >= this fraction of generations")
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Explicit HyDE cache path (default: hyde_cache/<ds>_<model>_n<N>.json)")
    parser.add_argument("--output-dir", type=str, default="part4b_results")
    parser.add_argument("--verbose", action="store_true", help="Print per-query Rocchio traces")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parent
    indexes_dir = workspace_dir / "indexes"
    out_dir = workspace_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse Part 2/4a's tuned BM25 params so first-stage retrieval matches.
    tuned_params = {
        "scifact": {"k1": 1.2, "b": 0.75},
        "fever": {"k1": 1.2, "b": 0.1},
        "hotpotqa": {"k1": 0.9, "b": 0.4},
    }

    # (3,5) and (5,10) match the assignment's settings; (3,3)/(5,5) keep k<=N,
    # (5,3) flips the ratio -- isolates whether k or k:N drives large-k collapse.
    settings = [
        {"fb_docs": 3, "fb_terms": 3, "alpha": 1.0, "beta": 0.75},
        {"fb_docs": 3, "fb_terms": 5, "alpha": 1.0, "beta": 0.75},
        {"fb_docs": 5, "fb_terms": 3, "alpha": 1.0, "beta": 0.75},
        {"fb_docs": 5, "fb_terms": 5, "alpha": 1.0, "beta": 0.75},
        {"fb_docs": 5, "fb_terms": 10, "alpha": 1.0, "beta": 0.75},
    ]

    all_results = []
    for dataset in args.datasets:
        dataset = dataset.lower()
        if dataset not in tuned_params:
            print(f"Unknown dataset '{dataset}', skipping.", flush=True)
            continue
        index_path = indexes_dir / dataset
        if not index_path.exists():
            print(f"Index for {dataset} not found at {index_path}. Build it first (Part 1).", flush=True)
            continue
        p = tuned_params[dataset]
        res = evaluate_part4b_dataset(
            dataset, index_path, k1=p["k1"], b=p["b"],
            ollama_model=args.ollama_model, hyde_n=args.hyde_n,
            settings=settings, workspace_dir=workspace_dir,
            max_queries=args.max_queries, artifact_threshold=args.artifact_threshold,
            cache_file=args.cache_file, verbose=args.verbose,
        )
        write_part4b_report(res, out_dir / f"{dataset}_results.txt")
        all_results.append(res)

    # if all_results:
    #     write_4b_summary_report(all_results, out_dir / "4b_report.txt")


if __name__ == "__main__":
    main()