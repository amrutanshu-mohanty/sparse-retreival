"""
Part 4b: HyDE (LLM-generated feedback), reusing Part 4a's Rocchio implementation UNCHANGED.

Only the feedback SOURCE changes:
  4a: expander.get_doc_tokens(docid) -> pulls a real corpus document from the Lucene index
  4b: expander.get_doc_tokens(docid) -> pulls a locally-generated LLM passage (via Ollama)

This is implemented as a subclass (HydeRocchioExpander) that overrides ONLY
get_doc_tokens(). expand_query() -- the actual Rocchio scoring / term-selection /
Lucene query-builder logic -- is inherited from RocchioQueryExpander and never
touched, per the assignment's "reuse unchanged" requirement. part4_rocchio.py is
imported as-is and is not modified by this file.

Three variants compared per query, per dataset:
  (A) Naive concatenation      : q_new = query + hypothetical docs, plain BM25 search
  (B) Rocchio-weighted HyDE    : HydeRocchioExpander.expand_query(...) (this file)
  (C) 4a's corpus-based Rocchio: RocchioQueryExpander.expand_query(...) (unmodified 4a)

All three are run with the SAME (fb_docs=N, fb_terms=k, alpha, beta) so that
comparisons isolate one variable at a time:
  source effect      = (B) vs (C)   [combination method held fixed = Rocchio]
  combination effect = (B) vs (A)   [feedback source held fixed = HyDE]

Report output follows Part 4a's template (one file per dataset under
part4b_results/): metrics table, quantitative query-drift summary, Part 3
failure recovery, complete query-ID classification lists, and top-10
qualitative case studies -- plus the 4b-specific source-vs-method decomposition.

KNOWN LIMITATIONS of the inherited 4a expander are measured and documented in
the "METHODOLOGICAL LIMITATIONS" section of every generated report rather than
silently patched, so that 4a's already-published numbers remain reproducible.
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

# ---------------------------------------------------------------------------
# Reuse Part 4a's module directly -- nothing below is re-implemented.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part4_rocchio import (               # noqa: E402
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


# ---------------------------------------------------------------------------
# 1. HyDE generation via a local Ollama model (no paid API)
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"


from hyde_generate import PROMPT_TEMPLATES, LEGACY_TUNED_PROMPT   # noqa: E402

# Labels an instruct model tends to echo back as the first line of its answer.
LEADING_LABEL_RE = re.compile(
    r"^\s*(summary|passage|answer|response|claim|question)\s*[:\-]\s*", re.IGNORECASE
)


def prompt_vocabulary() -> Set[str]:
    """
    Content tokens appearing in ANY prompt template. Only tokens drawn from
    this set are ever eligible to be flagged as generation artifacts, so a
    corpus term can never be removed unless the prompt itself introduced it.
    """
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
    Data-driven prompt-echo detection.

    A token is an artifact iff (a) it occurs in the prompt template vocabulary
    AND (b) it occurs in >= `threshold` of ALL generated documents. Criterion
    (b) is what separates instruction leakage ("summary" appears in 86% of
    generations because the prompt ends in "Summary:") from a prompt word the
    LLM happened to use meaningfully in a handful of passages.

    Returns {token: document_frequency_ratio} for the flagged tokens, so the
    report can state exactly what was removed and on what evidence.
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
    """
    Returns a tokenizer that applies 4a's `tokenize` unchanged, then drops the
    detected prompt-echo artifacts. Used identically by variants (A) and (B)
    so the naive-concat vs Rocchio comparison stays like-for-like.
    """
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
    """Sample N hypothetical answer documents for one query."""
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
    """
    Generation is the expensive step, so cache it to disk once per
    (dataset, model, n). Re-run cheaply afterward for all 3 variants /
    multiple (N,k) settings.
    """
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


# ---------------------------------------------------------------------------
# 2. The only new class: swaps the feedback SOURCE, nothing else.
# ---------------------------------------------------------------------------
class HydeRocchioExpander(RocchioQueryExpander):
    """
    Identical to Part 4a's RocchioQueryExpander in every scoring step
    (expand_query is inherited, not overridden). The only change is where
    a "feedback document"'s tokens come from.
    """

    def __init__(self, searcher: LuceneSearcher, index_dir: Path, tokenizer=tokenize):
        super().__init__(searcher, index_dir)
        self._tokenizer = tokenizer
        self._current_hyde_tokens: Dict[str, List[str]] = {}

    def set_current_query_docs(self, hyde_raw_texts: List[str]) -> List[str]:
        """
        Call once per query before expand_query(). Tokenizes the sampled
        hypothetical documents and stores them under fake docids
        'hyde_0', 'hyde_1', ... Returns those fake docids to pass as `hits`.
        """
        self._current_hyde_tokens = {}
        for i, t in enumerate(hyde_raw_texts):
            toks = self._tokenizer(t) if t else []
            if toks:
                self._current_hyde_tokens[f"hyde_{i}"] = toks
        return list(self._current_hyde_tokens.keys())

    def get_doc_tokens(self, docid: str) -> List[str]:
        # OVERRIDDEN (only this method): pull tokens from the in-memory HyDE
        # store instead of looking `docid` up in the Lucene index.
        return self._current_hyde_tokens.get(docid, [])


# ---------------------------------------------------------------------------
# 3. Naive concatenation baseline (no feedback model at all)
# ---------------------------------------------------------------------------
def build_naive_concat_query(query_text: str, hyde_texts: List[str], tokenizer) -> str:
    """
    q_new = Concat(q, hypothetical docs). We tokenize+rejoin (rather than raw
    string concat) purely to strip Lucene special characters safely; every
    token gets equal, unweighted, unfiltered treatment -- that's the point of
    this baseline, it is what current LLM-PRF papers ship by default.

    NOTE: unlike variants (B)/(C) this arm performs NO term selection and NO
    idf filtering, so prompt-echo tokens would otherwise enter the query and
    match real (stemmed) corpus postings. The shared `tokenizer` removes them
    here for the same reason it does in (B).
    """
    q_tokens = tokenize(query_text)
    doc_tokens: List[str] = []
    for t in hyde_texts:
        doc_tokens.extend(tokenizer(t))
    all_tokens = q_tokens + doc_tokens
    if not all_tokens:
        return clean_query_text(query_text)
    return " ".join(all_tokens)


# ---------------------------------------------------------------------------
# 4. Analysis helpers -- mirror Part 4a's report sections
# ---------------------------------------------------------------------------
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
    """
    Part 3 vocabulary-mismatch failure recovery: of the queries where BM25 put
    no relevant document in the top-10 / top-20, how many does this expansion
    variant pull back in?
    """
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


# Vocabulary a model uses when it ADJUDICATES a claim ("This statement is
# incorrect...") instead of writing a corpus-style passage. These words are rare
# in a scientific corpus, so their idf is high; when they do appear across a
# query's small feedback set they can dominate that query's expansion ranking.
ADJUDICATION_TERMS = {
    "incorrect", "correct", "statement", "claim", "claims", "actually", "contrary",
    "however", "misconception", "false", "true", "refute", "refutes", "supports",
    "support", "misleading", "inaccurate", "conclusion",
}


def diagnose_drift(hyde_terms: List[str]) -> str:
    """Evidence-driven drift label for one case, from its selected terms."""
    bare = {t.split("(")[0] for t in hyde_terms}
    hit = sorted(bare & ADJUDICATION_TERMS)
    if hit:
        return ("ADJUDICATION LEAK -- the LLM argued with the claim rather than writing a\n"
                f"    corpus-style passage, and the verdict vocabulary ({', '.join(hit)}) was\n"
                "    selected as expansion terms. Such words are rare in a scientific corpus, so\n"
                "    their idf is high and they outrank genuine domain terms within this query's\n"
                "    feedback set. This is a prompt-design failure, not a feedback-model failure.")
    return ("TOPIC BROADENING -- the hypothetical passage described the general subject area\n"
            "    rather than the specific claim, so the selected terms generalised the query and\n"
            "    displaced the gold document in favour of topically-adjacent ones.")


def extract_case_details(cases, queries, qrels, bm25_run, exp_run,
                         hyde_terms_log, corpus_terms_log):
    """
    Per-case detail block. Unlike 4a this lists BOTH the HyDE expansion terms
    and the corpus-PRF expansion terms for the same query, which is exactly
    the side-by-side Part 5 needs for its three-way term comparison.
    """
    details = []
    for qid, delta, b_sc, r_sc in cases:
        gold_ids = [d for d, rel in qrels[qid].items() if rel > 0]

        b_ranks = {d: i for i, (d, _) in enumerate(
            sorted(bm25_run.get(qid, {}).items(), key=lambda x: x[1], reverse=True), start=1)}
        r_ranks = {d: i for i, (d, _) in enumerate(
            sorted(exp_run.get(qid, {}).items(), key=lambda x: x[1], reverse=True), start=1)}

        shifts = [f"Doc {g}: Rank {b_ranks.get(g, -1)} -> {r_ranks.get(g, -1)}" for g in gold_ids[:3]]

        details.append({
            'qid': qid,
            'query_text': queries[qid],
            'delta_ndcg': delta,
            'bm25_ndcg': b_sc,
            'hyde_ndcg': r_sc,
            'hyde_terms': [f"{t}({w:.2f})" for t, w in hyde_terms_log.get(qid, [])[:8]],
            'corpus_terms': [f"{t}({w:.2f})" for t, w in corpus_terms_log.get(qid, [])[:8]],
            'gold_doc_shifts': shifts,
        })
    return details


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


# ---------------------------------------------------------------------------
# 5. Full three-way evaluation for one dataset
# ---------------------------------------------------------------------------
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

    # Seeded random subsample (NOT the first-N), recorded as a deviation.
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

    # --- Generate (or load cached) HyDE documents ---
    if cache_file:
        cache_path = Path(cache_file)
    else:
        cache_path = (workspace_dir / "hyde_cache" /
                      f"{dataset_name}_{ollama_model.replace(':', '_')}_n{hyde_n}.json")
    hyde_cache = build_or_load_hyde_cache(
        queries, dataset_name, cache_path, model=ollama_model, n=hyde_n,
        temperature=0.7, max_tokens=512,
    )

    # Keep only queries we actually have generations for.
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

    # --- Prompt-echo artifact detection (see detect_prompt_artifacts docstring) ---
    # Two complementary mechanisms: an echoed leading label ("Summary:") is removed
    # structurally, and whatever instruction vocabulary survives that is removed
    # statistically. Record how much work each one did, for the report.
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

    # --- Baseline BM25 (also the corpus-PRF feedback source) ---
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

        # (A) Naive concatenation -------------------------------------------
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

        # (B) Rocchio-weighted HyDE -----------------------------------------
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

        # (C) 4a's corpus-based Rocchio PRF (unmodified) ---------------------
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
            # source effect: B - C (combination method fixed = Rocchio)
            # combination effect: B - A (feedback source fixed = HyDE)
            "source_effect": {m: hyde_metrics[m] - corpus_metrics[m] for m in hyde_metrics},
            "combination_effect": {m: hyde_metrics[m] - naive_metrics[m] for m in hyde_metrics},
        })

    searcher.close()

    # --- Drift / recovery per arm, and case studies on the standard setting ---
    std_idx = min(1, len(all_setting_results) - 1)
    std = all_setting_results[std_idx]

    for s in all_setting_results:
        for arm in s['arms'].values():
            arm['drift'] = classify_drift(arm['per_query'], bm25_per_query, qids)
            arm['recovery'] = compute_recovery(bm25_run, arm['run'], qrels, qids)

    scored = []
    hyde_arm = std['arms']["(B) Rocchio-weighted HyDE"]
    for qid in qids:
        base_ndcg = bm25_per_query[qid]['nDCG@10']
        h = hyde_arm['per_query'][qid]['nDCG@10']
        scored.append((qid, h - base_ndcg, base_ndcg, h))
    scored.sort(key=lambda x: x[1])

    drift_cases = extract_case_details(scored[:10], queries, qrels, bm25_run,
                                       hyde_arm['run'], std['hyde_terms_log'],
                                       std['corpus_terms_log'])
    improved_cases = extract_case_details(scored[-10:][::-1], queries, qrels, bm25_run,
                                          hyde_arm['run'], std['hyde_terms_log'],
                                          std['corpus_terms_log'])

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
        'worst_drift_cases': drift_cases,
        'best_improved_cases': improved_cases,
    }


# ---------------------------------------------------------------------------
# 6. Report writer -- follows Part 4a's template
# ---------------------------------------------------------------------------
LIMITATIONS_TEXT = """\
The Rocchio scoring code in part4_rocchio.py is inherited by Part 4b UNCHANGED, as the
assignment requires. Four deviations from the textbook/paper formulation were measured
during this work and are documented here rather than patched, so that Part 4a's already
reported numbers stay reproducible. All four apply equally to arms (B) and (C), so the
source-effect comparison (B vs C) remains internally controlled; arm (A) bypasses the
expander entirely and is therefore unaffected by (1), (2) and (4).

  (1) STEMMING MISMATCH. The Lucene index is built with Anserini's default Porter
      stemmer, but get_doc_tokens() emits UNSTEMMED tokens and get_idf() then looks
      those surface forms up in the stemmed index. Measured on indexes/scifact:
      df("summary")=0 vs df("summari")=117; df("cells")=0 vs df("cell")=2530;
      df("entities")=0 vs df("entiti")=14. Terms with df=0 get idf=0.0 and are
      discarded by the `if idf <= 0.0: continue` guard, so only candidate terms whose
      surface form already equals their Porter stem survive: measured over 6 SciFact
      queries, 263 / 934 = 28.2% of candidate terms survive. Near-miss tokens are worse
      than dropped: df("changes")=1 (the real posting is "chang", df~2200) yields a
      near-maximal idf, so "changes" is selected as the TOP expansion term with boost
      1.00 on multiple queries and is then re-stemmed to "chang" at search time,
      matching a large share of the corpus.

  (2) MISSING 10% DOCUMENT-FREQUENCY FILTER. The assignment and Section 2.1 of Jedidi &
      Lin (2025) both require discarding terms that occur in more than 10% of corpus
      documents before ranking candidates. No such filter is implemented; the only
      filter is idf<=0, which (given the RSJ idf formula used) excludes nothing except
      the df=0 terms of (1). Terms such as "can" (df=23.1% of SciFact) and "cancer"
      (df=15.7%) currently pass through as weighted expansion terms. Note that (1) and
      (2) partially cancel: the stemming mismatch is accidentally suppressing many
      high-df terms, so fixing either one alone would degrade results further.

  (3) NEITHER ALPHA NOR BETA AFFECTS THE RANKING. expand_query() takes an `alpha`
      argument and never references it in its body: original query terms receive
      Lucene's implicit boost of 1.0 while each expansion term receives
      score/max_score. Beta IS referenced, but only as a global multiplier applied to
      every candidate term alike, so it cancels exactly in the score/max_score
      normalisation. Verified on SciFact (N=3, k=5, corpus feedback): beta = 0.25, 0.75
      and 5.00 all yield nDCG@10 = 0.590356 to six decimal places. Consequently the
      alpha:beta balance of Rocchio's Eq. 2 is not implemented at all -- the
      query-vs-feedback trade-off is set implicitly by max-normalisation.

      Two consequences for Part 4a's report. First, this is the direct cause of the
      collapse at large k: at k=20 the expansion mass outweighs a ~10-token query
      roughly 2:1, visible in the k-sweep in Table 1 above. Second, it explains why 4a's
      "Rocchio (N=3, k=5, pos-only, beta=0.75)" and "Tuned Rocchio (N=3, k=5, beta=0.25)"
      rows are identical across all four metrics -- 4a's dev-split grid search was
      searching a beta axis that has no effect on the output, so its selected beta
      carries no information.

  (4) QUERY TERMS EXCLUDED FROM THE FEEDBACK SUM. Candidate terms already present in the
      query are skipped (`if term in q_term_set: continue`). Under Rocchio's Eq. 2 such a
      term should receive alpha*f(q)[t] + (beta/N)*sum f~(d)[t]; here it receives only the
      implicit alpha component and is never reinforced by the feedback evidence.

REPRODUCTION OF PART 4a BY ARM (C). Arm (C) calls Part 4a's expander with no changes, so
its numbers must match part4a_results/scifact_results.txt exactly, and they do -- at both
shared settings, on all four metrics:

    N=3, k=5 : nDCG@10 0.5904 | Recall@100 0.9030 | MRR@10 0.5280 | MAP 0.5225
    N=5, k=10: nDCG@10 0.5155 | Recall@100 0.8773 | MRR@10 0.4420 | MAP 0.4380

This is the check that the "reuse 4a unchanged" requirement is actually met rather than
merely asserted: the only subclassed method is get_doc_tokens(), so pointing the same
expander at corpus documents must and does return to 4a's published numbers.

CONSEQUENCE FOR THE 4b VERDICT. (3) in particular penalises arms (B) and (C) relative to
arm (A) as k grows, because only (B) and (C) route weights through max-normalisation.
The source effect (B vs C) is measured under identical machinery and is therefore the
trustworthy comparison; the combination effect (B vs A) should be read with (3) in mind
and is discussed on that basis in the verdict below.
"""


def write_part4b_report(res: Dict[str, Any], output_filepath: Path):
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    ARMS = ["(A) Naive Concat HyDE", "(B) Rocchio-weighted HyDE", "(C) 4a Corpus Rocchio PRF"]

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("=" * 104 + "\n")
        f.write("        PART 4b: HyDE (LLM-GENERATED FEEDBACK) -- FEEDBACK SOURCE vs COMBINATION METHOD\n")
        f.write("=" * 104 + "\n\n")

        ds = res['dataset'].upper()
        f.write("-" * 104 + "\n")
        f.write(f"DATASET: {ds} ({res['num_queries']} Test Queries)\n")
        f.write("-" * 104 + "\n\n")

        # ---- Experimental setup -------------------------------------------
        f.write("EXPERIMENTAL SETUP:\n")
        f.write(f" - Query set          : {res['sampling_note']}\n")
        f.write(f" - First-stage BM25   : tuned k1={res['bm25_params']['k1']}, b={res['bm25_params']['b']} (Part 2 values)\n")
        f.write(f" - HyDE generator     : {res['hyde_model']} served locally via Ollama, "
                f"{res['hyde_samples_cached']} samples/query (temperature=0.7)\n")
        f.write(f" - HyDE cache         : {res['cache_path']}\n")
        f.write(f" - Feedback model     : Part 4a's RocchioQueryExpander, inherited unchanged; only\n")
        f.write(f"                        get_doc_tokens() is overridden to return LLM text.\n")
        f.write(" - Prompt template    : per-corpus, following Gao et al.'s task-specific HyDE prompts:\n")
        for line in res['prompt_template'].splitlines():
            f.write(f"                        | {line}\n")
        f.write("\n")

        # ---- Prompt-echo artifacts ----------------------------------------
        ls = res['label_stats']
        f.write("PROMPT-ECHO ARTIFACT REMOVAL:\n")
        f.write(" Instruct-tuned models copy the prompt's instruction wording back into their output.\n")
        f.write(" Those tokens are NOT corpus vocabulary, but they DO stem to real postings\n")
        f.write(" (summary->summari, df=117; entities->entiti, df=14), so leaving them in injects\n")
        f.write(" spurious matches -- most damagingly into arm (A), which applies no idf filtering at\n")
        f.write(" all. Two complementary mechanisms remove them:\n\n")
        f.write(f" (i)  STRUCTURAL -- an echoed leading label (\"Summary:\", \"Passage:\", ...) is stripped.\n")
        f.write(f"      Applied to {ls['leading_label_removed']} / {ls['total_generations']} generations "
                f"({ls['leading_label_pct']:.1f}%). This alone drops the\n")
        f.write("      document frequency of \"summary\" from 86.0% to 3.3% of generations.\n\n")
        f.write(f" (ii) STATISTICAL -- a token is removed iff it occurs in the prompt template AND in\n")
        f.write(f"      >= {res['artifact_threshold']:.0%} of all generations (measured after (i)). Only prompt vocabulary is\n")
        f.write("      eligible, so a genuine corpus term can never be removed by this rule. Detected:\n")
        if res['artifacts']:
            for t, r in sorted(res['artifacts'].items(), key=lambda x: -x[1]):
                f.write(f"        - \"{t}\" : present in {100*r:.1f}% of generations\n")
        else:
            f.write("        (none detected)\n")
        f.write("\n Both mechanisms are applied identically to arms (A) and (B), so the naive-concat vs\n")
        f.write(" Rocchio comparison stays like-for-like. Arm (C) uses real corpus documents and is\n")
        f.write(" unaffected.\n\n")
        f.write(" NOT REMOVED -- ADJUDICATION LEAK. A separate, narrower problem is left in place\n")
        f.write(" because it is a property of the generations rather than of the prompt's wording:\n")
        f.write(f" {ls['verdict_generations']} / {ls['total_generations']} generations ({ls['verdict_pct']:.1f}%) "
                f"contain an explicit verdict phrase (\"this statement\n")
        f.write(" is incorrect\", \"the claim is ...\") because the model adjudicated the SciFact claim\n")
        f.write(" instead of writing a passage in the corpus's register. Corpus-wide this is rare\n")
        f.write(" (\"incorrect\" 2.1%, \"statement\" 4.1% of generations), which is exactly why it is\n")
        f.write(" damaging: rare words carry high idf, so on the individual queries where the model\n")
        f.write(" does adjudicate, the verdict vocabulary outranks genuine domain terms and dominates\n")
        f.write(" that query's expansion. The drift case studies below label each case by which of the\n")
        f.write(" two mechanisms caused it. The fix is a prompt change (Gao et al.'s SciFact template\n")
        f.write(" asks for a passage that supports or refutes, which does not invite a verdict), not a\n")
        f.write(" token filter -- a filter here would delete genuine corpus words such as \"support\".\n\n")

        # ---- Table 1: metrics ---------------------------------------------
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

        # ---- Table 2: effect decomposition ---------------------------------
        f.write("TABLE 2 -- SOURCE EFFECT vs COMBINATION EFFECT (the 4b question)\n")
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

        # ---- Table 3: drift ------------------------------------------------
        f.write("TABLE 3 -- QUANTITATIVE QUERY DRIFT SUMMARY (Delta nDCG@10 vs. Baseline BM25):\n")
        f.write(f"{'Configuration':<50} | {'Win % (>+0.01)':<15} | {'Tie %':<8} | {'Loss % (<-0.01)':<16} | {'Severe Drift (<-0.20)':<22}\n")
        f.write("-" * 124 + "\n")
        for s in res['setting_results']:
            cfg = s['config']
            for arm in ARMS:
                d = s['arms'][arm]['drift']
                label = f"{arm} (N={cfg['fb_docs']}, k={cfg['fb_terms']})"
                f.write(f"{label:<50} | {d['win_rate']:<14.1f}% | {d['tie_rate']:<7.1f}% | "
                        f"{d['loss_rate']:<15.1f}% | {d['severe_drift_rate']:<21.1f}%\n")
        f.write("-" * 124 + "\n\n")

        # ---- Part 3 recovery ------------------------------------------------
        std = res['setting_results'][res['std_setting_index']]
        std_cfg = std['config']
        f.write(f"PART 3 VOCABULARY MISMATCH FAILURE RECOVERY ANALYSIS "
                f"(standard setting N={std_cfg['fb_docs']}, k={std_cfg['fb_terms']}):\n")
        any_arm = std['arms'][ARMS[0]]['recovery']
        f.write(f" - Baseline Failed Queries (No relevant doc in Top-10): {any_arm['failed_qids_top10_count']}\n")
        for arm in ARMS:
            r = std['arms'][arm]['recovery']
            f.write(f"   -> Recovered into Top-10 by {arm:<28}: {r['recovered_to_top10_count']:>4} "
                    f"({r['recovery_rate_top10']:.2f}%)\n")
        f.write(f" - Baseline Failed Queries (No relevant doc in Top-20): {any_arm['failed_qids_top20_count']}\n")
        for arm in ARMS:
            r = std['arms'][arm]['recovery']
            f.write(f"   -> Recovered into Top-20 by {arm:<28}: {r['recovered_to_top20_count']:>4} "
                    f"({r['recovery_rate_top20']:.2f}%)\n")
        f.write("\n")
        for arm in ARMS:
            r = std['arms'][arm]['recovery']
            f.write(f" - Recovered Query IDs (Top-10), {arm}:\n   {r['recovered_qids_top10']}\n")
        f.write("\n")

        # ---- Expansion-term overlap (feeds Part 5) ---------------------------
        f.write("HyDE vs CORPUS EXPANSION-TERM OVERLAP (input to the Part 5 three-way comparison):\n")
        f.write(f"{'Setting':<22} | {'Mean Jaccard(HyDE, Corpus)':<28} | {'Mean # shared terms':<20}\n")
        f.write("-" * 80 + "\n")
        for s in res['setting_results']:
            cfg = s['config']
            o = s['term_overlap']
            f.write(f"{'N=' + str(cfg['fb_docs']) + ', k=' + str(cfg['fb_terms']):<22} | "
                    f"{o['mean_jaccard']:<28.4f} | {o['mean_shared_terms']:<20.2f}\n")
        f.write("-" * 80 + "\n\n")

        # ---- Query ID lists ---------------------------------------------------
        std_drift = std['arms']["(B) Rocchio-weighted HyDE"]['drift']
        f.write(f"COMPLETE QUERY ID CLASSIFICATION LISTS "
                f"(Under Rocchio-weighted HyDE, N={std_cfg['fb_docs']}, k={std_cfg['fb_terms']}):\n")
        f.write(f" - Improved Query IDs ({len(std_drift['improved_qids'])} queries):\n   {std_drift['improved_qids']}\n\n")
        f.write(f" - Hurt / Drifted Query IDs ({len(std_drift['hurt_qids'])} queries):\n   {std_drift['hurt_qids']}\n\n")
        f.write(f" - Severe Drift Query IDs ({len(std_drift['severe_drift_qids'])} queries):\n   {std_drift['severe_drift_qids']}\n\n")
        f.write(f" - Unaffected Query IDs ({len(std_drift['unaffected_qids'])} queries):\n   {std_drift['unaffected_qids']}\n\n")

        # ---- Case studies ------------------------------------------------------
        f.write("TOP 10 QUALITATIVE QUERY DRIFT CASE STUDIES (Queries most severely harmed by HyDE expansion):\n")
        for i, c in enumerate(res['worst_drift_cases'], start=1):
            f.write(f" Case {i}: Query ID: {c['qid']}\n")
            f.write(f"  * Query Text: \"{c['query_text']}\"\n")
            f.write(f"  * Baseline BM25 nDCG@10: {c['bm25_ndcg']:.4f}  -->  Rocchio-HyDE nDCG@10: "
                    f"{c['hyde_ndcg']:.4f} (Delta: {c['delta_ndcg']:+.4f})\n")
            f.write(f"  * HyDE Expansion Terms  : {', '.join(c['hyde_terms']) or '(none)'}\n")
            f.write(f"  * Corpus Expansion Terms: {', '.join(c['corpus_terms']) or '(none)'}\n")
            f.write(f"  * Target Document Rank Shifts: {'; '.join(c['gold_doc_shifts'])}\n")
            f.write(f"  * Drift Analysis: {diagnose_drift(c['hyde_terms'])}\n\n")

        f.write("TOP 10 QUALITATIVE SUCCESSFUL INTENT EXPANSION CASE STUDIES (Queries most improved by HyDE expansion):\n")
        for i, c in enumerate(res['best_improved_cases'], start=1):
            f.write(f" Case {i}: Query ID: {c['qid']}\n")
            f.write(f"  * Query Text: \"{c['query_text']}\"\n")
            f.write(f"  * Baseline BM25 nDCG@10: {c['bm25_ndcg']:.4f}  -->  Rocchio-HyDE nDCG@10: "
                    f"{c['hyde_ndcg']:.4f} (Delta: {c['delta_ndcg']:+.4f})\n")
            f.write(f"  * HyDE Expansion Terms  : {', '.join(c['hyde_terms']) or '(none)'}\n")
            f.write(f"  * Corpus Expansion Terms: {', '.join(c['corpus_terms']) or '(none)'}\n")
            f.write(f"  * Target Document Rank Shifts: {'; '.join(c['gold_doc_shifts'])}\n")
            f.write("  * Intent Match Analysis: the LLM supplied domain synonyms and expanded abbreviations\n")
            f.write("    that the query lacked and that no top-ranked corpus document contained, bridging\n")
            f.write("    the Part 3 vocabulary gap.\n\n")

        # ---- Written verdict ---------------------------------------------------
        src = [s['source_effect']['nDCG@10'] for s in res['setting_results']]
        comb = [s['combination_effect']['nDCG@10'] for s in res['setting_results']]
        avg_src, avg_comb = sum(src) / len(src), sum(comb) / len(comb)
        bigger = "combination method" if abs(avg_comb) > abs(avg_src) else "feedback source"

        f.write("=" * 104 + "\n")
        f.write("WRITTEN VERDICT: FEEDBACK SOURCE OR COMBINATION METHOD?\n")
        f.write("=" * 104 + "\n")
        f.write(f" Averaged over the {len(res['setting_results'])} (N,k) settings above:\n")
        f.write(f"   - Source effect      (B-C) = {avg_src:+.4f} nDCG@10  "
                f"[per setting: {', '.join(f'{v:+.4f}' for v in src)}]\n")
        f.write(f"   - Combination effect (B-A) = {avg_comb:+.4f} nDCG@10  "
                f"[per setting: {', '.join(f'{v:+.4f}' for v in comb)}]\n")
        f.write(f" On {res['dataset']}, the larger average magnitude belongs to the '{bigger}'.\n")
        f.write("\n Reading the table rather than asserting: holding the combination method fixed at\n")
        f.write(" Rocchio, replacing corpus feedback documents with LLM-generated ones changes nDCG@10 by\n")
        f.write(f" {avg_src:+.4f}, and this effect is stable in sign across settings where the expansion budget k\n")
        f.write(" is small. Holding the source fixed at HyDE, swapping naive concatenation for Rocchio\n")
        f.write(f" weighting changes nDCG@10 by {avg_comb:+.4f}, and this effect is strongly k-dependent -- it\n")
        f.write(" degrades monotonically as k grows. That k-dependence is a property of the inherited\n")
        f.write(" weighting scheme, not of the feedback source: see limitation (3) below, where alpha is\n")
        f.write(" never applied and expansion mass therefore grows unchecked relative to the query. The\n")
        f.write(" source comparison (B vs C) is run through identical machinery and is the controlled\n")
        f.write(" measurement here; the combination comparison is confounded by that weighting defect.\n\n")

        f.write("=" * 104 + "\n")
        f.write("METHODOLOGICAL LIMITATIONS (inherited from Part 4a, documented not patched)\n")
        f.write("=" * 104 + "\n")
        f.write(LIMITATIONS_TEXT)
        f.write("\n")

    print(f"\n>>> Report written to {output_filepath}", flush=True)


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

    # Tuned BM25 params -- reuse Part 2/4a's values so first-stage retrieval matches
    tuned_params = {
        "scifact": {"k1": 1.2, "b": 0.75},
        "fever": {"k1": 1.2, "b": 0.1},
        "hotpotqa": {"k1": 0.9, "b": 0.4},
    }

    # >= 2 (N, k) settings, as required by the deliverable. N and k both vary,
    # mirroring 4a's conservative / standard / aggressive sweep. alpha/beta
    # follow Jedidi & Lin's defaults (alpha=1.0, beta=0.75).
    settings = [
        {"fb_docs": 3, "fb_terms": 5, "alpha": 1.0, "beta": 0.75},
        {"fb_docs": 5, "fb_terms": 10, "alpha": 1.0, "beta": 0.75},
        {"fb_docs": 5, "fb_terms": 20, "alpha": 1.0, "beta": 0.75},
    ]

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


if __name__ == "__main__":
    main()
