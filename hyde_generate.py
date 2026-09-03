"""
Part 4b: HyDE hypothetical-document generation via local Ollama.

Generates a few hypothetical "answer documents" per query using a local
LLM (default: qwen2.5:7b-instruct via Ollama), and caches them to disk so
part4b_hyde.py can reuse them without regenerating.

Usage:
    ollama pull qwen2.5:7b-instruct
    ollama serve                      # if not already running
    python hyde_generate.py --datasets scifact fever hotpotqa

Resumable: if interrupted, re-running skips queries already in the cache
file. Safe to Ctrl+C and restart.

For FEVER / HotpotQA (thousands of test queries), generation with a local
7B model can take hours. Use --limit N to subsample the test set for a
documented deviation (state this in the report, per the assignment's
"any deviation from the standard split" instruction).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

RNG_SEED = 42
OLLAMA_URL = "http://localhost:11434/api/generate"

# Per-dataset prompt templates -- the SINGLE SOURCE OF TRUTH for HyDE prompts.
# part4b_hyde.py imports these rather than defining its own, so the prompt used
# to generate a cache can never drift from the prompt the evaluator documents.
#
# Gao et al.'s HyDE paper does not use one prompt: it uses task-specific
# templates ("Please write a passage to answer the question" is only the
# web-search one; SciFact gets "Please write a scientific paper passage to
# support/refute the claim"). We follow that per-corpus approach so the
# generated tokens resemble the target corpus's register.
#
# Deliberately kept free of instruction nouns ("key entities", "precise
# terminology", ...): instruct-tuned models echo those words back into the
# generation, where they become spurious expansion terms. See
# LEGACY_TUNED_PROMPT below and the artifact detector in part4b_hyde.py.
PROMPT_TEMPLATES = {
    "scifact": (
        "Please write a scientific paper passage to support or refute the claim.\n"
        "Claim: {query}\n"
        "Passage:"
    ),
    "fever": (
        "Please write an encyclopedia passage to support or refute the claim.\n"
        "Claim: {query}\n"
        "Passage:"
    ),
    "hotpotqa": (
        "Please write a passage to answer the question.\n"
        "Question: {query}\n"
        "Passage:"
    ),
}

DEFAULT_PROMPT = PROMPT_TEMPLATES["hotpotqa"]

# The template that actually produced the currently cached SciFact generations.
# Retained ONLY so its instruction vocabulary feeds part4b_hyde.py's prompt-echo
# artifact detector -- it is not used for new generations.
LEGACY_TUNED_PROMPT = (
    "Write a concise, factual scientific summary to answer the following question. "
    "Focus heavily on key entities, synonyms, and precise terminology.\n"
    "Question: {query}\n"
    "Summary:"
)


def load_dataset_queries(dataset_id: str) -> Dict[str, str]:
    """Loads just the queries (no qrels needed here) via ir_datasets."""
    import ir_datasets
    ds = ir_datasets.load(dataset_id)
    return {q.query_id: q.text for q in ds.queries_iter()}


def call_ollama(model: str, prompt: str, temperature: float, max_retries: int = 3,
                 timeout: int = 60) -> str:
    """Calls the local Ollama generate endpoint, with basic retry on failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 180,
        },
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            if text:
                return text
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [warn] Ollama call failed after {max_retries} attempts: {last_err}", file=sys.stderr)
    return ""


def generate_for_dataset(dataset_name: str, model: str, num_samples: int,
                          limit: int, cache_path: Path, temperature: float = 0.8):
    """
    Generates `num_samples` hypothetical documents per query for a dataset,
    resuming from cache_path if it already has partial results.
    """
    dataset_id = f"beir/{dataset_name}/test"
    print(f"\n=== Generating HyDE docs for {dataset_name.upper()} ===")
    queries = load_dataset_queries(dataset_id)
    print(f"Loaded {len(queries)} test queries.")

    qids = list(queries.keys())
    if limit and len(qids) > limit:
        rng = random.Random(RNG_SEED)
        qids = rng.sample(qids, limit)
        print(f"Subsampled to {limit} queries (seed={RNG_SEED}). "
              f"NOTE: record this subsampling as a deviation in your report.")

    # Load existing cache (resume support)
    cache: Dict[str, List[str]] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Resuming: {len(cache)} queries already cached at {cache_path}")

    prompt_template = PROMPT_TEMPLATES.get(dataset_name, DEFAULT_PROMPT)

    remaining = [qid for qid in qids if qid not in cache]
    print(f"{len(remaining)} queries left to generate.")

    for i, qid in enumerate(tqdm(remaining, desc=f"HyDE gen [{dataset_name}]")):
        qtext = queries[qid]
        prompt = prompt_template.format(query=qtext)

        docs = []
        for _ in range(num_samples):
            doc = call_ollama(model, prompt, temperature=temperature)
            if doc:
                docs.append(doc)

        if docs:
            cache[qid] = docs

        # Checkpoint every 25 queries so a crash doesn't lose progress.
        if (i + 1) % 25 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    print(f"--> Saved {len(cache)} queries' worth of HyDE docs to {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate HyDE hypothetical documents via local Ollama")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fever", "hotpotqa"])
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct",
                         help="Ollama model tag (must already be pulled)")
    parser.add_argument("--num-samples", type=int, default=3,
                         help="Hypothetical documents to sample per query")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None,
                         help="Subsample test queries per dataset (recommended for FEVER/HotpotQA)")
    parser.add_argument("--cache-dir", type=str, default="hyde_cache")
    args = parser.parse_args()

    # Sanity check Ollama is reachable before burning time on dataset loads.
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("ERROR: Could not reach Ollama at http://localhost:11434. "
              "Is `ollama serve` running, and did you `ollama pull "
              f"{args.model}`?", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    for dataset in args.datasets:
        cache_path = cache_dir / f"{dataset}_hyde.json"
        generate_for_dataset(
            dataset_name=dataset,
            model=args.model,
            num_samples=args.num_samples,
            limit=args.limit,
            cache_path=cache_path,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()