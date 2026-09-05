import argparse
import os
os.environ.setdefault("OPENAI_API_KEY", "dummy")
import ir_datasets
import torch
import json
import time
import os
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pyserini.search.lucene import LuceneSearcher
from part5_splade import compute_metrics_and_per_query, SpladeTermExtractor

# Fix random seed
torch.manual_seed(42)
random.seed(42)


def compute_splade_vector(logits, attention_mask):
    """Computes SPLADE sparse vector w_j = max_i log(1 + ReLU(o_ij))."""
    relu_logits = torch.relu(logits)
    log_logits = torch.log1p(relu_logits)
    mask_expanded = attention_mask.unsqueeze(-1).expand(log_logits.size())
    log_logits = log_logits * mask_expanded
    sparse_vector, _ = torch.max(log_logits, dim=1)
    return sparse_vector


@torch.no_grad()
def evaluate_custom_splade_fever(
    model_dir: str = "splade_finetuned_fever",
    index_dir: str = "indexes/fever",
    max_eval_queries: int = 1000,
    top_k_candidates: int = 100,
    output_file: str = "part6_fever_results.txt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading custom fine-tuned FEVER model from {model_dir} on {device}...")
    
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Make sure training completed and saved to this path.")
        
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForMaskedLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    # 1. Load FEVER test dataset
    print("Loading beir/fever/test dataset...")
    dataset = ir_datasets.load("beir/fever/test")
    
    all_queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    all_qids = [qid for qid in all_queries.keys() if qid in qrels]
    
    if max_eval_queries and max_eval_queries < len(all_qids):
        eval_qids = all_qids[:max_eval_queries]
        print(f"Evaluating on {len(eval_qids)} test queries (subsampled from {len(all_qids)} total).")
    else:
        eval_qids = all_qids
        print(f"Evaluating on all {len(eval_qids)} test queries.")

    # 2. Candidate generation via tuned BM25 on FEVER Lucene index
    print(f"Initializing BM25 Searcher from {index_dir} (k1=1.2, b=0.1)...")
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(1.2, 0.1)

    # 3. Custom SPLADE Scoring / Reranking
    print(f"Scoring top-{top_k_candidates} BM25 candidates with custom fine-tuned SPLADE...")
    run = {}
    
    for qid in tqdm(eval_qids, desc="SPLADE Evaluation (FEVER)"):
        qtext = all_queries[qid]
        hits = searcher.search(qtext, k=top_k_candidates)
        if not hits:
            run[qid] = {}
            continue
            
        candidate_docids = [h.docid for h in hits]
        candidate_texts = []
        valid_docids = []
        
        for did in candidate_docids:
            doc = searcher.doc(did)
            if doc is not None:
                raw = doc.raw()
                content = ""
                if raw:
                    try:
                        content = json.loads(raw).get("contents", "")
                    except Exception:
                        pass
                if not content:
                    content = doc.contents() or ""
                candidate_texts.append(content)
                valid_docids.append(did)

        if not valid_docids:
            run[qid] = {}
            continue

        # Encode Query
        q_inp = tokenizer([qtext], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        q_rep = compute_splade_vector(model(**q_inp).logits, q_inp["attention_mask"]) # (1, vocab_size)

        # Encode Candidate Documents (in mini-batches)
        doc_batch_size = 32
        d_reps_list = []
        for b_idx in range(0, len(candidate_texts), doc_batch_size):
            b_texts = candidate_texts[b_idx:b_idx+doc_batch_size]
            d_inp = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            d_rep = compute_splade_vector(model(**d_inp).logits, d_inp["attention_mask"])
            d_reps_list.append(d_rep)
            
        d_reps = torch.cat(d_reps_list, dim=0) # (num_candidates, vocab_size)

        # SPLADE score: dot product q_rep * d_rep
        splade_scores = torch.matmul(q_rep, d_reps.transpose(0, 1)).squeeze(0).cpu().tolist()
        
        run[qid] = {did: float(score) for did, score in zip(valid_docids, splade_scores)}

    searcher.close()

    # 4. Compute Metrics
    metrics, _ = compute_metrics_and_per_query(run, qrels)
    print("\n" + "=" * 60)
    print("      CUSTOM FINE-TUNED SPLADE METRICS (FEVER)")
    print("=" * 60)
    print(f"nDCG@10    : {metrics['nDCG@10']:.4f}")
    print(f"Recall@100 : {metrics['Recall@100']:.4f}")
    print(f"MRR@10     : {metrics['MRR@10']:.4f}")
    print(f"MAP        : {metrics['MAP']:.4f}")

    # 5. Expansion Term Comparison (10 Queries: Custom vs Pretrained Part 5)
    print("\nComparing Expansion Terms (Custom Fine-tuned vs Pretrained)...")
    pretrained_model_name = "naver/splade-cocondenser-ensembledistil"
    pretrained_extractor = SpladeTermExtractor(pretrained_model_name)
    custom_extractor = SpladeTermExtractor(model_dir)
    
    sample_qids = eval_qids[:10]
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("   EXPANSION TERM COMPARISON: PRETRAINED VS FINE-TUNED SPLADE (FEVER)")
    report_lines.append("=" * 80 + "\n")
    
    overlap_scores = []
    for qid in sample_qids:
        qtext = all_queries[qid]
        report_lines.append(f"Query ID: {qid}")
        report_lines.append(f"  Claim Text : \"{qtext}\"")
        
        # Pretrained terms
        _, pt_terms = pretrained_extractor.extract_expansion_terms(qtext, top_k=10)
        pt_str = ", ".join([f"{t}({w:.2f})" for t, w in pt_terms])
        report_lines.append(f"  Pretrained : {pt_str}")
        
        # Custom fine-tuned terms
        _, ct_terms = custom_extractor.extract_expansion_terms(qtext, top_k=10)
        ct_str = ", ".join([f"{t}({w:.2f})" for t, w in ct_terms])
        report_lines.append(f"  Fine-tuned : {ct_str}")
        
        # Jaccard overlap
        pt_set = {t.lower() for t, _ in pt_terms}
        ct_set = {t.lower() for t, _ in ct_terms}
        union = pt_set | ct_set
        overlap = len(pt_set & ct_set) / len(union) if union else 0.0
        overlap_scores.append(overlap)
        report_lines.append(f"  Jaccard Overlap: {overlap:.3f}\n")
        
    mean_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    report_lines.append(f"Mean Jaccard Overlap across 10 queries: {mean_overlap:.4f}\n")
    
    print("\n".join(report_lines))

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Part 6: Custom Trained SPLADE Evaluation — FEVER\n")
        f.write("==================================================\n\n")
        f.write("TABLE 1 — RETRIEVAL METRICS\n")
        f.write(f"{'Method':<40} | {'nDCG@10':<9} | {'Recall@100':<11} | {'MRR@10':<9} | {'MAP':<9}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Custom Fine-tuned SPLADE (FEVER)':<40} | {metrics['nDCG@10']:<9.4f} | {metrics['Recall@100']:<11.4f} | {metrics['MRR@10']:<9.4f} | {metrics['MAP']:<9.4f}\n")
        f.write("-" * 90 + "\n\n")
        f.write("\n".join(report_lines))
        
    print(f"Results successfully saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Part 6: Evaluate Fine-Tuned SPLADE on FEVER")
    parser.add_argument("--model_dir", type=str, default="splade_finetuned_fever", help="Path to saved fine-tuned model")
    parser.add_argument("--index_dir", type=str, default="indexes/fever", help="Path to FEVER Lucene index")
    parser.add_argument("--max_eval_queries", type=int, default=1000, help="Number of test queries to evaluate (default: 1000)")
    parser.add_argument("--top_k", type=int, default=100, help="Top-K candidates per query (default: 100)")
    parser.add_argument("--output", type=str, default="part6_fever_results.txt", help="Output file path")
    args = parser.parse_args()

    evaluate_custom_splade_fever(
        model_dir=args.model_dir,
        index_dir=args.index_dir,
        max_eval_queries=args.max_eval_queries,
        top_k_candidates=args.top_k,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
