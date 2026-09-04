import argparse
import ir_datasets
import torch
import json
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from part5_splade import compute_metrics_and_per_query, init_impact_searcher, SpladeTermExtractor

# Fix random seed
torch.manual_seed(42)

def compute_splade_vector(logits, attention_mask):
    relu_logits = torch.relu(logits)
    log_logits = torch.log1p(relu_logits)
    mask_expanded = attention_mask.unsqueeze(-1).expand(log_logits.size())
    log_logits = log_logits * mask_expanded
    sparse_vector, _ = torch.max(log_logits, dim=1)
    return sparse_vector

@torch.no_grad()
def evaluate_custom_splade(model_dir: str, dataset_name: str = "scifact"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading custom model from {model_dir} to {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load dataset
    print(f"Loading {dataset_name} test dataset...")
    dataset = ir_datasets.load(f"beir/{dataset_name}/test")
    
    docs = {doc.doc_id: f"{doc.title} {doc.text}" for doc in dataset.docs_iter()}
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    doc_ids = list(docs.keys())
    query_ids = list(queries.keys())

    # Encode Corpus in batches
    print(f"Encoding {len(doc_ids)} documents...")
    batch_size = 32
    d_reps = []
    
    for i in tqdm(range(0, len(doc_ids), batch_size), desc="Encoding Corpus"):
        batch_ids = doc_ids[i:i+batch_size]
        batch_texts = [docs[did] for did in batch_ids]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        reps = compute_splade_vector(logits, inputs["attention_mask"])
        d_reps.append(reps.cpu())
        
    d_reps = torch.cat(d_reps, dim=0) # (num_docs, vocab_size)

    # Encode Queries
    print(f"Encoding {len(query_ids)} queries...")
    q_reps = []
    for i in tqdm(range(0, len(query_ids), batch_size), desc="Encoding Queries"):
        batch_ids = query_ids[i:i+batch_size]
        batch_texts = [queries[qid] for qid in batch_ids]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        reps = compute_splade_vector(logits, inputs["attention_mask"])
        q_reps.append(reps.cpu())
        
    q_reps = torch.cat(q_reps, dim=0) # (num_queries, vocab_size)

    # Compute dot products efficiently
    print("Computing retrieval scores (Query x Document matrix multiplication)...")
    # For memory reasons on larger datasets, sparse matmul is better, but SciFact fits in RAM easily
    # q_reps: (Q, V), d_reps: (D, V) -> scores: (Q, D)
    scores = torch.matmul(q_reps, d_reps.transpose(0, 1))
    
    # Extract top 100 for each query
    print("Extracting top 100 results per query...")
    top_k = min(100, scores.size(1))
    top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
    
    run = {}
    for q_idx, qid in enumerate(query_ids):
        run[qid] = {}
        for rank in range(top_k):
            d_idx = top_indices[q_idx, rank].item()
            score = top_scores[q_idx, rank].item()
            docid = doc_ids[d_idx]
            run[qid][docid] = score
            
    # Compute Metrics
    metrics, _ = compute_metrics_and_per_query(run, qrels)
    print("\n==================================================")
    print("      CUSTOM FINETUNED SPLADE METRICS")
    print("==================================================")
    print(f"nDCG@10    : {metrics['nDCG@10']:.4f}")
    print(f"Recall@100 : {metrics['Recall@100']:.4f}")
    print(f"MRR@10     : {metrics['MRR@10']:.4f}")
    print(f"MAP        : {metrics['MAP']:.4f}")
    
    # Generate expansion terms for 10 queries
    print("\nComparing Expansion Terms (Custom vs Pretrained from Part 5)...")
    
    # Load the pretrained Part 5 extractor
    pretrained_model_name = "naver/splade-cocondenser-ensembledistil"
    pretrained_extractor = SpladeTermExtractor(pretrained_model_name)
    
    # We can reuse SpladeTermExtractor for our custom model by passing the local path
    custom_extractor = SpladeTermExtractor(model_dir)
    
    sample_qids = query_ids[:10]
    
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("      EXPANSION TERM COMPARISON (10 QUERIES)")
    report_lines.append("==================================================\n")
    
    for qid in sample_qids:
        qtext = queries[qid]
        report_lines.append(f"Query ID: {qid}")
        report_lines.append(f"  Text: {qtext}")
        
        # Pretrained
        _, pt_terms = pretrained_extractor.extract_expansion_terms(qtext, top_k=10)
        pt_str = ", ".join([f"{term}({weight:.2f})" for term, weight in pt_terms])
        report_lines.append(f"  Pretrained: {pt_str}")
        
        # Custom
        _, ct_terms = custom_extractor.extract_expansion_terms(qtext, top_k=10)
        ct_str = ", ".join([f"{term}({weight:.2f})" for term, weight in ct_terms])
        report_lines.append(f"  Fine-tuned: {ct_str}")
        report_lines.append("")
        
    print("\n".join(report_lines))
    
    out_path = Path("part6_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Custom SPLADE Model Metrics (SciFact):\n")
        f.write(f"nDCG@10    : {metrics['nDCG@10']:.4f}\n")
        f.write(f"Recall@100 : {metrics['Recall@100']:.4f}\n")
        f.write(f"MRR@10     : {metrics['MRR@10']:.4f}\n")
        f.write(f"MAP        : {metrics['MAP']:.4f}\n\n")
        f.write("\n".join(report_lines))
        
    print(f"Saved complete report to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate custom SPLADE model on SciFact")
    parser.add_argument("--model_dir", type=str, default="splade_finetuned_scifact", help="Path to custom model")
    args = parser.parse_args()
    
    evaluate_custom_splade(args.model_dir)
