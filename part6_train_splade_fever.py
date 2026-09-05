import argparse
import os
os.environ.setdefault("OPENAI_API_KEY", "dummy")
import ir_datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.optim import AdamW
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import random
import os
import json
import time
from pathlib import Path

# Fix random seed for reproducibility
torch.manual_seed(42)
random.seed(42)


def compute_splade_vector(logits, attention_mask):
    """
    Computes the SPLADE sparse vector from MLM logits.
    w_j = max_i log(1 + ReLU(o_ij))
    """
    relu_logits = torch.relu(logits)
    log_logits = torch.log1p(relu_logits)
    mask_expanded = attention_mask.unsqueeze(-1).expand(log_logits.size())
    log_logits = log_logits * mask_expanded
    sparse_vector, _ = torch.max(log_logits, dim=1)
    return sparse_vector


class SpladeLoss(nn.Module):
    def __init__(self, lambda_q=0.01, lambda_d=0.008):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d

    def forward(self, q_rep, d_pos_rep, d_neg_rep):
        batch_size = q_rep.size(0)
        
        # InfoNCE Loss with in-batch negatives
        all_d = torch.cat([d_pos_rep, d_neg_rep], dim=0)  # (2*batch_size, vocab_size)
        scores = torch.matmul(q_rep, all_d.transpose(0, 1))  # (batch_size, 2*batch_size)
        labels = torch.arange(batch_size, device=q_rep.device)
        
        loss_rank = nn.functional.cross_entropy(scores, labels)
        
        # FLOPS Regularization: sum( mean_batch(w_j)^2 )
        q_flops = torch.sum(torch.mean(q_rep, dim=0) ** 2)
        d_flops = torch.sum(torch.mean(all_d, dim=0) ** 2)
        loss_reg = (self.lambda_q * q_flops) + (self.lambda_d * d_flops)
        
        return loss_rank, loss_reg, loss_rank.item(), loss_reg.item()


class FeverTrainDataset(Dataset):
    """
    Memory-efficient Dataset for FEVER:
    Uses Pyserini LuceneSearcher to fetch positive and negative document texts on-the-fly,
    avoiding storing 5.4M full documents in RAM.
    """
    def __init__(self, dataset_name="beir/fever/train", index_dir="indexes/fever", max_samples=10000):
        print(f"Loading {dataset_name} queries and qrels...")
        dataset = ir_datasets.load(dataset_name)
        
        self.queries = {q.query_id: q.text for q in dataset.queries_iter()}
        
        self.qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                if qrel.query_id not in self.qrels:
                    self.qrels[qrel.query_id] = []
                self.qrels[qrel.query_id].append(qrel.doc_id)
                
        all_qids = [qid for qid in self.qrels.keys() if qid in self.queries]
        random.shuffle(all_qids)
        
        if max_samples and max_samples < len(all_qids):
            self.qids = all_qids[:max_samples]
            print(f"Subsampled {len(self.qids)} training queries from {len(all_qids)} total.")
        else:
            self.qids = all_qids
            print(f"Loaded all {len(self.qids)} training queries.")
            
        print(f"Initializing BM25 Searcher for FEVER from {index_dir}...")
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(1.2, 0.1)  # FEVER tuned BM25 parameters
        
    def _get_doc_text(self, doc_id: str) -> str:
        doc = self.searcher.doc(doc_id)
        if doc is not None:
            raw = doc.raw()
            if raw:
                try:
                    data = json.loads(raw)
                    return data.get("contents", data.get("text", ""))
                except Exception:
                    pass
            contents = doc.contents()
            if contents:
                return contents
        return ""

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        qtext = self.queries[qid]
        
        # Positive document
        pos_doc_id = random.choice(self.qrels[qid])
        pos_doc_text = self._get_doc_text(pos_doc_id)
        if not pos_doc_text:
            pos_doc_text = qtext  # Fallback
            
        # Hard negative mining via BM25
        hits = self.searcher.search(qtext, k=25)
        neg_candidates = [hit.docid for hit in hits if hit.docid not in self.qrels[qid]]
        
        if neg_candidates:
            neg_doc_id = random.choice(neg_candidates)
            neg_doc_text = self._get_doc_text(neg_doc_id)
        else:
            neg_doc_text = ""
            
        if not neg_doc_text:
            neg_doc_text = "This is an unrelated statement."
            
        return qtext, pos_doc_text, neg_doc_text


def collate_fn(batch, tokenizer, max_len_q=64, max_len_d=128):
    q_texts = [item[0] for item in batch]
    pos_texts = [item[1] for item in batch]
    neg_texts = [item[2] for item in batch]
    
    q_enc = tokenizer(q_texts, padding=True, truncation=True, max_length=max_len_q, return_tensors="pt")
    pos_enc = tokenizer(pos_texts, padding=True, truncation=True, max_length=max_len_d, return_tensors="pt")
    neg_enc = tokenizer(neg_texts, padding=True, truncation=True, max_length=max_len_d, return_tensors="pt")
    
    return q_enc, pos_enc, neg_enc


def train_splade_fever():
    parser = argparse.ArgumentParser(description="Part 6: Train SPLADE on FEVER Dataset")
    parser.add_argument("--model_name", type=str, default="naver/splade-cocondenser-ensembledistil", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32 for 20-30GB GPU)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_train_samples", type=int, default=10000, help="Max queries to sample for training (default: 10000)")
    parser.add_argument("--max_len_q", type=int, default=64, help="Max token length for query (default: 64)")
    parser.add_argument("--max_len_d", type=int, default=128, help="Max token length for document (default: 128)")
    parser.add_argument("--lambda_q", type=float, default=0.01, help="Target lambda for query FLOPS (default: 0.01)")
    parser.add_argument("--lambda_d", type=float, default=0.008, help="Target lambda for doc FLOPS (default: 0.008)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Fraction of steps for quadratic warmup (default: 0.1)")
    parser.add_argument("--index_dir", type=str, default="indexes/fever", help="Path to FEVER Lucene index")
    parser.add_argument("--output_dir", type=str, default="splade_finetuned_fever", help="Output directory for checkpoint")
    args = parser.parse_args()

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base model
    model_name = args.model_name
    print(f"Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    # Dataset & Dataloader
    dataset = FeverTrainDataset(dataset_name="beir/fever/train", index_dir=args.index_dir, max_samples=args.max_train_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_len_q=args.max_len_q, max_len_d=args.max_len_d),
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = SpladeLoss(lambda_q=0.0, lambda_d=0.0)
    
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"\nStarting training: {args.epochs} epochs ({total_steps} total steps, {len(dataloader)} batches/epoch).")
    print(f"Batch size: {args.batch_size} (in-batch negatives = {2 * args.batch_size - 1})")
    print(f"Quadratic warmup over {warmup_steps} steps.")

    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_reg_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for q_enc, pos_enc, neg_enc in pbar:
                # Quadratic warmup scheduler for FLOPS regularization
                if global_step < warmup_steps:
                    ratio = (global_step / warmup_steps) ** 2
                else:
                    ratio = 1.0
                    
                loss_fn.lambda_q = args.lambda_q * ratio
                loss_fn.lambda_d = args.lambda_d * ratio
                
                q_enc = {k: v.to(device, non_blocking=True) for k, v in q_enc.items()}
                pos_enc = {k: v.to(device, non_blocking=True) for k, v in pos_enc.items()}
                neg_enc = {k: v.to(device, non_blocking=True) for k, v in neg_enc.items()}
                
                optimizer.zero_grad()
                
                # Forward passes
                q_logits = model(**q_enc).logits
                pos_logits = model(**pos_enc).logits
                neg_logits = model(**neg_enc).logits
                
                # Compute SPLADE vectors
                q_rep = compute_splade_vector(q_logits, q_enc["attention_mask"])
                pos_rep = compute_splade_vector(pos_logits, pos_enc["attention_mask"])
                neg_rep = compute_splade_vector(neg_logits, neg_enc["attention_mask"])
                
                # Contrastive Ranking Loss + FLOPS Regularization
                loss_rank, loss_reg, l_rank_val, l_reg_val = loss_fn(q_rep, pos_rep, neg_rep)
                loss = loss_rank + loss_reg
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_rank_loss += l_rank_val
                epoch_reg_loss += l_reg_val
                global_step += 1
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "rank": f"{l_rank_val:.4f}",
                    "reg": f"{l_reg_val:.4f}",
                    "l_q": f"{loss_fn.lambda_q:.4f}"
                })

        avg_loss = epoch_loss / len(dataloader)
        avg_rank = epoch_rank_loss / len(dataloader)
        avg_reg = epoch_reg_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} (Rank: {avg_rank:.4f}, Reg: {avg_reg:.4f})")

        # Save checkpoint after every epoch (overwrites output_dir so latest completed epoch is always usable)
        print(f"--> Saving checkpoint for Epoch {epoch+1} to {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Final save
    print(f"\nFinal training save to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    train_splade_fever()
