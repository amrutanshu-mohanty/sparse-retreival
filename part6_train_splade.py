import argparse
import ir_datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, AdamW
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import random
import os
import json
from pathlib import Path

# Fix random seed
torch.manual_seed(42)
random.seed(42)

def compute_splade_vector(logits, attention_mask):
    """
    Computes the SPLADE sparse vector from MLM logits.
    w_j = max_i log(1 + ReLU(o_ij))
    """
    # logits: (batch_size, seq_len, vocab_size)
    # attention_mask: (batch_size, seq_len)
    
    # Apply ReLU and log1p
    relu_logits = torch.relu(logits)
    log_logits = torch.log1p(relu_logits)
    
    # Mask out padding tokens
    mask_expanded = attention_mask.unsqueeze(-1).expand(log_logits.size())
    log_logits = log_logits * mask_expanded
    
    # Max pooling over sequence length
    sparse_vector, _ = torch.max(log_logits, dim=1)
    return sparse_vector

class SpladeLoss(nn.Module):
    def __init__(self, lambda_q=0.01, lambda_d=0.01):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d

    def forward(self, q_rep, d_pos_rep, d_neg_rep):
        batch_size = q_rep.size(0)
        
        # InfoNCE Loss (with in-batch negatives)
        # Score query against all documents in the batch (positives and negatives)
        all_d = torch.cat([d_pos_rep, d_neg_rep], dim=0)  # (2*batch_size, vocab_size)
        
        # Dot product scores: (batch_size, 2*batch_size)
        scores = torch.matmul(q_rep, all_d.transpose(0, 1))
        
        # Labels: the positive document for query i is at index i
        labels = torch.arange(batch_size, device=q_rep.device)
        
        loss_rank = nn.functional.cross_entropy(scores, labels)
        
        # FLOPS Regularization
        # FLOPS = sum( mean_batch(w_j)^2 )
        q_flops = torch.sum(torch.mean(q_rep, dim=0) ** 2)
        d_flops = torch.sum(torch.mean(all_d, dim=0) ** 2)
        
        loss_reg = (self.lambda_q * q_flops) + (self.lambda_d * d_flops)
        
        return loss_rank, loss_reg, loss_rank.item(), loss_reg.item()

class SciFactTrainDataset(Dataset):
    def __init__(self, dataset_name="beir/scifact/train", index_dir="indexes/scifact"):
        print(f"Loading {dataset_name} for training...")
        dataset = ir_datasets.load(dataset_name)
        
        self.corpus = {}
        for doc in dataset.docs_iter():
            self.corpus[doc.doc_id] = f"{doc.title} {doc.text}"
            
        self.queries = {q.query_id: q.text for q in dataset.queries_iter()}
        
        self.qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                if qrel.query_id not in self.qrels:
                    self.qrels[qrel.query_id] = []
                self.qrels[qrel.query_id].append(qrel.doc_id)
                
        self.qids = list(self.qrels.keys())
        
        print(f"Loaded {len(self.qids)} training queries with positive qrels.")
        
        # Initialize BM25 searcher to mine hard negatives
        print(f"Initializing BM25 Searcher for hard negative mining from {index_dir}...")
        self.bm25_searcher = LuceneSearcher(index_dir)
        self.bm25_searcher.set_bm25(0.9, 0.4) # SciFact tuned parameters

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        qtext = self.queries[qid]
        
        # Sample one positive document
        pos_doc_id = random.choice(self.qrels[qid])
        pos_doc_text = self.corpus[pos_doc_id]
        
        # Mine hard negative using BM25
        hits = self.bm25_searcher.search(qtext, k=20)
        neg_candidates = [hit.docid for hit in hits if hit.docid not in self.qrels[qid]]
        
        if neg_candidates:
            neg_doc_id = random.choice(neg_candidates)
        else:
            # Fallback to random corpus doc
            neg_doc_id = random.choice(list(self.corpus.keys()))
            
        neg_doc_text = self.corpus[neg_doc_id]
        
        return qtext, pos_doc_text, neg_doc_text

def collate_fn(batch, tokenizer):
    q_texts = [item[0] for item in batch]
    pos_texts = [item[1] for item in batch]
    neg_texts = [item[2] for item in batch]
    
    q_enc = tokenizer(q_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    pos_enc = tokenizer(pos_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    neg_enc = tokenizer(neg_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    return q_enc, pos_enc, neg_enc

def train_splade():
    parser = argparse.add_argument_group("Training Arguments")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lambda_q", type=float, default=0.01, help="Target lambda for query FLOPS")
    parser.add_argument("--lambda_d", type=float, default=0.01, help="Target lambda for doc FLOPS")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Fraction of steps for quadratic warmup")
    parser.add_argument("--output_dir", type=str, default="splade_finetuned_scifact", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base model
    model_name = "distilbert-base-uncased"
    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    # Load data
    dataset = SciFactTrainDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = SpladeLoss(lambda_q=0.0, lambda_d=0.0) # Will be updated by scheduler
    
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"Starting training for {args.epochs} epochs ({total_steps} steps).")
    print(f"Quadratic warmup over {warmup_steps} steps.")

    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_rank_loss = 0
        epoch_reg_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for q_enc, pos_enc, neg_enc in pbar:
                # Quadratic warmup scheduler for FLOPS regularization
                if global_step < warmup_steps:
                    ratio = (global_step / warmup_steps) ** 2
                else:
                    ratio = 1.0
                    
                loss_fn.lambda_q = args.lambda_q * ratio
                loss_fn.lambda_d = args.lambda_d * ratio
                
                q_enc = {k: v.to(device) for k, v in q_enc.items()}
                pos_enc = {k: v.to(device) for k, v in pos_enc.items()}
                neg_enc = {k: v.to(device) for k, v in neg_enc.items()}
                
                optimizer.zero_grad()
                
                # Forward passes
                q_logits = model(**q_enc).logits
                pos_logits = model(**pos_enc).logits
                neg_logits = model(**neg_enc).logits
                
                # Compute SPLADE vectors
                q_rep = compute_splade_vector(q_logits, q_enc["attention_mask"])
                pos_rep = compute_splade_vector(pos_logits, pos_enc["attention_mask"])
                neg_rep = compute_splade_vector(neg_logits, neg_enc["attention_mask"])
                
                # Loss computation
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

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f} "
              f"(Rank: {epoch_rank_loss/len(dataloader):.4f}, Reg: {epoch_reg_loss/len(dataloader):.4f})")

    # Save final model
    print(f"Saving fine-tuned model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom SPLADE model on SciFact")
    train_splade()
