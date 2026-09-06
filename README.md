# CS 6101: Programming Assignment 1
## Inverted Indexing, Sparse Retrieval, and Vocabulary Mismatch

This repository contains the complete implementation, evaluation pipelines, and analysis reports for **CS 6101 PA1**. The project explores lexical and neural sparse retrieval methods over the BEIR benchmark (**SciFact**, **FEVER**, **HotpotQA**, and **MSMARCO**), investigates the vocabulary mismatch problem, and implements both query-side and document-side expansion mechanisms (Rocchio, RM3, HyDE, Doc2Query, and custom fine-tuned SPLADE models).

---

## Table of Contents
1. [Prerequisites & System Setup](#prerequisites--system-setup)
2. [Quickstart (End-to-End Reproduction)](#quickstart-end-to-end-reproduction)
3. [Part 1: Build Lucene Inverted Indexes](#part-1-build-lucene-inverted-indexes)
4. [Part 2: Sparse Retrieval Baselines (BM25, TF-IDF, Grid Tuning)](#part-2-sparse-retrieval-baselines)
5. [Part 3: Vocabulary Mismatch & Jaccard Overlap Analysis](#part-3-vocabulary-mismatch--jaccard-overlap-analysis)
6. [Part 4a: Corpus-Retrieved Pseudo-Relevance Feedback (Rocchio & RM3)](#part-4a-corpus-retrieved-pseudo-relevance-feedback)
7. [Part 4b: LLM-Generated Feedback (HyDE)](#part-4b-llm-generated-feedback-hyde)
8. [Part 5: Sparse Retrieval with Pretrained SPLADE](#part-5-sparse-retrieval-with-pretrained-splade)
9. [Extra Credit 1: Document-Side Expansion (doc2query)](#extra-credit-1-document-side-expansion-doc2query)
10. [Extra Credit 2: Full-Scale MSMARCO Evaluation](#extra-credit-2-full-scale-msmarco-evaluation)
11. [Extra Credit 3: Custom SPLADE Training & Fine-Tuning](#extra-credit-3-custom-splade-training--fine-tuning)
12. [Verification Utilities](#verification-utilities)
13. [Repository Structure & Artifacts](#repository-structure--artifacts)

---

## Prerequisites & System Setup

- **Operating System**: Linux (Ubuntu/CentOS recommended) or macOS / Windows.
- **Python**: `3.9` to `3.12`.
- **Java**: **JDK 21 LTS** or **JDK 11+** (required for Pyserini/Lucene). 
  - *Note*: If `JAVA_HOME` is not set in your global environment, the codebase automatically detects and uses a bundled/local JDK 21 directory in the workspace (`jdk-21*`).
- **Disk Space**: ~25–30 GB of free disk space to store raw datasets, formatted JSONL collections, and Lucene inverted indexes across SciFact, FEVER, HotpotQA, and MSMARCO.
- **GPU (Optional but recommended)**: CUDA-enabled GPU with 16GB+ VRAM for local LLM generation (HyDE / Doc2Query) and SPLADE neural training.

### Environment Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate       # Linux/macOS
# or .\.venv\Scripts\Activate.ps1 # Windows PowerShell

# 3. Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set dummy OpenAI key (prevents pyserini eager import errors)
export OPENAI_API_KEY=dummy
```

---

## Quickstart (End-to-End Reproduction)

To execute the entire assignment pipeline sequentially on SciFact (testing all components in ~5–10 minutes):

```bash
# 1. Build Index
python part1_build_indexes.py --datasets scifact

# 2. Evaluate BM25 & TF-IDF baselines + Latency
python part2_evaluate_baselines.py --datasets scifact

# 3. Run Vocabulary Mismatch analysis & generate plots
python part3_analysis.py

# 4. Run Corpus PRF (Rocchio & RM3)
python part4_rocchio.py --datasets scifact --output part4a_results/scifact_results.txt

# 5. Run Pretrained SPLADE inference & term comparison
python part5_splade.py --datasets scifact

# 6. Train Custom SPLADE Model & Evaluate
python part6_train_splade.py --epochs 5 --batch_size 8 --output_dir splade_finetuned_scifact
python part6_eval_splade.py --model_dir splade_finetuned_scifact
```

---

## Part 1: Build Lucene Inverted Indexes

Builds Pyserini Lucene inverted indexes with `--storePositions`, `--storeDocvectors`, and `--storeRaw` enabled for downstream PRF and extraction.

```bash
# Build index for SciFact (~15 seconds)
python part1_build_indexes.py --datasets scifact

# Build indexes for FEVER and HotpotQA (~5-10 mins each)
python part1_build_indexes.py --datasets fever hotpotqa

# Build index for MSMARCO passage dataset (8.84M documents, ~6 mins)
python part1_build_indexes.py --datasets msmarco
```

Outputs index statistics (build time, document count, and on-disk size) to `report.txt` and `part1_report.txt`.

---

## Part 2: Sparse Retrieval Baselines

Evaluates Default BM25 ($k_1=0.9, b=0.4$), Classic TF-IDF, and Tuned BM25 ($k_1, b$ discovered via exhaustive grid search on held-out dev splits) across nDCG@10, Recall@100, MRR@10, MAP, and per-query latency (ms/query).

```bash
# Evaluate on SciFact
python part2_evaluate_baselines.py --datasets scifact

# Evaluate on FEVER and HotpotQA
python part2_evaluate_baselines.py --datasets fever hotpotqa

# Evaluate on MSMARCO (5,980 held-out dev queries)
python part2_evaluate_baselines.py --datasets msmarco
```

Results are saved to `part2_results.txt`.

---

## Part 3: Vocabulary Mismatch & Jaccard Overlap Analysis

Quantifies lexical overlap (Jaccard similarity coefficient) between queries and gold documents for successful ($k \le 10$) vs. failed ($k > 10$) queries.

```bash
# Run full analysis and generate distribution density plots
python part3_analysis.py
```

Outputs:
- Jaccard distribution histograms in `part3_plots/` (Full and sampled sets, with and without stopwords).
- Detailed statistical breakdown, failure modes (synonymy, entity granularity, multi-hop), and concrete query examples in `part3_analysis_report.txt`.

---

## Part 4a: Corpus-Retrieved Pseudo-Relevance Feedback

Implements Rocchio's expansion algorithm ($w_t = \alpha f(q)[t] + \frac{\beta}{N} \sum f(d)[t]$) and evaluates sensitivity across feedback documents ($N \in \{3, 5, 10\}$), expansion terms ($k \in \{5, 10, 20\}$), and Pyserini native RM3.

```bash
# Create output directory
mkdir -p part4a_results

# 1. SciFact (with dev grid search tuning)
python part4_rocchio.py --datasets scifact --output part4a_results/scifact_results.txt

# 2. FEVER (using tuned parameters)
python part4_rocchio.py --datasets fever --bypass-grid-search --output part4a_results/fever_results.txt

# 3. HotpotQA (using tuned parameters)
python part4_rocchio.py --datasets hotpotqa --bypass-grid-search --output part4a_results/hotpotqa_results.txt
```

---

## Part 4b: LLM-Generated Feedback (HyDE)

Uses a locally served open-weight LLM (`Qwen2.5-7B-Instruct` via Ollama) to generate zero-shot hypothetical documents per query, comparing Naive Concatenation vs. Rocchio/RM3 Term Weighting vs. Corpus PRF.

### Local Ollama Setup
```bash
# Install and start Ollama, then pull the model:
ollama pull qwen2.5:7b
```

### Execution
```bash
mkdir -p part4b_results hyde_cache

# Run HyDE evaluation (cached results load automatically)
python part4b_hyde.py --datasets scifact
python part4b_hyde.py --datasets fever --bypass-grid-search
python part4b_hyde.py --datasets hotpotqa --bypass-grid-search
```

Results are saved in `part4b_results/{dataset}_results.txt`.

---

## Part 5: Sparse Retrieval with Pretrained SPLADE

Uses the pretrained `naver/splade-cocondenser-ensembledistil` checkpoint and Pyserini prebuilt impact indexes. Generates the 3-way expansion term comparison table (SPLADE vs. Rocchio vs. HyDE).

```bash
mkdir -p part5_results

# Run on SciFact
python part5_splade.py --datasets scifact

# Run on FEVER and HotpotQA
python part5_splade.py --datasets fever hotpotqa

# Run on MSMARCO (using prebuilt msmarco-v1-passage.splade-pp-ed impact index)
python part5_splade.py --datasets msmarco
```

Outputs metrics tables, Jaccard overlap statistics, and qualitative disagreement case studies in `part5_results/{dataset}_results.txt`.

---

## Extra Credit 1: Document-Side Expansion (doc2query)

Generates predicted pseudo-queries per document at offline indexing time using `Qwen2.5-7B-Instruct`, rebuilds the Lucene index, and evaluates failure case recovery.

```bash
# Run doc2query pipeline on SciFact
python extra_credit_1_doc2query.py --datasets scifact
```

Outputs index trade-off metrics (index size overhead, query latency) and Part 3 failure recovery counts in `extra_credit_1_results.txt`.

---

## Extra Credit 2: Full-Scale MSMARCO Evaluation

Scales the complete pipeline to the full MSMARCO passage collection (8.84M passages, 5,980 evaluation queries).

```bash
# 1. Build Lucene index (372s, 4.02 GB)
python part1_build_indexes.py --datasets msmarco

# 2. Evaluate BM25 default, tuned, and classic TF-IDF + latency
python part2_evaluate_baselines.py --datasets msmarco

# 3. Evaluate SPLADE++ impact retrieval (0.4466 nDCG@10)
python part5_splade.py --datasets msmarco
```

---

## Extra Credit 3: Custom SPLADE Training & Fine-Tuning

Implements custom SPLADE neural training using an InfoNCE contrastive ranking loss with in-batch negatives and a quadratic warmup FLOPS regularizer ($\lambda_q=0.01, \lambda_d=0.008$).

### 1. SciFact Training & Evaluation (Local GPU / CPU)
```bash
# Train on SciFact (5 epochs)
python part6_train_splade.py \
    --epochs 5 \
    --batch_size 8 \
    --lr 2e-5 \
    --output_dir splade_finetuned_scifact

# Evaluate SciFact custom model
python part6_eval_splade.py --model_dir splade_finetuned_scifact
```

### 2. FEVER Large-Scale Fine-Tuning (HPC / Multi-Core GPU)
For multi-million document corpora, train using on-the-fly Lucene document fetching:

```bash
# Fine-tune warm-started SPLADE model on FEVER (1-5 epochs)
python part6_train_splade_fever.py \
    --model_name naver/splade-cocondenser-ensembledistil \
    --epochs 1 \
    --batch_size 32 \
    --lr 2e-5 \
    --max_train_samples 10000 \
    --output_dir splade_finetuned_fever

# Evaluate on FEVER test set (with candidate scoring & term comparison)
python part6_eval_splade_fever.py \
    --model_dir splade_finetuned_fever \
    --index_dir indexes/fever \
    --max_eval_queries 1000 \
    --output part6_fever_results.txt
```

### 3. Slurm HPC Batch Submission
To submit the FEVER training and evaluation job to an HPC cluster:
```bash
sbatch run_part6_fever.sh
```

---

## Verification Utilities

To inspect individual queries, lookup gold relevant documents from qrels, and verify exact term presence in the Lucene index:

```bash
# Lookup query claim and gold documents by Query ID
python verify_failures.py --dataset scifact --query_id 1
python verify_failures.py --dataset fever --query_id 163803

# Verify specific Document ID and term matching
python verify_failures.py --dataset scifact --doc_id 195689316 --terms Obesity decreases life quality
```

---

## Repository Structure & Artifacts

```
.
├── part1_build_indexes.py         # Part 1 & EC2: Lucene index builder
├── part2_evaluate_baselines.py     # Part 2 & EC2: BM25/TF-IDF baseline evaluator + latency
├── part3_analysis.py               # Part 3: Vocabulary mismatch & Jaccard overlap
├── part4_rocchio.py                # Part 4a: Rocchio & RM3 Corpus PRF implementation
├── part4b_hyde.py                  # Part 4b: LLM-generated feedback (HyDE)
├── part5_splade.py                 # Part 5 & EC2: SPLADE inference & term extractor
├── extra_credit_1_doc2query.py     # EC 1: Document-side expansion (doc2query)
├── part6_train_splade.py           # EC 3: SPLADE training on SciFact
├── part6_eval_splade.py            # EC 3: SPLADE evaluation on SciFact
├── part6_train_splade_fever.py     # EC 3: Memory-efficient SPLADE training on FEVER
├── part6_eval_splade_fever.py       # EC 3: SPLADE evaluation on FEVER
├── verify_failures.py              # Utility to inspect queries and gold doc contents
├── run_part6_fever.sh              # Slurm batch script for HPC cluster execution
├── report.tex                      # Complete assignment LaTeX report
├── chat_history.md                 # Full LLM conversation history for compliance
├── part1_report.txt                # Indexing build times and disk footprint
├── part2_results.txt               # Baseline retrieval metrics and latencies
├── part3_analysis_report.txt       # Failure modes and Jaccard statistics
├── part3_plots/                    # Generated overlap distribution histograms
├── part4a_results/                 # Corpus PRF results across datasets
├── part4b_results/                 # HyDE results across datasets
├── part5_results/                  # Pretrained SPLADE evaluation reports
└── part6_results.txt               # Custom trained SPLADE metrics on SciFact
```
