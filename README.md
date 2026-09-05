# BEIR Pyserini Indexing

This repository contains the setup and scripts required to build Pyserini (Lucene) inverted indexes for the BEIR benchmark datasets (SciFact, FEVER, HotpotQA) as part of the Information Retrieval assignment.

## Prerequisites

- **Python 3.9+**
- **Java 11 or 21**: Pyserini requires Java to build and read Lucene indexes. The script handles downloading a portable Java 21 JDK locally if JAVA_HOME is not set, so you don't need to install it globally.
- **Disk Space**: You will need approximately **15-20 GB** of free disk space. FEVER and HotpotQA contain ~5 million documents each. The raw data, intermediate JSONL formatting, and the final Lucene indexes will take up significant space.

## Installation Setup

It is highly recommended to use a Python virtual environment as the dependencies (like Transformers and PyTorch via Pyserini) are very large.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate

# 3. Install the requirements
pip install -r requirements.txt
```
## Running the Indexer

You can run the indexing script for one or multiple datasets:

```bash
# Run for SciFact (Small dataset, good for testing - takes ~15 seconds)
python build_indexes.py --datasets scifact

# Run for FEVER and HotpotQA (Large datasets - may take 5-10 mins each)
python build_indexes.py --datasets fever hotpotqa
```

## Running Baseline Evaluations (Part 2)

Evaluate Default BM25, Tuned BM25, and Classic TF-IDF baselines on the datasets:

```bash
# Evaluate SciFact baseline
python evaluate_baselines.py --datasets scifact

# Evaluate FEVER and HotpotQA baselines
python evaluate_baselines.py --datasets fever hotpotqa
```

The resulting metrics table (nDCG@10, Recall@100, MRR@10, MAP) is saved in `part2_results.txt`.

## Running Vocabulary Mismatch Analysis (Part 3)

Perform the vocabulary mismatch and Jaccard-overlap distribution analysis for the datasets:

```bash
# Run vocabulary mismatch and Jaccard-overlap calculations
python part3_analysis.py
```

This script will:
1. Retrieve top-50 results using the tuned BM25 parameters.
2. Label success vs. failure groups at $k \in \{10, 20, 50\}$.
3. Compute token Jaccard overlaps with and without stopwords.
4. Compare statistics (mean, median, std) between the full test set and a 500-query random sample.
5. Save Jaccard distribution plots (density histograms) in the `part3_plots/` directory.
6. Print and write lists of failed query IDs and concrete failure examples to `part3_analysis_report.txt`.

## Running Rocchio PRF, Query Expansion & Drift Analysis (Part 4)
## Running Rocchio PRF, Query Expansion & Drift Analysis (Part 4a)

Execute the Rocchio Pseudo-Relevance Feedback (PRF), parameter sensitivity, and query drift evaluation across the datasets:
Execute the Rocchio Pseudo-Relevance Feedback (PRF), parameter sensitivity, and query drift evaluation across the datasets. The results for each dataset can be saved separately under the `part4a_results/` folder:

```bash
# Run Part 4 on SciFact with dev grid search
python part4_rocchio.py --datasets scifact
# Ensure output directory exists
mkdir part4a_results

# Run Part 4 on FEVER and HotpotQA bypassing dev grid search (using preset tuned params)
python part4_rocchio.py --datasets fever hotpotqa --bypass-grid-search
# 1. Run Part 4a on SciFact (with Dev Grid Search Tuning)
python part4_rocchio.py --datasets scifact --output part4a_results/scifact_results.txt

# 2. Run Part 4a on FEVER (bypassing Dev Grid Search using tuned params)
python part4_rocchio.py --datasets fever --bypass-grid-search --output part4a_results/fever_results.txt

# 3. Run Part 4a on HotpotQA (bypassing Dev Grid Search using tuned params)
python part4_rocchio.py --datasets hotpotqa --bypass-grid-search --output part4a_results/hotpotqa_results.txt
```

Alternatively, to run all datasets together in a single run:
```bash
python part4_rocchio.py --datasets scifact fever hotpotqa --bypass-grid-search --output part4a_results/part4_results.txt
```

This script will:
1. Perform automated **hyperparameter grid search on the dev split** to find optimal $(\alpha, \beta, \gamma, N, k)$ parameters.
2. Evaluate 6 comparative retrieval regimes on the test split:
1. Perform automated **hyperparameter grid search on the dev split** (or load preset tuned parameters when `--bypass-grid-search` is passed).
2. Evaluate 7 comparative retrieval configurations on the test split:
   - Tuned BM25 Baseline (from Part 2)
   - Pyserini Native RM3 Relevance Model
   - Conservative Rocchio ($N=3, k=5$)
   - Standard Rocchio ($N=5, k=10$)
   - Aggressive Rocchio ($N=10, k=20$)
   - Rocchio with Negative Feedback ($\gamma=0.15$, ranks 91–100)
   - Dev Grid-Tuned Rocchio Model
   - Pyserini Native RM3 Relevance Model ($N=5, k=10, w=0.5$)
   - Conservative Rocchio ($N=3, k=5, \beta=0.75, \gamma=0.0$)
   - Standard Rocchio ($N=5, k=10, \beta=0.75, \gamma=0.0$)
   - Aggressive Rocchio ($N=10, k=20, \beta=0.75, \gamma=0.0$)
   - Rocchio with Negative Feedback ($N=5, k=10, \beta=0.75, \gamma=0.15$, ranks 91–100)
   - Dev Grid-Tuned Rocchio Model ($N^\ast, k^\ast, \beta^\ast, \gamma^\ast$)
3. Measure **quantitative query drift** metrics ($P_{\text{win}}, P_{\text{tie}}, P_{\text{loss}}, P_{\text{severe}}$).
4. Evaluate recovery of vocabulary mismatch failure cases identified in Part 3.
5. Log complete categorized query ID lists and top-10 qualitative case studies to `part4_results.txt`.
5. Log complete categorized query ID lists and top-10 qualitative case studies to the output report.

## Running HyDE (LLM-Generated Feedback) Analysis (Part 4b)

Part 4b replaces corpus-retrieved feedback with LLM-generated hypothetical documents (HyDE) while reusing the Part 4a Rocchio implementation unchanged. 

### 1. Ollama Setup (Local LLM)
This step uses a local LLM to avoid paid APIs. You must have Ollama installed and the required model downloaded before running the script.
1. Download and install [Ollama](https://ollama.com/).
2. Open your terminal and pull the required Qwen2.5 7B model:
   ```bash
   ollama pull qwen2.5:7b
   ```
      ``` bash
      # Run the full evaluation on SciFact (will take significant time to generate cache)
      python part4b_hyde.py --datasets scifact

      # Run on multiple datasets explicitly defining the model and number of docs (N=5)
       python part4b_hyde.py --datasets fever hotpotqa --ollama-model qwen2.5:7b --hyde-n 5
       ```

## Running SPLADE Sparse Retrieval (Part 5)

Part 5 uses a pretrained SPLADE checkpoint to encode queries and retrieve from prebuilt SPLADE impact indexes. It also compares expansion terms against Rocchio (Part 4a) and HyDE (Part 4b).

### Prerequisites
- `transformers` package (for loading the SPLADE model from HuggingFace)
- GPU recommended for query encoding (CPU works but is slower)
- Prebuilt Pyserini SPLADE indexes are auto-downloaded on first run

```bash
# Install additional dependency
pip install transformers>=4.30.0

# Run on SciFact (fastest, ~5 min)
python part5_splade.py --datasets scifact

# Run on all datasets
python part5_splade.py --datasets scifact fever hotpotqa

# Use a specific SPLADE model (default: naver/splade-cocondenser-ensembledistil)
python part5_splade.py --datasets scifact --model naver/splade-v3

# Control number of queries for expansion-term comparison (default: 15)
python part5_splade.py --datasets scifact --num-comparison 20
```

Results are saved in `part5_results/{dataset}_results.txt`.

## Extra Credit 1: Document-Side Expansion (doc2query)

Expands documents at **indexing time** by prompting an LLM to generate predicted search queries / questions per document, appending them to the document text, and rebuilding the Lucene index.

```bash
# 1. Run on SciFact using local Ollama (automatic CPU / GPU)
python extra_credit_1_doc2query.py --datasets scifact

# 2. Run with HuggingFace PyTorch backend on GPU
python extra_credit_1_doc2query.py --datasets scifact --backend hf --device cuda

# 3. Run on FEVER / HotpotQA with optional document limit
python extra_credit_1_doc2query.py --datasets fever --limit-docs 5000
```

This script will:
1. Generate pseudo-queries per document (with disk caching at `doc2query_cache/{dataset}_doc2query.json`).
2. Format `data/{dataset}_doc2query/corpus.jsonl` and compile an expanded Lucene index at `indexes/{dataset}_doc2query`.
3. Compare BM25 on the original index vs. doc2query index.
4. Output resource trade-off analysis (index build time, size overhead, sub-millisecond query latency) and Part 3 failure recovery statistics in `extra_credit_1_results.txt`.

## File Lifecycle & Artifacts

When you run the scripts, it generates several folders and files:

1. **data/**: Contains formatted `corpus.jsonl` files for Pyserini (including `data/{dataset}_doc2query/`). These can be safely deleted after the indexes are built to free up space.
2. **indexes/**: Contains the compiled Lucene indexes (e.g. `indexes/{dataset}` and `indexes/{dataset}_doc2query`). **DO NOT DELETE THESE**, as they are required for evaluations.
3. **~/.ir_datasets/**: Cache folder where raw zip source files are downloaded.
4. **report.txt**: Summary of final indexing metrics (size, build time, document count).
5. **part2_results.txt**: Contains retrieval metrics for all baseline runs.
6. **part3_analysis_report.txt**: Report detailing the Jaccard-overlap statistics, failed query ID lists, and categorized failure examples.
7. **part3_plots/**: Contains generated PNG distribution charts comparing lexical overlap for successful vs. failed queries.
8. **part4a_results/**: Directory containing individual evaluation reports for Part 4a:
   - `scifact_results.txt`: Full Part 4a metrics, query drift breakdown, and case studies for SciFact.
   - `fever_results.txt`: Full Part 4a metrics, query drift breakdown, and case studies for FEVER.
   - `hotpotqa_results.txt`: Full Part 4a metrics, query drift breakdown, and case studies for HotpotQA.
9. **part4b_results/**: Directory containing Part 4b HyDE evaluation reports.
10. **part5_results/**: Directory containing Part 5 SPLADE evaluation reports.
11. **doc2query_cache/**: Cached LLM-generated document pseudo-queries for Extra Credit 1.
12. **extra_credit_1_results.txt**: Full evaluation, trade-off, and failure recovery report for doc2query document-side expansion.

*Note: The data/, indexes/, and temporary logs (like *.log and output.txt) are listed in .gitignore so they are not pushed to GitHub.*



