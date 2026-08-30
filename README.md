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

## File Lifecycle & Artifacts

When you run the scripts, it generates several folders and files:

1. **data/**: Contains formatted `corpus.jsonl` files for Pyserini. These can be safely deleted after the indexes are built to free up space.
2. **indexes/**: Contains the compiled Lucene indexes. **DO NOT DELETE THESE**, as they are required for evaluations and subsequent steps (like querying and Pseudo-Relevance Feedback in Part 4).
3. **~/.ir_datasets/**: Cache folder where raw zip source files are downloaded.
4. **report.txt**: Summary of final indexing metrics (size, build time, document count).
5. **part2_results.txt**: Contains retrieval metrics for all baseline runs.
6. **part3_analysis_report.txt**: Report detailing the Jaccard-overlap statistics, failed query ID lists, and categorized failure examples.
7. **part3_plots/**: Contains generated PNG distribution charts comparing lexical overlap for successful vs. failed queries.

*Note: The data/, indexes/, and temporary logs (like *.log and output.txt) are listed in .gitignore so they are not pushed to GitHub.*
