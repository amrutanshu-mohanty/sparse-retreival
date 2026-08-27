# BEIR Pyserini Indexing

This repository contains the setup and scripts required to build Pyserini (Lucene) inverted indexes for the BEIR benchmark datasets (SciFact, FEVER, HotpotQA) as part of the Information Retrieval assignment.

## Prerequisites

- **Python 3.9+**
- **Java 11 or 21**: Pyserini requires Java to build and read Lucene indexes. The script handles downloading a portable Java 21 JDK locally if JAVA_HOME is not set, so you don't need to install it globally.
- **Disk Space**: You will need approximately **15-20 GB** of free disk space. FEVER and HotpotQA contain ~5 million documents each. The raw data, intermediate JSONL formatting, and the final Lucene indexes will take up significant space.

## Installation Setup

It is highly recommended to use a Python virtual environment as the dependencies (like Transformers and PyTorch via Pyserini) are very large.

`ash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate

# 3. Install the requirements
pip install -r requirements.txt
``n
## Running the Indexer

You can run the indexing script for one or multiple datasets. 

`ash
# Run for SciFact (Small dataset, good for testing - takes ~15 seconds)
python build_indexes.py --datasets scifact

# Run for FEVER and HotpotQA (Large datasets - may take 15-30+ mins each)
python build_indexes.py --datasets fever hotpotqa
``n
## File Lifecycle & Artifacts

When you run the script, it generates several folders:

1. **data/**: Contains corpus.jsonl files (the raw documents formatted for Pyserini). You **can** delete these files after the indexes are built if you need to free up space (they take up a few GBs).
2. **indexes/**: Contains the compiled Lucene indexes. **DO NOT DELETE THESE**. You will need these index files for subsequent parts of the assignment (like querying and Pseudo-Relevance Feedback in Part 4a).
3. **~/.ir_datasets/**: A hidden cache folder in your user directory where the raw dataset zip files are initially downloaded.
4. **eport.txt**: Contains the final deliverables (Corpus size, query count, build time, and on-disk size).

*Note: The data/ and indexes/ directories are intentionally added to .gitignore so you don't accidentally push gigabytes of data to GitHub.*
