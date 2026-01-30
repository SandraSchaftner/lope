# LOPE: LLM-driven Ontology-based Property Extraction & Evaluation Framework

LOPE is a comprehensive framework designed to automate the extraction of semantic properties from scientific publications using the Open Research Knowledge Graph (ORKG) ontology.

This repository implements a complete pipeline that:

1. **Generates** vector embeddings for thousands of existing ORKG properties using **Qwen3-Embedding-8B**.
2. **Extracts** new properties using **GPT-3.5** and the open-weight models **GPT-OSS-120B** via 8 different strategies.
3. **Evaluates** the extraction quality using an LLM-as-Judge approach (**DeepSeek-V3**).
4. **Matches** extracted terms to the ORKG ontology using Semantic Search (ChromaDB).

## Features

* **Multi-Strategy Extraction**: Implements 8 distinct strategies, including prompting techniques and Context-Augmented generation.
* **Semantic Matching**: Uses `Qwen/Qwen3-Embedding-8B` and **ChromaDB** to map extracted text to existing ORKG predicates (URI resolution).
* **Automated Evaluation**: Calculates Alignment, Deviation, and Mapping scores using DeepSeek-V3 as a judge.
* **Statistical Analysis**: Automatically generates Sign Tests, Chi-Square tests, and SVG and PNG visualizations (Heatmaps, Bar Charts).
* **Reproducibility**: Includes the original `publications.json` dataset and a robust scraper for verification.

## Prerequisites

* **Python 3.10+**
* **GPU** (Recommended for local embedding generation), though CPU is supported.
* **Google Chrome** (Only if you intend to run the optional scraper).

### API Keys

Create a `.env` file in the root directory to store your credentials:

```env
# OpenAI (Used for Extraction strategies & Guardrailing)
GPT_KEY=sk-...
ORGANIZATION=org-...

# Open Source Model Endpoint (Used for Extraction)
# Please adapt the 'base_url' in 'lope.py' (query_oss_agent) if using local vLLM/Ollama.
OSS_API_KEY=...

# DeepSeek (Used as the Evaluator / Judge)
DEEPSEEK_KEY=sk-...
```

## Installation

1. Clone the repository:

   ```bash
   git clone [https://github.com/SandraSchaftner/lope.git](https://github.com/SandraSchaftner/lope.git)
   cd lope
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Your `requirements.txt` should include: `pandas`, `numpy`, `scipy`, `python-dotenv`, `pyyaml`, `openai`, `torch`, `sentence-transformers`, `chromadb`, `matplotlib`. If you plan to use the scraper, also install `selenium`, `undetected-chromedriver`, `beautifulsoup4`, `requests`, and `pyautogui`.*

## Usage

### 1. Prepare Vector Database (Required for Matching)

Before running the main pipeline, you must generate the local vector store for ORKG properties. This uses the `Qwen/Qwen3-Embedding-8B` model.

```bash
# Step 1.1: Generate Parquet embeddings from raw data
python orkg_embeddings_Qwen3-Embedding-8B.py

# Step 1.2: Build the ChromaDB vector store
python orkg_chromadb_Qwen3-Embedding-8B.py
```

*Output:* A local folder `orkg_data/chroma_db_store_qwen` containing the vector index.

### 2. Run Main Pipeline (Extraction & Evaluation)

The core logic is contained in `lope.py`. This script performs the extraction strategies, sends results to the judge (DeepSeek), matches terms against the vector DB, and generates reports.

The `publications.json` file included in this repository contains the input data (Titles, Abstracts, Research Problems).

```bash
python lope.py
```

**Workflow:**

1. **Input:** Prompts for an output folder name (e.g., `run_experiments`) and LLM version selection.
2. **Extraction:** Runs all strategies (Baselines, Context-Augmented, Open Source models).
3. **Evaluation:** DeepSeek scores the extracted properties against the Gold Standard.
4. **Matching:** Maps terms to ORKG URIs using the ChromaDB built in Step 1.
5. **Reporting:** Saves final JSONs and generates SVG charts.

### 3. Optional: Data Scraping

The repository includes `publications.json`, so scraping is **not necessary** to run the pipeline.

However, if you wish to reproduce the data acquisition step or fetch new abstracts, you can use the included scraper. Note that this script mimics human behavior (mouse movements) to bypass bot detection and requires an active desktop session.

```bash
python get_abstract_title_doi.py
```

## Results & Output

All results are saved in the user-defined output folder created by `lope.py`.

* **`publications_final.json`**: The complete dataset containing extracted properties, judge scores, and ORKG matches.
* **`evaluation_report.txt`**: Detailed statistical significance tables (Sign Test results).
* **`matching_report.txt`**: Matching rates and Chi-Square test results.
* **Visualizations**:
  * `heatmap_*.svg`: Significance heatmaps comparing model performance.
  * `chart_*.svg`: Bar charts for Alignment, Deviation, and Matching rates.

## Project Structure

* `lope.py`: Main execution script (Extraction, Eval, Matching).
* `orkg_embeddings_Qwen3-Embedding-8B.py`: Generates embeddings from ORKG dump.
* `orkg_chromadb_Qwen3-Embedding-8B.py`: Loads embeddings into ChromaDB.
* `get_abstract_title_doi.py`: Selenium-based scraper (Optional).
* `publications.json`: Input dataset.
* `dimensions_system_prompts.yaml`: Prompt templates for extraction.
* `dimensions_eval_system_prompts.yaml`: Prompt templates for the judge.

## Credits

This script is based on research by Nechakhin et al., 2024:

Research paper: V. Nechakhin, J. D’Souza, and S. Eger, „Evaluating Large Language Models for Structured Science Summarization in the Open Research Knowledge Graph,“ Information, vol. 15, no. 6. MDPI AG, p. 328, Jun. 05, 2024. doi: 10.3390/info15060328. (19.01.2025)

Gold standard data set: Vladyslav Nechakhin, Jennifer D’Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n. (19.01.2025)

## License

CC BY-SA 4.0

## Acknowledgment

The authors gratefully acknowledge Nechakhin, D'Souza, and Eger for publishing the dataset, prompts, and experimental setup. This made it possible to replicate, understand, and build upon their study.

This work is supported by the European Union’s HORIZON Research and Innovation Programme under grant agreement No 101120657, project ENFIELD (European Lighthouse to Manifest Trustworthy and Green AI).