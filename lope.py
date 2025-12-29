"""
LOPE: LLM-driven Ontology-based Property Extraction & Evaluation Framework
================================================================

This script automates the extraction of semantic properties from scientific publications
for the Open Research Knowledge Graph (ORKG). It implements a complete pipeline:

1.  **Extraction**: Applies 8 different prompting strategies using GPT-3.5-turbo
    and the open weight LLM GPT-OSS-120B.
2.  **Evaluation**: Uses an LLM-as-a-Judge approach (DeepSeek-V3) to score extracted
    properties against a Gold Standard (Alignment, Deviation, Mapping).
3.  **Matching**: Semantically maps extracted properties to existing ORKG predicates
    using ChromaDB and the embedding model Qwen3-Embedding-8B.
4.  **Analysis**: Performs statistical significance tests (Sign Test, Chi-Square)
    and generates SVG and PNG visualizations.

Prerequisites:
- A `.env` file with API keys (GPT_KEY, OSS_API_KEY, DEEPSEEK_KEY).
- A `publications.json` input file.
- An initialized ChromaDB folder at `orkg_data/chroma_db_store_qwen`.
- YAML prompt files in the same directory.
"""

import math
import pandas as pd
import json
import sys
import os
import pickle
import yaml
import re
import ast
import torch
import chromadb
import numpy as np
from typing import List, Dict, Union, Optional
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import binomtest, chi2_contingency
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

#  CONFIGURATION & CONSTANTS 

# Path to the local vector database (ChromaDB) containing ORKG predicates.
# Ensure this folder exists and contains the indexed data before running.
CHROMA_DB_PATH = os.path.join("orkg_data", "chroma_db_store_qwen")

# Name of the collection within ChromaDB.
COLLECTION_NAME = "orkg_labels_qwen"

# HuggingFace model ID for embedding generation.
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Semantic distance threshold for accepting a match (lower = stricter).
MATCHING_THRESHOLD = 0.15

# Global counters for the self-correction mechanism (Guardrailing).
N_GUARDRAILING_SUCCESSFUL = 0
N_GUARDRAILING_UNSUCCESSFUL = 0
OUTPUT_DIR = ""  # Set at runtime based on user input.

class DualOutput:
    """
    Helper class to redirect stdout to both the console and a log file simultaneously.
    """
    def __init__(self, file):
        self.file = file
        self.console = sys.stdout

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

def get_output_path(filename: str) -> str:
    """
    Constructs the full file path by joining the global output directory with the filename.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The full absolute or relative path.
    """
    if not OUTPUT_DIR:
        return filename
    return os.path.join(OUTPUT_DIR, filename)

def get_p_text(p: float) -> str:
    """
    Formats a statistical p-value with standard scientific significance asterisks.

    Args:
        p (float): The p-value.

    Returns:
        str: Formatted string (e.g., "0.004**") or empty if not significant.
    """
    if p is None: return ""

    stars = ""
    if p < 0.001: stars = "***"
    elif p < 0.01: stars = "**"
    elif p < 0.05: stars = "*"
    else: return "" # Not significant

    if p < 0.001:
        val_str = f"{p:.1e}"
    else:
        val_str = f"{p:.3f}"

    return f"{val_str}{stars}"

def load_embedding_model():
    """
    Loads the SentenceTransformer model on the best available hardware (CUDA, MPS, or CPU).

    Returns:
        SentenceTransformer or None: The loaded model or None if loading failed.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Info: Loading Matching Model '{MODEL_NAME}' on {device}...")
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load embedding model: {e}")
        return None

def load_chroma_collection():
    """
    Connects to the persistent ChromaDB collection.

    Returns:
        Collection or None: The ChromaDB collection object or None on failure.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: ChromaDB path not found at {CHROMA_DB_PATH}")
        return None
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Info: Connected to ChromaDB collection '{COLLECTION_NAME}'.")
        return collection
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return None

def load_prompts_from_yaml(file_name: str) -> List[str]:
    """
    Loads a list of prompt templates from a YAML file located in the script's directory.

    Args:
        file_name (str): The filename (e.g., 'dimensions_system_prompts.yaml').

    Returns:
        List[str]: A list of prompt strings.
    """
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(file_path, 'r') as file:
            prompts = yaml.safe_load(file)
            return [p.strip() for p in prompts]
    except Exception as e:
        print(f"Error loading YAML {file_name}: {e}")
        return []

def query_gpt_agent(prompt: str, messages_history: List[Dict[str, str]], llm: int = 0) -> str:
    """
    Queries the OpenAI API (GPT-3.5 variants).

    Args:
        prompt (str): The user input prompt.
        messages_history (List[Dict]): The conversation history (system prompts, etc.).
        llm (int): Model selector (0 = gpt-3.5-turbo-0125, 1 = gpt-3.5-turbo-1106).

    Returns:
        str: The raw text response from the LLM.
    """
    api_key = os.getenv("GPT_KEY")
    organization = os.getenv("ORGANIZATION")
    if not api_key or not organization:
        print("Error: GPT credentials missing.")
        return ""

    client = OpenAI(organization=organization, api_key=api_key)
    model = "gpt-3.5-turbo-0125" if llm == 0 else "gpt-3.5-turbo-1106"

    messages_history.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_history,
            temperature=0,
            seed=4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""

def query_oss_agent(messages: List[Dict[str, str]], json_mode: bool = False) -> str:
    """
    Queries an Open Source Model hosted on an OpenAI-compatible endpoint.

    NOTE: The 'base_url' is currently set to a specific university server.
    Update this URL to your local vLLM/Ollama instance or API provider if needed.

    Args:
        messages (List[Dict]): The conversation history.
        json_mode (bool): Whether to enforce JSON object output format.

    Returns:
        str: The raw text response.
    """
    oss_key = os.getenv("OSS_API_KEY")
    if not oss_key:
        print("Error: OSS_API_KEY not found.")
        return ""

    # Adjust base_url if you are running models locally or elsewhere
    client = OpenAI(
        base_url="",
        api_key=oss_key
    )

    try:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OSS API: {e}")
        return ""

def query_deepseek(prompt: str, system_prompt: str) -> str:
    """
    Queries the DeepSeek API, primarily used here as the 'Judge' for evaluation.

    Args:
        prompt (str): The user content (usually extracted list vs gold standard).
        system_prompt (str): The evaluation instruction.

    Returns:
        str: The raw text response (usually a score).
    """
    api_key = os.getenv("DEEPSEEK_KEY")
    if not api_key:
        print("Error: DEEPSEEK_KEY not found.")
        return "0"

    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.0,
            max_tokens=5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying DeepSeek: {e}")
        return "0"

def postprocessing_response(response_text: str, llm_id: int, task_instruction: str) -> list:
    """
    Parses the LLM extraction output into a Python list. Includes a self-correction
    loop (Guardrailing) if the initial output is not valid Python syntax.

    Args:
        response_text (str): The raw output from the extraction LLM.
        llm_id (int): The ID of the LLM to use for repair (GPT-3.5).
        task_instruction (str): The original instruction context for the repair prompt.

    Returns:
        list: A list of extracted properties (strings). Returns empty list on failure.
    """
    global N_GUARDRAILING_SUCCESSFUL
    global N_GUARDRAILING_UNSUCCESSFUL

    # Attempt to find list syntax [...]
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if match:
        try:
            parsed_data = ast.literal_eval(match.group(0))
            if isinstance(parsed_data, list):
                return parsed_data
        except:
            pass

    # If parsing failed, trigger Guardrailing (Repair)
    print("   -> Guardrailing needed (Extraction)...")
    guardrailing_prompt = (
        f"I gave you the instruction: '{task_instruction}'\n"
        f"You replied: '{response_text}' which is not a valid Python list.\n"
        f"Please correct this and output ONLY the Python list."
    )
    # Use GPT-3.5 for repair
    repaired = query_gpt_agent(guardrailing_prompt, [{"role": "system", "content": "You output only valid Python lists."}], llm_id)

    match_repair = re.search(r'\[.*?\]', repaired, re.DOTALL)
    if match_repair:
        try:
            parsed = ast.literal_eval(match_repair.group(0))
            if isinstance(parsed, list):
                N_GUARDRAILING_SUCCESSFUL += 1
                return parsed
        except:
            pass

    N_GUARDRAILING_UNSUCCESSFUL += 1
    return []

def postprocessing_evaluation_deepseek(response_text: str, task_instruction: str) -> str:
    """
    Parses the DeepSeek evaluation output to ensure an integer score is returned.
    Does NOT count towards global extraction guardrailing stats.

    Args:
        response_text (str): Raw output from DeepSeek.
        task_instruction (str): Context for repair.

    Returns:
        str: The extracted integer score as a string (e.g., "5"), or "0" if failed.
    """
    match = re.search(r'\b\d+\b', response_text)
    if match:
        return match.group(0)

    print(f"   -> Guardrailing needed for DeepSeek: '{response_text}'")
    guard_prompt = (
        f"Instruction: '{task_instruction}'\n"
        f"Your invalid response: '{response_text}'\n"
        f"Output ONLY the integer score."
    )
    repaired = query_deepseek(guard_prompt, "You output only integers.")
    match_r = re.search(r'\b\d+\b', repaired)
    return match_r.group(0) if match_r else "0"

#  EXTRACTION STEPS 

def step_1_extract_gpt35_baselines(publication: dict, llm: int = 0) -> None:
    """
    Executes strategies 0-4 (Zero-Shot, Few-Shot, CoT variants) using the 'Research Problem' text.

    Args:
        publication (dict): The publication data object.
        llm (int): Model ID.
    """
    prompts = load_prompts_from_yaml("dimensions_system_prompts.yaml")
    research_problem = publication["research_problem"]
    dimensions = []

    # Iterate over strategies 0 to 4
    for i in range(5):
        msg_hist = [{"role": "system", "content": prompts[i]}]
        response = query_gpt_agent(research_problem, msg_hist, llm)
        extracted = postprocessing_response(response, llm, prompts[i])
        dimensions.append(extracted)
        print(f"   Strategy {i}: {len(extracted)} items extracted.")

    publication["dimensions"] = dimensions

def step_2_extract_gpt35_abstract(publication: dict, llm: int = 0) -> None:
    """
    Executes Strategy 5: Improved Prompt using 'Title + Abstract' via GPT-3.5.

    Args:
        publication (dict): The publication data object.
        llm (int): Model ID.
    """
    prompts = load_prompts_from_yaml("dimensions_system_prompts.yaml")
    # Index 5: Improved strategy + Abstract context
    prompt_instruction = prompts[5]

    title = publication.get("title", "")
    abstract = publication.get("abstract", "")
    res_prob = publication.get("research_problem", "")
    if not abstract or len(abstract) < 10: abstract = res_prob

    user_input = f'Title: "{title}", Abstract: "{abstract}"'
    msg_hist = [{"role": "system", "content": prompt_instruction}]

    print(f"Step 2 (GPT3.5+Abs): Processing '{title[:20]}...'")
    response = query_gpt_agent(user_input, msg_hist, llm)
    extracted = postprocessing_response(response, llm, prompt_instruction)

    publication["impr+abs"] = extracted
    print(f"   -> Extracted {len(extracted)} properties.")

def step_3_extract_120b_basic(publication: dict) -> None:
    """
    Executes Strategy 6: Basic extraction using the 120B Open Source Model on 'Research Problem'.
    Uses JSON-mode enforcement.

    Args:
        publication (dict): The publication data object.
    """
    prompts = load_prompts_from_yaml("dimensions_system_prompts.yaml")
    # Index 6: 120B Research Problem Only (JSON Prompt)
    prompt_instruction = prompts[6]

    user_input = publication.get("research_problem", "")

    messages = [
        {"role": "system", "content": prompt_instruction},
        {"role": "user", "content": user_input}
    ]

    print(f"Step 3 (120B Basic): Processing '{user_input[:20]}...'")
    content = query_oss_agent(messages, json_mode=True)

    try:
        data = json.loads(content)
        extracted = data.get("properties", [])
        # Fallback if key is missing, take first list found
        if not extracted:
             for v in data.values():
                 if isinstance(v, list): extracted = v; break

        if isinstance(extracted, list):
            publication["impr+120b"] = extracted
            print(f"   -> Extracted {len(extracted)} properties.")
        else: publication["impr+120b"] = []
    except:
        print("   -> JSON Error Step 3")
        publication["impr+120b"] = []

def step_4_extract_120b_abstract(publication: dict) -> None:
    """
    Executes Strategy 7: Extraction using the 120B Open Source Model on 'Title + Abstract'.

    Args:
        publication (dict): The publication data object.
    """
    prompts = load_prompts_from_yaml("dimensions_system_prompts.yaml")
    # Index 7: 120B Title + Abstract (JSON Prompt)
    json_prompt = prompts[7]

    title = publication.get("title", "")
    abstract = publication.get("abstract", "")
    res_prob = publication.get("research_problem", "")
    if not abstract or len(abstract) < 10: abstract = res_prob

    user_input = f'Title: "{title}", Abstract: "{abstract}"'

    messages = [
        {"role": "system", "content": json_prompt},
        {"role": "user", "content": user_input}
    ]

    print(f"Step 4 (120B+Abs): Processing '{title[:20]}...'")
    content = query_oss_agent(messages, json_mode=True)

    try:
        data = json.loads(content)
        extracted = data.get("properties", [])
        if not extracted:
             for v in data.values():
                 if isinstance(v, list): extracted = v; break

        if isinstance(extracted, list):
            publication["impr+120b+abs"] = extracted
            print(f"   -> Extracted {len(extracted)} properties.")
        else: publication["impr+120b+abs"] = []
    except:
        print("   -> JSON Error Step 4")
        publication["impr+120b+abs"] = []

#  EVALUATION 

def eval_dimensions_one(publication: dict) -> None:
    """
    Evaluates extracted dimensions against the Gold Standard using DeepSeek.
    Calculates 3 metrics: Alignment, Deviation, and Mappings.

    Args:
        publication (dict): The publication data object containing extracted lists.
    """
    eval_prompts = load_prompts_from_yaml("dimensions_eval_system_prompts.yaml")
    gold_standard = str(publication.get("orkg_properties", ""))

    candidates = []
    names = ["Zero", "Few", "CoT", "Comb", "Impr", "Impr+Abs", "120B", "120B+Abs"]

    # Gather results from strategies 0-4
    dims = publication.get("dimensions", [])
    for i in range(5):
        if i < len(dims): candidates.append(dims[i])
        else: candidates.append([])

    # Strategy 5: Improved + Abs
    candidates.append(publication.get("impr+abs", []))
    # Strategy 6: 120B Basic
    candidates.append(publication.get("impr+120b", []))
    # Strategy 7: 120B Abs
    candidates.append(publication.get("impr+120b+abs", []))

    all_scores = []

    for idx, cand_list in enumerate(candidates):
        scores = []
        if not cand_list:
            all_scores.append(["0", "0", "0"])
            continue

        cand_str = ", ".join(map(str, cand_list))
        user_msg = f"List 1 (Extracted): {cand_str}\nList 2 (Gold): {gold_standard}"

        # Loop through metrics: 0=Alignment, 1=Deviation, 2=Mappings
        for m_idx in range(3):
            raw = query_deepseek(user_msg, eval_prompts[m_idx])
            val = postprocessing_evaluation_deepseek(raw, eval_prompts[m_idx])
            scores.append(val)

        all_scores.append(scores)
        print(f"   {names[idx]:<12}: A={scores[0]} D={scores[1]} M={scores[2]}")

    publication["eval_deepseek"] = all_scores

def eval_dimensions_all(publications: list):
    """
    Iterates through all publications and triggers evaluation.
    """
    print("\n Starting DeepSeek Evaluation ")
    for i, pub in enumerate(publications):
        print(f"Evaluating {i+1}/{len(publications)}...")
        eval_dimensions_one(pub)

#  MATCHING 

def match_dimensions_to_orkg(publication: dict, model, collection) -> None:
    """
    Maps extracted text properties to valid ORKG predicates using vector similarity.

    Args:
        publication (dict): The publication data object.
        model (SentenceTransformer): The embedding model.
        collection (chromadb.Collection): The vector database collection.
    """
    if not model or not collection: return

    candidates = []
    # 0: Nechakhin (Previous baseline if exists)
    if "nechakhin_result" in publication:
        try:
            val = ast.literal_eval(publication["nechakhin_result"])
            candidates.append(val if isinstance(val, list) else [])
        except: candidates.append([])

    # 1-5: GPT3.5 Baselines
    dims = publication.get("dimensions", [])
    for i in range(5):
        if i < len(dims): candidates.append(dims[i])
        else: candidates.append([])

    # 6: GPT3.5 Abs
    candidates.append(publication.get("impr+abs", []))
    # 7: 120B Basic
    candidates.append(publication.get("impr+120b", []))
    # 8: 120B Abs
    candidates.append(publication.get("impr+120b+abs", []))

    all_matches = []
    all_rejected = []

    print(f"Matching for: {publication.get('research_problem', '')[:30]}...")

    for cand_list in candidates:
        matches = []
        rejected = []
        for item in cand_list:
            item_str = str(item).strip()
            if len(item_str) < 2: continue

            # Handle composite properties like "Accuracy/F1-Score"
            sub_dims = []
            if "/" in item_str: sub_dims = [d.strip() for d in item_str.split("/")]
            elif "(" in item_str: sub_dims = [item_str.split("(")[0].strip()]
            else: sub_dims = [item_str]

            for sub_dim in sub_dims:
                try:
                    emb = model.encode(sub_dim, normalize_embeddings=True).tolist()
                    res = collection.query(query_embeddings=[emb], n_results=1)

                    if res['ids'] and res['ids'][0]:
                        lbl = res['documents'][0][0]
                        rid = res['ids'][0][0]
                        dist = res['distances'][0][0]

                        obj = {"text": sub_dim, "id": rid, "label": lbl, "dist": float(dist), "uri": f"https://orkg.org/property/{rid}"}

                        if dist < MATCHING_THRESHOLD:
                            matches.append(obj)
                        else:
                            rejected.append(obj)
                except: pass

        all_matches.append(matches)
        all_rejected.append(rejected)

    publication["orkg_matches"] = all_matches
    publication["orkg_rejected"] = all_rejected

#  STATS & OUTPUT 

def calc_sign_test(publications: list, strat1: int, strat2: int) -> dict:
    """
    Calculates the Sign Test p-value between two strategies.
    Alternative 'greater' tests if strat1 > strat2 (Row > Col).

    Args:
        publications (list): List of evaluated publications.
        strat1 (int): Index of the first strategy.
        strat2 (int): Index of the second strategy.

    Returns:
        dict: p-values for 'alignment', 'deviation', 'mappings'.
    """
    def get_vals(s_idx, m_idx):
        vals = []
        for p in publications:
            try: vals.append(float(p["eval_deepseek"][s_idx][m_idx]))
            except: vals.append(0.0)
        return vals

    results = {}
    # Align & Map (Higher is better)
    for m_key, m_idx in [("alignment", 0), ("mappings", 2)]:
        s1 = get_vals(strat1, m_idx) # Row
        s2 = get_vals(strat2, m_idx) # Col
        diffs = [x - y for x, y in zip(s1, s2)]
        pos = sum(1 for d in diffs if d > 0)
        n = sum(1 for d in diffs if d != 0)
        if n > 0:
            results[m_key] = binomtest(pos, n, 0.5, alternative='greater').pvalue
        else: results[m_key] = None

    # Deviation (Lower is better) -> Test if Row < Col
    s1 = get_vals(strat1, 1)
    s2 = get_vals(strat2, 1)
    diffs = [x - y for x, y in zip(s1, s2)]
    neg = sum(1 for d in diffs if d < 0) # Row is smaller (better)
    n = sum(1 for d in diffs if d != 0)
    if n > 0:
        results["deviation"] = binomtest(neg, n, 0.5, alternative='greater').pvalue
    else: results["deviation"] = None

    return results

def save_rejected_json(publications: list):
    """Saves a JSON file containing only the rejected matches for manual review."""
    out_path = get_output_path("rejected_matches.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        simple_list = []
        for p in publications:
            simple_list.append({
                "id": p.get("paper_id"),
                "rejected": p.get("orkg_rejected", [])
            })
        json.dump(simple_list, f, indent=4)

def cell_color(value: Union[str, None]) -> str:
    """
    Determines heatmap cell color based on statistical significance.
    Grayscale logic: Darker = Lower p-value (More significant).
    """
    if value == "" or value is None: return "white"
    try:
        clean_val = str(value).replace('*', '')
        val = float(clean_val)
        if val <= 0.001: return "#252525"
        elif val <= 0.01: return "#636363"
        elif val <= 0.05: return "#bdbdbd"
        return "white"
    except: return "white"

def export_svg_bar_chart(label: str, x_min: float, x_max: float, strat_strings: list[str], chart_values: list[float], file_name: str) -> None:
    """Generates and saves a horizontal bar chart as SVG and PNG."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    y_pos = np.arange(len(chart_values))
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
    ax.barh(y_pos, chart_values, color="#405d85", height=0.65, zorder=3, edgecolor='black', linewidth=0.5)
    ax.grid(which='major', axis='x', linestyle='--', linewidth=0.5, color='#cccccc', zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strat_strings, fontsize=12)
    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    if x_max - x_min <= 1.0:
        ticks = np.arange(x_min, x_max + 0.001, 0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:.1f}" for x in ticks], fontsize=11)
    else:
        ticks = np.arange(math.floor(x_min), math.ceil(x_max) + 1, 1)
        if len(ticks) > 15: ticks = np.linspace(x_min, x_max, 11)
        ax.set_xticks(ticks)
        labels = [f"{int(x)}" if abs(x - round(x)) < 0.001 else f"{x:.1f}" for x in ticks]
        ax.set_xticklabels(labels, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    plt.savefig(file_name, format='svg', dpi=300)
    png_name = file_name.replace(".svg", ".png")
    if png_name == file_name: png_name += ".png"
    plt.savefig(png_name, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def export_svg_heatmap(strat_strings: list[str], heatmap_values: list[list[str]], file_name: str) -> None:
    """Generates and saves a significance heatmap as SVG and PNG."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    n = len(strat_strings)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(strat_strings, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(strat_strings, fontsize=11)
    for i in range(n):
        for j in range(n):
            value = heatmap_values[i][j]
            bg_color = cell_color(value)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=bg_color, edgecolor='#bfbfbf', linewidth=0.5)
            ax.add_patch(rect)
            if value != "":
                # Text rendering
                try:
                    # Check numerical value for text contrast (white text on dark background)
                    clean_val = str(value).replace('*', '')
                    num_val = float(clean_val)
                    text_color = "white" if num_val <= 0.05 else "black"

                    ax.text(j, i, value, ha="center", va="center", color=text_color, fontsize=10)
                except:
                    ax.text(j, i, value, ha="center", va="center", color="black", fontsize=10)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    plt.savefig(file_name, format="svg", bbox_inches="tight")
    png_name = file_name.replace(".svg", ".png")
    if png_name == file_name: png_name += ".png"
    plt.savefig(png_name, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def print_eval_details(publications: list):
    """
    Calculates average scores, significance matrices, and generates reports/charts.
    """
    file_path = get_output_path("evaluation_report.txt")
    strats = ["Zero", "Few", "CoT", "Comb", "Impr", "Impr+Abs", "120B", "120B+Abs"]
    n_strats = len(strats)

    # 1. Bar Charts
    means = {0: [], 1: [], 2: []}
    valid_count = 0
    sums = [[0,0,0] for _ in range(n_strats)]

    for pub in publications:
        if "eval_deepseek" in pub:
            valid_count += 1
            for i, score in enumerate(pub["eval_deepseek"]):
                if i < n_strats:
                    try:
                        sums[i][0] += float(score[0])
                        sums[i][1] += float(score[1])
                        sums[i][2] += float(score[2])
                    except: pass

    if valid_count > 0:
        for i in range(n_strats):
            means[0].append(sums[i][0]/valid_count)
            means[1].append(sums[i][1]/valid_count)
            means[2].append(sums[i][2]/valid_count)

    export_svg_bar_chart("Avg Alignment", 1, 5, strats, means[0], get_output_path("chart_align.svg"))
    export_svg_bar_chart("Avg Deviation", 1, 5, strats, means[1], get_output_path("chart_dev.svg"))
    export_svg_bar_chart("Avg Mappings", 0, max(means[2])+1, strats, means[2], get_output_path("chart_map.svg"))

    # 2. Significance Heatmaps (One-Sided with Stars)
    with open(file_path, 'w', encoding='utf-8') as f:
        dual = DualOutput(f)
        dual.write("\n Significance Legend \n")
        dual.write("* p < 0.05, ** p < 0.01, *** p < 0.001\n")

        metrics = ["alignment", "deviation", "mappings"]
        for m in metrics:
            dual.write(f"\nMetric: {m.upper()} (One-sided: Row > Col)\n")
            header = f"{'':<10} | " + " | ".join([f"{s[:6]:<8}" for s in strats])
            dual.write(header + "\n")

            heatmap_data = []
            for r in range(n_strats):
                row_str = f"{strats[r][:10]:<10} | "
                row_data = []
                for c in range(n_strats):
                    if r == c:
                        row_str += f"{'-':<8} | "
                        row_data.append("")
                        continue

                    res = calc_sign_test(publications, r, c)
                    p = res.get(m)

                    if p is not None and p < 0.05:
                        cell_txt = get_p_text(p)
                        row_str += f"{cell_txt:<8} | "
                        row_data.append(cell_txt)
                    else:
                        row_str += f"{'-':<8} | "
                        row_data.append("")

                dual.write(row_str + "\n")
                heatmap_data.append(row_data)

            export_svg_heatmap(strats, np.array(heatmap_data), get_output_path(f"heatmap_{m}.svg"))

def eval_matching_stats(publications: list):
    """
    Calculates matching rates and performs Chi-Square significance tests for the matching step.
    """
    file_path = get_output_path("matching_report.txt")
    # Add Nechakhin (legacy baseline) if present in data
    has_nechakhin = "nechakhin_result" in publications[0]
    strats = []
    if has_nechakhin: strats.append("Nechakhin")
    strats.extend(["Zero", "Few", "CoT", "Comb", "Impr", "Impr+Abs", "120B", "120B+Abs"])
    n_strats = len(strats)
    counts_match = [0] * n_strats
    counts_total = [0] * n_strats

    with open(file_path, 'w', encoding='utf-8') as f:
        dual = DualOutput(f)

        for pub in publications:
            matches_lists = pub.get("orkg_matches", [])
            r_lists = pub.get("orkg_rejected", [])
            for i in range(min(len(matches_lists), n_strats)):
                m_list = matches_lists[i]
                r_count = len(r_lists[i]) if i < len(r_lists) else 0
                total = len(m_list) + r_count
                counts_match[i] += len(m_list)
                counts_total[i] += total

        rates = []
        dual.write("\n Matching Rates \n")
        for i in range(n_strats):
            rate = counts_match[i] / counts_total[i] if counts_total[i] > 0 else 0
            rates.append(rate)
            dual.write(f"{strats[i]:<12}: {counts_match[i]}/{counts_total[i]} ({rate:.2%})\n")

        export_svg_bar_chart("Matching Rate", 0, 1, strats, rates, get_output_path("chart_match_rates.svg"))

        dual.write("\n Chi-Square Significance (Row > Col) \n")
        dual.write("* p < 0.05, ** p < 0.01, *** p < 0.001\n")

        header = f"{'':<12} | " + " | ".join([f"{s[:6]:<8}" for s in strats])
        dual.write(header + "\n")

        heatmap_data = []
        for r in range(n_strats):
            row_vals = []
            row_data = []
            for c in range(n_strats):
                if r == c:
                    row_vals.append("-")
                    row_data.append("")
                    continue

                rate_a = rates[r]; rate_b = rates[c]
                if rate_a <= rate_b: # Only test if row is better
                    row_vals.append("")
                    row_data.append("")
                    continue

                suc_a = counts_match[r]; fail_a = counts_total[r] - suc_a
                suc_b = counts_match[c]; fail_b = counts_total[c] - suc_b

                try:
                    obs = np.array([[suc_a, fail_a], [suc_b, fail_b]])
                    chi2, p, dof, ex = chi2_contingency(obs)
                    if p < 0.05:
                        txt = get_p_text(p)
                        cell_val = txt # String with stars
                    else:
                        txt = ""
                        cell_val = ""
                    row_vals.append(txt)
                    row_data.append(cell_val)
                except:
                    row_vals.append("Err")
                    row_data.append("")

            dual.write(f"{strats[r]:<12} | " + " | ".join([f"{v:<8}" for v in row_vals]) + "\n")
            heatmap_data.append(row_data)

        export_svg_heatmap(strats, np.array(heatmap_data), get_output_path("heatmap_match_sig.svg"))

def main():
    """
    Main orchestration function.
    1. Sets up output directories.
    2. Loads publications and models.
    3. Runs Extraction (GPT-3.5 & OSS 120B).
    4. Runs Evaluation (DeepSeek).
    5. Runs Matching (ChromaDB).
    6. Generates Reports.
    """
    global OUTPUT_DIR
    load_dotenv()

    # 1. Output Folder Setup
    folder_name = input("Enter output folder name (e.g. 'run_01'): ").strip()
    if not folder_name: folder_name = "output_default"
    OUTPUT_DIR = folder_name
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created folder: {OUTPUT_DIR}")

    # 2. LLM Selection
    llm = 10
    while llm == 10:
        llm_ab = input("Which LLM should be used?\nA: gpt-3.5-turbo-0125\nB: gpt-3.5-turbo-1106\n")
        if llm_ab.lower() == "a": llm = 0
        if llm_ab.lower() == "b": llm = 1

    input_file = "publications.json"
    if not os.path.exists(input_file):
        print("publications.json not found!")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        publications = json.load(f)

    #  PROCESSING 
    print("\n Step 1: GPT-3.5 Baselines ")
    for pub in publications:
        step_1_extract_gpt35_baselines(pub, llm)

    print("\n Step 2: GPT-3.5 (Title + Abstract) ")
    for pub in publications:
        step_2_extract_gpt35_abstract(pub, llm)

    print("\n Step 3: 120B (Research Problem Only) ")
    for pub in publications:
        step_3_extract_120b_basic(pub)

    print("\n Step 4: 120B (Title + Abstract) ")
    for pub in publications:
        step_4_extract_120b_abstract(pub)

    with open(get_output_path("publications_extracted.json"), 'w') as f:
        json.dump(publications, f, indent=4)

    #  EVALUATION 
    print("\n Starting DeepSeek Evaluation ")
    eval_dimensions_all(publications)
    print_eval_details(publications)

    #  MATCHING 
    print("\n Starting Matching  ")
    emb_model = load_embedding_model()
    chroma = load_chroma_collection()

    if emb_model and chroma:
        for pub in publications:
            match_dimensions_to_orkg(pub, emb_model, chroma)

        eval_matching_stats(publications)
        save_rejected_json(publications)

    # Final Output Guardrailing Counts
    print(f"\nGuardrailing Successful: {N_GUARDRAILING_SUCCESSFUL}")
    print(f"Guardrailing Unsuccessful: {N_GUARDRAILING_UNSUCCESSFUL}")

    with open(get_output_path("publications_final.json"), 'w') as f:
        json.dump(publications, f, indent=4)

    print("Output dir:", OUTPUT_DIR)

if __name__ == "__main__":
    main()