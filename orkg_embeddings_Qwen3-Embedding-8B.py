"""
Generates embeddings for ORKG (Open Research Knowledge Graph) property labels
using the local Qwen/Qwen3-Embedding-8B model and stores them in Parquet files.

Files and Directories:
----------------------
Input Files:
  - ORKG_PICKLE_FILE (e.g., "api_orkg_properties.pickle"):
    A Pickle file containing a list of dictionaries. Each dictionary must represent
    an ORKG property and contain at least "id" and "label".

Output Files/Directories:
  - OUTPUT_EMBEDDINGS_BASE_DIR:
    Directory where the generated Parquet files (e.g., "orkg_embeddings_qwen_batch_0.parquet")
    will be saved.
  - PROGRESS_FILE:
    Tracks the index of the last processed item to allow resuming.

Prerequisites:
--------------
  - sentence-transformers, torch, pandas, numpy
  - A machine with sufficient RAM (approx. 32GB+) for full precision loading
    or appropriate configuration for half-precision.
"""

import os
import pickle
import time
import warnings
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# --- Configuration ---

# Suppress specific runtime warnings from sklearn/numpy on Apple Silicon
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input/Output Paths
ORKG_PICKLE_FILE = "api_orkg_properties.pickle"
OUTPUT_EMBEDDINGS_BASE_DIR = os.path.join("orkg_data", "embeddings_parquet_qwen")
PROGRESS_FILE = os.path.join("orkg_data", "embedding_progress_qwen.txt")

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 32
ITEMS_PER_PARQUET_FILE = 10000


def get_local_model() -> SentenceTransformer:
    """
    Initializes and returns the local SentenceTransformer model.
    Checks for Apple Silicon (MPS) availability.
    """
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Info: Apple Silicon GPU (MPS) detected. Using device: {device}")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Info: NVIDIA GPU (CUDA) detected. Using device: {device}")
    else:
        device = "cpu"
        print("Warning: No GPU detected. Using CPU (inference will be slow).")

    print(f"Info: Loading model '{MODEL_NAME}'...")
    print("Info: Loading in standard float32 precision for mathematical stability.")

    try:
        model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True,
            device=device
        )
        print("Info: Model successfully loaded.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {MODEL_NAME}: {e}")


def load_orkg_data(pickle_file_path: str) -> List[Dict[str, Any]]:
    """
    Loads ORKG properties data from a Pickle file.
    """
    print(f"Info: Loading ORKG data from {pickle_file_path}...")

    if not os.path.isabs(pickle_file_path):
        # Check if it exists relative to current working dir first
        if os.path.exists(pickle_file_path):
            absolute_pickle_path = os.path.abspath(pickle_file_path)
        else:
            # Fallback: check relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_pickle_path = os.path.join(script_dir, pickle_file_path)
    else:
        absolute_pickle_path = pickle_file_path

    if not os.path.exists(absolute_pickle_path):
        print(f"Error: Pickle file not found at: {absolute_pickle_path}")
        return []

    try:
        with open(absolute_pickle_path, 'rb') as f:
            orkg_data = pickle.load(f)

        if not isinstance(orkg_data, list):
            print("Error: ORKG data is not a list.")
            return []

        # Basic validation of first item
        if orkg_data and not (isinstance(orkg_data[0], dict) and "id" in orkg_data[0] and "label" in orkg_data[0]):
            print("Error: Data format incorrect. Expected list of dicts with 'id' and 'label'.")
            return []

        print(f"Info: Successfully loaded {len(orkg_data)} ORKG property entries.")
        return orkg_data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return []


def get_last_processed_index(progress_file: str) -> int:
    """Reads the index of the last successfully processed item."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            print(f"Warning: Progress file corrupted or unreadable. Starting from scratch.")
            return -1
    return -1


def update_progress_file(progress_file: str, index: int) -> None:
    """Updates the progress file."""
    try:
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'w') as f:
            f.write(str(index))
    except Exception as e:
        print(f"Error updating progress file: {e}")


def create_embeddings_local(model: SentenceTransformer, texts: List[str]) -> Optional[np.ndarray]:
    """
    Generates embeddings using the local SentenceTransformer model.
    """
    try:
        # normalize_embeddings=True is crucial for cosine similarity later
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,  # Disable internal tqdm to keep main loop clean
            batch_size=BATCH_SIZE
        )

        return embeddings.astype(np.float32)

    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None


def main():
    # Setup directories
    os.makedirs(OUTPUT_EMBEDDINGS_BASE_DIR, exist_ok=True)
    print(f"Info: Output directory set to: {OUTPUT_EMBEDDINGS_BASE_DIR}")

    # Load Model
    try:
        model = get_local_model()
    except RuntimeError as e:
        print(f"Critical Error: {e}")
        return

    # Load Data
    orkg_data = load_orkg_data(ORKG_PICKLE_FILE)
    if not orkg_data:
        print("Error: No data to process. Exiting.")
        return

    total_items = len(orkg_data)
    start_index = get_last_processed_index(PROGRESS_FILE) + 1

    # Calculate parquet file counter based on progress
    parquet_file_counter = 0
    if start_index > 0:
        parquet_file_counter = start_index // ITEMS_PER_PARQUET_FILE

    processed_data_buffer = []

    print(f"Info: Starting processing from index {start_index} of {total_items}.")

    # Main Processing Loop
    for i in range(start_index, total_items, BATCH_SIZE):
        batch_slice = orkg_data[i: min(i + BATCH_SIZE, total_items)]

        valid_items = []
        labels = []

        # Filter valid items
        for item in batch_slice:
            label = item.get("label")
            orkg_id = item.get("id")
            if label and isinstance(label, str) and label.strip() and orkg_id:
                labels.append(label)
                valid_items.append(item)

        if not labels:
            # Update progress even if batch was empty/invalid to prevent infinite loops
            last_idx = i + len(batch_slice) - 1
            update_progress_file(PROGRESS_FILE, last_idx)
            continue

        # Generate Embeddings
        t0 = time.time()
        embeddings = create_embeddings_local(model, labels)
        duration = time.time() - t0

        if embeddings is not None and len(embeddings) == len(labels):
            # Store results in buffer
            for idx, embedding in enumerate(embeddings):
                original_item = valid_items[idx]
                processed_data_buffer.append({
                    "id": original_item["id"],
                    "label": original_item["label"],
                    "embedding": embedding
                })

            last_processed_idx = i + len(batch_slice) - 1
            update_progress_file(PROGRESS_FILE, last_processed_idx)

            # Log progress every 100 items or so to keep output readable but informative
            if i % 100 < BATCH_SIZE:
                print(f"Info: Processed items up to index {last_processed_idx}/{total_items}. "
                      f"Batch time: {duration:.2f}s.")

            # Check if it's time to write a Parquet file
            buffer_full = len(processed_data_buffer) >= ITEMS_PER_PARQUET_FILE
            is_last_batch = (i + len(batch_slice)) >= total_items

            if (buffer_full or (is_last_batch and processed_data_buffer)):
                df = pd.DataFrame(processed_data_buffer)

                parquet_filename = f"orkg_embeddings_qwen_batch_{parquet_file_counter}.parquet"
                parquet_path = os.path.join(OUTPUT_EMBEDDINGS_BASE_DIR, parquet_filename)

                df.to_parquet(parquet_path)
                print(f"Info: Saved {len(processed_data_buffer)} items to {parquet_path}")

                processed_data_buffer = []  # Clear buffer
                parquet_file_counter += 1

        else:
            print(f"Error: Failed to generate embeddings for batch starting at {i}. Stopping.")
            break

    print("Info: Processing complete.")


if __name__ == "__main__":
    main()