import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import os
import json
from dotenv import load_dotenv

from .detector import HallucinationVerifierBatch

# Load environment variables
load_dotenv(".env")


# This code extracts responses from the different models, than it embedds them using BERT, stems the responses using the Porter Stemmer, and links to claude API to detect hallucinations or not. 

def read_model_ids(file_path="model_ids.json"):
    """
    Read model IDs from JSON file.
    
    Args:
        file_path (str): Path to the model_ids.json file
        
    Returns:
        dict: Dictionary of model IDs
    """
    try:
        with open(file_path, 'r') as f:
            model_ids = json.load(f)
        print(f"Successfully loaded {len(model_ids)} model IDs")
        return model_ids
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading model_ids.json: {e}")
        return None


def read_model_dataset(model_key):
    """
    Read dataset.json for a specific model.
    
    Args:
        model_key (str): Original model key (e.g., 'provider/model-name')
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Transform model key by replacing / with _
    model_folder = model_key.replace('/', '_')
    file_path = os.path.join('responses', model_folder, 'dataset.json')
    
    try:
        with open(file_path, 'r') as f:
            df = pd.read_json(f, lines=True)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_all_models(model_ids):
    """
    Process dataset.json files for all models.
    
    Args:
        model_ids (dict): Dictionary of model IDs
        
    Returns:
        dict: Dictionary mapping model keys to DataFrames
    """
    results = {}
    
    for model_key in model_ids['model_id'].keys():
        print(f"\nProcessing model: {model_key}")
        df = read_model_dataset(model_key)
        if df is not None:
            results[model_key] = df
    
    return results


def read_parquet(file_path):
    """
    Read a parquet file and return as pandas DataFrame.
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None


def display_dataset_info(df):
    """Display basic information about the dataset."""
    if df is not None:
        print("\nDataset shape:", df.shape)
        print("\nColumn names:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)


def concatenate_datasets(all_datasets):
    """
    Concatenate all model datasets vertically and save as parquet.
    
    Args:
        all_datasets (dict): Dictionary mapping model keys to DataFrames
        
    Returns:
        pd.DataFrame: Combined dataset
    """
    if not all_datasets:
        print("No datasets to concatenate")
        return None
    
    # Add model_key as a column to each dataframe before concatenating
    dfs_with_model = []
    for _, df in all_datasets.items():
        df_copy = df.copy()
        dfs_with_model.append(df_copy)
    
    # Concatenate vertically
    combined_df = pd.concat(dfs_with_model, ignore_index=True)

    print(f"\nCombined dataset shape: {combined_df.shape}")

    return combined_df


if __name__ == "__main__":
    # Load config for verification parameters
    try:
        with open("metadata/config.json", 'r') as f:
            config = json.load(f)
        verification_params = config.get("verification_params", {})
        batch_size = verification_params.get("batch_size", 10000)
        check_interval = verification_params.get("check_interval", 60)
    except Exception as e:
        print(f"Warning: Could not load verification params from config: {e}")
        batch_size = 10000
        check_interval = 60

    # Read model IDs
    model_ids = read_model_ids("metadata/model_ids.json")
    
    if model_ids:
        # Process all models
        all_datasets = process_all_models(model_ids)
        
        # Concatenate datasets
        combined_df = concatenate_datasets(all_datasets)
        
        if combined_df is not None:
            # Run hallucination detection
            print("\n" + "="*60)
            print("Running Hallucination Detection")
            print("="*60)

            print(f"\nDataframe shape before filtering: {combined_df.shape}")
            print(f"Columns: {combined_df.columns.tolist()}")
            print(f"Sample data:\n{combined_df.head()}")

            combined_df = combined_df[combined_df['response'].notna() & (combined_df['response'] != '')]
            print(f"Dataframe shape after filtering: {combined_df.shape}")

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            verifier = HallucinationVerifierBatch(
                api_key=api_key,
                batch_size=batch_size
            )
            
            # Detect hallucinations - each batch saved automatically
            verifier.run_verification(combined_df)
            
            print("\n" + "="*60)
            print("All batches processed and saved to dataset/ folder")
            print("="*60)
