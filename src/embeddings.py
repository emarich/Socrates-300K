import os
import json
import pandas as pd
import logging
import nltk
from nltk.stem import PorterStemmer
from tqdm import tqdm
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
import torch

# Initialize Python porter stemmer
nltk.download("punkt")
ps = PorterStemmer()




def extract_year_from_prompt(prompt):
    if pd.isna(prompt):
        return None
    match = re.search(r'(202[0-2])', str(prompt))
    return int(match.group(1)) if match else None


def stem_text(text):
    """
    Stem a text using Porter Stemmer.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Stemmed text
    """
    if pd.isna(text):
        return ""
    
    # Tokenize and stem
    tokens = nltk.word_tokenize(str(text).lower())
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def add_stemmed_responses(df):
    """
    Add a stemmed_responses column to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'response' column
        
    Returns:
        pd.DataFrame: Dataframe with added 'stemmed_responses' column
    """
    print("\nStemming responses...")
    df['stemmed_response'] = df['response'].apply(stem_text)
    print("Stemming completed")
    return df


def get_bert_embeddings(texts, model_name, batch_size):
    """
    Extract embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        encoded = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # Use [CLS] token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_concatenate_batches(dataset_dir="dataset"):
    """
    Read all batch_{i}.parquet files from dataset folder and concatenate them vertically.
    
    Args:
        dataset_dir (str): Directory containing batch parquet files
        
    Returns:
        pd.DataFrame: Combined dataframe from all batches
    """
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} does not exist!")
        return None
    
    # Get all batch parquet files and sort them numerically
    batch_files = []
    for filename in os.listdir(dataset_dir):
        if filename.startswith('batch_') and filename.endswith('.parquet'):
            batch_files.append(filename)
    
    # Sort by batch number
    batch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not batch_files:
        logger.error(f"No batch files found in {dataset_dir}")
        return None
    
    logger.info(f"Found {len(batch_files)} batch files")
    
    # Load all batches
    all_batches = []
    for batch_file in batch_files:
        batch_path = os.path.join(dataset_dir, batch_file)
        try:
            batch_df = pd.read_parquet(batch_path)
            all_batches.append(batch_df)
            logger.info(f"Loaded {batch_file}: {len(batch_df)} rows")
        except Exception as e:
            logger.error(f"Error loading {batch_file}: {str(e)}")
    
    if not all_batches:
        logger.error("No batches were successfully loaded")
        return None
    
    # Concatenate all batches vertically
    combined_df = pd.concat(all_batches, ignore_index=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Successfully concatenated {len(all_batches)} batches")
    logger.info(f"Total rows: {len(combined_df)}")
    logger.info(f"Columns: {combined_df.columns.tolist()}")
    logger.info(f"{'='*60}")
    
    return combined_df


if __name__ == "__main__":
    # Load config
    try:
        with open("metadata/config.json", 'r') as f:
            config = json.load(f)
        embeddings_params = config.get("embeddings_params", {})
        bert_model_name = embeddings_params.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        batch_size = embeddings_params.get("batch_size", 32)
        logger.info(f"Loaded embeddings config: model={bert_model_name}, batch_size={batch_size}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        batch_size = 32

    # Step 1: Load all batch files
    logger.info("Loading batch files from dataset folder...")
    df = load_and_concatenate_batches("dataset")
    df = df[df['hallucination'].isin([0, 1])]
    df['year'] = df['prompt'].apply(extract_year_from_prompt)

    df = df.drop(['prompt'], axis=1)


    if df is not None:
        # Step 2: Add stemmed responses
        df = add_stemmed_responses(df)

        # Step 3: Generate embeddings for responses
        logger.info("\nGenerating embeddings for responses...")
        response_texts = df['response'].fillna("").tolist()
        response_embeddings = get_bert_embeddings(response_texts, model_name=bert_model_name, batch_size=batch_size)

        # Step 4: Generate embeddings for stemmed_responses
        logger.info("\nGenerating embeddings for stemmed_responses...")
        stemmed_texts = df['stemmed_response'].fillna("").tolist()
        stemmed_response_embeddings = get_bert_embeddings(stemmed_texts, model_name=bert_model_name, batch_size=batch_size)
        
        # Step 5: Add embeddings to dataframe as new columns
        logger.info("\nAdding embeddings to dataframe...")
        df['response_embeddings'] = list(response_embeddings)
        df['stemmed_response_embeddings'] = list(stemmed_response_embeddings)
        
        # Step 6: Save enriched dataset
        output_path = "socrates-300k.parquet"
        logger.info(f"\nSaving enriched dataset to {output_path}")
        df.to_parquet(output_path, index=False)
        logger.info("Complete!")
        
        # Display summary
        logger.info("\nDataset summary:")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Response embeddings shape: {response_embeddings.shape}")
        logger.info(f"Stemmed response embeddings shape: {stemmed_response_embeddings.shape}")
