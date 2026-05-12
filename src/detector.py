import json
import time
from typing import List, Dict, Any
import os
from anthropic import Anthropic
from datetime import datetime
import re
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HallucinationVerifierBatch:
    def __init__(self, 
                 api_key: str, 
                 model: str = "claude-sonnet-4-5-20250929",
                 batch_size: int = 10000):
        
        """Initialize the hallucination verifier with Claude Batch API."""
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.output_dir = "dataset"
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Batch results will be saved to {self.output_dir}/")
    
    def create_verification_prompt(self, 
                                   prompt: str, 
                                   response: str) -> str:
        """Create a prompt for Claude to verify if a response is hallucinated."""
        verification_prompt = f"""Analyze this response and classify it into one of three categories.

QUESTION: {prompt}

RESPONSE: {response}

Classify the response as:
- 0: CORRECT (factually accurate)
- 1: HALLUCINATED (contains false claims, contradictions, fabricated information or does not answer the question)
- 2: NOT_KNOWN (model admits it doesn't know)

Provide your verdict:
CATEGORY: [0, 1, or 2]
"""
        
        return verification_prompt
    
    def create_batch_requests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create batch requests from dataframe."""
        requests = []
        
        for i, row in data.iterrows():
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            
            if not prompt or not response:
                logging.warning(f"Missing prompt or response at row {i}. Skipping.")
                continue
            
            verification_prompt = self.create_verification_prompt(
                prompt=prompt,
                response=response
            )
            
            request = {
                "custom_id": f"request_{i}",
                "params": {
                    "model": self.model,
                    "max_tokens": 20,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": verification_prompt
                        },
                        {
                            "role": "assistant",
                            "content": "CATEGORY:"  # Prefill forces immediate answer
                    }
                    ]
                }
            }
            requests.append(request)
        
        return requests
    
    def submit_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Submit batch request and return batch ID."""
        logging.info(f"Submitting batch with {len(requests)} requests...")

        if len(requests) == 0:
            raise ValueError("No valid requests to submit. Check that 'prompt' and 'response' columns exist and contain data.")

        message_batch = self.client.messages.batches.create(
            requests=requests
        )
        
        logging.info(f"Batch submitted! Batch ID: {message_batch.id}")
        logging.info(f"Status: {message_batch.processing_status}")
        
        return message_batch.id
    
    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check the status of a batch."""
        batch = self.client.messages.batches.retrieve(batch_id)
        
        return {
            "id": batch.id,
            "processing_status": batch.processing_status,
            "request_counts": {
                "processing": batch.request_counts.processing,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
                "canceled": batch.request_counts.canceled,
                "expired": batch.request_counts.expired,
            },
            "ended_at": batch.ended_at,
            "created_at": batch.created_at
        }
    
    def wait_for_batch(self, batch_id: str, check_interval: int = 60):
        """Wait for batch to complete, checking status periodically."""
        logging.info(f"Waiting for batch {batch_id} to complete...")
        logging.info(f"This may take up to 24 hours. Checking every {check_interval} seconds.")
        
        while True:
            status = self.check_batch_status(batch_id)
            
            logging.info(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Status: {status['processing_status']}")
            logging.info(f"Succeeded: {status['request_counts']['succeeded']}")
            logging.info(f"Processing: {status['request_counts']['processing']}")
            logging.info(f"Errored: {status['request_counts']['errored']}")
            
            if status['processing_status'] == 'ended':
                logging.info("\nBatch processing completed!")
                return status
            
            time.sleep(check_interval)
    
    def retrieve_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieve results from a completed batch."""
        logging.info(f"Retrieving results for batch {batch_id}...")
        
        results = []
        
        for result in self.client.messages.batches.results(batch_id):
            results.append(result)
        
        logging.info(f"Retrieved {len(results)} results")
        return results
    
    def parse_claude_response(self, response_text: str) -> int:
        """Parse Claude's response to extract category (0, 1, or 2)."""
        # Get all valid digits in the string
        valid_digits = [int(char) for char in response_text if char in '012']
        
        if valid_digits:
            return valid_digits[0]  # Return first occurrence
        
        return -1  # Unable to parse
    
    def process_batch_results(self, data: pd.DataFrame, 
                             batch_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process batch results and merge with original dataframe."""
        verified_data = data.copy()
        
        # Initialize new columns
        verified_data['hallucination'] = -1
        verified_data['verification'] = False
        
        # Create mapping from custom_id to result
        results_map = {}
        for result in batch_results:
            custom_id = result.custom_id
            index = int(custom_id.split('_')[1])
            
            if result.result.type == "succeeded":
                message = result.result.message
                if message.content and len(message.content) > 0:
                    claude_response = message.content[0].text
                    results_map[index] = {
                        "hallucination": self.parse_claude_response(claude_response),
                        "verification": True
                    }
                else:
                    # Handle empty or invalid content
                    results_map[index] = {
                        "hallucination": -1,
                        "verification": False
                    }
                    logging.error(f"Empty or invalid content for request {index}")
            else:
                error = result.result.error
                results_map[index] = {
                    "hallucination": -1,
                    "verification": False
                }
                logging.error(f"Error for request {index}: {error.type} - {error.message}")
        
        # Merge results with original data
        for idx in results_map:
            if idx in verified_data.index:
                for key, value in results_map[idx].items():
                    verified_data.at[idx, key] = value
        
        return verified_data

    def run_verification(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run complete verification pipeline on dataframe."""
        logging.info(f"Starting verification for {len(data)} rows")
        
        # Calculate number of batches
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        logging.info(f"Processing in {num_batches} batches of {self.batch_size} items each")
        
        # Submit all batches
        for batch_num in range(num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(data))
            batch_data = data.iloc[start_idx:end_idx].copy()
            
            logging.info(f"\nSubmitting Batch {batch_num + 1}/{num_batches} (rows {start_idx}-{end_idx})")
            
            requests = self.create_batch_requests(batch_data)
            batch_id = self.submit_batch(requests)
            logging.info(f"Waiting for batch {batch_num + 1}/{num_batches}...")
            self.wait_for_batch(batch_id, check_interval=60)
            
            logging.info(f"Retrieving results for batch {batch_num + 1}/{num_batches}...")
            batch_results = self.retrieve_batch_results(batch_id)
            
            verified_batch = self.process_batch_results(batch_data, batch_results)
            
            
            # Save this batch to parquet immediately
            batch_filename = f"batch_{batch_num:03d}.parquet"
            batch_filepath = os.path.join(self.output_dir, batch_filename)
            verified_batch.to_parquet(batch_filepath, index=False)
            logging.info(f"Saved batch to {batch_filepath}")
        
        logging.info(f"\nAll {num_batches} batches processed and saved to {self.output_dir}/")
        return data  # Return original data since batches are saved separately