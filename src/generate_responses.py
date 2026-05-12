from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import json
import os
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from dotenv import load_dotenv

def load_env_variables():
    """Load environment variables from .env file"""
    env_path = ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Environment variables loaded from {env_path}")
    else:
        print(f"Warning: {env_path} not found. Using system environment variables.")

def load_config(config_path="config.json"):
    """Load configuration from JSON file and inject HuggingFace token from .env"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")

        # Inject HuggingFace token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            print("HuggingFace token loaded from environment")
        else:
            print("Warning: HUGGINGFACE_TOKEN not found in environment")

        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using default values.")
        return None


class ResponseGenerator:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for model: {model_name}")
        
        if self.device == "cuda":
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self._init_model()



    def _init_model(self):
        """Initialize model and tokenizer with 4-bit quantization"""
        
        # Verify bitsandbytes
        try:
            import bitsandbytes as bnb
            print(f"✓ bitsandbytes installed: {bnb.__version__}")
        except ImportError:
            raise ImportError("bitsandbytes not installed! Run: pip install bitsandbytes")
        
        # Setup quantization config
        quant_config = self.config["model_params"]["quantization"]
        compute_dtype = torch.float16 if quant_config["bnb_4bit_compute_dtype"] == "float16" else torch.float32
        
        print("Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        
        print(f"Quantization config: {bnb_config}")
        
        # Initialize tokenizer
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=os.getenv("HUGGINGFACE_TOKEN"),
            trust_remote_code=True,
            padding_side=self.config["model_params"]["padding_side"]
        )
        
        # Setup tokenizer pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with quantization
        print(f"Loading model {self.model_name} with 4-bit quantization...")
        print("This may take a few minutes...")
        
        # Standard loading for all models
        compute_dtype = torch.float16 if self.config["model_params"]["quantization"]["bnb_4bit_compute_dtype"] == "float16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully!")
        
        # VERIFY quantization worked
        print("\n" + "="*50)
        print("VERIFICATION:")
        print("="*50)
        
        # Check if model is quantized
        is_quantized = False
        for name, param in self.model.named_parameters():
            if hasattr(param, 'quant_state'):
                is_quantized = True
                break
        
        if is_quantized:
            print("✓ Model IS quantized (4-bit)")
        else:
            print("✗ WARNING: Model NOT quantized!")
        
        # Check memory usage
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU Memory allocated: {memory_allocated:.2f} GB")
            print(f"GPU Memory reserved: {memory_reserved:.2f} GB")
        
        print("="*50 + "\n")

    def generate_responses(self, questions, responses_per_question=150):
        """Generate multiple responses for each question individually with progress bar"""
        all_responses = []
        gen_params = self.config["generation_params"]
        proc_params = self.config["processing_params"]
        min_new_tokens = gen_params.get("min_new_tokens", 10)
        max_retries = 5
        
        # Calculate total number of responses to generate
        total_responses = len(questions) * responses_per_question
        
        # Progress bar for this model's response generation
        with tqdm(total=total_responses, desc=f"Generating responses for {self.model_name}", unit="response") as pbar:
            for question in questions:
                question_responses = []
                
                # Generate multiple responses for this question
                for response_idx in range(responses_per_question):
                    retry_count = 0
                    response = ""
                    
                    while retry_count < max_retries:
                        try:
                            # Prepare prompt using the template
                            formatted_prompt = self.config["prompt_template"].format(question=question)
                            
                            # Tokenize input
                            inputs = self.tokenizer(
                                formatted_prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=proc_params["max_length"]
                            ).to(self.device)
                            

                            # Store input length to extract only new tokens
                            input_length = inputs["input_ids"].shape[1]
                            

                            with torch.no_grad():
                                    outputs = self.model.generate(
                                        inputs["input_ids"],
                                        min_new_tokens=min_new_tokens,
                                        max_new_tokens=gen_params["max_tokens"],
                                        attention_mask=inputs["attention_mask"],
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        temperature=gen_params["temperature"],
                                        do_sample=gen_params["do_sample"],
                                        top_p=gen_params["top_p"],
                                        top_k=gen_params["top_k"],
                                        num_return_sequences=1,
                                        repetition_penalty=gen_params["repetition_penalty"]
                                    )
                            
                            # Decode only the newly generated tokens (skip the input prompt)
                            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
                            
                            # Clean up GPU memory after each generation
                            if self.device == "cuda":
                                del outputs
                                del inputs
                                torch.cuda.empty_cache()
                                gc.collect()
                            
                            # Check if response is empty or too short
                            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                            if len(response) == 0 or len(response_tokens) < min_new_tokens:
                                retry_count += 1
                                if retry_count < max_retries:
                                    print(f"\nResponse too short or empty (length: {len(response_tokens)} tokens, required: {min_new_tokens}). Retry {retry_count}/{max_retries}...")
                                    continue
                                else:
                                    print(f"\nMax retries reached. Using last response (length: {len(response_tokens)} tokens).")
                            
                            # Response is valid, break out of retry loop
                            break
                            
                        except RuntimeError as e:
                            if "CUDA" in str(e) or "assert" in str(e):
                                print(f"\nCUDA error detected for question: {question[:50]}...")
                                print(f"Error: {str(e)}")
                                response = "Error: CUDA assertion failure"
                                break
                            else:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    print(f"\nError generating response {response_idx + 1} for question: {question[:50]}...")
                                    print(f"Error: {str(e)}")
                                    response = "Error generating response"
                                    break
                                else:
                                    print(f"\nError occurred. Retry {retry_count}/{max_retries}...")
                                    continue
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"\nError generating response {response_idx + 1} for question: {question[:50]}...")
                                print(f"Error: {str(e)}")
                                response = "Error generating response"
                                break
                            else:
                                print(f"\nError occurred. Retry {retry_count}/{max_retries}...")
                                continue
                    
                    question_responses.append(response)
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Add all responses for this question to the main list
                all_responses.extend(question_responses)
        
        return all_responses

    def cleanup(self):
        """Clean up model resources"""
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


def read_prompts_from_excel(excel_file):
    """Read prompts from Excel file"""
    try:
        df = pd.read_excel(excel_file)
        if 'question' not in df.columns:
            raise ValueError("Excel file must contain a 'question' column")
        prompts = df['question'].tolist()
        id = df['id'].tolist()
        return prompts, id
    except Exception as e:
        print(f"Error reading prompts from Excel: {str(e)}")
        return [], []


def generate_dataset(model_name, 
                     num_responses, 
                     config_path="config.json", 
                     model_ids_path="model_ids.json", 
                     prompts_file="prompts.xlsx"):

    """Generate dataset for a single model and save incrementally per prompt"""
    
    # Load configuration
    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    
    # Load model IDs mapping
    try:
        with open(model_ids_path, 'r', encoding='utf-8') as f:
            model_ids_config = json.load(f)
            model_ids = model_ids_config.get("model_id", {})
    except Exception as e:
        print(f"Error loading model IDs: {str(e)}")
        return
    
    # Get model ID
    if model_name not in model_ids:
        print(f"Model '{model_name}' not found in model_ids.json")
        print(f"Available models: {list(model_ids.keys())}")
        return
    
    model_id = model_ids[model_name]
    print(f"Model ID for {model_name}: {model_id}")
    
    # Read prompts from Excel
    prompts, ids = read_prompts_from_excel(prompts_file)
    if not prompts:
        print("No prompts found. Exiting.")
        return
    
    print(f"Found {len(prompts)} prompts")
    print(f"Generating {num_responses} responses per prompt")
    
    # Get temperature from config
    temperature = config["generation_params"]["temperature"]
    
    # Create output directory structure
    model_safe_name = model_name.replace("/", "_")
    output_dir = os.path.join("responses", model_safe_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving responses to directory: {output_dir}")
    
    # Initialize response generator
    generator = ResponseGenerator(model_name=model_name, config=config)
    
    # Generate and save responses for each prompt individually
    print(f"Generating and saving responses incrementally...")
    
    for id_val, prompt in zip(ids, prompts):
        print(f"\nProcessing prompt {id_val}")
        
        # Extract year if present
        year = None
        if '2020' in prompt or '2022' in prompt:
            year = '2020' if '2020' in prompt else '2022'
        
        # Generate responses for this single prompt
        prompt_responses = generator.generate_responses([prompt], num_responses)
        
        # Create data for this prompt
        prompt_data = []
        for resp_id, response in enumerate(prompt_responses):
            prompt_data.append({
                "model_id": model_id,
                "prompt": prompt,
                "prompt_id": id_val,
                "year": year,
                "response_index": resp_id,
                "response": response,
                "temperature": temperature,
            })
        
        # Save this prompt's responses immediately
        df = pd.DataFrame(prompt_data)
        df.to_json(f'{output_dir}/dataset.json', orient="records", force_ascii=False, 
                   default_handler=None, lines=True, mode="a")
        
        print(f"Saved {len(prompt_data)} responses")
    
    
    # Cleanup
    generator.cleanup()


def main():
    # Load environment variables from .env
    load_env_variables()

    parser = argparse.ArgumentParser(description='Generate responses from a language model')
    parser.add_argument('--model', type=str,
                        default = 'google/gemma-2-9b',
                        help='Model name (e.g., mistralai/Mistral-7B-v0.1)')
    parser.add_argument('--num_responses', type=int,
                        help='Number of responses to generate per prompt', default=150)
    parser.add_argument('--config', type=str, default='metadata/config.json',
                        help='Path to config.json file (default: metadata/config.json)')
    parser.add_argument('--model_ids', type=str, default='metadata/model_ids.json',
                        help='Path to model_ids.json file (default: metadata/model_ids.json)')
    parser.add_argument('--prompts', type=str, default='metadata/prompts.xlsx',
                        help='Path to prompts Excel file (default: metadata/prompts.xlsx)')

    args = parser.parse_args()

    try:
        generate_dataset(
            model_name=args.model,
            num_responses=args.num_responses,
            config_path=args.config,
            model_ids_path=args.model_ids,
            prompts_file=args.prompts
        )
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()