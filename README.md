# Socrates-300K
This repository contains the official code and data for "A Geometric Analysis of Small-sized Language Model Hallucinations," accepted to ICML 2026. It includes the complete pipeline for text generation, dataset creation, and tagging.

**Citation:**

```
@misc{https://doi.org/10.48550/arxiv.2602.14778,
  doi = {10.48550/ARXIV.2602.14778},
  url = {https://arxiv.org/abs/2602.14778},
  author = {Ricco,  Emanuele and Onofri,  Elia and Cima,  Lorenzo and Cresci,  Stefano and Di Pietro,  Roberto},
  keywords = {Computation and Language (cs.CL),  Artificial Intelligence (cs.AI),  Computers and Society (cs.CY),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {A Geometric Analysis of Small-sized Language Model Hallucinations},
  publisher = {arXiv},
  year = {2026},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```
## Repository Structure
This repository is organized as follows:

```
Socrates-300K/
├── src/                           # Python source modules
│   ├── generate_responses.py      # LLM response generation
│   ├── detector.py                # Hallucination detection library (Claude Sonnet 4.5 API)
│   ├── analysis.py                # Response tagging and batches generation
│   ├── embeddings.py              # BERT embeddings and generation of the final dataset
│   └── .env-template              # API keys (ANTHROPIC_API_KEY, HUGGINGFACE_TOKEN)
├── metadata/
│   ├── config.json                # Model & generation parameters
│   ├── model_ids.json             # Model identifiers
│   └── prompts.xlsx               # 200 input prompts
├── CLAUDE.md                      # Project documentation
├── .gitignore                     # Git rules (ignores .env, models, cache)
├── README.md                      # this file
├── requirements.txt               # Python dependencies
├── LICENSE                        # CC BY 4.0 License
├── changelog.md                   # version history
│
│ Generated during pipeline execution:
├── responses/                     # Generated model responses (created by generate_responses.py)
│   ├── mistralai_Mistral-7B-v0.1/
│   │   └── dataset.json           # JSONL format responses
│   ├── google_gemma-2-9b/
│   │   └── dataset.json
│   └── ... (one folder per model)
├── dataset/                       # Hallucination detection results (created by analysis.py)
│   ├── batch_000.parquet
│   ├── batch_001.parquet
│   └── ... (results from Claude batch processing)
└── socrates-300k.parquet          # Final enriched dataset with BERT embeddings (created by embeddings.py)
```

## Getting Started

### 1. Create Anaconda Environment

Create a new conda environment called `socrates` and install all dependencies:

```bash
# Create the environment
conda create -n socrates python=3.11

# Activate the environment
conda activate socrates

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file with your API credentials:

```bash
# Copy the template
cp .env-template .env

# Edit .env and add:
HUGGINGFACE_TOKEN=your_hf_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Available Models

The following language models are configured in `metadata/model_ids.json`:

| Model | ID | Size |
|-------|----|----|
| mistralai/Mistral-7B-v0.1 | 0 | 7B |
| google/gemma-2-9b | 1 | 9B |
| Upstage/SOLAR-10.7B-v1.0 | 2 | 10.7B |
| microsoft/phi-4 | 3 | 14B |
| Qwen/Qwen2.5-14B | 4 | 14B |
| google/gemma-2-27b | 5 | 27B |
| Qwen/Qwen2.5-32B | 6 | 32B |
| deepseek-ai/deepseek-llm-7b-base | 7 | 7B |
| meta-llama/Llama-3.1-8B | 8 | 8B |
| 01-ai/Yi-1.5-9B | 9 | 9B |

## Generating Responses

### Basic Usage

```bash
# Generate 150 responses per prompt from default model (Gemma-2-9B)
python -m src.generate_responses

# Generate 150 responses per prompt from a specific model
python -m src.generate_responses --model "google/gemma-2-9b"

# Generate 5 responses per prompt from a specified model (Gemma-2-9B)
python -m src.generate_responses --model "google/gemma-2-9b" --num_responses 5
```

### Generating Full Socrates-300K Dataset

To generate responses from all 10 models with 150 responses per 200 prompt, for a total of 300k dataset:

```bash
# Generate responses for all models
for model in \
  "mistralai/Mistral-7B-v0.1" \
  "google/gemma-2-9b" \
  "Upstage/SOLAR-10.7B-v1.0" \
  "microsoft/phi-4" \
  "Qwen/Qwen2.5-14B" \
  "google/gemma-2-27b" \
  "Qwen/Qwen2.5-32B" \
  "deepseek-ai/deepseek-llm-7b-base" \
  "meta-llama/Llama-3.1-8B" \
  "01-ai/Yi-1.5-9B"; do
  echo "Generating responses for: $model"
  python -m src.generate_responses --model "$model" --num_responses 150
done
```

This will create response datasets for all models in `responses/{model_name}/dataset.json`.

### Output

Responses are saved to `responses/{model_name}/dataset.json` in JSONL format with fields:
- `model_id`: Numeric identifier of the model
- `prompt`: Prompt added to the model
- `prompt_id`: Input prompt ID
- `response`: Generated text
- `temperature`: Generation temperature
- `response_index`: Response number for the prompt

Example output path:
```
responses/google_gemma-2-9b/dataset.json
responses/mistralai_Mistral-7B-v0.1/dataset.json
```

### Configuration

Model generation parameters are in `metadata/config.json`:
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`, `top_k`: Nucleus and top-k sampling
- `quantization`: 4-bit quantization settings for memory efficiency

## Detecting Hallucinations

### Workflow

After generating responses from all models, use Claude API to classify each response as:
- **0**: CORRECT (factually accurate)
- **1**: HALLUCINATED (false claims, contradictions, fabricated information)
- **2**: NOT_KNOWN (model admits uncertainty)

### Running Detection

```bash
# Reads all responses from responses/{model_name}/ directories
# Processes them through Claude Sonnet API using batches
# Saves results to dataset/batch_*.parquet files
python -m src.analysis
```

### Process

1. **Load responses**: Reads all generated responses from `responses/` directory
2. **Combine datasets**: Merges responses from all models into single dataframe
3. **Batch verification**: Uses Claude Batch API to classify hallucinations efficiently
4. **Save results**: Saves verified data as parquet files in `dataset/` folder

### Output

Results saved in `dataset/` directory:
```
dataset/
├── batch_000.parquet
├── batch_001.parquet
└── ...
```

Each parquet file contains:
- Original response data
- `hallucination`: Classification (0, 1, or 2)
- `verification`: Success flag

**Note:** Requires `ANTHROPIC_API_KEY` in `.env` file for Claude API access.

## Generating Embeddings

After hallucination detection, generate semantic embeddings for responses using 🤗 [Sentence Transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

### Running Embeddings Generation

```bash
# Loads all batch files from dataset/ directory
# Generates BERT embeddings for responses
# Produces final enriched dataset
python -m src.embeddings
```

### Process

1. **Load verified batches**: Reads all batch parquet files from `dataset/` directory
2. **Filter responses**: Keeps only correct (0) and hallucinated (1) responses
3. **Extract metadata**: Extracts year from prompts
4. **Stem responses**: Applies Porter Stemmer for text normalization
5. **Generate embeddings**: Creates BERT embeddings using `sentence-transformers/all-MiniLM-L6-v2`
6. **Save dataset**: Outputs enriched dataset with embeddings

### Output

Results saved as single parquet file:
```
socrates-300k.parquet
```

Contains:
- Response and model metadata
- `hallucination`: Classification (0 or 1)
- `stemmed_response`: Porter stemmed text
- `response_embeddings`: BERT embeddings for original text (384-dim)
- `stemmed_response_embeddings`: BERT embeddings for stemmed text (384-dim)
