# Changelog

All notable changes to this dataset will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dataset releases are versioned as `MAJOR.MINOR.PATCH`:

- **MAJOR** — incompatible structural changes (e.g., renamed folders, changed filename convention, removed images)
- **MINOR** — backwards-compatible additions (e.g., new metadata columns, additional splits, new generator subset)
- **PATCH** — corrections to metadata, checksums, or documentation with no change to image content

---

## [1.0.0] — 2026-05-12

### Added

#### Source Code
- **src/generate_responses.py** — LLM response generation from prompts with 4-bit quantization and multi-model support
- **src/detector.py** — Hallucination detection using Claude Batch API with classification into correct/hallucinated/unknown
- **src/analysis.py** — Response aggregation and batch processing orchestration for hallucination detection
- **src/embeddings.py** — BERT embedding generation for responses and stemmed text using sentence-transformers/all-MiniLM-L6-v2


#### Configuration
- **metadata/config.json** — Centralized configuration for model parameters, generation, verification, and embedding settings
- **metadata/model_ids.json** — Mapping of 10 language models (Mistral, Gemma, Qwen, Phi, SOLAR, DeepSeek, Llama, Yi)
- **metadata/prompts.xlsx** — 200 input prompts for response generation

#### Documentation & Setup
- **README.md** — Complete pipeline documentation with usage examples and full dataset generation instructions
- **.env-template** — Template for environment variables (ANTHROPIC_API_KEY, HUGGINGFACE_TOKEN)
- **requirements.txt** — Python dependencies (transformers, torch, anthropic, pandas, etc.)
- **LICENSE** — CC BY 4.0 License
- **changelog.md** — This file
