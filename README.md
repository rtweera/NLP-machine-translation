# NLP Machine Translation

This repository contains a multi-stage machine translation workflow built with Python and notebooks.  
It includes:

- A baseline SMT spelling corrector (`app/smt.py`)
- An improved SMT spelling corrector with guardrails (`app/smt_improved.py`)
- A Gemini-powered dataset generator (`utils/smt-data-gen.py`)
- A complete 3-stage translation notebook (`final-3-stage-machine-translation.ipynb`)

## Repository structure

- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/app/` — main SMT implementations
- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/data/` — parallel corpus, LM corpus, and test text
- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/utils/` — dataset generation utilities
- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/final-3-stage-machine-translation.ipynb` — end-to-end pipeline notebook

## Requirements

- Python `>=3.13`
- Poetry (recommended)

Dependencies are defined in:
- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/pyproject.toml`

## Setup

From the repository root:

```bash
poetry install
```

## Run the SMT corrector

Baseline version:

```bash
poetry run python app/smt.py
```

Improved version:

```bash
poetry run python app/smt_improved.py
```

## 3-stage notebook pipeline

Use the notebook below for the full end-to-end system:

- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/final-3-stage-machine-translation.ipynb`

The notebook is organized into three stages:

1. **Stage 1: Statistical spelling correction preprocessing**
   - Builds and tests a statistical spelling corrector
   - Exposes helper functions and an object usable in a larger pipeline
2. **Stage 2: Complex English to simpler English conversion**
   - Includes environment setup, utility functions, optional finetuning, and inference
   - Provides a `convert_lang_eng(...)` function for pipeline use
3. **Stage 3: Final neural translation**
   - Uses a quantized MBART translation pipeline
   - Connects all stages in a full system flow and includes a Gradio interface

## Generate training data with Gemini (optional)

Set your API key first:

```bash
export GEMINI_API_KEY="your_api_key"
```

Then run:

```bash
poetry run python utils/smt-data-gen.py
```

This generates `parallel_corpus.txt` and `lm_corpus.txt` in the current working directory.
