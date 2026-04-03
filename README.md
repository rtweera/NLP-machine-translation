# NLP Machine Translation

This repository contains an SMT-based spelling correction workflow built with Python.  
It includes:

- A baseline SMT spelling corrector (`app/smt.py`)
- An improved SMT spelling corrector with guardrails (`app/smt_improved.py`)
- A Gemini-powered dataset generator (`utils/smt-data-gen.py`)

## Repository structure

- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/app/` — main SMT implementations
- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/data/` — parallel corpus, LM corpus, and test text
- `/home/runner/work/NLP-machine-translation/NLP-machine-translation/utils/` — dataset generation utilities

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
