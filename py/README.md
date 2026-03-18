# Python Utilities

This directory contains small Python utility scripts for text processing and machine learning tasks.

## Setup

Ensure you have the required dependencies installed:

```bash
pip install -r py/requirements.txt
```

## Scripts

### `tokenize_sentences.py`

Tokenizes input sentences using Hugging Face models and prints the resulting tokens and token IDs.

#### Usage

```bash
python3 py/tokenize_sentences.py [sentences...] [--model MODEL_ID]
```

#### Examples

**Tokenize specific sentences:**
```bash
python3 py/tokenize_sentences.py "Hello world" "How are you?"
```

**Use a specific model:**
```bash
python3 py/tokenize_sentences.py --model bert-base-uncased "Hello world"
```

**Read from stdin:**
```bash
echo "This is a test sentence." | python3 py/tokenize_sentences.py
```

#### Configuration
- `--model`: The Hugging Face model ID (default: `tencent/KaLM-Embedding-Gemma3-12B-2511`).
- `sentences`: Positional arguments for sentences. If omitted, the script reads from standard input.
