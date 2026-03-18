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

### `embed_sentence.py`

Generates embeddings for sentences using Hugging Face models. Uses last-token pooling and normalization.

#### Usage

```bash
python3 py/embed_sentence.py [sentences...] [--model MODEL_ID] [--query]
```

#### Examples

**Generate embeddings for sentences:**
```bash
python3 py/embed_sentence.py "Hello world" "Machine learning is fun."
```

**Save embedding to a file:**
```bash
python3 py/embed_sentence.py --output vector.txt "Example sentence"
```

**Generate embeddings for a query (with instruction prefix):**
```bash
python3 py/embed_sentence.py --query "How does tokenization work?"
```

#### Configuration
- `--model`: The Hugging Face model ID (default: `tencent/KaLM-Embedding-Gemma3-12B-2511`).
- `--query`: If set, applies the recommended instruction prefix for the default model.
- `--output`: File path to write the full embedding (one value per line). Suppresses full embedding output in stdout.
- `sentences`: Positional arguments for sentences. If omitted, the script reads from standard input.
