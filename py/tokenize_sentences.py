import argparse
import sys
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize sentences using a Hugging Face model.")
    parser.add_argument(
        "--model",
        type=str,
        default="tencent/KaLM-Embedding-Gemma3-12B-2511",
        help="The Hugging Face model ID (default: tencent/KaLM-Embedding-Gemma3-12B-2511)"
    )
    parser.add_argument(
        "sentences",
        nargs="*",
        help="Sentences to tokenize. If omitted, reads from stdin."
    )

    args = parser.parse_args()

    try:
        print(f"Loading tokenizer for model: {args.model}...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    sentences = args.sentences
    if not sentences:
        if sys.stdin.isatty():
            print("Enter sentences (one per line, Ctrl-D to finish):", file=sys.stderr)
        sentences = [line.strip() for line in sys.stdin if line.strip()]

    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")
        tokens = tokenizer.tokenize(sentence)
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {ids}")

if __name__ == "__main__":
    main()
