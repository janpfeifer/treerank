import argparse
import sys
import torch
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for sentences using Sentence Transformers.")
    parser.add_argument(
        "--model",
        type=str,
        default="tencent/KaLM-Embedding-Gemma3-12B-2511",
        help="The Hugging Face model ID (default: tencent/KaLM-Embedding-Gemma3-12B-2511)"
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Embed the sentences as queries (adds instruction prefix)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="File path to write the full embedding (one value per line). Suppresses full embedding output in stdout."
    )
    parser.add_argument(
        "sentences",
        nargs="*",
        help="Sentences to embed. If omitted, reads from stdin."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"Loading model: {args.model}...", file=sys.stderr)
        # trust_remote_code=True is often needed for newer architectures
        model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    sentences = args.sentences
    if not sentences:
        if sys.stdin.isatty():
            print("Enter sentences (one per line, Ctrl-D to finish):", file=sys.stderr)
        sentences = [line.strip() for line in sys.stdin if line.strip()]

    # Use encode_query if --query is set, otherwise encode (which is for documents)
    # SentenceTransformer handles the specific prefixes for models like KaLM automatically
    # if they are configured in the model's metadata.
    if args.query:
        embeddings = model.encode(sentences, prompt_name="query") if hasattr(model, "prompts") and "query" in model.prompts else model.encode(sentences)
        # Note: For KaLM specifically, SentenceTransformer usually maps encode_query 
        # to the correct "Instruct: ...\nQuery:" prefix if the model is loaded correctly.
        # If the model doesn't have named prompts, we'd manually prefix, 
        # but modern ST models usually have this.
    else:
        embeddings = model.encode(sentences)

    for i, (orig, emb) in enumerate(zip(sentences, embeddings)):
        emb_list = emb.tolist()
        
        print(f"\nSentence {i+1}: {orig}")
        print(f"Embedding Shape: {len(emb_list)}")
        print(f"Embedding (first 5 values): {emb_list[:5]}")
        
        if args.output:
            # If multiple sentences are provided, this will overwrite for each.
            # To be safe, we append index if multiple, but following previous logic:
            out_path = args.output if len(sentences) == 1 else f"{args.output}.{i}"
            try:
                with open(out_path, "w") as f:
                    for val in emb_list:
                        f.write(f"{val}\n")
                print(f"Full embedding written to: {out_path}")
            except Exception as e:
                print(f"Error writing to file {out_path}: {e}", file=sys.stderr)
        else:
            print(f"Full Embedding: {emb_list}")

if __name__ == "__main__":
    main()
