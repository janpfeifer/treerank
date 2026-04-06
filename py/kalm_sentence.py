import argparse
import sys
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for sentences using local HuggingFace transformers.")
    parser.add_argument(
        "--model",
        type=str,
        default="tencent/KaLM-Embedding-Gemma3-12B-2511",
        help="The Hugging Face model ID (default: tencent/KaLM-Embedding-Gemma3-12B-2511)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="File path to write the embedding space (for first layer output)."
    )
    parser.add_argument(
        "sentences",
        nargs="*",
        default=["Hello World"],
        help="Sentences to embed."
    )

    args = parser.parse_args()

    # Make sure we use the local transformers
    transformers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../transformers/src'))
    if transformers_path not in sys.path:
        sys.path.insert(0, transformers_path)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer: {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print(f"Loading model: {args.model}...", file=sys.stderr)
    start_time = time.time()
    # We use torch_dtype="auto" and device_map="auto" to handle the 12B model gracefully if possible.
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.", file=sys.stderr)
    
    sentences = args.sentences
    if not sentences:
        if sys.stdin.isatty():
            print("Enter sentences (one per line, Ctrl-D to finish):", file=sys.stderr)
        sentences = [line.strip() for line in sys.stdin if line.strip()]

    print(f"Tokenizing {len(sentences)} sentences...", file=sys.stderr)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)
    
    print("Tokens:", inputs["input_ids"], file=sys.stderr)

    print("Running forward pass (output_hidden_states=True)...", file=sys.stderr)
    
    debug_dict = {}
    def hook_fn(name):
        def _hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            debug_dict[name] = output.detach().cpu().reshape(-1).tolist()[:5]
        return _hook

    layer0 = model.layers[0]
    layer0.input_layernorm.register_forward_hook(hook_fn("input_norm"))
    layer0.self_attn.q_proj.register_forward_hook(hook_fn("q_proj"))
    layer0.self_attn.k_proj.register_forward_hook(hook_fn("k_proj"))
    layer0.self_attn.v_proj.register_forward_hook(hook_fn("v_proj"))
    if hasattr(layer0.self_attn, 'q_norm'):
        layer0.self_attn.q_norm.register_forward_hook(hook_fn("q_norm"))
    if hasattr(layer0.self_attn, 'k_norm'):
        layer0.self_attn.k_norm.register_forward_hook(hook_fn("k_norm"))
    layer0.self_attn.o_proj.register_forward_hook(hook_fn("o_proj"))
    layer0.post_attention_layernorm.register_forward_hook(hook_fn("post_attn_norm"))
    layer0.mlp.gate_proj.register_forward_hook(hook_fn("gate_proj"))
    layer0.mlp.up_proj.register_forward_hook(hook_fn("up_proj"))
    layer0.mlp.down_proj.register_forward_hook(hook_fn("down_proj"))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    print("\n--- DEBUG LAYER 0 ---")
    for key, val in debug_dict.items():
        print(f"Layer 0 {key}: {val}")
    print("---------------------\n")

    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        for i, orig in enumerate(sentences):
            if args.output:
                out_path = args.output if len(sentences) == 1 else f"{args.output}.{i}"
                try:
                    with open(out_path, "w") as f:
                        f.write(f"# Tokens: {inputs['input_ids'][i].tolist()}\n")
                        f.write(f"# Shape: {outputs.hidden_states[0][i].shape}\n")
                        for layer_idx, layer_out in enumerate(outputs.hidden_states):
                            if layer_idx == 0:
                                f.write("# Token Embeddings\n")
                            else:
                                f.write(f"# Layer {layer_idx} Output\n")
                            
                            layer_vals = layer_out[i].view(-1).tolist()
                            for val in layer_vals:
                                f.write(f"{val}\n")
                    print(f"Full layer embeddings written to: {out_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Error writing to file {out_path}: {e}", file=sys.stderr)
            else:
                print(f"\nSentence {i+1}: {orig}")
                print(f"Token embeddings (first 5 flat values): {outputs.hidden_states[0][i].view(-1).tolist()[:5]}")
                if len(outputs.hidden_states) > 1:
                    print(f"First layer output (first 5 flat values): {outputs.hidden_states[1][i].view(-1).tolist()[:5]}")
    else:
        print("Model did not return hidden_states. You might need to change `transformers/models/gemma3/modeling_gemma3.py` to return them.", file=sys.stderr)

if __name__ == "__main__":
    main()
