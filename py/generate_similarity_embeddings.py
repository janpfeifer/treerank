import subprocess
import os
import sys

queries = [
    "What is the capital of China?",
    "Explain gravity",
]
docs = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

out_file = "pkg/kalmgemma3/similarity_embeddings.txt"

# Make sure it's run from the project root
if not os.path.exists("py/embed_sentence.py"):
    sys.exit("Please run from project root")

with open(out_file, "w") as f_out:
    for i, q in enumerate(queries):
        print(f"Embedding query {i+1}...")
        f_out.write(f"# Query {i+1}\n")
        subprocess.run(["python", "py/embed_sentence.py", "--query", "--output", f"{out_file}.tmp", q], check=True)
        with open(f"{out_file}.tmp", "r") as f_in:
            f_out.write(f_in.read())
            
    for i, d in enumerate(docs):
        print(f"Embedding doc {i+1}...")
        f_out.write(f"# Doc {i+1}\n")
        subprocess.run(["python", "py/embed_sentence.py", "--output", f"{out_file}.tmp", d], check=True)
        with open(f"{out_file}.tmp", "r") as f_in:
            f_out.write(f_in.read())

if os.path.exists(f"{out_file}.tmp"):
    os.remove(f"{out_file}.tmp")

print(f"Done! Embeddings saved to {out_file}")
