from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    trust_remote_code=True,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        # "attn_implementation": "flash_attention_2",  # Optional
    },
)
model.max_seq_length = 512  

queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)

similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
