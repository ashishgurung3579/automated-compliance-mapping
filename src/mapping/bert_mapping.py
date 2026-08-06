"""
BERT (bert-base-uncased) compliance mapping using mean-pooled embeddings.

Kept as an entry point; the work is in src.mapping.run_embedding, which runs
every embedding method from its entry in src.methods.
"""
from src.mapping.run_embedding import run

if __name__ == "__main__":
    run("bert")
