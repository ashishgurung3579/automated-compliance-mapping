"""
SecureBERT (ehsanaghaei/SecureBERT) compliance mapping.
RoBERTa-based model pre-trained on cybersecurity text.
Downloads ~500 MB from HuggingFace on first run.

Kept as an entry point; the work is in src.mapping.run_embedding, which runs
every embedding method from its entry in src.methods.
"""
from src.mapping.run_embedding import run

if __name__ == "__main__":
    run("securebert")
