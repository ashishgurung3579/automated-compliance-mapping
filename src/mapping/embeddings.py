"""
Shared embedding utilities for BERT-family models and SBERT.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


def embed_with_transformers(
    model_name: str,
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 512,
    use_safetensors: bool | None = None,
) -> np.ndarray:
    """Mean-pool last hidden state, L2-normalise. Works for bert-base-uncased and SecureBERT."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=use_safetensors)
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_emb = output.last_hidden_state
        summed = (token_emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = (summed / counts).cpu().numpy()
        all_embeddings.append(embeddings)

    emb = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-9)
    return emb


def embed_with_sbert(model_name: str, texts: list[str]) -> np.ndarray:
    """Wrapper around SentenceTransformer.encode(). Returns L2-normalised embeddings."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast cosine similarity for already-normalised embeddings: a @ b.T"""
    return a @ b.T


def score_to_relation_semantic(score: float) -> str:
    """Semantic similarity thresholds for dense embedding models."""
    if score >= 0.85:
        return "EQUIVALENCE"
    if score >= 0.70:
        return "OVERLAP"
    if score >= 0.50:
        return "COMPLEMENTARITY"
    return "NO_RELATION"
