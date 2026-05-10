"""
Gemini Embedding API compliance mapping.
Uses the synchronous Gemini embedding API by default for this small dataset.
Set GEMINI_EMBEDDING_MODE=batch to use managed Batch API jobs instead.
Requires GEMINI_API_KEY env var.
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from google import genai

from src.mapping.gemini_batch import (
    embedding_values,
    inline_embedding_responses,
    response_embeddings,
    wait_for_batch,
)
from src.mapping.embeddings import cosine_sim_matrix, score_to_relation_semantic

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
MODEL = "gemini-embedding-2"
THRESHOLD = 0.50
BATCH_CONFIG = {"task_type": "SEMANTIC_SIMILARITY"}
EMBEDDING_MODE = os.environ.get("GEMINI_EMBEDDING_MODE", "sync").strip().lower()
SYNC_RETRY_SECONDS = int(os.environ.get("GEMINI_EMBEDDING_RETRY_SECONDS", "10"))
SYNC_MAX_RETRIES = int(os.environ.get("GEMINI_EMBEDDING_MAX_RETRIES", "3"))


def _normalise_vectors(vectors: list[list[float]], label: str, expected_count: int) -> np.ndarray:
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} embeddings for {label}, got {len(vectors)}"
        )

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


def embed_all_sync(client, texts: list[str], label: str = "") -> np.ndarray:
    print(f"Embedding {label} provisions synchronously ({len(texts)} texts)")
    vectors = []
    for index, text in enumerate(texts, start=1):
        for attempt in range(1, SYNC_MAX_RETRIES + 1):
            try:
                response = client.models.embed_content(
                    model=MODEL,
                    contents=text,
                    config=BATCH_CONFIG,
                )
                embeddings = response_embeddings(response)
                if len(embeddings) != 1:
                    raise RuntimeError(
                        f"Expected one embedding for {label} item {index}, "
                        f"got {len(embeddings)}"
                    )
                vectors.append(embedding_values(embeddings[0]))
                break
            except Exception:
                if attempt == SYNC_MAX_RETRIES:
                    raise
                print(
                    f"  Gemini embedding request failed for {label} item {index}; "
                    f"retrying in {SYNC_RETRY_SECONDS}s ({attempt}/{SYNC_MAX_RETRIES})"
                )
                time.sleep(SYNC_RETRY_SECONDS)
        if index == len(texts) or index % 25 == 0:
            print(f"  Embedded {index}/{len(texts)} {label} provisions")

    return _normalise_vectors(vectors, label, len(texts))


def embed_all_batch(client, texts: list[str], label: str = "") -> np.ndarray:
    print(f"Submitting {label} embedding batch ({len(texts)} texts)")
    job = client.batches.create_embeddings(
        model=MODEL,
        src={
            "inlined_requests": {
                "contents": texts,
                "config": BATCH_CONFIG,
            }
        },
        config={"display_name": f"compliance-{label}-embeddings"},
    )
    job = wait_for_batch(client, job)

    vectors = []
    for inline_response in inline_embedding_responses(job):
        if getattr(inline_response, "error", None):
            raise RuntimeError(f"Embedding batch item failed: {inline_response.error}")
        response = inline_response.response
        for embedding in response_embeddings(response):
            vectors.append(embedding_values(embedding))

    return _normalise_vectors(vectors, label, len(texts))


def embed_all(client, texts: list[str], label: str = "") -> np.ndarray:
    if EMBEDDING_MODE == "batch":
        return embed_all_batch(client, texts, label)
    if EMBEDDING_MODE not in {"sync", "direct"}:
        raise ValueError(
            "GEMINI_EMBEDDING_MODE must be 'sync', 'direct', or 'batch' "
            f"(got {EMBEDDING_MODE!r})"
        )
    return embed_all_sync(client, texts, label)


def load_provisions() -> tuple[list[dict], list[dict]]:
    p1 = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    p2 = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    return p1, p2


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY env var before running")

    client = genai.Client(api_key=api_key)

    src, tgt = load_provisions()
    print(f"Embedding {len(src)} src + {len(tgt)} tgt provisions with Gemini API")
    print(f"Model: {MODEL}")
    print(f"Mode: {EMBEDDING_MODE}")

    src_texts = [p["text"] for p in src]
    tgt_texts = [p["text"] for p in tgt]

    print("Embedding src provisions...")
    src_emb = embed_all(client, src_texts, label="src")
    print("Embedding tgt provisions...")
    tgt_emb = embed_all(client, tgt_texts, label="tgt")

    sim = cosine_sim_matrix(src_emb, tgt_emb)

    out_dir = DATA / "mappings"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "gemini_embedding_similarity_matrix.npy", sim)

    rows = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            score = float(sim[i, j])
            if score >= THRESHOLD:
                rows.append({
                    "src_id": sp["provision_id"],
                    "tgt_id": tp["provision_id"],
                    "gemini_embed_score": round(score, 4),
                    "predicted_rel": score_to_relation_semantic(score),
                })

    df = pd.DataFrame(rows).sort_values("gemini_embed_score", ascending=False)
    df.to_csv(out_dir / "gemini_embedding_output.csv", index=False)
    print(f"Candidates (threshold={THRESHOLD}): {len(df)}")
    print(df["predicted_rel"].value_counts().to_dict())
    print(f"Saved: {out_dir / 'gemini_embedding_output.csv'}")


if __name__ == "__main__":
    main()
