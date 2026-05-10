"""
BERT (bert-base-uncased) compliance mapping using mean-pooled embeddings.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.mapping.embeddings import embed_with_transformers, cosine_sim_matrix, score_to_relation_semantic

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
MODEL = "bert-base-uncased"
THRESHOLD = 0.70


def load_provisions() -> tuple[list[dict], list[dict]]:
    p1 = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    p2 = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    return p1, p2


if __name__ == "__main__":
    src, tgt = load_provisions()
    print(f"Mapping {len(src)} EN 303 645 provisions to {len(tgt)} EN 304 223 provisions")

    src_texts = [p["text"] for p in src]
    tgt_texts = [p["text"] for p in tgt]

    print(f"Encoding with {MODEL}...")
    src_emb = embed_with_transformers(MODEL, src_texts)
    tgt_emb = embed_with_transformers(MODEL, tgt_texts)

    sim = cosine_sim_matrix(src_emb, tgt_emb)

    out_dir = DATA / "mappings"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            score = float(sim[i, j])
            if score >= THRESHOLD:
                rows.append({
                    "src_id": sp["provision_id"],
                    "tgt_id": tp["provision_id"],
                    "bert_score": round(score, 4),
                    "predicted_rel": score_to_relation_semantic(score),
                })

    df = pd.DataFrame(rows).sort_values("bert_score", ascending=False)
    df.to_csv(out_dir / "bert_output.csv", index=False)
    print(f"Candidates (threshold={THRESHOLD}): {len(df)}")
    print(df["predicted_rel"].value_counts().to_dict())
    print(f"Saved: {out_dir / 'bert_output.csv'}")
