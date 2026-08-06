"""
Run one embedding method over the full provision pair space.

Every embedding method does the same three things -- encode both standards,
take the cosine similarity of every pair, and keep the pairs above the family
threshold -- so they share this runner and differ only by their entry in
src.methods. The similarity matrix is written whole, before thresholding, since
the threshold sensitivity analysis and the corpus recalibration both need the
scores of pairs no threshold kept.

    python3 -m src.mapping.run_embedding sbert
    python3 -m src.mapping.run_embedding --all
"""
import sys
import time

import numpy as np
import pandas as pd

from src.mapping.embeddings import (cosine_sim_matrix, embed_with_sbert,
                                    embed_with_transformers, load_provisions,
                                    score_to_relation_semantic)
from src.methods import METHODS, embedding_methods


def encode(method, texts: list[str]) -> np.ndarray:
    if method.encoder == "sbert":
        return embed_with_sbert(method.model_id, texts)
    return embed_with_transformers(method.model_id, texts,
                                   use_safetensors=method.use_safetensors)


def run(key: str) -> float:
    method = METHODS[key]
    src, tgt = load_provisions()
    print(f"=== {method.long_label}")
    print(f"Mapping {len(src)} EN 303 645 provisions to {len(tgt)} EN 304 223 "
          f"provisions ({len(src) * len(tgt)} pairs)")

    started = time.perf_counter()
    src_emb = encode(method, [p["text"] for p in src])
    tgt_emb = encode(method, [p["text"] for p in tgt])
    sim = cosine_sim_matrix(src_emb, tgt_emb)
    elapsed = time.perf_counter() - started

    method.matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(method.matrix, sim)

    rows = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            score = float(sim[i, j])
            if score >= method.threshold:
                rows.append({
                    "src_id": sp["provision_id"],
                    "tgt_id": tp["provision_id"],
                    method.score_column: round(score, 4),
                    "predicted_rel": score_to_relation_semantic(score),
                })

    df = pd.DataFrame(rows).sort_values(method.score_column, ascending=False)
    df.to_csv(method.predictions, index=False)
    print(f"Candidates (threshold={method.threshold}): {len(df)}")
    print(df["predicted_rel"].value_counts().to_dict())
    print(f"Saved: {method.predictions}  [{elapsed:.1f} s]\n")
    return elapsed


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(f"usage: run_embedding <key|--all>\n"
                         f"keys: {', '.join(embedding_methods())}")

    keys = list(embedding_methods()) if args[0] == "--all" else args
    timings = {k: run(k) for k in keys}
    if len(timings) > 1:
        print("wall-clock seconds")
        for k, t in timings.items():
            print(f"  {k:<14}{t:>8.1f}")
