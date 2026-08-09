"""
Draw a screened, score-banded set of provision pairs for annotation.

The pilot reference set in gt.csv was assembled purposively, so it concentrates on
pairs that already looked related. This module widens the annotation pool: pairs are
ranked by mean similarity across every method and drawn from three bands of that
ranking, so the material judged spans the likely-related and the plainly-unrelated
regions of the space instead of only the top of it.

The band sizes below carry no weighting role in the current evaluation -- they only
control how many pairs each region contributes. Provenance for the labels this
produced, retained so the reference set can be traced back.

Writes data/baseline/reference_pool.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.analysis import SIM_MATRICES, load_provision_ids

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
OUT = DATA / "baseline" / "reference_pool.csv"

SEED = 20260804

# Fractions of the ranked candidate space, and how many pairs to draw from each.
# Allocation is deliberately disproportionate: relations concentrate at the top of the
# ranking, so most of the annotation budget goes there. Every band still contributes,
# which is what keeps hard negatives in the pool.
STRATA = [
    ("H", 0.10, 250),
    ("M", 0.30, 200),
    ("L", 0.60, 200),
]


def percentile_ranks(matrix: np.ndarray) -> np.ndarray:
    # Average ranking for ties: the TF-IDF matrix is mostly zeros, and breaking those
    # ties by index order would let array layout rather than similarity drive the strata.
    ranks = pd.Series(matrix.ravel()).rank(method="average", pct=True).to_numpy()
    return ranks.reshape(matrix.shape)


def screening_scores(src_ids: list[str], tgt_ids: list[str]) -> pd.DataFrame:
    """Mean percentile rank across every similarity method.

    Averaging keeps the stratification method-agnostic; stratifying on a single
    method's scores would sample densely exactly where that method is confident and
    quietly favour it in the evaluation.
    """
    ranked = []
    for method, path in sorted(SIM_MATRICES.items()):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run the mapping stages first")
        ranked.append(percentile_ranks(np.load(path)))
    mean_rank = np.mean(ranked, axis=0)

    rows = [
        {"src_id": s, "tgt_id": t, "screening_score": float(mean_rank[i, j])}
        for i, s in enumerate(src_ids)
        for j, t in enumerate(tgt_ids)
    ]
    return pd.DataFrame(rows)


def assign_strata(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.sort_values("screening_score", ascending=False).reset_index(drop=True)
    total = len(pool)

    labels = np.empty(total, dtype=object)
    start = 0
    for i, (name, fraction, _) in enumerate(STRATA):
        end = total if i == len(STRATA) - 1 else start + round(fraction * total)
        labels[start:end] = name
        start = end
    pool["stratum"] = labels
    return pool


def draw(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    picked = []
    for name, _, n in STRATA:
        block = pool[pool.stratum == name]
        take = rng.choice(block.index.to_numpy(), size=n, replace=False)
        chosen = block.loc[sorted(take)].copy()
        chosen["N_stratum"] = len(block)
        chosen["n_stratum"] = n
        chosen["weight"] = len(block) / n
        picked.append(chosen)
    return pd.concat(picked, ignore_index=True)


if __name__ == "__main__":
    src_ids, tgt_ids = load_provision_ids()
    pool = assign_strata(screening_scores(src_ids, tgt_ids))
    sample = draw(pool)

    pilot = pd.read_csv(DATA / "baseline" / "gt.csv")
    pilot_pairs = set(zip(pilot.src_id, pilot.tgt_id))
    sample["in_pilot"] = [
        (r.src_id, r.tgt_id) in pilot_pairs for r in sample.itertuples()
    ]

    sample = sample[[
        "src_id", "tgt_id", "stratum", "screening_score",
        "N_stratum", "n_stratum", "weight", "in_pilot",
    ]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)

    print(f"Candidate space: {len(pool)} pairs")
    for name, _, _ in STRATA:
        block = pool[pool.stratum == name]
        drawn = sample[sample.stratum == name]
        print(f"  stratum {name}: N={len(block):>5}  n={len(drawn):>4}  "
              f"p={len(drawn)/len(block):.3f}  weight={drawn.weight.iloc[0]:.2f}")
    print(f"Sampled {len(sample)} pairs; weights sum to {sample.weight.sum():.0f} "
          f"(candidate space is {len(pool)})")
    print(f"Overlap with pilot set: {int(sample.in_pilot.sum())} pairs")
    print(f"Saved {OUT}")
