"""
Ranking metrics for the shortlisting task: for each EN 303 645 provision, how well
does a method rank its EN 304 223 counterparts?

Unlike precision and F1, these do not move with the positive rate of the reference
set, so they stay meaningful even though related pairs are rare in the corpus.

Writes data/evaluation/ranking.json. Run after src.evaluation.evaluate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.analysis import SIM_MATRICES, load_provision_ids

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
EVAL_DIR = DATA / "evaluation"

K_VALUES = [1, 3, 5, 10]


def relevance_lookup(gt: pd.DataFrame) -> dict[tuple[str, str], bool]:
    return {
        (r.src_id, r.tgt_id): r.relationship != "NO_RELATION"
        for r in gt.itertuples()
    }


def dcg(gains: list[int]) -> float:
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def rank_one_provision(scores: np.ndarray, tgt_ids: list[str], src_id: str,
                       relevance: dict) -> dict | None:
    """Metrics for a single source provision, or None if it has no known relations."""
    order = np.argsort(-scores)
    ranked = [tgt_ids[j] for j in order]

    # Pairs absent from the reference set are unjudged. Following standard pooled
    # test-collection practice they count as non-relevant, and judged_at_k below
    # records how much of the ranking that assumption actually covers.
    gains = [1 if relevance.get((src_id, t), False) else 0 for t in ranked]
    judged = [1 if (src_id, t) in relevance else 0 for t in ranked]
    total_relevant = sum(gains)
    if total_relevant == 0:
        return None

    out = {"n_relevant": total_relevant}
    for k in K_VALUES:
        hits = sum(gains[:k])
        out[f"p_at_{k}"] = hits / k
        out[f"judged_at_{k}"] = sum(judged[:k]) / k
    out["r_at_10"] = sum(gains[:10]) / total_relevant

    hit_positions = [i for i, g in enumerate(gains) if g]
    out["ap"] = float(np.mean([
        sum(gains[: i + 1]) / (i + 1) for i in hit_positions
    ]))

    ideal = sorted(gains, reverse=True)[:10]
    denom = dcg(ideal)
    out["ndcg_at_10"] = dcg(gains[:10]) / denom if denom else 0.0
    return out


def evaluate_method(sim: np.ndarray, src_ids: list[str], tgt_ids: list[str],
                    relevance: dict) -> dict:
    per_provision = [
        m for i, src_id in enumerate(src_ids)
        if (m := rank_one_provision(sim[i], tgt_ids, src_id, relevance)) is not None
    ]

    summary = {"n_provisions_scored": len(per_provision)}
    keys = [f"p_at_{k}" for k in K_VALUES] + [f"judged_at_{k}" for k in K_VALUES]
    for key in keys + ["r_at_10", "ap", "ndcg_at_10"]:
        summary[key] = round(float(np.mean([m[key] for m in per_provision])), 4)
    summary["map"] = summary.pop("ap")

    # Diagnostic, not a result. When this sits at 1.0 the reference set judged almost
    # nothing that a method ranked highly and got wrong, so P@k is tracking judging
    # coverage rather than ranking quality and must not be read as method performance.
    summary["precision_among_judged_at_10"] = round(
        summary["p_at_10"] / summary["judged_at_10"], 4
    ) if summary["judged_at_10"] else None
    return summary


if __name__ == "__main__":
    gt = pd.read_csv(DATA / "baseline" / "gt.csv")
    src_ids, tgt_ids = load_provision_ids()
    relevance = relevance_lookup(gt)

    results = {}
    for method, path in SIM_MATRICES.items():
        if not path.exists():
            print(f"Skipping {method}: {path.name} not found")
            continue
        results[method] = evaluate_method(np.load(path), src_ids, tgt_ids, relevance)

    header = f"{'method':<20}{'P@1':>7}{'P@5':>7}{'P@10':>7}{'R@10':>7}{'MAP':>7}{'nDCG@10':>9}{'judged@10':>11}"
    print(header)
    for method, r in results.items():
        print(f"{method:<20}{r['p_at_1']:>7.3f}{r['p_at_5']:>7.3f}{r['p_at_10']:>7.3f}"
              f"{r['r_at_10']:>7.3f}{r['map']:>7.3f}{r['ndcg_at_10']:>9.3f}"
              f"{r['judged_at_10']:>11.3f}")

    (EVAL_DIR / "ranking.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {EVAL_DIR / 'ranking.json'}")
