"""
Extended analysis on top of evaluate.py output:

  - per-method confusion matrices (normalised label set)
  - bootstrap 95% confidence intervals for detection F1 and macro-F1
  - detection-F1 threshold sensitivity from saved similarity matrices
  - cross-validated operating points, and the metrics each method reaches there
  - paired bootstrap intervals for a prespecified set of between-method contrasts

Writes data/evaluation/analysis.json. Run after src.evaluation.evaluate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.methods import keys as method_keys, with_matrix

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
EVAL_DIR = DATA / "evaluation"
MAP_DIR = DATA / "mappings"
REFERENCE = DATA / "baseline" / "reference_set.csv"

LABELS = ["EQUIVALENCE", "OVERLAP", "SUBSUMPTION", "COMPLEMENTARITY", "NO_RELATION"]

METHODS = method_keys()

# Similarity matrices available for threshold sweeps (rows = EN 303 645 order,
# cols = EN 304 223 order, matching the extracted provision JSON files).
SIM_MATRICES = {k: m.matrix for k, m in with_matrix().items()}

N_BOOT = 2000
N_REPEATS = 20
SEED = 42


def load_merged(method: str) -> pd.DataFrame:
    """Reference pairs with true and predicted (normalised) labels."""
    return pd.read_csv(EVAL_DIR / f"{method}_predictions_vs_gt.csv")


def detection_f1(df: pd.DataFrame) -> float:
    tp = ((df["true_rel"] != "NO_RELATION") & (df["pred_rel"] != "NO_RELATION")).sum()
    fp = ((df["true_rel"] == "NO_RELATION") & (df["pred_rel"] != "NO_RELATION")).sum()
    fn = ((df["true_rel"] != "NO_RELATION") & (df["pred_rel"] == "NO_RELATION")).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_f1(df: pd.DataFrame) -> float:
    scores = []
    for label in LABELS:
        tp = ((df["true_rel"] == label) & (df["pred_rel"] == label)).sum()
        fp = ((df["true_rel"] != label) & (df["pred_rel"] == label)).sum()
        fn = ((df["true_rel"] == label) & (df["pred_rel"] != label)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores))


def confusion_matrix(df: pd.DataFrame) -> dict:
    counts = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in df.itertuples():
        counts[r.true_rel][r.pred_rel] += 1
    return counts


def bootstrap_ci(df: pd.DataFrame, metric, rng) -> dict:
    n = len(df)
    stats = []
    for _ in range(N_BOOT):
        sample = df.iloc[rng.integers(0, n, n)]
        stats.append(metric(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": round(metric(df), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


def load_provision_ids() -> tuple[list[str], list[str]]:
    src = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    tgt = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    return [p["provision_id"] for p in src], [p["provision_id"] for p in tgt]


def pair_scores(sim: np.ndarray, gt: pd.DataFrame,
                src_ids: list[str], tgt_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    src_idx = {p: i for i, p in enumerate(src_ids)}
    tgt_idx = {p: j for j, p in enumerate(tgt_ids)}
    scores = np.array([sim[src_idx[r.src_id], tgt_idx[r.tgt_id]] for r in gt.itertuples()])
    positive = np.array([r.relationship != "NO_RELATION" for r in gt.itertuples()])
    return scores, positive


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Cut points taken as quantiles of a method's own scores.

    The methods occupy very different ranges, so a shared absolute grid would give
    one of them many usable cut points and another almost none.
    """
    return np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 200)))


def threshold_sweep(sim: np.ndarray, gt: pd.DataFrame,
                    src_ids: list[str], tgt_ids: list[str]) -> list[dict]:
    """Detection precision/recall/F1 across the method's own quantile grid."""
    scores, positive = pair_scores(sim, gt, src_ids, tgt_ids)

    rows = []
    for t in candidate_thresholds(scores):
        pred = scores >= t
        tp = int((pred & positive).sum())
        fp = int((pred & ~positive).sum())
        fn = int((~pred & positive).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"threshold": round(float(t), 4), "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f1, 4)})
    return rows


def _f1_at(scores: np.ndarray, positive: np.ndarray, t: float) -> float:
    pred = scores >= t
    tp = int((pred & positive).sum())
    fp = int((pred & ~positive).sum())
    fn = int((~pred & positive).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def metrics_at(scores: np.ndarray, positive: np.ndarray, t: float) -> dict:
    pred = scores >= t
    tp = int((pred & positive).sum())
    fp = int((pred & ~positive).sum())
    fn = int((~pred & positive).sum())
    tn = int((~pred & ~positive).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "predicted_positive": int(pred.sum()),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
    }


def cv_threshold(scores: np.ndarray, positive: np.ndarray) -> dict:
    """Held-out detection F1 when the threshold is tuned only on training folds.

    Unlike threshold_sweep(), the threshold never sees the fold it is scored on.
    The reported value is the median of the selections, not the in-sample optimum.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)

    held_out, chosen = [], []
    for train, test in cv.split(scores.reshape(-1, 1), positive):
        grid = candidate_thresholds(scores[train])
        best_t = max(grid, key=lambda t: _f1_at(scores[train], positive[train], t))
        held_out.append(_f1_at(scores[test], positive[test], best_t))
        chosen.append(float(best_t))

    threshold = float(np.median(chosen))
    in_sample = max(_f1_at(scores, positive, t) for t in candidate_thresholds(scores))
    return {
        "threshold": round(threshold, 4),
        "threshold_sd": round(float(np.std(chosen, ddof=1)), 4),
        "cv_f1_mean": round(float(np.mean(held_out)), 4),
        "cv_f1_sd": round(float(np.std(held_out, ddof=1)), 4),
        "in_sample_best_f1": round(float(in_sample), 4),
        "optimism": round(float(in_sample - np.mean(held_out)), 4),
        "n_splits": 5,
        "n_repeats": N_REPEATS,
        "calibrated": metrics_at(scores, positive, threshold),
        "calibrated_ci": calibrated_ci(scores, positive, threshold),
    }


def calibrated_ci(scores: np.ndarray, positive: np.ndarray, t: float) -> dict:
    """Bootstrap intervals for the metrics at the calibrated operating point.

    The threshold is held fixed across draws, so the interval is the uncertainty
    of a fixed operating point, not of the selection procedure.
    """
    rng = np.random.default_rng(SEED)
    n = len(scores)
    draws = {"precision": [], "recall": [], "f1": [], "specificity": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = metrics_at(scores[idx], positive[idx], t)
        for key in draws:
            draws[key].append(m[key])
    return {
        key: [round(float(lo), 4), round(float(hi), 4)]
        for key, vals in draws.items()
        for lo, hi in [np.percentile(vals, [2.5, 97.5])]
    }


def contrast_set(names: list[str], reference: str) -> list[tuple[str, str]]:
    """The comparisons that get reported, fixed by rule rather than exhaustive.

    Fifty-five uncorrected intervals would contain false findings, so only two
    contrasts are computed per method: against the all-positive baseline, and
    against the best-scoring method. The reference is the sample maximum, so
    differences against it are biased in its favour and are descriptive only.
    """
    pairs = [(n, "all_positive") for n in names]
    pairs += [(n, reference) for n in names if n != reference]
    return pairs


def paired_differences(positive: np.ndarray, preds: dict[str, np.ndarray],
                       rng, contrasts: list[tuple[str, str]]) -> dict:
    """Bootstrap CIs for between-method differences, both methods on the same draw.

    The methods are scored on identical pairs, so the paired variance is much
    smaller than the marginals and overlapping marginals prove nothing.
    """
    n = len(positive)
    names = list(preds)

    def rates(idx):
        a = positive[idx]
        out = {}
        for name in names:
            p = preds[name][idx]
            tp = int((p & a).sum())
            fp = int((p & ~a).sum())
            fn = int((~p & a).sum())
            tn = int((~p & ~a).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            out[name] = {
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
            }
        return out

    draws = [rates(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    point = rates(np.arange(n))

    out = {}
    for a, b in contrasts:
        entry = {}
        for metric in ("f1", "specificity"):
            diffs = [d[a][metric] - d[b][metric] for d in draws]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            entry[metric] = {
                "difference": round(point[a][metric] - point[b][metric], 4),
                "ci": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        out[f"{a}__vs__{b}"] = entry
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    gt = pd.read_csv(REFERENCE)
    src_ids, tgt_ids = load_provision_ids()

    analysis = {}
    for method in METHODS:
        path = EVAL_DIR / f"{method}_predictions_vs_gt.csv"
        if not path.exists():
            print(f"Skipping {method}: no predictions_vs_gt file")
            continue
        df = load_merged(method)
        entry = {
            "confusion_matrix": confusion_matrix(df),
            "bootstrap": {
                "detection_f1": bootstrap_ci(df, detection_f1, rng),
                "macro_f1": bootstrap_ci(df, macro_f1, rng),
            },
        }
        sim_path = SIM_MATRICES.get(method)
        if sim_path and sim_path.exists():
            sim = np.load(sim_path)
            entry["threshold_sweep"] = threshold_sweep(sim, gt, src_ids, tgt_ids)
            scores, positive = pair_scores(sim, gt, src_ids, tgt_ids)
            entry["cv_threshold"] = cv_threshold(scores, positive)
        analysis[method] = entry
        b = entry["bootstrap"]
        print(f"{method:<20} det-F1 {b['detection_f1']['point']:.3f} "
              f"[{b['detection_f1']['ci_low']:.3f}, {b['detection_f1']['ci_high']:.3f}]   "
              f"macro-F1 {b['macro_f1']['point']:.3f} "
              f"[{b['macro_f1']['ci_low']:.3f}, {b['macro_f1']['ci_high']:.3f}]")

    print(f"\n{'method':<20}{'threshold':>11}{'in-sample':>11}{'CV held-out':>14}{'optimism':>10}")
    for method, entry in analysis.items():
        cv = entry.get("cv_threshold")
        if cv:
            print(f"{method:<20}{cv['threshold']:>11.3f}{cv['in_sample_best_f1']:>11.3f}"
                  f"{cv['cv_f1_mean']:>9.3f} ±{cv['cv_f1_sd']:.3f}"
                  f"{cv['optimism']:>10.3f}")

    # Preregistered contrasts, computed at each method's calibrated operating point.
    positive = (gt.relationship != "NO_RELATION").to_numpy()
    preds = {"all_positive": np.ones(len(gt), dtype=bool)}
    for method, entry in analysis.items():
        cv = entry.get("cv_threshold")
        sim_path = SIM_MATRICES.get(method)
        if cv and sim_path and sim_path.exists():
            scores, _ = pair_scores(np.load(sim_path), gt, src_ids, tgt_ids)
            preds[method] = scores >= cv["threshold"]

    scored = [n for n in preds if n != "all_positive"]
    if scored:
        best = max(scored, key=lambda n: analysis[n]["cv_threshold"]["calibrated"]["f1"])
        contrasts = contrast_set(scored, best)
        analysis["_contrasts"] = {
            "reference": best,
            "rule": "every method vs. the all-positive baseline, and vs. the best method",
            "results": paired_differences(positive, preds, rng, contrasts),
        }
        print(f"\nContrast reference (highest calibrated F1): {best}")
        print(f"{len(contrasts)} prespecified contrasts, {N_BOOT} paired bootstrap draws")

    (EVAL_DIR / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"Saved {EVAL_DIR / 'analysis.json'}")
