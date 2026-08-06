"""
Corpus-level detection metrics estimated from the stratified sample.

The pilot reference set has no inclusion probabilities, so its precision and F1 can
only describe the annotated pairs themselves. Here each sampled pair carries the
weight N_stratum/n_stratum, which lets Horvitz-Thompson estimation recover what the
same method would score over all 4,968 candidate pairs.

Two operating points are reported for every method with a continuous score. The
shipped one comes from a threshold tuned on the pilot set, where 79% of pairs are
related; scoring it here, where 15% are, mixes the method's behaviour with an
operating point chosen at the wrong base rate. The corpus-calibrated one selects the
threshold by weighted cross-validation on this sample instead, so the comparison
between methods is at least made at comparable operating points.

Writes data/evaluation/weighted_metrics.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.methods import METHODS, with_matrix

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
EVAL_DIR = DATA / "evaluation"
MAP_DIR = DATA / "mappings"

N_BOOT = 2000
N_REPEATS = 20
SEED = 42
CANDIDATE_SPACE = 4968

# Rows follow EN 303 645 provision order, columns EN 304 223, matching the
# extracted JSON files. Same matrices analysis.py sweeps on the pilot set.
SIM_MATRICES = {k: m.matrix for k, m in with_matrix().items()}

PREDICTION_FILES = {k: m.predictions for k, m in METHODS.items()}


def load_predictions(path: Path) -> set[tuple[str, str]]:
    """Pairs a method calls related. Absent pairs count as NO_RELATION."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "predicted_rel" in df.columns:
        df = df[df.predicted_rel != "NO_RELATION"]
    return set(zip(df.src_id, df.tgt_id))


def confusion(sample: pd.DataFrame, predicted: set) -> pd.DataFrame:
    out = sample.copy()
    out["actual"] = (out.relationship != "NO_RELATION").astype(int)
    out["pred"] = [
        int((r.src_id, r.tgt_id) in predicted) for r in out.itertuples()
    ]
    return out


def weighted_rates(df: pd.DataFrame) -> dict:
    w = df.weight
    tp = (w * (df.pred & df.actual)).sum()
    fp = (w * (df.pred & ~df.actual.astype(bool))).sum()
    fn = (w * (~df.pred.astype(bool) & df.actual)).sum()
    tn = (w * (~df.pred.astype(bool) & ~df.actual.astype(bool))).sum()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
        "est_true_positives": tp, "est_false_positives": fp,
    }


def stratified_bootstrap(df: pd.DataFrame, fn, rng) -> dict:
    """Resample within stratum, since the strata have different sampling fractions."""
    groups = [g for _, g in df.groupby("stratum")]
    stats = {k: [] for k in fn(df)}
    for _ in range(N_BOOT):
        draw = pd.concat([
            g.iloc[rng.integers(0, len(g), len(g))] for g in groups
        ])
        for k, v in fn(draw).items():
            stats[k].append(v)
    return {
        k: [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]
        for k, v in stats.items()
    }


def prevalence(sample: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    strict = {"EQUIVALENCE", "OVERLAP", "SUBSUMPTION_A_BROADER", "SUBSUMPTION_B_BROADER"}
    out = {}
    for name, mask in [
        ("any_relationship", sample.relationship != "NO_RELATION"),
        ("substantive_only", sample.relationship.isin(strict)),
    ]:
        d = sample.assign(flag=mask.astype(int))
        point = (d.flag * d.weight).sum() / d.weight.sum()
        ci = stratified_bootstrap(
            d, lambda x: {"p": (x.flag * x.weight).sum() / x.weight.sum()}, rng)["p"]
        out[name] = {
            "prevalence": round(float(point), 4),
            "ci": ci,
            "implied_pairs": round(float(point) * CANDIDATE_SPACE),
        }
    return out


def load_provision_ids() -> tuple[list[str], list[str]]:
    src = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    tgt = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    return [p["provision_id"] for p in src], [p["provision_id"] for p in tgt]


def sample_scores(sim: np.ndarray, sample: pd.DataFrame,
                  src_ids: list[str], tgt_ids: list[str]) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(src_ids)}
    tgt_idx = {p: j for j, p in enumerate(tgt_ids)}
    return np.array([sim[src_idx[r.src_id], tgt_idx[r.tgt_id]] for r in sample.itertuples()])


def weighted_f1_at(scores, actual, weight, t: float) -> float:
    pred = scores >= t
    tp = weight[pred & actual].sum()
    fp = weight[pred & ~actual].sum()
    fn = weight[~pred & actual].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def corpus_threshold(scores: np.ndarray, sample: pd.DataFrame) -> dict:
    """Operating point selected on this sample rather than on the pilot set.

    Folds are stratified by sampling stratum so each one carries all three weight
    classes; stratifying on the label as well would leave fewer than five positives
    in L to split. The reported threshold is the median of the selected values, not
    the in-sample optimum, so the point estimate that follows is not tuned on the
    pairs it scores.

    The candidate thresholds are quantiles of the scores rather than a fixed absolute
    grid. The methods occupy very different ranges --- TF-IDF tops out at 0.14 while
    SecureBERT never drops below 0.91 --- so a shared absolute grid would give some
    methods a hundred candidate cut points and others three.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    actual = (sample.relationship != "NO_RELATION").to_numpy()
    weight = sample.weight.to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)

    def candidates(s):
        return np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))

    held_out, chosen = [], []
    for train, test in cv.split(scores.reshape(-1, 1), sample.stratum):
        grid = candidates(scores[train])
        best = max(grid, key=lambda t: weighted_f1_at(
            scores[train], actual[train], weight[train], t))
        held_out.append(weighted_f1_at(scores[test], actual[test], weight[test], best))
        chosen.append(float(best))

    in_sample = max(weighted_f1_at(scores, actual, weight, t)
                    for t in candidates(scores))
    return {
        "threshold": round(float(np.median(chosen)), 4),
        "threshold_sd": round(float(np.std(chosen, ddof=1)), 4),
        "cv_weighted_f1_mean": round(float(np.mean(held_out)), 4),
        "cv_weighted_f1_sd": round(float(np.std(held_out, ddof=1)), 4),
        "in_sample_best_weighted_f1": round(float(in_sample), 4),
        "optimism": round(float(in_sample - np.mean(held_out)), 4),
        "n_splits": 5,
        "n_repeats": N_REPEATS,
    }


def design_effect(sample: pd.DataFrame) -> dict:
    """Kish's deff. Weights range 1.99 to 14.91, so n does not act like n."""
    w = sample.weight.to_numpy()
    n = len(w)
    deff = n * (w ** 2).sum() / (w.sum() ** 2)
    return {
        "n": n,
        "kish_deff": round(float(deff), 3),
        "n_effective": round(float(n / deff), 1),
        "weight_range": [round(float(w.min()), 2), round(float(w.max()), 2)],
    }


def contrast_set(preds: dict[str, np.ndarray], reference: str) -> list[tuple[str, str]]:
    """The comparisons that get reported, fixed by rule rather than exhaustive.

    All pairs of eleven methods and their recalibrated variants is over two
    hundred intervals, and reporting two hundred uncorrected 95% intervals
    guarantees false findings among them. Two questions carry the argument --
    does a method beat the all-positive baseline, and does anything beat the best
    method -- so those are the contrasts computed.

    The rule is fixed, but the reference is not: it is whichever method scores
    highest on this sample. Differences measured against a selected maximum are
    biased in that maximum's favour, because the selection absorbs some of the
    sampling noise. The intervals here are therefore read as descriptive rather
    than as a test that the reference is best.
    """
    names = [n for n in preds if n != "all_positive"]
    pairs = [(n, "all_positive") for n in names]
    pairs += [(n, reference) for n in names if n != reference]
    return pairs


def paired_differences(sample: pd.DataFrame, preds: dict[str, np.ndarray],
                       rng, contrasts: list[tuple[str, str]]) -> dict:
    """Bootstrap CIs for between-method differences, both methods on the same draw.

    Comparing two marginal intervals for overlap is not a test of the difference:
    the methods are scored on identical pairs, so the paired variance is much
    smaller than the marginals suggest and a ranking can be significant while the
    intervals overlap.
    """
    actual = (sample.relationship != "NO_RELATION").to_numpy()
    weight = sample.weight.to_numpy()
    strata = [np.where(sample.stratum.to_numpy() == s)[0]
              for s in sorted(sample.stratum.unique())]
    names = list(preds)

    def rates(idx):
        a, w = actual[idx], weight[idx]
        out = {}
        for name in names:
            p = preds[name][idx]
            tp, fp = w[p & a].sum(), w[p & ~a].sum()
            fn, tn = w[~p & a].sum(), w[~p & ~a].sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            out[name] = {
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
            }
        return out

    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in strata])
        draws.append(rates(idx))

    point = rates(np.arange(len(sample)))
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
    sample = pd.read_csv(DATA / "baseline" / "gt_sample.csv")
    sample = sample[sample.stratum.notna()]
    rng = np.random.default_rng(SEED)

    results = {
        "n_annotated": int(len(sample)),
        "by_stratum": sample.stratum.value_counts().to_dict(),
        "design_effect": design_effect(sample),
        "confidence_levels": {
            str(int(k)): int(v) for k, v in sample.confidence.value_counts().items()
        },
        "prevalence": prevalence(sample),
        "methods": {},
    }

    d = results["design_effect"]
    print(f"Design effect {d['kish_deff']:.2f} -> effective n = {d['n_effective']:.0f} "
          f"of {d['n']} annotated")

    print(f"Sampled pairs annotated: {len(sample)}")
    for name, p in results["prevalence"].items():
        print(f"  prevalence ({name}): {p['prevalence']:.3f} {p['ci']} "
              f"-> ~{p['implied_pairs']} of {CANDIDATE_SPACE} pairs")

    src_ids, tgt_ids = load_provision_ids()
    paired_preds = {}

    print(f"\n{'method':<22}{'precision':>11}{'recall':>9}{'specif.':>9}{'F1':>8}")
    for method, filename in PREDICTION_FILES.items():
        predicted = load_predictions(filename)
        if predicted is None:
            print(f"{method:<22} (no prediction file)")
            continue
        df = confusion(sample, predicted)
        point = weighted_rates(df)
        ci = stratified_bootstrap(df, weighted_rates, rng)
        results["methods"][method] = {
            "point": {k: round(float(v), 4) for k, v in point.items()},
            "ci": ci,
        }
        paired_preds[method] = df.pred.to_numpy().astype(bool)
        print(f"{method:<22}{point['precision']:>11.3f}{point['recall']:>9.3f}"
              f"{point['specificity']:>9.3f}{point['f1']:>8.3f}")

    print(f"\ncorpus-calibrated operating points")
    print(f"{'method':<22}{'thresh.':>9}{'precision':>11}{'recall':>9}"
          f"{'specif.':>9}{'F1':>8}")
    for method, path in SIM_MATRICES.items():
        if not path.exists() or method not in results["methods"]:
            continue
        scores = sample_scores(np.load(path), sample, src_ids, tgt_ids)
        cal = corpus_threshold(scores, sample)
        df = sample.assign(
            actual=(sample.relationship != "NO_RELATION").astype(int),
            pred=(scores >= cal["threshold"]).astype(int))
        point = weighted_rates(df)
        results["methods"][method]["corpus_calibrated"] = cal | {
            "point": {k: round(float(v), 4) for k, v in point.items()},
            "ci": stratified_bootstrap(df, weighted_rates, rng),
        }
        paired_preds[f"{method}@corpus"] = df.pred.to_numpy().astype(bool)
        print(f"{method:<22}{cal['threshold']:>9.3f}{point['precision']:>11.3f}"
              f"{point['recall']:>9.3f}{point['specificity']:>9.3f}{point['f1']:>8.3f}")

    # Baselines a method has to beat before its score means anything.
    all_positive = sample.assign(pred=1, actual=(sample.relationship != "NO_RELATION").astype(int))
    results["baseline_all_positive"] = {
        k: round(float(v), 4) for k, v in weighted_rates(all_positive).items()
    }
    b = results["baseline_all_positive"]
    print(f"{'[all-positive baseline]':<22}{b['precision']:>11.3f}{b['recall']:>9.3f}"
          f"{b['specificity']:>9.3f}{b['f1']:>8.3f}")

    paired_preds["all_positive"] = np.ones(len(sample), dtype=bool)

    reference = max((n for n in paired_preds if n != "all_positive"),
                    key=lambda n: weighted_rates(sample.assign(
                        actual=(sample.relationship != "NO_RELATION").astype(int),
                        pred=paired_preds[n].astype(int)))["f1"])
    contrasts = contrast_set(paired_preds, reference)
    results["paired_reference"] = reference
    results["paired_differences"] = paired_differences(
        sample, paired_preds, rng, contrasts)

    significant = [k for k, v in results["paired_differences"].items()
                   if v["f1"]["excludes_zero"]]
    print(f"\nReference for paired comparisons: {reference}")
    print(f"Paired F1 differences excluding zero: {len(significant)} of "
          f"{len(results['paired_differences'])} comparisons")

    (EVAL_DIR / "weighted_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {EVAL_DIR / 'weighted_metrics.json'}")
