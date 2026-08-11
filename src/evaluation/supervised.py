"""
What supervision buys, measured against the same reference set.

The eleven evaluated methods are all zero-shot: they turn a similarity score
into a label through fixed bands, and Section 3.7.4 already notes that a
symmetric scalar cannot express the distinctions the taxonomy is built on. That
leaves an obvious question open. The project annotated 1,027 pairs, only 200 of
which are scored on, so 827 labelled pairs are available as training data. If a
classifier trained on those still cannot type a relationship, the difficulty is
the task; if it can, the difficulty was the zero-shot framing.

This is a control, not a twelfth method. It trains nothing new: the features are
the similarity scores the eleven methods already produced, so no model is
downloaded, fine-tuned, or called over the network.

Two properties of the split shape what can be claimed:

  - The 827 training pairs hold NO_RELATION, OVERLAP and COMPLEMENTARITY only.
    Every EQUIVALENCE and SUBSUMPTION pair in the corpus sits in the evaluation
    set, so the typed classifier cannot reach those two classes by construction.
    Typed results are therefore reported twice: over all 200 pairs with the 13
    rare-class pairs counted as errors, and over the 187 pairs whose class the
    training data covers.
  - The training pool is 73% negative and the evaluation set is 50% negative by
    design, so the classifiers are fit with balanced class weights.

Writes data/evaluation/supervised.json. Run after src.evaluation.analysis.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline.build_reference import pool
from src.evaluation import analysis as A
from src.methods import METHODS, with_matrix

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
EVAL_DIR = DATA / "evaluation"
MAP_DIR = DATA / "mappings"
REFERENCE = DATA / "baseline" / "reference_set.csv"

SEED = 42
TYPED_LABELS = ["NO_RELATION", "OVERLAP", "COMPLEMENTARITY"]
GEMINI_LLM_LABELS = ["EQUIVALENCE", "OVERLAP", "SUBSUMPTION",
                     "COMPLEMENTARITY", "NO_RELATION"]

SUBSUMPTION_VARIANTS = {"SUBSUMPTION_A_BROADER", "SUBSUMPTION_B_BROADER"}


def normalise(rel: str) -> str:
    return "SUBSUMPTION" if rel in SUBSUMPTION_VARIANTS else rel


def build_features(pairs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One row per pair: every stored score the eleven methods produced.

    Nine cosine or entailment matrices, both NLI directions separately so the
    asymmetry survives, the Jaccard score (absent from its sparse output file
    means the pair scored below its floor, hence zero), and the Gemini LLM's
    label as indicators.
    """
    src_ids, tgt_ids = A.load_provision_ids()
    src_idx = {p: i for i, p in enumerate(src_ids)}
    tgt_idx = {p: j for j, p in enumerate(tgt_ids)}
    rows = [(src_idx[r.src_id], tgt_idx[r.tgt_id]) for r in pairs.itertuples()]
    i_arr = np.array([r[0] for r in rows])
    j_arr = np.array([r[1] for r in rows])

    columns, names = [], []

    for key, method in with_matrix().items():
        columns.append(np.load(method.matrix)[i_arr, j_arr])
        names.append(key)

    directions = np.load(MAP_DIR / "nli_entailment_directions.npy")
    columns.append(directions[0][i_arr, j_arr])
    names.append("nli_forward")
    columns.append(directions[1][i_arr, j_arr])
    names.append("nli_backward")

    jaccard = pd.read_csv(METHODS["rule_based_jaccard"].predictions)
    lookup = {(r.src_id, r.tgt_id): r.jaccard_score for r in jaccard.itertuples()}
    columns.append(np.array([lookup.get((r.src_id, r.tgt_id), 0.0)
                             for r in pairs.itertuples()]))
    names.append("jaccard")

    llm = pd.read_csv(METHODS["gemini_llm"].predictions)
    llm_lookup = {(r.src_id, r.tgt_id): normalise(r.predicted_rel)
                  for r in llm.itertuples()}
    llm_labels = [llm_lookup.get((r.src_id, r.tgt_id), "NO_RELATION")
                  for r in pairs.itertuples()]
    for label in GEMINI_LLM_LABELS:
        columns.append(np.array([1.0 if l == label else 0.0 for l in llm_labels]))
        names.append(f"gemini_llm_{label.lower()}")

    return np.column_stack(columns).astype(np.float64), names


def models() -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000,
                               random_state=SEED),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            random_state=SEED, class_weight="balanced"),
    }


def detection_metrics(true_positive: np.ndarray, pred_positive: np.ndarray) -> dict:
    tp = int((true_positive & pred_positive).sum())
    fp = int((~true_positive & pred_positive).sum())
    fn = int((true_positive & ~pred_positive).sum())
    tn = int((~true_positive & ~pred_positive).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "predicted_positive": int(pred_positive.sum()),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
    }


def macro_f1_over(df: pd.DataFrame, labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = ((df["true_rel"] == label) & (df["pred_rel"] == label)).sum()
        fp = ((df["true_rel"] != label) & (df["pred_rel"] == label)).sum()
        fn = ((df["true_rel"] == label) & (df["pred_rel"] != label)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores))


def typed_report(df: pd.DataFrame, rng) -> dict:
    """Typed metrics over all 200 pairs and over the three covered classes."""
    covered = df[df["true_rel"].isin(TYPED_LABELS)]
    return {
        "all_200": {
            "n": len(df),
            "accuracy": round(float((df["true_rel"] == df["pred_rel"]).mean()), 4),
            "macro_f1_5class": round(A.macro_f1(df), 4),
            "macro_f1_ci": A.bootstrap_ci(df, A.macro_f1, rng),
        },
        "three_class_only": {
            "n": len(covered),
            "accuracy": round(float(
                (covered["true_rel"] == covered["pred_rel"]).mean()), 4),
            "macro_f1": round(macro_f1_over(covered, TYPED_LABELS), 4),
            "macro_f1_ci": A.bootstrap_ci(
                covered, lambda d: macro_f1_over(d, TYPED_LABELS), rng),
        },
        "confusion_matrix": {
            t: {p: int(((df["true_rel"] == t) & (df["pred_rel"] == p)).sum())
                for p in A.LABELS}
            for t in A.LABELS
        },
    }


def gemini_calibrated_predictions(gt: pd.DataFrame) -> np.ndarray | None:
    """The leader's calibrated decisions, as the contrast for the paired bootstrap."""
    analysis = json.loads((EVAL_DIR / "analysis.json").read_text())
    cv = analysis.get("gemini_embedding", {}).get("cv_threshold")
    if cv is None:
        return None
    src_ids, tgt_ids = A.load_provision_ids()
    matrix = np.load(METHODS["gemini_embedding"].matrix)
    scores, _ = A.pair_scores(matrix, gt, src_ids, tgt_ids)
    return scores >= cv["threshold"]


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    judged = pool()
    judged["relationship"] = judged["relationship"].apply(normalise)
    gt = pd.read_csv(REFERENCE)
    gt["relationship_norm"] = gt["relationship"].apply(normalise)

    evaluated_pairs = set(zip(gt.src_id, gt.tgt_id))
    train = judged[~judged.apply(
        lambda r: (r.src_id, r.tgt_id) in evaluated_pairs, axis=1)].reset_index(drop=True)

    print(f"Judged pool: {len(judged)}   evaluation set: {len(gt)}   "
          f"training pairs: {len(train)}")
    print(f"Training labels: {train.relationship.value_counts().to_dict()}")
    print(f"Evaluation labels: {gt.relationship_norm.value_counts().to_dict()}\n")

    X_train, feature_names = build_features(train)
    X_eval, _ = build_features(gt)

    y_train_det = (train.relationship != "NO_RELATION").to_numpy()
    y_eval_det = (gt.relationship_norm != "NO_RELATION").to_numpy()
    y_train_typed = train.relationship.to_numpy()

    results = {
        "split": {
            "judged_pool": len(judged),
            "training_pairs": len(train),
            "evaluation_pairs": len(gt),
            "training_labels": {k: int(v) for k, v in
                                train.relationship.value_counts().items()},
            "training_positive_rate": round(float(y_train_det.mean()), 4),
            "evaluation_positive_rate": round(float(y_eval_det.mean()), 4),
        },
        "features": feature_names,
        "detection": {},
        "typed": {},
    }

    baseline_det = detection_metrics(y_eval_det, np.ones(len(gt), dtype=bool))
    results["detection"]["all_positive_baseline"] = baseline_det
    print(f"{'model':<22}{'P':>8}{'R':>8}{'Spec':>8}{'F1':>8}")
    print(f"{'all-positive baseline':<22}{baseline_det['precision']:>8.3f}"
          f"{baseline_det['recall']:>8.3f}{baseline_det['specificity']:>8.3f}"
          f"{baseline_det['f1']:>8.3f}")

    detection_preds = {}
    for name, model in models().items():
        model.fit(X_train, y_train_det)
        pred = model.predict(X_eval).astype(bool)
        detection_preds[name] = pred
        metrics = detection_metrics(y_eval_det, pred)
        merged = pd.DataFrame({
            "true_rel": np.where(y_eval_det, "RELATED", "NO_RELATION"),
            "pred_rel": np.where(pred, "RELATED", "NO_RELATION"),
        })
        metrics["f1_ci"] = A.bootstrap_ci(merged, A.detection_f1, rng)
        results["detection"][name] = metrics
        print(f"{name:<22}{metrics['precision']:>8.3f}{metrics['recall']:>8.3f}"
              f"{metrics['specificity']:>8.3f}{metrics['f1']:>8.3f}")

    # Which stored score the linear model leans on, for the RQ2 discussion.
    lr = models()["logistic_regression"]
    lr.fit(X_train, y_train_det)
    coefficients = dict(sorted(
        zip(feature_names, lr[-1].coef_[0].round(4).tolist()),
        key=lambda kv: abs(kv[1]), reverse=True))
    results["detection"]["logistic_regression_coefficients"] = coefficients
    print("\nLargest standardised coefficients (detection):")
    for feature, weight in list(coefficients.items())[:6]:
        print(f"  {feature:<28}{weight:>+9.3f}")

    print("\nTyped classification:")
    for name, model in models().items():
        model.fit(X_train, y_train_typed)
        pred = model.predict(X_eval)
        df = pd.DataFrame({"true_rel": gt.relationship_norm, "pred_rel": pred})
        report = typed_report(df, rng)
        results["typed"][name] = report
        print(f"  {name:<22} all 200: acc {report['all_200']['accuracy']:.3f}  "
              f"macro-F1 {report['all_200']['macro_f1_5class']:.3f}   "
              f"three-class: acc {report['three_class_only']['accuracy']:.3f}  "
              f"macro-F1 {report['three_class_only']['macro_f1']:.3f}")

    majority = pd.DataFrame({
        "true_rel": gt.relationship_norm,
        "pred_rel": ["NO_RELATION"] * len(gt),
    })
    results["typed"]["majority_class_baseline"] = {
        "accuracy": round(float((majority.true_rel == majority.pred_rel).mean()), 4),
        "macro_f1_5class": round(A.macro_f1(majority), 4),
        "macro_f1_three_class": round(macro_f1_over(
            majority[majority.true_rel.isin(TYPED_LABELS)], TYPED_LABELS), 4),
    }
    print(f"  {'majority-class baseline':<22} "
          f"all 200: acc {results['typed']['majority_class_baseline']['accuracy']:.3f}  "
          f"macro-F1 {results['typed']['majority_class_baseline']['macro_f1_5class']:.3f}")

    gemini = gemini_calibrated_predictions(gt)
    if gemini is not None:
        preds = {"all_positive": np.ones(len(gt), dtype=bool),
                 "gemini_embedding": gemini, **detection_preds}
        contrasts = [(n, "all_positive") for n in detection_preds]
        contrasts += [(n, "gemini_embedding") for n in detection_preds]
        results["contrasts"] = {
            "rule": "each supervised model vs. the all-positive baseline and vs. "
                    "the best zero-shot method at its calibrated threshold",
            "results": A.paired_differences(y_eval_det, preds, rng, contrasts),
        }
        print("\nPaired bootstrap contrasts (detection F1):")
        for key, entry in results["contrasts"]["results"].items():
            f1 = entry["f1"]
            flag = "*" if f1["excludes_zero"] else " "
            print(f"  {flag} {key:<48}{f1['difference']:>+8.3f}  "
                  f"[{f1['ci'][0]:+.3f}, {f1['ci'][1]:+.3f}]")

    (EVAL_DIR / "supervised.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {EVAL_DIR / 'supervised.json'}")
