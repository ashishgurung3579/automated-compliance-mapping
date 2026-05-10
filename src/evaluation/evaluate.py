"""
Evaluation of automated compliance mapping against expert ground truth.

Metrics:
  - Pair detection: precision, recall, F1, coverage
  - Relationship classification: per-class precision, recall, F1
  - Error analysis: false positives, false negatives with details
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
GT_PATH = DATA / "baseline" / "gt.csv"
OUT_DIR = DATA / "evaluation"

# Treat the two directional subsumption labels as one class for metrics.
SUBSUMPTION_VARIANTS = {"SUBSUMPTION_A_BROADER", "SUBSUMPTION_B_BROADER"}


def _normalise_rel(rel: str) -> str:
    if rel in SUBSUMPTION_VARIANTS:
        return "SUBSUMPTION"
    return rel


def load_gt() -> pd.DataFrame:
    df = pd.read_csv(GT_PATH)
    df["relationship_norm"] = df["relationship"].apply(_normalise_rel)
    return df


def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["predicted_rel_norm"] = df["predicted_rel"].apply(_normalise_rel)
    return df


def pair_detection_metrics(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """
    Positive = any relationship != NO_RELATION.
    Treats non-predicted GT pairs as NO_RELATION.
    """
    gt_positive = set(
        zip(gt[gt["relationship"] != "NO_RELATION"]["src_id"],
            gt[gt["relationship"] != "NO_RELATION"]["tgt_id"])
    )
    gt_negative = set(
        zip(gt[gt["relationship"] == "NO_RELATION"]["src_id"],
            gt[gt["relationship"] == "NO_RELATION"]["tgt_id"])
    )
    pred_positive = set(
        zip(pred[pred["predicted_rel"] != "NO_RELATION"]["src_id"],
            pred[pred["predicted_rel"] != "NO_RELATION"]["tgt_id"])
    )

    tp = len(gt_positive & pred_positive)
    fp = len(pred_positive - gt_positive - gt_negative)
    fp_on_negative = len(pred_positive & gt_negative)
    fn = len(gt_positive - pred_positive)

    precision = tp / (tp + fp + fp_on_negative) if (tp + fp + fp_on_negative) > 0 else 0.0
    recall = tp / len(gt_positive) if gt_positive else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    coverage = recall

    return {
        "gt_positive_pairs": len(gt_positive),
        "gt_negative_pairs": len(gt_negative),
        "predicted_positive_pairs": len(pred_positive),
        "true_positives": tp,
        "false_positives_unannotated": fp,
        "false_positives_on_negatives": fp_on_negative,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "coverage": round(coverage, 4),
    }


def classification_metrics(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """
    For GT pairs that appear in predictions, compare normalised relationship labels.
    GT pairs not found in predictions are treated as predicted NO_RELATION.
    """
    pred_lookup = {
        (r.src_id, r.tgt_id): r.predicted_rel_norm
        for r in pred.itertuples()
    }

    rows = []
    for r in gt.itertuples():
        predicted = pred_lookup.get((r.src_id, r.tgt_id), "NO_RELATION")
        rows.append({
            "src_id": r.src_id,
            "tgt_id": r.tgt_id,
            "true_rel": r.relationship_norm,
            "pred_rel": predicted,
            "correct": r.relationship_norm == predicted,
        })
    merged = pd.DataFrame(rows)

    overall_accuracy = merged["correct"].mean()

    per_class = {}
    all_labels = sorted(merged["true_rel"].unique())
    for label in all_labels:
        tp = ((merged["true_rel"] == label) & (merged["pred_rel"] == label)).sum()
        fp = ((merged["true_rel"] != label) & (merged["pred_rel"] == label)).sum()
        fn = ((merged["true_rel"] == label) & (merged["pred_rel"] != label)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        support = (merged["true_rel"] == label).sum()
        per_class[label] = {
            "support": int(support),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        }

    macro_p = np.mean([v["precision"] for v in per_class.values()])
    macro_r = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    return {
        "overall_accuracy": round(float(overall_accuracy), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "_merged": merged,
    }


def error_analysis(gt: pd.DataFrame, pred: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    gt_pairs = set(zip(gt["src_id"], gt["tgt_id"]))

    errors = []

    for r in merged.itertuples():
        if r.true_rel != "NO_RELATION" and r.pred_rel == "NO_RELATION":
            errors.append({
                "error_type": "FALSE_NEGATIVE",
                "src_id": r.src_id,
                "tgt_id": r.tgt_id,
                "true_rel": r.true_rel,
                "pred_rel": r.pred_rel,
                "note": "GT pair missed entirely by model",
            })
        elif r.true_rel != r.pred_rel and r.pred_rel != "NO_RELATION":
            errors.append({
                "error_type": "WRONG_LABEL",
                "src_id": r.src_id,
                "tgt_id": r.tgt_id,
                "true_rel": r.true_rel,
                "pred_rel": r.pred_rel,
                "note": "Pair found but relationship label incorrect",
            })
        elif r.true_rel == "NO_RELATION" and r.pred_rel != "NO_RELATION":
            errors.append({
                "error_type": "FALSE_POSITIVE_ON_NEGATIVE",
                "src_id": r.src_id,
                "tgt_id": r.tgt_id,
                "true_rel": r.true_rel,
                "pred_rel": r.pred_rel,
                "note": "Model predicted relation on GT-negative pair",
            })

    for r in pred.itertuples():
        if (r.src_id, r.tgt_id) not in gt_pairs and r.predicted_rel != "NO_RELATION":
            errors.append({
                "error_type": "FALSE_POSITIVE_UNANNOTATED",
                "src_id": r.src_id,
                "tgt_id": r.tgt_id,
                "true_rel": "UNKNOWN",
                "pred_rel": r.predicted_rel,
                "note": "Predicted pair not in GT at all (may or may not be correct)",
            })

    return pd.DataFrame(errors)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_gt()
    print(f"Ground truth: {len(gt)} pairs")
    print(f"  Relationships: {gt['relationship'].value_counts().to_dict()}\n")

    methods = {
        "rule_based_tfidf":     DATA / "mappings" / "rule_based_tfidf.csv",
        "rule_based_jaccard":   DATA / "mappings" / "rule_based_jaccard.csv",
        "sbert":                DATA / "mappings" / "sbert_output.csv",
        "bert":                 DATA / "mappings" / "bert_output.csv",
        "securebert":           DATA / "mappings" / "securebert_output.csv",
        "gemini_embedding":     DATA / "mappings" / "gemini_embedding_output.csv",
        "gemini_llm":           DATA / "mappings" / "gemini_output.csv",
    }

    summary = {}

    for name, path in methods.items():
        if not path.exists():
            print(f"Skipping {name}: file not found")
            continue

        pred = load_predictions(path)
        # Ensure predicted_rel column exists (tfidf/jaccard CSVs use same column name)
        if "predicted_rel" not in pred.columns:
            print(f"Skipping {name}: no predicted_rel column")
            continue

        print(f"=== {name} ({len(pred)} predictions) ===")

        det = pair_detection_metrics(gt, pred)
        print(f"  Pair detection: P={det['precision']:.3f}  R={det['recall']:.3f}  F1={det['f1']:.3f}  Coverage={det['coverage']:.3f}")
        print(f"  TP={det['true_positives']}  FP(unannotated)={det['false_positives_unannotated']}  "
              f"FP(on negatives)={det['false_positives_on_negatives']}  FN={det['false_negatives']}")

        cls = classification_metrics(gt, pred)
        merged_df = cls.pop("_merged")
        print(f"  Classification: Accuracy={cls['overall_accuracy']:.3f}  "
              f"Macro-F1={cls['macro_f1']:.3f}")
        print("  Per-class:")
        for lbl, m in cls["per_class"].items():
            print(f"    {lbl:<28} support={m['support']:3d}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

        errors_df = error_analysis(gt, pred, merged_df)
        print(f"  Errors: {errors_df['error_type'].value_counts().to_dict()}\n")

        result = {"detection": det, "classification": cls}
        summary[name] = result

        (OUT_DIR / f"{name}_eval.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        errors_df.to_csv(OUT_DIR / f"{name}_errors.csv", index=False)
        merged_df.to_csv(OUT_DIR / f"{name}_predictions_vs_gt.csv", index=False)

    (OUT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"Results saved to {OUT_DIR}")
