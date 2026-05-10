"""
Report generator for automated compliance mapping evaluation.

Reads data/evaluation/evaluation_summary.json and writes
data/evaluation/report.md.
"""
import json
from datetime import date
from pathlib import Path

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
SUMMARY_PATH = DATA / "evaluation" / "evaluation_summary.json"
REPORT_PATH = DATA / "evaluation" / "report.md"

METHOD_ORDER = [
    "rule_based_tfidf",
    "rule_based_jaccard",
    "sbert",
    "bert",
    "securebert",
    "gemini_embedding",
    "gemini_llm",
]

METHOD_LABELS = {
    "rule_based_tfidf":     "Rule-Based TF-IDF",
    "rule_based_jaccard":   "Rule-Based Jaccard",
    "sbert":                "SBERT (all-MiniLM-L6-v2)",
    "bert":                 "BERT (bert-base-uncased)",
    "securebert":           "SecureBERT (ehsanaghaei/SecureBERT)",
    "gemini_embedding":     "Gemini Embedding (gemini-embedding-2)",
    "gemini_llm":           "Gemini LLM (gemini-2.5-flash-lite)",
}


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _f(v: float) -> str:
    return f"{v:.3f}"


def best_method(summary: dict) -> str:
    return max(
        (k for k in METHOD_ORDER if k in summary),
        key=lambda k: summary[k]["classification"]["macro_f1"],
    )


def render_report(summary: dict) -> str:
    lines = []
    today = date.today().isoformat()

    lines += [
        "# Automated Compliance Mapping - Evaluation Report",
        "",
        f"**Generated**: {today}  ",
        "**Standards**: ETSI EN 303 645 (IoT security) x ETSI EN 304 223 (AI security)  ",
        "**Ground truth**: 107 annotated pairs (85 positive, 22 negative)  ",
        "",
        "---",
        "",
        "## Method Comparison",
        "",
        "| Method | Det. P | Det. R | Det. F1 | Cls. Accuracy | Macro-F1 |",
        "|--------|-------:|-------:|--------:|--------------:|---------:|",
    ]

    for key in METHOD_ORDER:
        if key not in summary:
            continue
        det = summary[key]["detection"]
        cls = summary[key]["classification"]
        label = METHOD_LABELS.get(key, key)
        lines.append(
            f"| {label} "
            f"| {_pct(det['precision'])} "
            f"| {_pct(det['recall'])} "
            f"| {_pct(det['f1'])} "
            f"| {_pct(cls['overall_accuracy'])} "
            f"| {_pct(cls['macro_f1'])} |"
        )

    lines += ["", ""]

    best = best_method(summary)
    best_cls = summary[best]["classification"]
    best_det = summary[best]["detection"]
    lines += [
        "## Best Performing Method",
        "",
        f"**{METHOD_LABELS.get(best, best)}** achieved the highest macro-F1 of "
        f"**{_pct(best_cls['macro_f1'])}** on classification, with pair detection "
        f"precision {_pct(best_det['precision'])} / recall {_pct(best_det['recall'])} / "
        f"F1 {_pct(best_det['f1'])}.",
        "",
        "---",
        "",
        "## Per-Method Detail",
        "",
    ]

    for key in METHOD_ORDER:
        if key not in summary:
            continue
        det = summary[key]["detection"]
        cls = summary[key]["classification"]
        label = METHOD_LABELS.get(key, key)

        lines += [
            f"### {label}",
            "",
            "**Pair detection**",
            "",
            f"- Precision: {_f(det['precision'])}  ",
            f"- Recall: {_f(det['recall'])}  ",
            f"- F1: {_f(det['f1'])}  ",
            f"- True positives: {det['true_positives']} / {det['gt_positive_pairs']}  ",
            f"- False positives (on GT negatives): {det['false_positives']}  ",
            f"- False negatives: {det['false_negatives']}  ",
            f"- Predicted positive pairs within GT scope: {det['predicted_positive_in_gt']}  ",
            "",
            "**Classification**",
            "",
            f"- Overall accuracy: {_f(cls['overall_accuracy'])}  ",
            f"- Macro-F1: {_f(cls['macro_f1'])}  ",
            "",
            "**Per-class F1**",
            "",
            "| Class | Support | Precision | Recall | F1 |",
            "|-------|--------:|----------:|-------:|---:|",
        ]
        for cls_name, m in cls["per_class"].items():
            lines.append(
                f"| {cls_name} | {m['support']} "
                f"| {_f(m['precision'])} "
                f"| {_f(m['recall'])} "
                f"| {_f(m['f1'])} |"
            )
        lines += ["", ""]

    lines += [
        "---",
        "",
        "## Limitations and Notes",
        "",
        "- All metrics computed against the 107 annotated GT pairs only; "
        "predictions outside GT scope are excluded from evaluation.",
        "- SUBSUMPTION variants (A_BROADER / B_BROADER) are merged into a single "
        "SUBSUMPTION class for classification metrics.",
        "- Threshold values for each method were set heuristically; systematic "
        "threshold search may improve precision/recall balance.",
        "- Gemini Embedding API results may vary across API versions or rate-limit conditions.",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found: {SUMMARY_PATH}\n"
            "Run src/evaluation/evaluate.py first."
        )

    summary = json.loads(SUMMARY_PATH.read_text())
    report_text = render_report(summary)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Report written to: {REPORT_PATH}")
    print(f"  Methods included: {[k for k in METHOD_ORDER if k in summary]}")
