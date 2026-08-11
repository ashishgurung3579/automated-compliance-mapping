"""
Guard against the thesis drifting from the data: parses the result tables in
thesis/chapters/05_results.tex and the sampling table in 03_methodology.tex, and
compares every number against the stored evaluation, agreement and annotation data.

Exits non-zero if any value disagrees. Run before every submission build.
"""
import csv
import json
import re
import sys
from pathlib import Path

from src.methods import THESIS_ROWS

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
EVAL = DATA / "evaluation"
BASELINE = DATA / "baseline"
RESULTS_TEX = BASE / "thesis" / "chapters" / "05_results.tex"
METHOD_TEX = BASE / "thesis" / "chapters" / "03_methodology.tex"

# thesis row label -> method key in the JSON files
ROWS = THESIS_ROWS

# tab:agreement row -> key in data/validation/agreement.json
AGREEMENT_ROWS = {
    "Second round vs.\\ reviewed pilot": "sample_vs_pilot",
    "Evaluated LLM vs.\\ itself, identical input": "llm_run_to_run",
}

# tab:sampling row -> label in the annotation CSVs
SAMPLING_ROWS = {
    "No relation": "NO_RELATION",
    "Overlap": "OVERLAP",
    "Complementarity": "COMPLEMENTARITY",
    "Subsumption (B broader)": "SUBSUMPTION_B_BROADER",
    "Subsumption (A broader)": "SUBSUMPTION_A_BROADER",
    "Equivalence": "EQUIVALENCE",
}

# tab:ablation row -> method key in data/ablation/ablation.json. The delta columns
# are differences of the four checked ones, so checking those four covers the row.
ABLATION_ROWS = {
    "TF--IDF": "rule_based_tfidf",
    "Sentence-BERT": "sbert",
    "MPNet": "mpnet",
    "BGE": "bge",
    "BERT": "bert",
    "SecureBERT": "securebert",
    "CySecBERT": "cysecbert",
    "NLI cross-encoder": "nli",
}

# tab:supervised row -> model key in data/evaluation/supervised.json
SUPERVISED_ROWS = {
    "Logistic regression": "logistic_regression",
    "Gradient boosting": "gradient_boosting",
}

# tab:runtime row -> key in data/evaluation/runtimes.json
RUNTIME_ROWS = {
    "Provision extraction (both PDFs)": "extraction",
    "TF--IDF + keyword mapping": "rule_based_tfidf_and_jaccard",
    "Sentence-BERT mapping": "sbert",
    "MPNet mapping": "mpnet",
    "BGE mapping": "bge",
    "BERT mapping": "bert",
    "SecureBERT mapping": "securebert",
    "CySecBERT mapping": "cysecbert",
    "NLI cross-encoder (9{,}936 forward passes)": "nli",
    "Evaluation": "evaluate",
    "Calibration and bootstrap analysis": "analysis",
}


def strip_tex(cell: str) -> str:
    cell = re.sub(r"\\(?:textbf|emph|textsc)\{([^}]*)\}", r"\1", cell)
    return cell.replace("\\,", "").replace("$-$", "-").strip()


def parse_rows(tex: str, label: str, keys=None) -> dict[str, list[str]]:
    """Return {row label: [cells]} for the tabular carrying the given \\label."""
    keys = ROWS if keys is None else keys
    block = re.search(r"\\begin\{table\}.*?\\label\{" + re.escape(label)
                      + r"\}.*?\\end\{table\}", tex, re.S)
    if not block:
        raise SystemExit(f"table {label} not found")
    rows = {}
    for line in block.group(0).splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        cells = [strip_tex(c) for c in line.replace("\\\\", "").split("&")]
        if cells[0] in keys:
            rows[cells[0]] = cells[1:]
    return rows


def judged_pool() -> dict[str, int]:
    """Label counts over the 1,027 judged pairs, pilot labels taking precedence.

    Reproduces the Judged column of tab:sampling, including the precedence rule
    documented in Section 3.4.3.
    """
    def read(name):
        return list(csv.DictReader((BASELINE / name).open()))

    pool = {}
    for row in read("gt_sample.csv") + read("rare_class_pairs.csv"):
        pool[(row["src_id"], row["tgt_id"])] = row["relationship"]
    for row in read("gt.csv"):                      # pilot wins where they overlap
        pool[(row["src_id"], row["tgt_id"])] = row["relationship"]

    counts: dict[str, int] = {}
    for label in pool.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


def reference_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in csv.DictReader((BASELINE / "reference_set.csv").open()):
        counts[row["relationship"]] = counts.get(row["relationship"], 0) + 1
    return counts


def check(name: str, thesis: str, data, fmt: str, errors: list):
    expected = format(data, fmt)
    if thesis != expected:
        errors.append(f"{name}: thesis says {thesis!r}, data says {expected!r}")


if __name__ == "__main__":
    tex = RESULTS_TEX.read_text()
    summary = json.loads((EVAL / "evaluation_summary.json").read_text())
    analysis = json.loads((EVAL / "analysis.json").read_text())
    errors: list[str] = []
    checked = 0

    # Only methods absent from the data are skipped; a run method with no thesis
    # row is the drift this guard catches.
    evaluated = {label: key for label, key in ROWS.items() if key in summary}

    det = parse_rows(tex, "tab:detection")
    for label, key in evaluated.items():
        d = summary[key]["detection"]
        cells = det.get(label)
        if cells is None:
            errors.append(f"detection table: row {label!r} missing")
            continue
        for i, (field, fmt) in enumerate([
            ("predicted_positive_in_gt", "d"), ("true_positives", "d"),
            ("false_positives", "d"), ("false_negatives", "d"),
            ("precision", ".3f"), ("recall", ".3f"), ("f1", ".3f"),
        ]):
            check(f"detection/{label}/{field}", cells[i], d[field], fmt, errors)
            checked += 1

    cls = parse_rows(tex, "tab:classification")
    for label, key in evaluated.items():
        c = summary[key]["classification"]
        b = analysis[key]["bootstrap"]["macro_f1"]
        cells = cls.get(label)
        if cells is None:
            errors.append(f"classification table: row {label!r} missing")
            continue
        check(f"classification/{label}/accuracy", cells[0],
              c["overall_accuracy"], ".3f", errors)
        check(f"classification/{label}/macro_f1", cells[1],
              c["macro_f1"], ".3f", errors)
        expected_ci = f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]"
        if cells[2] != expected_ci:
            errors.append(f"classification/{label}/CI: thesis says {cells[2]!r}, "
                          f"data says {expected_ci!r}")
        checked += 3

    cal = parse_rows(tex, "tab:calibrated")
    for label, key in evaluated.items():
        cv = analysis.get(key, {}).get("cv_threshold")
        if cv is None:
            continue
        cells = cal.get(label)
        if cells is None:
            errors.append(f"calibrated table: row {label!r} missing")
            continue
        check(f"calibrated/{label}/threshold", cells[0], cv["threshold"], ".3f", errors)
        for i, field in enumerate(["precision", "recall", "specificity", "f1"], start=1):
            check(f"calibrated/{label}/{field}", cells[i], cv["calibrated"][field],
                  ".3f", errors)
        check(f"calibrated/{label}/held_out", cells[5], cv["cv_f1_mean"], ".3f", errors)
        checked += 6

    # Trailer ablation: a separate run, so it may legitimately be absent.
    ablation_path = DATA / "ablation" / "ablation.json"
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text())["methods"]
        abl = parse_rows(tex, "tab:ablation", ABLATION_ROWS)
        for label, key in ABLATION_ROWS.items():
            cells = abl.get(label)
            if cells is None:
                errors.append(f"ablation table: row {label!r} missing")
                continue
            entry = ablation[key]
            for i, (condition, field) in enumerate([
                ("full_text", "range_width"), ("requirement_only", "range_width"),
            ]):
                check(f"ablation/{label}/{condition}.{field}", cells[i],
                      entry[condition][field], ".3f", errors)
            for i, condition in [(3, "full_text"), (4, "requirement_only")]:
                check(f"ablation/{label}/{condition}.f1", cells[i],
                      entry[condition]["calibrated_f1"], ".3f", errors)
            checked += 4

    # Supervised control: a separate run, so it may legitimately be absent.
    supervised_path = EVAL / "supervised.json"
    if supervised_path.exists():
        supervised = json.loads(supervised_path.read_text())
        sup = parse_rows(tex, "tab:supervised", SUPERVISED_ROWS)
        for label, key in SUPERVISED_ROWS.items():
            cells = sup.get(label)
            if cells is None:
                errors.append(f"supervised table: row {label!r} missing")
                continue
            det = supervised["detection"][key]
            typed = supervised["typed"][key]
            check(f"supervised/{label}/precision", cells[0], det["precision"], ".3f", errors)
            check(f"supervised/{label}/f1", cells[1], det["f1"], ".3f", errors)
            check(f"supervised/{label}/accuracy", cells[2],
                  typed["all_200"]["accuracy"], ".3f", errors)
            check(f"supervised/{label}/macro_f1", cells[3],
                  typed["all_200"]["macro_f1_5class"], ".3f", errors)
            check(f"supervised/{label}/macro_f1_187", cells[4],
                  typed["three_class_only"]["macro_f1"], ".3f", errors)
            checked += 5

    # Agreement table: the kappas are computed elsewhere and were transcribed by hand.
    agreement = json.loads((DATA / "validation" / "agreement.json").read_text())
    agr = parse_rows(tex, "tab:agreement", AGREEMENT_ROWS)
    for label, key in AGREEMENT_ROWS.items():
        cells = agr.get(label)
        if cells is None:
            errors.append(f"agreement table: row {label!r} missing")
            continue
        block = agreement[key]
        check(f"agreement/{label}/n", cells[0], block["six_label"]["n"]
              if "n" in block["six_label"] else block["n"], "d", errors)
        for i, (part, field) in enumerate([
            ("six_label", "percent_agreement"), ("six_label", "cohen_kappa"),
            ("binary_detection", "percent_agreement"), ("binary_detection", "cohen_kappa"),
        ], start=1):
            check(f"agreement/{label}/{part}.{field}", cells[i],
                  block[part][field], ".2f", errors)
            checked += 1
        checked += 1

    # Sampling table lives in the methodology chapter, not the results chapter.
    meth = METHOD_TEX.read_text()
    smp = parse_rows(meth, "tab:sampling", SAMPLING_ROWS)
    pool, refset = judged_pool(), reference_counts()
    for label, key in SAMPLING_ROWS.items():
        cells = smp.get(label)
        if cells is None:
            errors.append(f"sampling table: row {label!r} missing")
            continue
        check(f"sampling/{label}/judged", cells[0], pool.get(key, 0), "d", errors)
        check(f"sampling/{label}/in_set", cells[1], refset.get(key, 0), "d", errors)
        checked += 2

    # Runtime table: printed to whichever precision reads well, so compare rounded.
    runtimes = json.loads((EVAL / "runtimes.json").read_text())
    rt = parse_rows(tex, "tab:runtime", RUNTIME_ROWS)
    for label, key in RUNTIME_ROWS.items():
        cells = rt.get(label)
        if cells is None:
            errors.append(f"runtime table: row {label!r} missing")
            continue
        printed = cells[0].replace("s", "").strip()
        stored = runtimes[key]
        if printed not in {f"{stored:.1f}", f"{round(stored):d}"}:
            errors.append(f"runtime/{label}: thesis says {printed!r}, "
                          f"data says {stored}")
        checked += 1

    if errors:
        print("Thesis numbers do not match the data:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"All {checked} table values match the evaluation data.")
