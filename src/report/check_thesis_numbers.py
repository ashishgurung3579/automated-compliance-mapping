"""
Guard against the thesis drifting from the data: parses the result tables in
thesis/chapters/05_results.tex and compares every number against
data/evaluation/evaluation_summary.json and analysis.json.

Exits non-zero if any value disagrees. Run before every submission build.
"""
import json
import re
import sys
from pathlib import Path

from src.methods import THESIS_ROWS

BASE = Path(__file__).parents[2]
EVAL = BASE / "data" / "evaluation"
RESULTS_TEX = BASE / "thesis" / "chapters" / "05_results.tex"

# thesis row label -> method key in the JSON files
ROWS = THESIS_ROWS


def strip_tex(cell: str) -> str:
    return re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell).strip()


def parse_rows(tex: str, label: str) -> dict[str, list[str]]:
    """Return {row label: [cells]} for the tabular carrying the given \\label."""
    block = re.search(r"\\begin\{table\}.*?\\label\{" + re.escape(label)
                      + r"\}.*?\\end\{table\}", tex, re.S)
    if not block:
        raise SystemExit(f"table {label} not found in {RESULTS_TEX.name}")
    rows = {}
    for line in block.group(0).splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        cells = [strip_tex(c) for c in line.replace("\\\\", "").split("&")]
        if cells[0] in ROWS:
            rows[cells[0]] = cells[1:]
    return rows


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

    if errors:
        print("Thesis numbers do not match the data:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"All {checked} table values match the evaluation data.")
