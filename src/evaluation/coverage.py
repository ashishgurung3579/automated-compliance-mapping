"""
Coverage and gap analysis over the full candidate space.

Reports which provisions of one standard have a predicted counterpart in the other
and which have none, the automated form of the coverage statement and
unmapped-requirement list a manual comparative study produces.

Everything here is a prediction from the best-scoring method at its calibrated
threshold, not a verified mapping. That threshold was fitted on a reference set
built to be half positive, so the precision printed in the captions is an upper
bound on its precision across all 4,968 candidates.

Writes data/evaluation/coverage.json, thesis/figures/coverage_heatmap.pdf and
the generated tables under thesis/tables/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.mapping.embeddings import load_provisions
from src.methods import METHODS

BASE = Path(__file__).parents[2]
EVAL_DIR = BASE / "data" / "evaluation"
FIG_DIR = BASE / "thesis" / "figures"
TAB_DIR = BASE / "thesis" / "tables"

INK = "#0b0b0b"
MUTED = "#898781"

# Clause headings, shortened to fit an axis. The numbering is the standards' own.
IOT_CLAUSES = {
    "5.0": "Reporting implementation",
    "5.1": "No universal default passwords",
    "5.2": "Vulnerability disclosure",
    "5.3": "Keep software updated",
    "5.4": "Store parameters securely",
    "5.5": "Communicate securely",
    "5.6": "Minimise attack surface",
    "5.7": "Software integrity",
    "5.8": "Personal data security",
    "5.9": "Resilience to outage",
    "5.10": "System telemetry",
    "5.11": "Deletion of user data",
    "5.12": "Installation and maintenance",
    "5.13": "Input validation",
}

AI_PHASES = {
    "5.1": "Secure design",
    "5.2": "Secure development",
    "5.3": "Secure deployment",
    "5.4": "Secure maintenance",
    "5.5": "Secure end of life",
}


def iot_clause(provision_id: str) -> str:
    return provision_id.split("-")[0]


def ai_phase(provision_id: str) -> str:
    return ".".join(provision_id.split("-")[0].split(".")[:2])


def reference_method(analysis: dict) -> tuple[str, float, dict]:
    """The method the coverage map is built from, and the precision that qualifies it.

    Chosen as the highest F1 at a cross-validated operating point, which is the same
    reference the paired contrasts use.
    """
    name = analysis["_contrasts"]["reference"]
    cv = analysis[name].get("cv_threshold")
    if cv is None:
        raise SystemExit(f"reference method {name} has no calibrated operating point")
    return name, cv["threshold"], cv


def predict(method, threshold: float) -> np.ndarray:
    return np.load(method.matrix) >= threshold


def escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        text = text.replace(a, b)
    return text


def truncate(text: str, limit: int = 78) -> str:
    """Shorten to a line, then escape. The ellipsis is added last, because it is
    LaTeX rather than content and escaping it would print the macro."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return escape(text)
    return escape(text[:limit - 1].rstrip()) + r"\ldots"


def write_heatmap(matrix: np.ndarray, src, tgt, method_label: str,
                  precision: float, ci: list[float]):
    rows = list(IOT_CLAUSES)
    cols = list(AI_PHASES)
    grid = np.zeros((len(rows), len(cols)))
    counts = np.zeros_like(grid)
    for i, sp in enumerate(src):
        r = rows.index(iot_clause(sp["provision_id"]))
        for j, tp in enumerate(tgt):
            c = cols.index(ai_phase(tp["provision_id"]))
            counts[r, c] += 1
            grid[r, c] += matrix[i, j]
    share = np.divide(grid, counts, out=np.zeros_like(grid), where=counts > 0)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_seq", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r in range(len(rows)):
        for c in range(len(cols)):
            if counts[r, c]:
                ax.text(c, r, f"{share[r, c]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color=INK if share[r, c] < 0.55 else "#ffffff")
    ax.set_xticks(range(len(cols)),
                  [f"{k}\n{AI_PHASES[k]}" for k in cols], fontsize=6.5)
    ax.set_yticks(range(len(rows)),
                  [f"{k}  {IOT_CLAUSES[k]}" for k in rows], fontsize=6.5)
    ax.set_xlabel("EN 304 223 life-cycle phase", fontsize=8)
    ax.set_ylabel("EN 303 645 provision clause", fontsize=8)
    ax.set_title(f"Predicted relation density, {method_label}\n"
                 f"precision {precision:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] "
                 f"on the balanced reference set — an upper bound here",
                 fontsize=8.5, color=MUTED)
    ax.grid(visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_heatmap.pdf")
    plt.close(fig)
    return share, counts


def write_unmapped(matrix: np.ndarray, src, tgt, method_label: str) -> dict:
    src_unmapped = [p for i, p in enumerate(src) if not matrix[i].any()]
    tgt_unmapped = [p for j, p in enumerate(tgt) if not matrix[:, j].any()]

    lines = [
        "% Generated by src.evaluation.coverage -- do not edit by hand.",
        r"\begin{tabular}{@{}llp{0.56\linewidth}@{}}",
        r"\toprule",
        r"Standard & Provision & Requirement \\",
        r"\midrule",
    ]
    for standard, provisions in [("EN 303 645", src_unmapped),
                                 ("EN 304 223", tgt_unmapped)]:
        for n, p in enumerate(provisions):
            first = standard if n == 0 else ""
            lines.append(f"{first} & {escape(p['provision_id'])} & "
                         f"{truncate(p['text'])} \\\\")
        if provisions and standard == "EN 303 645" and tgt_unmapped:
            lines.append(r"\midrule")
    if not src_unmapped and not tgt_unmapped:
        lines.append(r"\multicolumn{3}{@{}l@{}}{"
                     r"Every provision received at least one predicted counterpart.} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "unmapped_provisions.tex").write_text("\n".join(lines) + "\n")

    return {
        "en303645_unmapped": [p["provision_id"] for p in src_unmapped],
        "en304223_unmapped": [p["provision_id"] for p in tgt_unmapped],
    }


def write_predicted_mapping(matrix: np.ndarray, sim: np.ndarray, src, tgt) -> int:
    pairs = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            if matrix[i, j]:
                pairs.append((sp["provision_id"], tp["provision_id"],
                              float(sim[i, j])))
    pairs.sort(key=lambda r: (r[0], -r[2]))

    lines = [
        "% Generated by src.evaluation.coverage -- do not edit by hand.",
        r"\begin{longtable}{@{}lll@{}}",
        r"\toprule",
        r"EN 303 645 & EN 304 223 & Score \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"EN 303 645 & EN 304 223 & Score \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for a, b, s in pairs:
        lines.append(f"{escape(a)} & {escape(b)} & {s:.3f} \\\\")
    lines.append(r"\end{longtable}")
    (TAB_DIR / "predicted_mapping.tex").write_text("\n".join(lines) + "\n")
    return len(pairs)


if __name__ == "__main__":
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    analysis = json.loads((EVAL_DIR / "analysis.json").read_text())
    name, threshold, calibrated = reference_method(analysis)
    method = METHODS[name]
    precision = calibrated["calibrated"]["precision"]
    ci = calibrated["calibrated_ci"]["precision"]

    src, tgt = load_provisions()
    sim = np.load(method.matrix)
    predicted = predict(method, threshold)

    print(f"Reference method: {method.long_label}")
    print(f"  calibrated threshold {threshold:.3f}, "
          f"precision {precision:.3f} {ci} on the reference set (upper bound)")
    print(f"  predicted related pairs: {int(predicted.sum())} of {predicted.size}")

    share, counts = write_heatmap(predicted, src, tgt, method.label, precision, ci)
    gaps = write_unmapped(predicted, src, tgt, method.label)
    n_pairs = write_predicted_mapping(predicted, sim, src, tgt)

    results = {
        "method": name,
        "method_label": method.long_label,
        "threshold": threshold,
        "precision": precision,
        "precision_ci": ci,
        "predicted_pairs": int(predicted.sum()),
        "candidate_pairs": int(predicted.size),
        "clause_density": {
            iot: {ai: round(float(share[r, c]), 4)
                  for c, ai in enumerate(AI_PHASES)}
            for r, iot in enumerate(IOT_CLAUSES)
        },
        **gaps,
    }
    results["en303645_unmapped_count"] = len(gaps["en303645_unmapped"])
    results["en304223_unmapped_count"] = len(gaps["en304223_unmapped"])
    (EVAL_DIR / "coverage.json").write_text(json.dumps(results, indent=2))

    print(f"  EN 303 645 provisions with no predicted counterpart: "
          f"{results['en303645_unmapped_count']} of {len(src)}")
    print(f"  EN 304 223 provisions with no predicted counterpart: "
          f"{results['en304223_unmapped_count']} of {len(tgt)}")
    print(f"Wrote coverage.json, coverage_heatmap.pdf and {n_pairs} mapping rows")
