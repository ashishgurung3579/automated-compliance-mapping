"""
Reliability of the annotated reference set.

Three measurements, none of which requires claiming an expert study that was not run:

  1. The second annotation round versus the pilot set. The pilot's 107 pairs were
     reviewed by both authors, so agreement against them is the closest thing to an
     external human check the project has. Pilot pairs are mixed unmarked into the
     later annotation stream, so the comparison is blind: it measures whether an
     independent pass over the same pairs reproduces the reviewed labels.
  2. Confidence calibration. Recorded confidence should predict where the two sources
     disagree; if it does not, the field carries no information and is reported so.
  3. Machine-annotator self-agreement, as context. Two runs of the evaluated LLM on
     identical input give a reference point for what an unreliable annotator looks like.

Writes data/validation/agreement.json and a LaTeX table fragment.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
VAL = DATA / "validation"

SUBSUMPTION = {"SUBSUMPTION_A_BROADER", "SUBSUMPTION_B_BROADER"}
LANDIS_KOCH = [
    (0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
    (0.21, "fair"), (0.01, "slight"), (-1.0, "poor"),
]


def merge(label: str) -> str:
    return "SUBSUMPTION" if label in SUBSUMPTION else label


def band(kappa: float) -> str:
    return next(name for threshold, name in LANDIS_KOCH if kappa >= threshold)


def compare(x: pd.Series, y: pd.Series) -> dict:
    binary_x, binary_y = x != "NO_RELATION", y != "NO_RELATION"
    six = {
        "percent_agreement": round(float((x == y).mean()), 4),
        "cohen_kappa": round(float(cohen_kappa_score(x, y)), 4),
    }
    merged_x, merged_y = x.map(merge), y.map(merge)
    return {
        "n": int(len(x)),
        "six_label": six | {"band": band(six["cohen_kappa"])},
        "merged_five_label": {
            "percent_agreement": round(float((merged_x == merged_y).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(merged_x, merged_y)), 4),
        },
        "binary_detection": {
            "percent_agreement": round(float((binary_x == binary_y).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(binary_x, binary_y)), 4),
            "band": band(float(cohen_kappa_score(binary_x, binary_y))),
        },
    }


def sample_vs_pilot() -> tuple[dict, pd.DataFrame]:
    sample = pd.read_csv(DATA / "baseline" / "gt_sample.csv")
    pilot = pd.read_csv(DATA / "baseline" / "gt.csv")[["src_id", "tgt_id", "relationship"]]
    merged = sample.merge(pilot, on=["src_id", "tgt_id"], suffixes=("_sample", "_pilot"))
    if len(merged) < 10:
        raise SystemExit(
            f"Only {len(merged)} pilot pairs re-annotated so far; annotate more batches "
            "before the comparison is meaningful.")
    return compare(merged.relationship_sample, merged.relationship_pilot), merged


def confidence_calibration(merged: pd.DataFrame) -> dict:
    merged = merged.assign(agree=(merged.relationship_sample == merged.relationship_pilot))
    by_conf = merged.groupby("confidence").agree.agg(["mean", "size"])
    return {
        str(int(c)): {"agreement": round(float(r["mean"]), 4), "n": int(r["size"])}
        for c, r in by_conf.iterrows()
    }


def llm_self_agreement() -> dict | None:
    runs = [DATA / "mappings" / f for f in ("gemini_output.csv", "gemini_output_rep2.csv")]
    if not all(p.exists() for p in runs):
        return None
    a, b = (pd.read_csv(p) for p in runs)
    m = a.merge(b, on=["src_id", "tgt_id"], suffixes=("_1", "_2"))
    return compare(m.predicted_rel_1, m.predicted_rel_2)


if __name__ == "__main__":
    VAL.mkdir(parents=True, exist_ok=True)
    results = {}

    pilot_cmp, merged = sample_vs_pilot()
    results["sample_vs_pilot"] = pilot_cmp
    results["confidence_calibration"] = confidence_calibration(merged)

    llm = llm_self_agreement()
    if llm:
        results["llm_run_to_run"] = llm

    p = pilot_cmp
    print(f"Second round vs author-reviewed pilot (n={p['n']}):")
    for scheme in ("six_label", "merged_five_label", "binary_detection"):
        s = p[scheme]
        extra = f"  [{s['band']}]" if "band" in s else ""
        print(f"  {scheme:<20} agreement {s['percent_agreement']:.3f}  "
              f"kappa {s['cohen_kappa']:.3f}{extra}")

    print("\nConfidence calibration (agreement with pilot by recorded confidence):")
    for conf, r in sorted(results["confidence_calibration"].items()):
        print(f"  confidence {conf}: agreement {r['agreement']:.3f} (n={r['n']})")

    if llm:
        print(f"\nFor contrast, the evaluated LLM against itself on identical input "
              f"(n={llm['n']}):")
        print(f"  six_label            agreement {llm['six_label']['percent_agreement']:.3f}  "
              f"kappa {llm['six_label']['cohen_kappa']:.3f}  [{llm['six_label']['band']}]")

    (VAL / "agreement.json").write_text(json.dumps(results, indent=2))

    rows = [("Second round vs.\\ reviewed pilot", pilot_cmp)]
    if llm:
        rows.append(("Evaluated LLM vs.\\ itself, identical input", llm))
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Six labels} & \\multicolumn{2}{c}{Related vs.\\ not} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Comparison & $n$ & Agreement & $\\kappa$ & Agreement & $\\kappa$ \\\\",
        "\\midrule",
    ]
    for label, r in rows:
        lines.append(
            f"{label} & {r['n']} & {r['six_label']['percent_agreement']:.2f} & "
            f"{r['six_label']['cohen_kappa']:.2f} & "
            f"{r['binary_detection']['percent_agreement']:.2f} & "
            f"{r['binary_detection']['cohen_kappa']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (VAL / "agreement_table.tex").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {VAL / 'agreement.json'} and agreement_table.tex")
