"""
Publication figures for the thesis. Reads data/evaluation/ and writes PDF
figures to thesis/figures/. Run after src.evaluation.evaluate and
src.evaluation.analysis.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parents[2]
EVAL_DIR = BASE / "data" / "evaluation"
FIG_DIR = BASE / "thesis" / "figures"

# Fixed categorical order (validated); grid/ink roles from the same system.
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

METHOD_LABELS = {
    "rule_based_tfidf": "TF-IDF",
    "rule_based_jaccard": "Jaccard",
    "sbert": "SBERT",
    "bert": "BERT",
    "securebert": "SecureBERT",
    "gemini_embedding": "Gemini emb.",
    "gemini_llm": "Gemini LLM",
}
ORDER = list(METHOD_LABELS)

LABELS = ["EQUIVALENCE", "OVERLAP", "SUBSUMPTION", "COMPLEMENTARITY", "NO_RELATION"]
SHORT = {"EQUIVALENCE": "EQV", "OVERLAP": "OVL", "SUBSUMPTION": "SUB",
         "COMPLEMENTARITY": "CMP", "NO_RELATION": "NONE"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)


def load_summary() -> dict:
    return json.loads((EVAL_DIR / "evaluation_summary.json").read_text())


def load_analysis() -> dict:
    return json.loads((EVAL_DIR / "analysis.json").read_text())


def fig_detection_metrics(summary):
    methods = [m for m in ORDER if m in summary]
    x = np.arange(len(methods))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for off, key, color, name in [(-w, "precision", C1, "Precision"),
                                  (0, "recall", C2, "Recall"),
                                  (w, "f1", C3, "F1")]:
        vals = [summary[m]["detection"][key] for m in methods]
        ax.bar(x + off, vals, width=w - 0.03, color=color, label=name)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncols=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
    style_axes(ax)
    fig.savefig(FIG_DIR / "detection_metrics.pdf")
    plt.close(fig)


def fig_f1_ci(analysis):
    methods = [m for m in ORDER if m in analysis]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
    for ax, key, title in [(axes[0], "detection_f1", "Pair-detection F1"),
                           (axes[1], "macro_f1", "Classification macro-F1")]:
        pts = [analysis[m]["bootstrap"][key]["point"] for m in methods]
        lo = [analysis[m]["bootstrap"][key]["ci_low"] for m in methods]
        hi = [analysis[m]["bootstrap"][key]["ci_high"] for m in methods]
        y = np.arange(len(methods))[::-1]
        err = [np.array(pts) - lo, np.array(hi) - pts]
        ax.errorbar(pts, y, xerr=err, fmt="o", color=C1, ecolor=C1,
                    elinewidth=1.2, capsize=2.5, markersize=4.5)
        ax.set_yticks(y, [METHOD_LABELS[m] for m in methods])
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f1_confidence_intervals.pdf")
    plt.close(fig)


def fig_confusion(analysis, methods=("gemini_embedding", "bert", "gemini_llm")):
    fig, axes = plt.subplots(1, len(methods), figsize=(6.2, 2.4))
    for ax, m in zip(axes, methods):
        cm = analysis[m]["confusion_matrix"]
        mat = np.array([[cm[t][p] for p in LABELS] for t in LABELS], dtype=float)
        row_sum = mat.sum(axis=1, keepdims=True)
        norm = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
        ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue_seq", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"]),
            vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if mat[i, j] > 0:
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                            fontsize=7, color=INK if norm[i, j] < 0.55 else "#ffffff")
        ax.set_xticks(range(len(LABELS)), [SHORT[l] for l in LABELS],
                      fontsize=6.5, rotation=45)
        ax.set_yticks(range(len(LABELS)),
                      [SHORT[l] for l in LABELS] if ax is axes[0] else [])
        if ax is axes[0]:
            ax.set_ylabel("Ground truth", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_title(METHOD_LABELS[m], fontsize=9)
        ax.grid(visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.pdf")
    plt.close(fig)


def fig_similarity_distributions():
    gt = pd.read_csv(BASE / "data" / "baseline" / "gt.csv")
    src = json.loads((BASE / "data" / "extracted" / "en303645_provisions.json").read_text())
    tgt = json.loads((BASE / "data" / "extracted" / "en304223_provisions.json").read_text())
    src_idx = {p["provision_id"]: i for i, p in enumerate(src)}
    tgt_idx = {p["provision_id"]: j for j, p in enumerate(tgt)}

    mats = {
        "SBERT": "sbert_similarity_matrix.npy",
        "BERT": "bert_similarity_matrix.npy",
        "SecureBERT": "securebert_similarity_matrix.npy",
        "Gemini emb.": "gemini_embedding_similarity_matrix.npy",
    }
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.9), sharey=False)
    bins = np.linspace(0, 1, 29)
    for ax, (name, fn) in zip(axes, mats.items()):
        sim = np.load(BASE / "data" / "mappings" / fn)
        pos, neg = [], []
        for r in gt.itertuples():
            s = sim[src_idx[r.src_id], tgt_idx[r.tgt_id]]
            (neg if r.relationship == "NO_RELATION" else pos).append(s)
        ax.hist(pos, bins=bins, density=True, color=C1, alpha=0.75, label="Related")
        ax.hist(neg, bins=bins, density=True, color=C2, alpha=0.75, label="No relation")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Cosine similarity", fontsize=7.5)
        style_axes(ax)
        ax.spines["left"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "similarity_distributions.pdf")
    plt.close(fig)


def fig_threshold_sweep(analysis):
    styles = {
        "rule_based_tfidf": (C4, ":"),
        "sbert": (C3, "-."),
        "bert": (C1, "-"),
        "securebert": (C5, (0, (4, 1.5))),
        "gemini_embedding": (C2, "-"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, (color, ls) in styles.items():
        sweep = analysis.get(m, {}).get("threshold_sweep")
        if not sweep:
            continue
        t = [row["threshold"] for row in sweep]
        f1 = [row["f1"] for row in sweep]
        ax.plot(t, f1, color=color, linestyle=ls, linewidth=1.6,
                label=METHOD_LABELS[m])
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Pair-detection F1")
    ax.set_xlim(0.05, 0.975)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "threshold_sweep.pdf")
    plt.close(fig)


def fig_gt_distribution():
    gt = pd.read_csv(BASE / "data" / "baseline" / "gt.csv")
    counts = gt["relationship"].value_counts()
    order = ["OVERLAP", "COMPLEMENTARITY", "NO_RELATION", "SUBSUMPTION_A_BROADER",
             "SUBSUMPTION_B_BROADER", "EQUIVALENCE"]
    names = {"SUBSUMPTION_A_BROADER": "Subsumption (IoT broader)",
             "SUBSUMPTION_B_BROADER": "Subsumption (AI broader)",
             "OVERLAP": "Overlap", "COMPLEMENTARITY": "Complementarity",
             "NO_RELATION": "No relation", "EQUIVALENCE": "Equivalence"}
    vals = [counts.get(k, 0) for k in order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.barh(y, vals, height=0.62, color=C1)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, str(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [names[k] for k in order])
    ax.set_xlabel("Annotated pairs")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "gt_distribution.pdf")
    plt.close(fig)


def load_weighted() -> dict:
    return json.loads((EVAL_DIR / "weighted_metrics.json").read_text())


def fig_precision_prevalence(weighted):
    """Precision implied by each method's sensitivity and specificity as the base
    rate varies. Sensitivity and specificity do not depend on prevalence, so a
    single measured pair fixes the whole curve."""
    shown = {"sbert": (C3, "-"), "gemini_llm": (C2, "-."),
             "bert": (C1, "--"), "securebert": (C5, ":")}
    pi = np.linspace(0.01, 0.99, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, (color, ls) in shown.items():
        p = weighted["methods"][m]["point"]
        sens, spec = p["recall"], p["specificity"]
        prec = pi * sens / (pi * sens + (1 - pi) * (1 - spec))
        ax.plot(pi, prec, color=color, linestyle=ls, linewidth=1.6,
                label=METHOD_LABELS[m])

    corpus = weighted["prevalence"]["any_relationship"]["prevalence"]
    for x, y, text in [(corpus, 0.02, f"corpus {corpus:.2f}"),
                       (0.794, 0.93, "pilot 0.79")]:
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(x + 0.012, y, text, fontsize=7, color=MUTED, va="bottom")
    lo, hi = weighted["prevalence"]["any_relationship"]["ci"]
    ax.axvspan(lo, hi, color=MUTED, alpha=0.12, linewidth=0)

    ax.set_xlabel("Prevalence of related pairs")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.text(0.66, 0.55, "precision = prevalence:\nno discrimination", fontsize=7,
            color=MUTED, rotation=30, ha="center", va="center")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "precision_vs_prevalence.pdf")
    plt.close(fig)


def fig_stratum_yield(weighted):
    sample = pd.read_csv(BASE / "data" / "baseline" / "gt_sample.csv").dropna(subset=["stratum"])
    sample = sample.assign(positive=sample.relationship != "NO_RELATION")
    strata = ["H", "M", "L"]
    names = {"H": "H\ntop 10%", "M": "M\nnext 30%", "L": "L\nbottom 60%"}
    grouped = sample.groupby("stratum")
    frame = grouped.agg(n=("positive", "size"), pos=("positive", "sum"),
                        N=("N_stratum", "first")).reindex(strata)
    x = np.arange(len(strata))

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    ax = axes[0]
    ax.bar(x - 0.2, frame.N, width=0.36, color=C1, label="In corpus")
    ax.bar(x + 0.2, frame.n, width=0.36, color=C4, label="Annotated")
    for xi, (big, small) in enumerate(zip(frame.N, frame.n)):
        ax.text(xi - 0.2, big + 60, f"{int(big)}", ha="center", fontsize=7, color=INK)
        ax.text(xi + 0.2, small + 60, f"{int(small)}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [names[s] for s in strata], fontsize=8)
    ax.set_ylabel("Pairs")
    ax.set_ylim(0, frame.N.max() * 1.18)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style_axes(ax)

    ax = axes[1]
    rate = frame.pos / frame.n
    ax.bar(x, rate, width=0.5, color=C3)
    for xi, (r, p, n) in enumerate(zip(rate, frame.pos, frame.n)):
        ax.text(xi, r + 0.015, f"{r:.2f}  ({int(p)}/{int(n)})", ha="center",
                fontsize=7, color=INK)
    corpus = weighted["prevalence"]["any_relationship"]["prevalence"]
    ax.axhline(corpus, color=C2, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.text(len(strata) - 0.45, corpus + 0.02, f"corpus {corpus:.2f}",
            ha="right", fontsize=7, color=C2)
    ax.set_xticks(x, [names[s] for s in strata], fontsize=8)
    ax.set_xlim(-0.6, len(strata) - 0.4)
    ax.set_ylabel("Share related")
    ax.set_ylim(0, 0.55)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "stratum_yield.pdf")
    plt.close(fig)


def fig_cv_threshold(analysis):
    methods = [m for m in ORDER if analysis.get(m, {}).get("cv_threshold")]
    x = np.arange(len(methods))
    cv = [analysis[m]["cv_threshold"] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))

    ax = axes[0]
    ax.bar(x - 0.19, [c["in_sample_best_f1"] for c in cv], width=0.34, color=MUTED,
           label="Tuned in sample")
    ax.bar(x + 0.19, [c["cv_f1_mean"] for c in cv], width=0.34, color=C1,
           yerr=[c["cv_f1_sd"] for c in cv], error_kw={"elinewidth": 1, "capsize": 2.5,
                                                       "ecolor": INK},
           label="Held out")
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pair-detection F1")
    ax.legend(frameon=False, fontsize=7.5, ncols=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
    style_axes(ax)

    ax = axes[1]
    ax.errorbar([c["threshold_mean"] for c in cv], x[::-1],
                xerr=[c["threshold_sd"] for c in cv], fmt="o", color=C2, ecolor=C2,
                elinewidth=1.2, capsize=2.5, markersize=4.5)
    ax.set_yticks(x[::-1], [METHOD_LABELS[m] for m in methods])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Threshold selected on training folds")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_threshold.pdf")
    plt.close(fig)


def fig_calibration_gap(weighted):
    """Corpus F1 at the shipped operating point against one calibrated on the sample.

    The shipped thresholds were tuned on the pilot set, where 79% of pairs are
    related. Applying them to a corpus where 12% are is what produces the collapse
    onto the all-positive baseline; the second bar shows how much of that collapse
    is calibration rather than the model.
    """
    methods = [m for m in ORDER if "corpus_calibrated" in weighted["methods"].get(m, {})]
    x = np.arange(len(methods))
    w = 0.36

    shipped = [weighted["methods"][m]["point"]["f1"] for m in methods]
    tuned = [weighted["methods"][m]["corpus_calibrated"]["point"]["f1"] for m in methods]
    err = [[t - weighted["methods"][m]["corpus_calibrated"]["ci"]["f1"][0] for t, m
            in zip(tuned, methods)],
           [weighted["methods"][m]["corpus_calibrated"]["ci"]["f1"][1] - t for t, m
            in zip(tuned, methods)]]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w / 2, shipped, w, color=MUTED, label="Pilot-calibrated threshold")
    ax.bar(x + w / 2, tuned, w, yerr=err, color=C3, ecolor=INK, capsize=2.5,
           error_kw={"elinewidth": 0.8}, label="Corpus-calibrated threshold")

    base = weighted["baseline_all_positive"]["f1"]
    ax.axhline(base, color=C2, linestyle=(0, (4, 3)), linewidth=1.1,
               label=f"All-positive baseline ({base:.2f})")

    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Corpus pair-detection F1")
    ax.set_ylim(0, 0.68)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2,
              columnspacing=1.2)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_gap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    analysis = load_analysis()
    weighted = load_weighted()
    fig_detection_metrics(summary)
    fig_f1_ci(analysis)
    fig_confusion(analysis)
    fig_similarity_distributions()
    fig_threshold_sweep(analysis)
    fig_gt_distribution()
    fig_precision_prevalence(weighted)
    fig_stratum_yield(weighted)
    fig_cv_threshold(analysis)
    fig_calibration_gap(weighted)
    print(f"Figures written to {FIG_DIR}")
