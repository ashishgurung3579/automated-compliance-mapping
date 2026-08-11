"""
Re-run the local methods on requirement text alone, without the explanatory
trailers EN 303 645 prints beneath many of its provisions.

Section 3.2.2 of the thesis keeps those trailers, and Section 5.9 notes that
they are 53% of that standard's extracted words, so the compression documented
in Section 5.3 could be the input rather than the model geometry. This module
settles which, by re-scoring the same pairs on text stripped to its normative
sentences and recalibrating each method exactly as Section 5.4 does.

The two hosted methods are archived as stored outputs and cannot be re-scored,
so they are absent here by necessity rather than by choice.

Writes data/ablation/. Never touches data/extracted/ or data/mappings/: every
number already reported in the thesis is computed from those, and this
experiment must not be able to move them.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import analysis as A
from src.mapping.embeddings import (
    cosine_sim_matrix,
    embed_with_sbert,
    embed_with_transformers,
    load_provisions,
)
from src.mapping.rule_based import compute_tfidf_matrix
from src.methods import METHODS, embedding_methods

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
OUT_DIR = DATA / "ablation"
REFERENCE = DATA / "baseline" / "reference_set.csv"

# ETSI states requirements with "shall" (mandatory) and "should" (recommended).
# "may" is deliberately excluded: it appears inside explanatory prose too
# ("these interfaces may have been used during development"), so keying on it
# would pull trailers back in.
_MODAL = re.compile(r"\b(shall|should)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.;])\s+")


def requirement_only(text: str) -> str:
    """The normative sentences of a provision, dropping unlabelled explanation.

    The trailers carry no marker of their own -- the extractor already removed
    the ones EN 303 645 labels NOTE or EXAMPLE -- so they are identified by the
    absence of a normative modal. A provision with no modal at all keeps its
    first sentence, which leaves no provision empty on this corpus.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    sentences = _SENTENCE.split(flat)
    kept = [s for s in sentences if _MODAL.search(s)]
    return " ".join(kept) if kept else sentences[0]


def strip_provisions(provisions: list[dict]) -> list[dict]:
    return [{**p, "text": requirement_only(p["text"])} for p in provisions]


def text_stats(original: list[dict], stripped: list[dict]) -> dict:
    words = lambda ps: sum(len(p["text"].split()) for p in ps)
    shortened = sum(
        1 for a, b in zip(original, stripped)
        if len(b["text"].split()) < len(a["text"].split())
    )
    return {
        "provisions": len(original),
        "words_full": words(original),
        "words_requirement_only": words(stripped),
        "share_dropped": round(1 - words(stripped) / words(original), 4),
        "provisions_shortened": shortened,
    }


def build_matrices(src: list[dict], tgt: list[dict], with_nli: bool) -> dict[str, float]:
    """Score every pair on the stripped text, one matrix per local method."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtimes = {}

    started = time.perf_counter()
    np.save(OUT_DIR / "rule_based_tfidf_requirement_only.npy",
            compute_tfidf_matrix(src, tgt))
    runtimes["rule_based_tfidf"] = round(time.perf_counter() - started, 1)
    print("  rule_based_tfidf done")

    src_texts = [p["text"] for p in src]
    tgt_texts = [p["text"] for p in tgt]

    for key, method in embedding_methods().items():
        print(f"  {key} ({method.model_id}) ...", flush=True)
        started = time.perf_counter()
        if method.encoder == "sbert":
            src_emb = embed_with_sbert(method.model_id, src_texts)
            tgt_emb = embed_with_sbert(method.model_id, tgt_texts)
        else:
            src_emb = embed_with_transformers(
                method.model_id, src_texts, use_safetensors=method.use_safetensors)
            tgt_emb = embed_with_transformers(
                method.model_id, tgt_texts, use_safetensors=method.use_safetensors)
        np.save(OUT_DIR / f"{key}_requirement_only.npy",
                cosine_sim_matrix(src_emb, tgt_emb))
        runtimes[key] = round(time.perf_counter() - started, 1)

    if with_nli:
        runtimes["nli"] = _build_nli(src_texts, tgt_texts)

    return runtimes


def _build_nli(src_texts: list[str], tgt_texts: list[str]) -> float:
    """The cross-encoder condition, reusing the scorer of src.mapping.nli_mapping."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.mapping.nli_mapping import _device, entailment_probabilities

    method = METHODS["nli"]
    print(f"  nli ({method.model_id}), 2 x {len(src_texts) * len(tgt_texts)} passes ...",
          flush=True)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(method.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(method.model_id)
    model.eval()
    device = _device()
    model.to(device)
    entail_idx = next(i for i, name in model.config.id2label.items()
                      if name.lower().startswith("entail"))

    premises, hypotheses = [], []
    for s in src_texts:
        for t in tgt_texts:
            premises.append(s)
            hypotheses.append(t)

    fwd = entailment_probabilities(premises, hypotheses, tokenizer, model,
                                   device, entail_idx)
    bwd = entailment_probabilities(hypotheses, premises, tokenizer, model,
                                   device, entail_idx)
    shape = (len(src_texts), len(tgt_texts))
    p_fwd, p_bwd = fwd.reshape(shape), bwd.reshape(shape)
    np.save(OUT_DIR / "nli_requirement_only.npy", np.maximum(p_fwd, p_bwd))
    np.save(OUT_DIR / "nli_entailment_directions_requirement_only.npy",
            np.stack([p_fwd, p_bwd]))
    return round(time.perf_counter() - started, 1)


def condition(matrix: np.ndarray, gt: pd.DataFrame,
              src_ids: list[str], tgt_ids: list[str]) -> dict:
    """Range width over the whole corpus, plus the calibrated operating point.

    The calibration is the same procedure Section 5.4 applies to the full-text
    condition, so the two are comparable: the threshold is re-selected by the
    same repeated stratified cross-validation rather than carried over.
    """
    scores, positive = A.pair_scores(matrix, gt, src_ids, tgt_ids)
    cv = A.cv_threshold(scores, positive)
    return {
        "score_min": round(float(matrix.min()), 4),
        "score_max": round(float(matrix.max()), 4),
        "range_width": round(float(matrix.max() - matrix.min()), 4),
        "threshold": cv["threshold"],
        "calibrated_f1": cv["calibrated"]["f1"],
        "calibrated_precision": cv["calibrated"]["precision"],
        "calibrated_recall": cv["calibrated"]["recall"],
        "held_out_f1": cv["cv_f1_mean"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-nli", action="store_true",
                        help="include the cross-encoder (about 5 minutes)")
    parser.add_argument("--scores-only", action="store_true",
                        help="skip re-scoring, recompute the comparison from stored matrices")
    args = parser.parse_args()

    src_full, tgt_full = load_provisions()
    src, tgt = strip_provisions(src_full), strip_provisions(tgt_full)

    stats = {
        "EN 303 645": text_stats(src_full, src),
        "EN 304 223": text_stats(tgt_full, tgt),
    }
    for standard, s in stats.items():
        print(f"{standard}: {s['words_full']} -> {s['words_requirement_only']} words "
              f"({s['share_dropped']:.0%} dropped), "
              f"{s['provisions_shortened']}/{s['provisions']} provisions shortened")

    runtimes = {}
    if not args.scores_only:
        print("\nRe-scoring on requirement text only:")
        runtimes = build_matrices(src, tgt, args.with_nli)

    print("\nComparing conditions on the 200-pair reference set:")
    gt = pd.read_csv(REFERENCE)
    src_ids, tgt_ids = A.load_provision_ids()

    comparison = {}
    for key, method in METHODS.items():
        ablated_path = OUT_DIR / f"{key}_requirement_only.npy"
        if method.matrix is None or not method.matrix.exists() or not ablated_path.exists():
            continue
        full = condition(np.load(method.matrix), gt, src_ids, tgt_ids)
        ablated = condition(np.load(ablated_path), gt, src_ids, tgt_ids)
        comparison[key] = {
            "label": method.label,
            "family": method.family,
            "full_text": full,
            "requirement_only": ablated,
            "delta_f1": round(ablated["calibrated_f1"] - full["calibrated_f1"], 4),
            "delta_range_width": round(
                ablated["range_width"] - full["range_width"], 4),
        }

    header = f"{'method':<20}{'F1 full':>9}{'F1 req':>9}{'delta':>8}" \
             f"{'width full':>12}{'width req':>11}{'delta':>8}"
    print(header)
    for key, c in comparison.items():
        print(f"{c['label']:<20}{c['full_text']['calibrated_f1']:>9.3f}"
              f"{c['requirement_only']['calibrated_f1']:>9.3f}{c['delta_f1']:>+8.3f}"
              f"{c['full_text']['range_width']:>12.3f}"
              f"{c['requirement_only']['range_width']:>11.3f}"
              f"{c['delta_range_width']:>+8.3f}")

    payload = {
        "text_statistics": stats,
        "runtimes_seconds": runtimes,
        "methods": comparison,
        "excluded": {
            "gemini_embedding": "hosted, archived as stored output",
            "gemini_llm": "hosted, archived as stored output",
            "rule_based_jaccard": "no stored score matrix",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT_DIR / 'ablation.json'}")
