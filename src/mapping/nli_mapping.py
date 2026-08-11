"""
Natural language inference as compliance mapping.

A cross-encoder reads both provisions together and answers whether the first
entails the second, so scoring each pair in both directions recovers subsumption
direction, which a symmetric cosine cannot express.

Each pair is scored as (src premise, tgt hypothesis) and the reverse, and the
entailment probabilities decide the label:

    both directions entail          -> EQUIVALENCE
    src entails tgt only            -> SUBSUMPTION_B_BROADER  (tgt is the weaker claim)
    tgt entails src only            -> SUBSUMPTION_A_BROADER
    neither, but some entailment    -> OVERLAP
    no entailment either way        -> NO_RELATION

Both cut points are fixed before the evaluation. The stored similarity matrix is
the larger of the two entailment probabilities, so the method can be swept and
recalibrated like the embedding methods.

Costs 2 x 4,968 forward passes; a few minutes on Apple Silicon.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.mapping.embeddings import load_provisions
from src.methods import METHODS

METHOD = METHODS["nli"]
ENTAILMENT_T = 0.50   # a direction counts as entailed
FLOOR_T = METHOD.threshold   # below this, no relation is emitted at all

# The longest tokenised pair in this corpus is 304 tokens, so nothing is
# truncated at 320. Batches pad to their own longest sequence, which is well
# under that for most pairs.
MAX_LENGTH = 320

# Small on purpose: on 8 GB of unified memory a larger batch swaps and throughput
# collapses. Override on a machine with more headroom.
BATCH_SIZE = int(os.environ.get("NLI_BATCH_SIZE", "8"))


def _device() -> torch.device:
    override = os.environ.get("NLI_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def entailment_probabilities(premises: list[str], hypotheses: list[str],
                             tokenizer, model, device, entail_idx: int) -> np.ndarray:
    out = []
    started = time.perf_counter()
    total = len(premises)
    for i in range(0, len(premises), BATCH_SIZE):
        if i and i % (BATCH_SIZE * 20) == 0:
            rate = i / (time.perf_counter() - started)
            print(f"    {i}/{total} pairs  {rate:.0f}/s  "
                  f"eta {(total - i) / rate:.0f} s", flush=True)
        encoded = tokenizer(
            premises[i:i + BATCH_SIZE], hypotheses[i:i + BATCH_SIZE],
            padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[:, entail_idx]
        out.append(probs.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def label(p_forward: float, p_backward: float) -> str:
    forward = p_forward >= ENTAILMENT_T
    backward = p_backward >= ENTAILMENT_T
    if forward and backward:
        return "EQUIVALENCE"
    if forward:
        return "SUBSUMPTION_B_BROADER"
    if backward:
        return "SUBSUMPTION_A_BROADER"
    if max(p_forward, p_backward) >= FLOOR_T:
        return "OVERLAP"
    return "NO_RELATION"


if __name__ == "__main__":
    src, tgt = load_provisions()
    print(METHOD.long_label)
    print(f"Scoring {len(src) * len(tgt)} pairs in both directions")

    tokenizer = AutoTokenizer.from_pretrained(METHOD.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(METHOD.model_id)
    model.eval()
    device = _device()
    model.to(device)
    print(f"  Device: {device}")

    # Label order differs between NLI checkpoints, so read it rather than assume.
    entail_idx = next(i for i, name in model.config.id2label.items()
                      if name.lower().startswith("entail"))

    src_texts = [p["text"] for p in src]
    tgt_texts = [p["text"] for p in tgt]
    premises_fwd, hypotheses_fwd = [], []
    for s in src_texts:
        for t in tgt_texts:
            premises_fwd.append(s)
            hypotheses_fwd.append(t)

    started = time.perf_counter()
    print("  Direction 1: EN 303 645 entails EN 304 223 ...")
    fwd = entailment_probabilities(premises_fwd, hypotheses_fwd,
                                   tokenizer, model, device, entail_idx)
    print("  Direction 2: EN 304 223 entails EN 303 645 ...")
    bwd = entailment_probabilities(hypotheses_fwd, premises_fwd,
                                   tokenizer, model, device, entail_idx)
    elapsed = time.perf_counter() - started

    shape = (len(src), len(tgt))
    p_fwd = fwd.reshape(shape)
    p_bwd = bwd.reshape(shape)
    sim = np.maximum(p_fwd, p_bwd)

    METHOD.matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(METHOD.matrix, sim)
    np.save(METHOD.matrix.with_name("nli_entailment_directions.npy"),
            np.stack([p_fwd, p_bwd]))

    rows = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            rel = label(float(p_fwd[i, j]), float(p_bwd[i, j]))
            if rel == "NO_RELATION":
                continue
            rows.append({
                "src_id": sp["provision_id"],
                "tgt_id": tp["provision_id"],
                METHOD.score_column: round(float(sim[i, j]), 4),
                "p_src_entails_tgt": round(float(p_fwd[i, j]), 4),
                "p_tgt_entails_src": round(float(p_bwd[i, j]), 4),
                "predicted_rel": rel,
            })

    df = pd.DataFrame(rows).sort_values(METHOD.score_column, ascending=False)
    df.to_csv(METHOD.predictions, index=False)
    print(f"Candidates (floor={FLOOR_T}, entailment={ENTAILMENT_T}): {len(df)}")
    print(df["predicted_rel"].value_counts().to_dict())
    print(f"Saved: {METHOD.predictions}  [{elapsed:.1f} s]")
