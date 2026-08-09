"""
Targeted search for the rare relationship classes.

Ranks the pairs no earlier round judged by mean percentile similarity across every
method, so no single method is privileged, and emits the top ones for annotation.
The harvest is conditioned on that screen, so per-class recall for the classes it
fills is not unbiased.

Usage:
    python3 -m src.baseline.rare_class_search emit     # rank and write batches
    python3 -m src.baseline.rare_class_search ingest   # merge returned labels
    python3 -m src.baseline.rare_class_search status
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.baseline.annotate_sample import LABELS, load_provisions
from src.baseline.sample_reference import screening_scores
from src.evaluation.analysis import load_provision_ids

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
WORK = DATA / "annotation" / "rare"
OUT = DATA / "baseline" / "rare_class_pairs.csv"

BATCH_SIZE = 40
N_CANDIDATES = 300


def annotated_pairs() -> set[tuple[str, str]]:
    """Every pair any earlier round already judged."""
    seen: set[tuple[str, str]] = set()
    for name in ("gt.csv", "gt_sample.csv"):
        path = DATA / "baseline" / name
        if path.exists():
            df = pd.read_csv(path)
            seen |= set(zip(df.src_id, df.tgt_id))
    return seen


def candidates() -> pd.DataFrame:
    src_ids, tgt_ids = load_provision_ids()
    pool = screening_scores(src_ids, tgt_ids)

    seen = annotated_pairs()
    mask = [(s, t) not in seen for s, t in zip(pool.src_id, pool.tgt_id)]
    pool = pool[mask]

    pool = pool.sort_values("screening_score", ascending=False).head(N_CANDIDATES)
    pool = pool.reset_index(drop=True)
    pool["batch"] = pool.index // BATCH_SIZE
    return pool


def emit(work: pd.DataFrame) -> None:
    src_map, tgt_map = load_provisions()
    WORK.mkdir(parents=True, exist_ok=True)

    for batch_id, block in work.groupby("batch"):
        if (WORK / f"rare_{batch_id:03d}_labels.json").exists():
            continue
        items = [
            {
                "pair_id": f"{r.src_id}||{r.tgt_id}",
                "requirement_A_en303645": src_map[r.src_id]["text"].strip(),
                "requirement_B_en304223": tgt_map[r.tgt_id]["text"].strip(),
            }
            for r in block.itertuples()
        ]
        path = WORK / f"rare_{batch_id:03d}.json"
        path.write_text(json.dumps({"batch": int(batch_id), "items": items}, indent=2))

    print(f"{work.batch.nunique()} batches of up to {BATCH_SIZE} pairs in {WORK}")


def ingest(work: pd.DataFrame) -> None:
    records = {}
    for path in sorted(WORK.glob("rare_*_labels.json")):
        payload = json.loads(path.read_text())
        for item in payload["items"]:
            label = item["relationship"].strip().upper()
            if label not in LABELS:
                raise ValueError(f"{path.name}: unknown label {label!r} for {item['pair_id']}")
            src_id, tgt_id = item["pair_id"].split("||")
            records[(src_id, tgt_id)] = {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "relationship": label,
                "justification": item.get("justification", "").strip(),
                "confidence": int(item.get("confidence", 2)),
                "batch": payload["batch"],
            }

    if not records:
        raise SystemExit("No completed batches found. Run emit, label, then ingest.")

    annotated = pd.DataFrame(records.values())
    annotated["source"] = "rare_search"
    annotated.to_csv(OUT, index=False)

    print(f"Annotated {len(annotated)} of {len(work)} candidates")
    print(annotated.relationship.value_counts().to_dict())
    print(f"Saved {OUT}")


def status(work: pd.DataFrame) -> None:
    done = len(list(WORK.glob("rare_*_labels.json")))
    total = work.batch.nunique()
    print(f"{done}/{total} batches labelled ({done * BATCH_SIZE} of {len(work)} candidates)")
    pending = [b for b in range(total) if not (WORK / f"rare_{b:03d}_labels.json").exists()]
    if pending:
        print(f"Next pending: {pending[:8]}{' ...' if len(pending) > 8 else ''}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"emit": emit, "ingest": ingest, "status": status}[command](candidates())
