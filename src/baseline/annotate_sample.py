"""
Annotation harness for the stratified reference sample.

Emits batches of provision pairs as JSON, ingests returned labels, and assembles
data/baseline/gt_sample.csv. Splitting emit from ingest keeps the work resumable and
leaves an auditable record of exactly which pairs were judged in which batch.

Usage:
    python3 -m src.baseline.annotate_sample emit          # write pending batches
    python3 -m src.baseline.annotate_sample ingest        # merge completed batches
    python3 -m src.baseline.annotate_sample status
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
WORK = DATA / "annotation"
OUT = DATA / "baseline" / "gt_sample.csv"

BATCH_SIZE = 40
SEED = 20260804

LABELS = [
    "EQUIVALENCE",
    "SUBSUMPTION_A_BROADER",
    "SUBSUMPTION_B_BROADER",
    "OVERLAP",
    "COMPLEMENTARITY",
    "NO_RELATION",
]


def load_provisions() -> tuple[dict, dict]:
    src = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    tgt = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    return ({p["provision_id"]: p for p in src}, {p["provision_id"]: p for p in tgt})


def worklist() -> pd.DataFrame:
    """Sampled pairs plus the pilot pairs, shuffled into a single blind pass.

    The pilot pairs ride along so that agreement against the earlier human-reviewed
    labels is measured on all 107 of them rather than the 23 the sample happened to
    hit. They are not marked in the emitted batches.
    """
    sample = pd.read_csv(DATA / "baseline" / "reference_pool.csv")
    sample["source"] = "sample"

    pilot = pd.read_csv(DATA / "baseline" / "gt.csv")[["src_id", "tgt_id"]]
    pilot["source"] = "pilot"

    combined = pd.concat([sample, pilot], ignore_index=True)
    combined = combined.drop_duplicates(subset=["src_id", "tgt_id"], keep="first")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(combined))
    combined = combined.iloc[order].reset_index(drop=True)
    combined["batch"] = combined.index // BATCH_SIZE
    return combined


def emit(work: pd.DataFrame) -> None:
    src_map, tgt_map = load_provisions()
    WORK.mkdir(parents=True, exist_ok=True)

    for batch_id, block in work.groupby("batch"):
        path = WORK / f"batch_{batch_id:03d}.json"
        if (WORK / f"batch_{batch_id:03d}_labels.json").exists():
            continue
        items = [
            {
                "pair_id": f"{r.src_id}||{r.tgt_id}",
                "requirement_A_en303645": src_map[r.src_id]["text"].strip(),
                "requirement_B_en304223": tgt_map[r.tgt_id]["text"].strip(),
            }
            for r in block.itertuples()
        ]
        path.write_text(json.dumps({"batch": int(batch_id), "items": items}, indent=2))
    print(f"{work.batch.nunique()} batches of up to {BATCH_SIZE} pairs in {WORK}")


def ingest(work: pd.DataFrame) -> None:
    records = {}
    for path in sorted(WORK.glob("batch_*_labels.json")):
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
    merged = work.merge(annotated, on=["src_id", "tgt_id"], how="inner")
    merged.to_csv(OUT, index=False)

    print(f"Annotated {len(merged)} of {len(work)} pairs")
    print(merged.relationship.value_counts().to_dict())
    print(f"Saved {OUT}")


def status(work: pd.DataFrame) -> None:
    done = len(list(WORK.glob("batch_*_labels.json")))
    total = work.batch.nunique()
    print(f"{done}/{total} batches labelled ({done * BATCH_SIZE} of {len(work)} pairs)")
    pending = [
        b for b in range(total)
        if not (WORK / f"batch_{b:03d}_labels.json").exists()
    ]
    if pending:
        print(f"Next pending: {pending[:8]}{' ...' if len(pending) > 8 else ''}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    work = worklist()
    {"emit": emit, "ingest": ingest, "status": status}[command](work)
