"""
Assemble the reference set the evaluation runs against.

Pools every pair judged in the pilot, screened and targeted rounds, then draws
under one rule: negatives are half the set, positives balanced across the five
positive classes as far as the material allows. Equivalence and subsumption fall
short of their share and enter at their full available count.

Writes data/baseline/reference_set.csv.
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
OUT = DATA / "baseline" / "reference_set.csv"

SEED = 42
TARGET_SIZE = 200

POSITIVE_LABELS = [
    "EQUIVALENCE",
    "SUBSUMPTION_A_BROADER",
    "SUBSUMPTION_B_BROADER",
    "OVERLAP",
    "COMPLEMENTARITY",
]
NEGATIVE_LABEL = "NO_RELATION"

COLUMNS = [
    "src_id", "tgt_id", "relationship", "justification",
    "confidence", "round", "author_reviewed",
]


def pool() -> pd.DataFrame:
    """Every judged pair, one row each, earliest round winning on duplicates.

    The pilot goes first: its labels are the reviewed ones. A later round's second
    judgement of the same pair goes to the agreement analysis instead.
    """
    frames = []

    pilot = pd.read_csv(DATA / "baseline" / "gt.csv")
    pilot["round"] = "pilot"
    pilot["author_reviewed"] = True
    frames.append(pilot)

    screened = pd.read_csv(DATA / "baseline" / "gt_sample.csv")
    screened["round"] = "screened"
    screened["author_reviewed"] = False
    frames.append(screened)

    rare_path = DATA / "baseline" / "rare_class_pairs.csv"
    if rare_path.exists():
        rare = pd.read_csv(rare_path)
        rare["round"] = "rare_search"
        rare["author_reviewed"] = False
        frames.append(rare)

    combined = pd.concat([f[COLUMNS] for f in frames], ignore_index=True)
    return combined.drop_duplicates(subset=["src_id", "tgt_id"], keep="first")


def allocate(available: dict[str, int]) -> dict[str, int]:
    """Target count per label under the composition rule.

    A class that cannot meet its share contributes everything it has, and the
    remainder is redistributed across the classes that can still absorb it.
    """
    n_negative = TARGET_SIZE // 2
    quota = {NEGATIVE_LABEL: min(n_negative, available[NEGATIVE_LABEL])}

    remaining = TARGET_SIZE - quota[NEGATIVE_LABEL]
    unfilled = list(POSITIVE_LABELS)

    while unfilled and remaining > 0:
        share = remaining // len(unfilled)
        if share == 0:
            break
        capped = [lab for lab in unfilled if available.get(lab, 0) <= share]
        if not capped:
            for lab in unfilled:
                quota[lab] = share
            remaining -= share * len(unfilled)
            break
        for lab in capped:
            quota[lab] = available.get(lab, 0)
            remaining -= quota[lab]
            unfilled.remove(lab)

    # Whatever is still unallocated goes to the largest class that can take it.
    if remaining > 0 and unfilled:
        spare = max(unfilled, key=lambda lab: available.get(lab, 0) - quota.get(lab, 0))
        quota[spare] = quota.get(spare, 0) + remaining

    return quota


if __name__ == "__main__":
    judged = pool()
    available = judged.relationship.value_counts().to_dict()
    quota = allocate(available)

    drawn = []
    for label, n in quota.items():
        block = judged[judged.relationship == label]
        take = min(n, len(block))
        drawn.append(block.sample(n=take, random_state=SEED))

    reference = pd.concat(drawn, ignore_index=True)
    reference = reference.sort_values(["src_id", "tgt_id"]).reset_index(drop=True)
    reference.to_csv(OUT, index=False)

    positives = (reference.relationship != NEGATIVE_LABEL).sum()
    rate = positives / len(reference)
    print(f"Pool: {len(judged)} judged pairs across {judged['round'].nunique()} rounds")
    for label in [NEGATIVE_LABEL] + POSITIVE_LABELS:
        got = int((reference.relationship == label).sum())
        have = int(available.get(label, 0))
        flag = "  (all available)" if got == have and got < TARGET_SIZE // 10 else ""
        print(f"  {label:<24} {got:>4} of {have:>4} judged{flag}")
    print(f"Reference set: {len(reference)} pairs, positive rate {rate:.3f}")
    print(f"All-positive baseline F1: {2 * rate / (1 + rate):.3f}")
    print(f"Author-reviewed: {int(reference.author_reviewed.sum())}")
    print(f"Saved {OUT}")
