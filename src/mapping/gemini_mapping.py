"""
Gemini LLM compliance mapping using the managed Gemini Batch API.

Gemini independently classifies every EN 303 645 provision against every
EN 304 223 provision in one asynchronous batch job.

Requires GEMINI_API_KEY env var.
"""
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

from src.mapping.gemini_batch import inline_generate_responses, response_text, wait_for_batch

BASE = Path(__file__).parents[2]
DATA = BASE / "data"
LLM_MODEL = "gemini-2.5-flash-lite"

VALID_LABELS = [
    "EQUIVALENCE",
    "SUBSUMPTION_A_BROADER",
    "SUBSUMPTION_B_BROADER",
    "OVERLAP",
    "COMPLEMENTARITY",
    "NO_RELATION",
]

PROMPT_TEMPLATE = """\
You are a cybersecurity compliance expert. Classify the relationship between these two security requirements.

Requirement A (EN 303 645 - IoT security): {src_text}

Requirement B (EN 304 223 - AI security): {tgt_text}

Relationship options:
- EQUIVALENCE: Same security concern, essentially same scope
- SUBSUMPTION_A_BROADER: Requirement A fully covers and goes beyond Requirement B
- SUBSUMPTION_B_BROADER: Requirement B fully covers and goes beyond Requirement A
- OVERLAP: Partially shared scope, each has unique elements
- COMPLEMENTARITY: Related but different concerns that work together
- NO_RELATION: Fundamentally different security concerns

Respond with exactly one label from the list above. No explanation."""


def load_provisions() -> tuple[list[dict], list[dict]]:
    src = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    tgt = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    return src, tgt


def _parse_label(text: str) -> str:
    label = text.strip().upper()
    if label in VALID_LABELS:
        return label
    for vl in VALID_LABELS:
        if vl in label:
            return vl
    return "NO_RELATION"


def build_request(src_text: str, tgt_text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(src_text=src_text, tgt_text=tgt_text)
    return {
        "contents": [
            {
                "parts": [{"text": prompt}],
                "role": "user",
            }
        ],
        "config": {
            "temperature": 0,
            "candidate_count": 1,
        },
    }


def build_candidate_pairs(src: list[dict], tgt: list[dict]) -> pd.DataFrame:
    rows = []
    for sp in src:
        for tp in tgt:
            rows.append({
                "src_id": sp["provision_id"],
                "tgt_id": tp["provision_id"],
                "src_text": sp["text"],
                "tgt_text": tp["text"],
            })
    return pd.DataFrame(rows)


def invoke_batch(client, requests: list[dict]):
    job = client.batches.create(
        model=LLM_MODEL,
        src=requests,
        config={"display_name": "compliance-gemini-independent-mapping"},
    )
    return wait_for_batch(client, job)


def classify_all(
    client,
    candidates: pd.DataFrame,
) -> list[dict]:
    requests = [
        build_request(
            src_text=row.src_text,
            tgt_text=row.tgt_text,
        )
        for row in candidates.itertuples()
    ]

    job = invoke_batch(client, requests)
    inline_responses = inline_generate_responses(job)
    if len(inline_responses) != len(candidates):
        raise RuntimeError(
            f"Expected {len(candidates)} LLM responses, got {len(inline_responses)}"
        )

    results = []
    for row, inline_response in zip(candidates.itertuples(), inline_responses):
        if getattr(inline_response, "error", None):
            print(
                f"  Batch item failed for {row.src_id}->{row.tgt_id}: "
                f"{inline_response.error}; defaulting to NO_RELATION"
            )
            label = "NO_RELATION"
        else:
            label = _parse_label(response_text(inline_response.response))

        results.append({
            "src_id": row.src_id,
            "tgt_id": row.tgt_id,
            "predicted_rel": label,
        })
    return results


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY env var before running")

    client = genai.Client(api_key=api_key)

    src, tgt = load_provisions()
    candidates = build_candidate_pairs(src, tgt)
    print(f"Loaded {len(src)} src + {len(tgt)} tgt provisions")
    print(f"Gemini independent candidate pairs: {len(candidates)}")
    print(f"Invoking Gemini Batch API LLM classify (model={LLM_MODEL})")

    results = classify_all(client, candidates)

    out_dir = DATA / "mappings"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "gemini_output.csv", index=False)
    print(f"Classified {len(df)} pairs.")
    print(df["predicted_rel"].value_counts().to_dict())
    print(f"Saved: {out_dir / 'gemini_output.csv'}")


if __name__ == "__main__":
    main()
