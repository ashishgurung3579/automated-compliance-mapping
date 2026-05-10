"""Rule-based compliance mapping using lexical and keyword similarity."""
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parents[2]
DATA = BASE / "data"


# Security domain vocabulary.

SECURITY_KEYWORDS = {
    # Authentication & credentials
    "password", "authentication", "credential", "token", "certificate",
    "cryptography", "cipher", "hash", "signature", "tls", "ssl", "biometric",
    "brute", "force",
    # Software lifecycle
    "update", "patch", "firmware", "upgrade", "rollback",
    # Vulnerability management
    "vulnerability", "cve", "disclosure", "exploit", "remediation",
    # Core properties
    "integrity", "authenticity", "availability", "confidentiality",
    # Access control
    "authorization", "privilege", "permission",
    # Monitoring
    "monitoring", "logging", "audit", "telemetry", "anomaly",
    # Supply chain
    "supply", "vendor",
    # AI/ML specific
    "poisoning", "adversarial", "inference", "inversion", "hallucination",
    "dataset",
    # Input handling
    "injection", "validation", "sanitize",
    # Storage
    "encryption", "persistent", "sensitive", "parameter", "hardcoded",
    # Network
    "network", "protocol", "firewall", "interface",
    # Resilience
    "resilience", "outage", "recovery", "redundancy",
    # Privacy
    "privacy", "deletion", "retention",
    # Lifecycle
    "lifecycle", "disposal", "decommission",
    # Risk
    "risk", "assessment", "oversight", "threat",
}


def _extract_keywords(text: str) -> set[str]:
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    return words & SECURITY_KEYWORDS


def _provision_text(p: dict) -> str:
    return p["text"]


def load_provisions() -> tuple[list[dict], list[dict]]:
    p1 = json.loads((DATA / "extracted" / "en303645_provisions.json").read_text())
    p2 = json.loads((DATA / "extracted" / "en304223_provisions.json").read_text())
    p1 = [p for p in p1 if not p.get("is_void", False)]
    return p1, p2


def tfidf_mapping(
    src: list[dict],
    tgt: list[dict],
    threshold: float = 0.10,
) -> pd.DataFrame:
    src_texts = [_provision_text(p) for p in src]
    tgt_texts = [_provision_text(p) for p in tgt]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    all_texts = src_texts + tgt_texts
    vectorizer.fit(all_texts)

    src_vecs = vectorizer.transform(src_texts)
    tgt_vecs = vectorizer.transform(tgt_texts)

    sim_matrix = cosine_similarity(src_vecs, tgt_vecs)

    rows = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                rows.append({
                    "src_id": sp["provision_id"],
                    "tgt_id": tp["provision_id"],
                    "tfidf_score": round(score, 4),
                    "predicted_rel": _score_to_relation(score),
                })

    return pd.DataFrame(rows).sort_values("tfidf_score", ascending=False)


def keyword_jaccard_mapping(
    src: list[dict],
    tgt: list[dict],
    threshold: float = 0.05,
) -> pd.DataFrame:
    src_kw = [_extract_keywords(_provision_text(p)) for p in src]
    tgt_kw = [_extract_keywords(_provision_text(p)) for p in tgt]

    rows = []
    for i, sp in enumerate(src):
        for j, tp in enumerate(tgt):
            a, b = src_kw[i], tgt_kw[j]
            union = a | b
            if not union:
                continue
            score = len(a & b) / len(union)
            if score >= threshold:
                rows.append({
                    "src_id": sp["provision_id"],
                    "tgt_id": tp["provision_id"],
                    "jaccard_score": round(score, 4),
                    "shared_keywords": sorted(a & b),
                    "predicted_rel": _score_to_relation(score),
                })

    return pd.DataFrame(rows).sort_values("jaccard_score", ascending=False)


def _score_to_relation(score: float) -> str:
    """Heuristic relationship classification from similarity score."""
    if score >= 0.6:
        return "EQUIVALENCE"
    if score >= 0.35:
        return "OVERLAP"
    if score >= 0.15:
        return "COMPLEMENTARITY"
    return "NO_RELATION"


def compute_tfidf_matrix(src: list[dict], tgt: list[dict]) -> np.ndarray:
    src_texts = [_provision_text(p) for p in src]
    tgt_texts = [_provision_text(p) for p in tgt]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(src_texts + tgt_texts)
    return cosine_similarity(vec.transform(src_texts), vec.transform(tgt_texts))


if __name__ == "__main__":
    src, tgt = load_provisions()
    print(f"Mapping {len(src)} EN 303 645 provisions to {len(tgt)} EN 304 223 provisions")

    out_dir = DATA / "mappings"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_tfidf = tfidf_mapping(src, tgt)
    print(f"\nTF-IDF candidates (threshold=0.10): {len(df_tfidf)}")
    print(df_tfidf.head(10).to_string(index=False))
    df_tfidf.to_csv(out_dir / "rule_based_tfidf.csv", index=False)

    df_jac = keyword_jaccard_mapping(src, tgt)
    print(f"\nJaccard candidates (threshold=0.05): {len(df_jac)}")
    print(df_jac.head(10).to_string(index=False))
    df_jac.to_csv(out_dir / "rule_based_jaccard.csv", index=False)

    matrix = compute_tfidf_matrix(src, tgt)
    np.save(out_dir / "tfidf_similarity_matrix.npy", matrix)
    print(f"\nSaved similarity matrix: {matrix.shape}")
