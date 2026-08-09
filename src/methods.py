"""
The set of mapping methods under evaluation, in one place.

The mapping runners, the evaluation, the calibration analysis, the figures and
the thesis-table guard all import from here, so a method is defined once.
Thresholds are set per family, not per model.
"""
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).parents[1]
DATA = BASE / "data"
MAP_DIR = DATA / "mappings"

# Family -> threshold applied to every member, fixed before the evaluation.
FAMILY_LABELS = {
    "lexical": "Lexical",
    "sentence-embedding": "Sentence embedding",
    "contextual-embedding": "Contextual embedding",
    "cross-encoder": "Cross-encoder",
    "api": "Hosted API",
}


@dataclass(frozen=True)
class Method:
    key: str
    label: str            # short, for figures and thesis table rows
    long_label: str       # with the model identifier, for the report
    family: str
    model_id: str | None  # None for the two lexical baselines
    threshold: float | None
    predictions: Path     # data/mappings/*.csv
    matrix: Path | None   # similarity matrix, when the method produces a score
    score_column: str | None
    # "sbert" for a SentenceTransformer head, "transformers" for mean-pooled MLMs.
    encoder: str | None = None
    use_safetensors: bool | None = None


def _m(key, label, long_label, family, model_id, threshold,
       predictions, matrix=None, score_column=None,
       encoder=None, use_safetensors=None) -> Method:
    if encoder is None:
        encoder = {"sentence-embedding": "sbert",
                   "contextual-embedding": "transformers"}.get(family)
    return Method(
        key=key, label=label, long_label=long_label, family=family,
        model_id=model_id, threshold=threshold,
        predictions=MAP_DIR / predictions,
        matrix=MAP_DIR / matrix if matrix else None,
        score_column=score_column,
        encoder=encoder, use_safetensors=use_safetensors,
    )


# Insertion order is display order everywhere.
METHODS: dict[str, Method] = {m.key: m for m in [
    _m("rule_based_tfidf", "TF-IDF", "TF-IDF cosine similarity",
       "lexical", None, 0.10,
       "rule_based_tfidf.csv", "tfidf_similarity_matrix.npy", "tfidf_score"),
    _m("rule_based_jaccard", "Jaccard", "Security-keyword Jaccard overlap",
       "lexical", None, 0.05,
       "rule_based_jaccard.csv", None, "jaccard_score"),

    _m("sbert", "SBERT", "Sentence-BERT (all-MiniLM-L6-v2)",
       "sentence-embedding", "all-MiniLM-L6-v2", 0.30,
       "sbert_output.csv", "sbert_similarity_matrix.npy", "sbert_score"),
    _m("mpnet", "MPNet", "Sentence-BERT (all-mpnet-base-v2)",
       "sentence-embedding", "sentence-transformers/all-mpnet-base-v2", 0.30,
       "mpnet_output.csv", "mpnet_similarity_matrix.npy", "mpnet_score"),
    _m("bge", "BGE", "BGE (BAAI/bge-base-en-v1.5)",
       "sentence-embedding", "BAAI/bge-base-en-v1.5", 0.30,
       "bge_output.csv", "bge_similarity_matrix.npy", "bge_score"),

    _m("bert", "BERT", "BERT (bert-base-uncased), mean-pooled",
       "contextual-embedding", "bert-base-uncased", 0.70,
       "bert_output.csv", "bert_similarity_matrix.npy", "bert_score"),
    _m("securebert", "SecureBERT", "SecureBERT (ehsanaghaei/SecureBERT)",
       "contextual-embedding", "ehsanaghaei/SecureBERT", 0.70,
       "securebert_output.csv", "securebert_similarity_matrix.npy",
       "securebert_score", use_safetensors=False),
    _m("cysecbert", "CySecBERT", "CySecBERT (markusbayer/CySecBERT)",
       "contextual-embedding", "markusbayer/CySecBERT", 0.70,
       "cysecbert_output.csv", "cysecbert_similarity_matrix.npy",
       "cysecbert_score"),

    # Entailment probability, not cosine. See src.mapping.nli_mapping.
    _m("nli", "NLI cross-enc.", "NLI cross-encoder (nli-deberta-v3-base)",
       "cross-encoder", "cross-encoder/nli-deberta-v3-base", 0.20,
       "nli_output.csv", "nli_similarity_matrix.npy", "nli_score"),

    _m("gemini_embedding", "Gemini emb.", "Gemini embeddings (gemini-embedding-2)",
       "api", "gemini-embedding-2", 0.50,
       "gemini_embedding_output.csv", "gemini_embedding_similarity_matrix.npy",
       "gemini_embed_score"),
    _m("gemini_llm", "Gemini LLM", "Gemini LLM classification (gemini-2.5-flash-lite)",
       "api", "gemini-2.5-flash-lite", None,
       "gemini_output.csv", None, None),
]}

# Thesis table row label -> method key. Must match the first cell of each row exactly.
THESIS_ROWS = {
    "TF--IDF": "rule_based_tfidf",
    "Keyword Jaccard": "rule_based_jaccard",
    "Sentence-BERT": "sbert",
    "MPNet": "mpnet",
    "BGE": "bge",
    "BERT": "bert",
    "SecureBERT": "securebert",
    "CySecBERT": "cysecbert",
    "NLI cross-encoder": "nli",
    "Gemini embeddings": "gemini_embedding",
    "Gemini LLM": "gemini_llm",
}

# Named subsets for figures that would be unreadable with all eleven methods.
CONFUSION_SUBSET = ("gemini_embedding", "bge", "bert", "nli")
PREVALENCE_SUBSET = ("sbert", "bge", "gemini_llm", "securebert")


def keys() -> list[str]:
    return list(METHODS)


def labels() -> dict[str, str]:
    return {k: m.label for k, m in METHODS.items()}


def with_matrix() -> dict[str, Method]:
    """Methods that expose a continuous score, so can be swept and recalibrated."""
    return {k: m for k, m in METHODS.items() if m.matrix is not None}


def by_family() -> dict[str, list[Method]]:
    out: dict[str, list[Method]] = {}
    for m in METHODS.values():
        out.setdefault(m.family, []).append(m)
    return out


def embedding_methods() -> dict[str, Method]:
    """Methods run by src.mapping.run_embedding (one model, cosine over provisions)."""
    return {k: m for k, m in METHODS.items()
            if m.family in ("sentence-embedding", "contextual-embedding")}
