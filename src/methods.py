"""
The set of mapping methods under evaluation, in one place.

Every stage of the pipeline -- the mapping runners, the evaluation, the weighted
corpus estimates, the figures, and the guard that checks the thesis tables --
used to carry its own copy of this list, which is how a method ends up in a
figure but not in a table. They all import from here instead.

Methods are grouped into families because the grouping is a result, not just
housekeeping: models trained for sentence similarity behave differently from
mean-pooled masked language models, and the shipped thresholds were set per
family before any evaluation rather than per model afterwards.
"""
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).parents[1]
DATA = BASE / "data"
MAP_DIR = DATA / "mappings"

# Family -> the a priori threshold applied to every member. Chosen before the
# evaluation from the range each family's scores occupy, not tuned on results;
# Section 5 reports what happens when they meet the corpus base rate.
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
    # "sbert" for models with a SentenceTransformer head, "transformers" for raw
    # masked language models that have to be mean-pooled by hand.
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


# Insertion order is display order everywhere: family by family, and within a
# family in the order the models were published.
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

    # Threshold is on entailment probability, not cosine: 0.20 is the weak-evidence
    # floor below which no relation is emitted. See src.mapping.nli_mapping.
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

# Thesis table row label -> method key. The guard in src.report.check_thesis_numbers
# matches on these, so they must be the exact first cell of each row.
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

# Figures that would be unreadable with every method get a named subset instead
# of a subset buried in the plotting code.
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
