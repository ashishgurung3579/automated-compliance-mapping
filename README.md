# Automated Compliance Mapping for IoT and AI Security Standards

Code and data for the master's degree project *Automated Compliance Mapping for IoT and AI
Security Standards: An NLP-Based Approach for ETSI EN 303 645 and EN 304 223*
(University West, Department of Engineering Science).

The project investigates how well NLP and LLM techniques can identify semantic
relationships (equivalence, overlap, subsumption, complementarity) between security
provisions in two ETSI standards:

- **ETSI EN 303 645** — Cyber Security for Consumer Internet of Things (69 extracted provisions)
- **ETSI EN 304 223** — Securing Artificial Intelligence, Baseline Cyber Security (72 extracted provisions)

Eleven automated mapping methods, spanning five families — lexical, sentence-embedding,
contextual-embedding, cross-encoder, and hosted API — are compared against a single
**200-pair reference set**, drawn from 1,027 pairs judged across three annotation rounds:

- a **pilot** of 107 purposively selected pairs, reviewed by the authors;
- a **screened round** of 650 pairs drawn from three bands of a method-agnostic similarity ranking;
- a **targeted round** of the 300 top-ranked unjudged pairs, searched for the rare classes.

All were produced through LLM-assisted annotation (Claude Opus 5) against the written codebook
in `docs/annotation_codebook.md`. Only the pilot was reviewed pair by pair; the reliability of
the rest rests on a blind comparison against it (see `src/validation/`).

The reference set is built under one rule: **negatives are half the set; positives are balanced
across the five positive classes as far as the material allows.** Half negatives fixes the score
of the trivial "everything is related" classifier at F1 0.667, which every detection table
carries as its final row. Because the set is balanced by design while the full candidate space is
sparse, every reported figure is an **upper bound** on corpus-scale performance.

The targeted round returned no equivalence or subsumption pairs at all. Across all 1,027 judged
pairs those two classes total 13, so for this pair of standards the typed taxonomy is effectively
a three-class problem — a finding, not a sampling shortfall.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install poppler        # provides pdftotext, needed for extraction
```

Gemini-based methods additionally need an API key:

```bash
cp .env.example .env        # then set GEMINI_API_KEY
```

## Pipeline

Run the stages in order from the repository root:

| Stage | Command | Output |
|---|---|---|
| 1. Extract provisions | `python3 -m src.extraction.provision_extractor` | `data/extracted/*.json` |
| 2. Build reference set | `python3 -m src.baseline.build_reference` | `data/baseline/reference_set.csv` |
| 3a. Rule-based (TF-IDF, Jaccard) | `python3 -m src.mapping.rule_based` | `data/mappings/rule_based_*.csv` |
| 3b. Embedding models (SBERT, MPNet, BGE, BERT, SecureBERT, CySecBERT) | `python3 -m src.mapping.run_embedding --all` | `data/mappings/<key>_output.csv` |
| 3c. NLI cross-encoder | `python3 -m src.mapping.nli_mapping` | `data/mappings/nli_output.csv` |
| 3d. Gemini embeddings (API) | `python3 -m src.mapping.gemini_embedding_mapping` | `data/mappings/gemini_embedding_output.csv` |
| 3e. Gemini LLM classification (API) | `python3 -m src.mapping.gemini_mapping` | `data/mappings/gemini_output.csv` |
| 4. Evaluate | `python3 -m src.evaluation.evaluate` | `data/evaluation/*_eval.json`, summary |
| 5. Extended analysis | `python3 -m src.evaluation.analysis` | `data/evaluation/analysis.json` |
| 6. Ranking metrics | `python3 -m src.evaluation.ranking` | `data/evaluation/ranking.json` |
| 7. Coverage and gaps | `python3 -m src.evaluation.coverage` | `data/evaluation/coverage.json`, `thesis/tables/*.tex` |
| 8. Report | `python3 -m src.report.generate_report` | `data/evaluation/report.md` |
| 9. Figures | `python3 -m src.report.figures` | `thesis/figures/*.pdf` |

Stages 3d/3e call the Gemini API (cost + `GEMINI_API_KEY` required); all other stages run
locally. First runs of 3b–3c download about 2 GB of models from Hugging Face.

Every method — its model, family, threshold, and output filenames — is defined once in
`src/methods.py`. Adding a method means adding one entry there; the evaluation, the calibration
analysis, the figures, and the guard that checks the thesis tables all read from it.

Stage 7's outputs are *predictions*, not verified mappings: they come from the best-scoring
method at its calibrated threshold, whose precision is well below 1 — and was measured on the
balanced set, so at corpus scale it is an upper bound. The generated tables and figure say so.

Stage 6's output is computed but not reported in the thesis — the judged fraction of each ranking is thin
enough that precision among judged candidates is 1.000 for every method, so the numbers describe
pooling coverage rather than ranking quality.

## Repository layout

```
data/raw/         original ETSI standard PDFs
data/extracted/   provisions extracted from the PDFs (JSON)
data/baseline/    reference_set.csv (evaluated), plus the three rounds it draws from
data/annotation/  the 19 emitted annotation batches and their labels
data/mappings/    predictions per method + similarity matrices (.npy)
data/evaluation/  metrics, corpus estimates, error analyses, prediction-vs-GT comparisons
data/validation/  inter-annotator agreement results
src/extraction/   PDF → provision JSON
src/baseline/     ground-truth construction, sampling, annotation harness
src/mapping/      the eleven mapping methods (registry in src/methods.py)
src/validation/   inter-rater agreement
src/evaluation/   metrics against ground truth, corpus estimation, ranking
src/report/       report generation, figures, table-value guard
docs/             degree project plan, thesis PDF, pipeline notes (PIPELINE.md)
thesis/tables/    LaTeX tables generated by src.evaluation.coverage
thesis/           LaTeX source of the thesis
```

## Ground truth

`data/baseline/reference_set.csv` holds the 200 evaluated pairs (100 related, 100
`NO_RELATION`) with one of six labels: `EQUIVALENCE`, `OVERLAP`, `SUBSUMPTION_A_BROADER`,
`SUBSUMPTION_B_BROADER`, `COMPLEMENTARITY`, `NO_RELATION`, plus a justification, a 1–3
confidence score, the round each pair came from, and whether the authors reviewed it. For
evaluation the two subsumption directions are merged into one `SUBSUMPTION` class.

The rounds it draws from are kept as provenance: `gt.csv` (pilot, 107), `gt_sample.csv`
(screened round plus the pilot pairs re-judged blind, 727) and `rare_class_pairs.csv` (targeted
round, 300). The confidence scale is defined 1–3 throughout, but only levels 2 and 3 occur, so
the calibration result is a two-level contrast rather than a curve.

## Evaluation

Metrics are computed only within the annotated 200-pair scope; predictions outside it are
ignored. Two views are reported per method:

- **Pair detection** — precision / recall / F1 on whether a related pair is found at all.
- **Relationship classification** — accuracy and macro-averaged P/R/F1 over the five
  merged labels.

Stage 5 adds per-method confusion matrices, bootstrap 95 % confidence intervals
(2,000 resamples, seed 42), and threshold sensitivity sweeps computed from the stored
similarity matrices. See `data/evaluation/report.md` for the current results table.

Stage 5 also reports each continuous method at two operating points — the threshold fixed per
family before evaluation, and one re-selected by repeated stratified cross-validation over
quantiles of the method's own scores — because several methods ship at a point that places every
pair on one side of the line. Method orderings are claimed only where a paired
bootstrap on the difference excludes zero.

## Probability sample and reference-set reliability

```bash
python3 -m src.baseline.sample_reference        # stratified sample of all 4,968 pairs
python3 -m src.baseline.annotate_sample emit    # next unlabelled batch of 40
python3 -m src.baseline.annotate_sample ingest  # fold batch_NNN_labels.json back in
python3 -m src.baseline.annotate_sample status  # progress across the 19 batches
python3 -m src.validation.agreement             # kappa, confidence calibration, LaTeX table
```

The pilot's 107 pairs are mixed into the annotation stream unmarked, so
`agreement.py` compares the new labels against the author-reviewed originals
blind. The stored batches are complete, so the sampling and annotation steps only need
re-running to rebuild the reference set from scratch.

Run-to-run agreement of the LLM classifier is measured by producing a second pass over
byte-identical prompts at temperature 0:

```bash
GEMINI_OUTPUT_SUFFIX=rep2 python3 -m src.mapping.gemini_mapping
```

`docs/expert_survey.md` holds the full expert-survey instrument and analysis plan
designed for a larger external validation. The survey was designed but not conducted; no result
in this repository comes from it.

## Thesis

LaTeX source lives in `thesis/`. Build with:

```bash
cd thesis
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

(`latexmk -pdf main.tex` does the same thing where it is installed.)

Before every submission build, verify that the reported tables still match the data:

```bash
python3 -m src.report.check_thesis_numbers
```
