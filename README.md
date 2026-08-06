# Automated Compliance Mapping for IoT and AI Security Standards

Code and data for the master's degree project *Automated Compliance Mapping for IoT and AI
Security Standards: An NLP-Based Approach for ETSI EN 303 645 and EN 304 223*
(University West, Department of Engineering Science).

The project investigates how well NLP and LLM techniques can identify semantic
relationships (equivalence, overlap, subsumption, complementarity) between security
provisions in two ETSI standards:

- **ETSI EN 303 645** — Cyber Security for Consumer Internet of Things (69 extracted provisions)
- **ETSI EN 304 223** — Securing Artificial Intelligence, Baseline Cyber Security (72 extracted provisions)

Seven automated mapping methods are compared against two reference sets:

- a **pilot** of 107 purposively selected pairs, reviewed by the authors;
- a **stratified probability sample** of 650 pairs drawn over all 4,968 candidate pairs with
  known inclusion probabilities, which is what supports any statement about the corpus as a
  whole.

Both were produced through LLM-assisted annotation (Claude Opus 5) against the written codebook
in `docs/annotation_codebook.md`. The pilot was reviewed pair by pair; the probability sample was
not, and its reliability rests instead on a blind comparison against the pilot (see
`src/validation/`). The two sets differ sharply in base rate — 79 % of pilot pairs are related
against 12.4 % of the corpus — and that gap drives most of the results.

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
| 2. Build ground truth | `python3 -m src.baseline.create_baseline` | `data/baseline/gt.csv` |
| 3a. Rule-based (TF-IDF, Jaccard) | `python3 -m src.mapping.rule_based` | `data/mappings/rule_based_*.csv` |
| 3b. SBERT | `python3 -m src.mapping.sbert_mapping` | `data/mappings/sbert_output.csv` |
| 3c. BERT | `python3 -m src.mapping.bert_mapping` | `data/mappings/bert_output.csv` |
| 3d. SecureBERT | `python3 -m src.mapping.securebert_mapping` | `data/mappings/securebert_output.csv` |
| 3e. Gemini embeddings (API) | `python3 -m src.mapping.gemini_embedding_mapping` | `data/mappings/gemini_embedding_output.csv` |
| 3f. Gemini LLM classification (API) | `python3 -m src.mapping.gemini_mapping` | `data/mappings/gemini_output.csv` |
| 4. Evaluate | `python3 -m src.evaluation.evaluate` | `data/evaluation/*_eval.json`, summary |
| 5. Extended analysis | `python3 -m src.evaluation.analysis` | `data/evaluation/analysis.json` |
| 6. Corpus estimates | `python3 -m src.evaluation.weighted_metrics` | `data/evaluation/weighted_metrics.json` |
| 7. Ranking metrics | `python3 -m src.evaluation.ranking` | `data/evaluation/ranking.json` |
| 8. Report | `python3 -m src.report.generate_report` | `data/evaluation/report.md` |
| 9. Figures | `python3 -m src.report.figures` | `thesis/figures/*.pdf` |

Stages 3e/3f call the Gemini API (cost + `GEMINI_API_KEY` required); all other stages run
locally. First runs of 3b–3d download models from Hugging Face.

Stage 6 depends on the probability sample being annotated; see the section below. Stage 7's
output is computed but not reported in the thesis — the judged fraction of each ranking is thin
enough that precision among judged candidates is 1.000 for every method, so the numbers describe
pooling coverage rather than ranking quality.

## Repository layout

```
data/raw/         original ETSI standard PDFs
data/extracted/   provisions extracted from the PDFs (JSON)
data/baseline/    pilot annotations (gt.csv), probability sample (gt_sample.csv)
data/annotation/  the 19 emitted annotation batches and their labels
data/mappings/    predictions per method + similarity matrices (.npy)
data/evaluation/  metrics, corpus estimates, error analyses, prediction-vs-GT comparisons
data/validation/  inter-annotator agreement results
src/extraction/   PDF → provision JSON
src/baseline/     ground-truth construction, sampling, annotation harness
src/mapping/      the seven mapping methods (shared helpers in embeddings.py)
src/validation/   inter-rater agreement
src/evaluation/   metrics against ground truth, corpus estimation, ranking
src/report/       report generation, figures, table-value guard
docs/             degree project plan, thesis PDF, pipeline notes (PIPELINE.md)
thesis/           LaTeX source of the thesis
```

## Ground truth

`data/baseline/gt.csv` holds the 107-pair pilot (85 related, 22 `NO_RELATION`) with one of
six labels: `EQUIVALENCE`, `OVERLAP`, `SUBSUMPTION_A_BROADER`, `SUBSUMPTION_B_BROADER`,
`COMPLEMENTARITY`, `NO_RELATION`, plus a justification and a 1–3 confidence score per pair.
For evaluation the two subsumption directions are merged into one `SUBSUMPTION` class.

`data/baseline/gt_sample.csv` holds the probability sample in the same schema, with the
sampling stratum, inclusion weight, and an `in_pilot` flag added. The confidence scale is
defined 1–3 throughout, but only levels 2 and 3 occur in the sample, so the calibration result
is a two-level contrast rather than a curve.

## Evaluation

Pilot metrics are computed only within the annotated 107-pair scope; predictions outside it are
ignored. Two views are reported per method:

- **Pair detection** — precision / recall / F1 on whether a related pair is found at all.
- **Relationship classification** — accuracy and macro-averaged P/R/F1 over the five
  merged labels.

Stage 5 adds per-method confusion matrices, bootstrap 95 % confidence intervals
(2,000 resamples, seed 42), and threshold sensitivity sweeps computed from the stored
similarity matrices. See `data/evaluation/report.md` for the current results table.

Stage 6 estimates the same quantities over the full candidate space from the probability sample,
using Horvitz-Thompson estimators with within-stratum bootstrap intervals. It reports each
continuous method at two operating points — the threshold tuned on the pilot, and one re-selected
by repeated stratified cross-validation on the sample — because the base-rate difference between
the two sets moves the optimum substantially. Method orderings are claimed only where a paired
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
