# Automated Compliance Mapping - Project Notes

This project compares two ETSI security standards and checks which requirements are related to each other.

- **ETSI EN 303 645**: cybersecurity requirements for consumer IoT devices.
- **ETSI EN 304 223**: cybersecurity requirements for AI systems.

Main question:

> "Does this IoT security requirement match, overlap with, or relate to this AI security requirement?"

The project tests eleven mapping methods, compares them against two reference sets, and writes an evaluation report.

---

## 1. Main Folder Structure

```text
automated-compliance-mapping/
+-- requirements.txt
+-- docs/
+-- thesis/
+-- src/
|   +-- methods.py
|   +-- extraction/
|   +-- baseline/
|   +-- mapping/
|   +-- validation/
|   +-- evaluation/
|   +-- report/
+-- data/
    +-- raw/
    +-- extracted/
    +-- baseline/
    +-- annotation/
    +-- mappings/
    +-- evaluation/
    +-- validation/
```

What each folder means:

- `data/raw/`: original PDF standards.
- `data/extracted/`: provisions extracted from the PDFs as JSON.
- `data/baseline/`: the two reference sets, also called ground truth.
- `data/annotation/`: the 19 emitted annotation batches and their labels.
- `data/mappings/`: prediction files created by the different mapping methods.
- `data/evaluation/`: evaluation metrics, error files, comparison files, corpus estimates, and final report.
- `data/validation/`: inter-annotator agreement results.
- `src/methods.py`: the registry where every method is defined once.
- `src/extraction/`: code that reads PDFs and extracts provisions.
- `src/baseline/`: code that runs the three annotation rounds and assembles the reference set from them.
- `src/mapping/`: code for all automatic mapping methods.
- `src/validation/`: code for inter-rater agreement.
- `src/evaluation/`: code that compares predictions against the reference sets.
- `src/report/`: code that creates the Markdown report, the thesis figures, and the table-value guard.
- `thesis/`: LaTeX source of the thesis, with generated figures and tables.

---

## 2. Required Setup

Python packages are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

The extraction step also needs `pdftotext`, which comes from Poppler.

On macOS:

```bash
brew install poppler
```

Gemini-based scripts also need:

```text
GEMINI_API_KEY
```

Usually this is stored in a `.env` file (see `.env.example`).

---

## 3. Raw Input Files

The project starts with two PDF files:

```text
data/raw/etsi303645v030103p.pdf
data/raw/etsi304223v020101p.pdf
```

These are the original standards.

Nothing can be mapped until the provisions are extracted from these PDFs.

---

## 4. Step 1 - Extract Provisions From PDFs

Script:

```text
src/extraction/provision_extractor.py
```

Run command:

```bash
python3 -m src.extraction.provision_extractor
```

This script uses helper code from:

```text
src/extraction/utils/pdf_parser.py
```

What happens:

1. `pdf_parser.py` calls `pdftotext` to convert each PDF into plain text.
2. It cleans the text by removing repeated ETSI headers and extra blank lines.
3. `provision_extractor.py` searches the text for provision IDs.
4. It extracts each provision's:
   - provision ID
   - standard name
   - section ID
   - section title
   - provision text
   - modality, such as `shall`, `should`, `may`, or `unknown`
5. It saves the extracted provisions as JSON files.

Created files:

```text
data/extracted/en303645_provisions.json
data/extracted/en304223_provisions.json
```

Current extracted counts:

- `en303645_provisions.json`: 69 provisions.
- `en304223_provisions.json`: 72 provisions.

Total possible pairs:

```text
69 x 72 = 4,968 possible provision pairs
```

---

## 5. Step 2 - Create the Reference Sets (Ground Truth)

Script:

```text
src/baseline/create_baseline.py
```

Run command:

```bash
python3 -m src.baseline.create_baseline
```

What happens:

1. The script writes the 107-pair pilot reference set.
2. Each pair says how one EN 303 645 provision relates to one EN 304 223 provision.
3. The relationship labels can be:
   - `EQUIVALENCE`
   - `OVERLAP`
   - `SUBSUMPTION_A_BROADER`
   - `SUBSUMPTION_B_BROADER`
   - `COMPLEMENTARITY`
   - `NO_RELATION`
4. Each row also includes:
   - a short justification
   - confidence score from 1 to 3
5. The script writes everything to a CSV file.

Created file:

```text
data/baseline/gt.csv
```

Current baseline count:

- 107 annotated pairs.
- 85 positive relation pairs.
- 22 `NO_RELATION` negative pairs.

Important: the annotations were produced with LLM assistance (Claude Opus 5)
against the written codebook in `docs/annotation_codebook.md`, and the 107 pilot
pairs were then reviewed pair by pair by the authors. This is disclosed in the
thesis. Two further annotation rounds widen the pool (section 15), and the
200-pair set actually evaluated is assembled from all three by
`build_reference.py`. Reliability of the unreviewed rounds is measured in
section 16.

This file is the pilot round's output, retained as provenance.

---

## 6. Step 3 - Run Automatic Mapping Methods

The mapping scripts read:

```text
data/extracted/en303645_provisions.json
data/extracted/en304223_provisions.json
```

Then they produce prediction files in:

```text
data/mappings/
```

Each method tries to predict related provision pairs.

Every method — its model, family, threshold, and output filenames — is defined
once in `src/methods.py`. Adding a method means adding one entry there; the
evaluation, the corpus estimates, the figures, and the guard that checks the
thesis tables all read from it.

---

## 7. Mapping Methods 1-2 - Rule-Based Mapping

Script:

```text
src/mapping/rule_based.py
```

Run command:

```bash
python3 -m src.mapping.rule_based
```

What happens:

The script runs two rule-based approaches.

First, **TF-IDF cosine similarity**:

- Converts provision text into word vectors.
- Compares provisions based on shared words and phrases.
- Good for simple lexical similarity.

Second, **keyword Jaccard similarity**:

- Looks only for predefined cybersecurity keywords.
- Examples: `password`, `authentication`, `encryption`, `vulnerability`, `logging`, `risk`.
- Compares how many security keywords two provisions share.

Created files:

```text
data/mappings/rule_based_tfidf.csv
data/mappings/rule_based_jaccard.csv
data/mappings/tfidf_similarity_matrix.npy
```

What each file means:

- `rule_based_tfidf.csv`: predictions from TF-IDF only.
- `rule_based_jaccard.csv`: predictions from cybersecurity keyword matching.
- `tfidf_similarity_matrix.npy`: raw TF-IDF similarity scores for all provision pairs.

---

## 8. Mapping Methods 3-8 - Embedding Models

Script:

```text
src/mapping/run_embedding.py
```

Run commands:

```bash
python3 -m src.mapping.run_embedding sbert
python3 -m src.mapping.run_embedding --all
```

The three original entry points still work and simply call the runner:

```bash
python3 -m src.mapping.sbert_mapping
python3 -m src.mapping.bert_mapping
python3 -m src.mapping.securebert_mapping
```

Models used:

```text
sbert       all-MiniLM-L6-v2                        sentence-embedding
mpnet       sentence-transformers/all-mpnet-base-v2 sentence-embedding
bge         BAAI/bge-base-en-v1.5                   sentence-embedding
bert        bert-base-uncased                       contextual-embedding
securebert  ehsanaghaei/SecureBERT                  contextual-embedding
cysecbert   markusbayer/CySecBERT                   contextual-embedding
```

Every model, its threshold, its output filenames and its family live in one
place, `src/methods.py`. Adding a model means adding one entry there; nothing
else in the pipeline needs to change.

What happens:

1. Each provision text is converted into an embedding.
2. Sentence-embedding models use their SentenceTransformer head; contextual-embedding models are mean-pooled over token outputs and L2-normalized.
3. The script compares every EN 303 645 embedding with every EN 304 223 embedding.
4. The full similarity matrix is saved before thresholding, because the sensitivity analysis and the corpus recalibration need the scores of pairs no threshold kept.
5. Pairs above the family threshold are written with a relationship label.

Created files, one pair per model:

```text
data/mappings/<key>_output.csv
data/mappings/<key>_similarity_matrix.npy
```

The first run of each new model downloads it from Hugging Face (about 2 GB in
total across the embedding and NLI models).

---

## 9. Mapping Method 9 - NLI Cross-Encoder

Script:

```text
src/mapping/nli_mapping.py
```

Run command:

```bash
python3 -m src.mapping.nli_mapping
```

Model used:

```text
cross-encoder/nli-deberta-v3-base
```

What happens:

1. Unlike every embedding method, this one reads both provisions together instead of embedding them separately.
2. Each pair is scored twice: does provision A entail provision B, and does B entail A.
3. Entailment in both directions means equivalence; entailment in one direction means subsumption, with the entailed provision as the narrower one.
4. Weak entailment in either direction means overlap; none means no relation.
5. This is the only method in the study that can express direction, so it is the only one that can predict subsumption at all.

Created files:

```text
data/mappings/nli_output.csv
data/mappings/nli_similarity_matrix.npy
data/mappings/nli_entailment_directions.npy
```

Cost: 2 x 4,968 forward passes, a few minutes on Apple Silicon.

---

## 10. Mapping Method 10 - Gemini Embedding Mapping

Script:

```text
src/mapping/gemini_embedding_mapping.py
```

Run command:

```bash
python3 -m src.mapping.gemini_embedding_mapping
```

Model used:

```text
gemini-embedding-2
```

What happens:

1. The script reads the extracted provisions.
2. It sends each provision text to the Gemini Embedding API.
3. Gemini returns an embedding vector for each provision.
4. The script calculates cosine similarity between all source and target embeddings.
5. Pairs above the threshold are saved.

Created files:

```text
data/mappings/gemini_embedding_output.csv
data/mappings/gemini_embedding_similarity_matrix.npy
```

Important:

This script requires `GEMINI_API_KEY`.

It can run in two modes:

- sync mode: sends requests directly one by one.
- batch mode: uses Gemini Batch API.

The batch helper code is in:

```text
src/mapping/gemini_batch.py
```

---

## 11. Mapping Method 11 - Gemini LLM Mapping

Script:

```text
src/mapping/gemini_mapping.py
```

Run command:

```bash
python3 -m src.mapping.gemini_mapping
```

Model used:

```text
gemini-2.5-flash-lite
```

What happens:

1. The script creates all 4,968 possible provision pairs.
2. For each pair, it builds a prompt asking Gemini to classify the relationship.
3. Gemini must answer with exactly one label:
   - `EQUIVALENCE`
   - `SUBSUMPTION_A_BROADER`
   - `SUBSUMPTION_B_BROADER`
   - `OVERLAP`
   - `COMPLEMENTARITY`
   - `NO_RELATION`
4. The script submits the work using the Gemini Batch API.
5. It waits until the batch job finishes.
6. It saves Gemini's predicted label for every pair.

Created file:

```text
data/mappings/gemini_output.csv
```

Important:

This script requires `GEMINI_API_KEY`.

Batch polling and response parsing are handled by:

```text
src/mapping/gemini_batch.py
```

Run-to-run agreement of the classifier is measured by producing a second pass
over byte-identical prompts at temperature 0:

```bash
GEMINI_OUTPUT_SUFFIX=rep2 python3 -m src.mapping.gemini_mapping
```

This creates `data/mappings/gemini_output_rep2.csv`.

---

## 12. Shared Embedding Helper

Script:

```text
src/mapping/embeddings.py
```

This is not usually run directly.

It provides shared functions used by the embedding and Gemini embedding scripts.

What it does:

- loads transformer models
- creates embeddings
- normalizes embeddings
- calculates cosine similarity
- converts similarity scores into relationship labels

Relationship label thresholds:

- score >= `0.85`: `EQUIVALENCE`
- score >= `0.70`: `OVERLAP`
- score >= `0.50`: `COMPLEMENTARITY`
- otherwise: `NO_RELATION`

---

## 13. Step 4 - Evaluate Predictions

Script:

```text
src/evaluation/evaluate.py
```

Run command:

```bash
python3 -m src.evaluation.evaluate
```

What this script reads:

```text
data/baseline/gt.csv
data/mappings/  (the prediction CSV of every method registered in src/methods.py)
```

What happens:

1. It loads the pilot reference set.
2. It loads each method's prediction CSV — all eleven registered methods.
3. It filters predictions to only those within the 107 GT pairs; predictions outside GT are ignored.
4. It calculates pair-detection metrics:
   - precision (TP / (TP + FP), within GT scope only)
   - recall (TP / 85 GT positive pairs)
   - F1
5. It calculates relationship classification metrics:
   - overall accuracy
   - macro precision
   - macro recall
   - macro F1
   - per-class scores
6. It creates error-analysis files.

Important detail:

For evaluation, these two labels are merged into one `SUBSUMPTION` class:

```text
SUBSUMPTION_A_BROADER
SUBSUMPTION_B_BROADER
```

Created files, one set per method:

```text
data/evaluation/<key>_eval.json
data/evaluation/<key>_errors.csv
data/evaluation/<key>_predictions_vs_gt.csv
data/evaluation/evaluation_summary.json
```

(The Gemini LLM method uses the key `gemini_llm`, so its files are
`gemini_llm_eval.json` and so on.)

What these mean:

- `*_eval.json`: metrics for one method.
- `*_errors.csv`: mistakes made by one method.
- `*_predictions_vs_gt.csv`: side-by-side comparison between predicted and true labels.
- `evaluation_summary.json`: combined metrics for all methods.

---

## 14. Step 5 - Extended Analysis

Script:

```text
src/evaluation/analysis.py
```

Run command:

```bash
python3 -m src.evaluation.analysis
```

What happens:

1. It reads each method's `*_predictions_vs_gt.csv`.
2. It builds a confusion matrix per method over the five merged labels.
3. It computes bootstrap 95% confidence intervals (2,000 resamples, seed 42) for
   detection F1 and classification macro-F1.
4. For methods with a stored similarity matrix, it sweeps the threshold from 0.05 to
   0.975 and records precision, recall, and F1 at each step.

Created file:

```text
data/evaluation/analysis.json
```

---

## 15. Probability Sample and Reference-Set Reliability

Scripts:

```text
src/baseline/sample_reference.py
src/baseline/annotate_sample.py
src/baseline/rare_class_search.py
src/baseline/build_reference.py
src/validation/agreement.py
```

`sample_reference.py` ranks all 4,968 pairs by their mean percentile rank across every
similarity matrix, cuts the ordering into three bands (top 10 %, next 30 %, bottom 60 %)
and draws 250/200/200 pairs without replacement under seed 20260804. Averaging the ranks
keeps the screen neutral between methods; screening on one model's scores would sample
densely wherever that model is confident and flatter it at evaluation.

`annotate_sample.py` emits fixed-size batches with the full text of both provisions,
validates the label vocabulary on ingest, and is resumable. Batch order is shuffled under
the same seed. The 107 pilot pairs are folded into the stream unmarked, which is what makes
the agreement check blind. All 19 batches were annotated: 650 screened pairs plus 77
pilot-only pairs, 727 rows in total.

`rare_class_search.py` runs the third round. It re-uses the same method-agnostic screen over
the pairs no round has judged, emits the top 300 as eight further batches, and ingests them.
It returned no equivalence or subsumption pairs at all — across all 1,027 judged pairs those
two classes total 13, which is a property of the standard pair rather than a shortfall of the
search.

`build_reference.py` assembles the evaluated set from all three rounds under one rule:
negatives are half the set, positives are balanced across the five positive classes as far as
the material allows. Classes that cannot meet their share contribute everything they have and
the remainder is redistributed. Seed 42.

Commands:

```bash
python3 -m src.baseline.sample_reference          # screened round over all 4,968 pairs
python3 -m src.baseline.annotate_sample emit      # next unlabelled batch of 40
python3 -m src.baseline.annotate_sample ingest    # fold batch_NNN_labels.json back in
python3 -m src.baseline.rare_class_search status  # targeted round, 8 batches
python3 -m src.baseline.build_reference           # -> reference_set.csv
```

The stored batches are complete, so only `build_reference` is needed to rebuild the
evaluated set; the annotation steps are for rebuilding a round from scratch.

---

## 16. Step 6 - Reference-Set Reliability

`src/validation/agreement.py` reports agreement with the author-reviewed
pilot labels, confidence calibration, and the evaluated LLM's run-to-run self-agreement
as a contrast:

```bash
python3 -m src.validation.agreement
```

Created files:

```text
data/baseline/reference_pool.csv
data/baseline/gt_sample.csv
data/baseline/rare_class_pairs.csv
data/baseline/reference_set.csv
data/validation/agreement.json
data/validation/agreement_table.tex
```

(`reference_pool.csv` and `gt_sample.csv` come from the sampling and annotation
steps in section 15.)

---

## 17. Step 7 - Ranking Metrics

Script:

```text
src/evaluation/ranking.py
```

Run command:

```bash
python3 -m src.evaluation.ranking
```

Created file:

```text
data/evaluation/ranking.json
```

This output is computed but not reported in the thesis: the judged fraction of
each ranking is thin enough that precision among judged candidates is 1.000 for
every method, so the numbers describe pooling coverage rather than ranking
quality.

---

## 18. Step 8 - Coverage and Gap Analysis

Script:

```text
src/evaluation/coverage.py
```

Run command:

```bash
python3 -m src.evaluation.coverage
```

Uses the best-scoring method at its calibrated threshold to predict a full
mapping between the two standards, then reports which provisions have no
predicted counterpart.

Created files:

```text
data/evaluation/coverage.json
thesis/figures/coverage_heatmap.pdf
thesis/tables/predicted_mapping.tex
thesis/tables/unmapped_provisions.tex
```

Important: these outputs are *predictions*, not verified mappings. Precision at
that operating point is well below 1, so the generated tables and figure carry
that precision and its confidence interval in their captions.

---

## 19. Step 9 - Generate Final Report

Script:

```text
src/report/generate_report.py
```

Run command:

```bash
python3 -m src.report.generate_report
```

What it reads:

```text
data/evaluation/evaluation_summary.json
```

What happens:

1. It loads all evaluation results.
2. It creates a method comparison table.
3. It identifies the best method by classification macro-F1.
4. It writes detailed per-method metrics.
5. It writes notes and limitations.

Created file:

```text
data/evaluation/report.md
```

This is the final Markdown evaluation report.

---

## 20. Step 10 - Thesis Figures

Script:

```text
src/report/figures.py
```

Run command:

```bash
python3 -m src.report.figures
```

Created files (10 PDFs):

```text
thesis/figures/detection_metrics.pdf
thesis/figures/f1_confidence_intervals.pdf
thesis/figures/confusion_matrices.pdf
thesis/figures/similarity_distributions.pdf
thesis/figures/threshold_sweep.pdf
thesis/figures/gt_distribution.pdf
thesis/figures/cv_threshold.pdf
thesis/figures/calibration_gap.pdf
```

---

## 21. Checking the Thesis Numbers

Script:

```text
src/report/check_thesis_numbers.py
```

Run command:

```bash
python3 -m src.report.check_thesis_numbers
```

This guard re-reads every number printed in a thesis results table straight from
the stored evaluation JSON and exits non-zero on any mismatch. It currently
checks 199 values and should be run before every submission build.

---

## 22. Measured Runtimes

File:

```text
data/evaluation/runtimes.json
```

This is a frozen measurement, not a pipeline output: wall-clock seconds per
stage on an 8 GB Apple Silicon MacBook, recorded once with models already
cached. No script regenerates it; it exists as provenance for the runtime
claims made in the thesis and in `explainer.md`.

---

## 23. Full End-to-End Order

The pipeline is meant to be run one stage at a time. That makes it easier to
check outputs before moving on, especially before using API-based Gemini steps.

If starting from the raw PDFs, the full order is:

```bash
python3 -m src.extraction.provision_extractor
python3 -m src.baseline.create_baseline
python3 -m src.mapping.rule_based
python3 -m src.mapping.run_embedding --all
python3 -m src.mapping.nli_mapping
python3 -m src.evaluation.evaluate
python3 -m src.evaluation.analysis
python3 -m src.evaluation.ranking
python3 -m src.evaluation.coverage
python3 -m src.report.generate_report
python3 -m src.report.figures
python3 -m src.report.check_thesis_numbers
```

If you also want Gemini results, run these before evaluation:

```bash
python3 -m src.mapping.gemini_embedding_mapping
python3 -m src.mapping.gemini_mapping
```

Only run the Gemini commands when `GEMINI_API_KEY` is configured and external
API calls are intended.

The probability-sample reference set is already complete in the repo. To rebuild
it from scratch, run the sampling and annotation commands from section 15 before
and `python3 -m src.validation.agreement` afterwards.

---

## 24. Complete File Flow

```text
Raw PDFs
  |
  | src/extraction/provision_extractor.py
  v
Extracted provision JSON files
  |
  | src/baseline/build_reference.py  (three annotation rounds, section 15)
  v
Reference-set CSV files
  |
  | src/mapping/*.py
  v
Automatic mapping CSV files
  |
  | src/evaluation/evaluate.py
  v
Evaluation JSON, error CSV, prediction-vs-ground-truth CSV
  |
  | src/evaluation/analysis.py, ranking.py, coverage.py
  v
Corpus estimates, ranking metrics, coverage tables and heatmap
  |
  | src/report/generate_report.py, figures.py
  v
Final Markdown report and thesis figures
```

---

## 25. Current Evaluation Summary

Current results from:

```text
data/evaluation/evaluation_summary.json
```

Metrics are computed only against the 107 annotated pilot pairs. Predictions outside GT scope are excluded from all calculations.

| Method             | Detection Precision | Detection Recall | Detection F1 | Classification Accuracy | Macro F1 |
| ------------------ | ------------------: | ---------------: | -----------: | ----------------------: | -------: |
| Rule-based TF-IDF  |               1.000 |            0.035 |        0.068 |                   0.215 |    0.086 |
| Rule-based Jaccard |               1.000 |            0.071 |        0.132 |                   0.224 |    0.179 |
| SBERT              |               1.000 |            0.176 |        0.300 |                   0.224 |    0.099 |
| MPNet              |               1.000 |            0.165 |        0.283 |                   0.224 |    0.100 |
| BGE                |               0.833 |            0.882 |        0.857 |                   0.243 |    0.149 |
| BERT               |               0.809 |            1.000 |        0.895 |                   0.411 |    0.172 |
| SecureBERT         |               0.794 |            1.000 |        0.885 |                   0.028 |    0.011 |
| CySecBERT          |               0.794 |            1.000 |        0.885 |                   0.196 |    0.087 |
| NLI cross-encoder  |               1.000 |            0.059 |        0.111 |                   0.215 |    0.102 |
| Gemini Embedding   |               0.794 |            1.000 |        0.885 |                   0.477 |    0.279 |
| Gemini LLM         |               0.777 |            0.859 |        0.816 |                   0.187 |    0.088 |

In the current saved results, **Gemini Embedding** has the highest classification macro-F1.

Two caveats matter when reading this table:

- Detection precision counts false positives only on the 22 annotated negatives, so a
  method that predicts almost nothing (TF-IDF, Jaccard, SBERT at its default threshold)
  can reach precision 1.000 while missing nearly every true pair.
- With 85 of the 107 annotated pairs being positives, predicting "related" for everything
  already yields detection F1 = 0.885. BERT, SecureBERT, CySecBERT, and Gemini Embedding
  are at or near that value, so detection F1 barely separates them; `analysis.json`
  (confusion matrices, threshold sweeps, calibrated operating points, bootstrap intervals)
  (corpus-level estimates) carry the informative differences.

---

## 26. Simple Explanation of the Whole Project

This project starts with two security-standard PDFs.

First, it extracts individual security provisions from both PDFs and saves them as JSON.

Then, reference answer keys are created in `gt.csv` (a 107-pair pilot) and
`reference_set.csv` (the 200 evaluated pairs). They say which
provision pairs are truly related and what kind of relationship they have.

After that, eleven automatic methods try to find related provision pairs:

- TF-IDF word matching
- cybersecurity keyword matching (Jaccard)
- SBERT, MPNet, and BGE sentence embeddings
- BERT, SecureBERT, and CySecBERT contextual embeddings
- an NLI cross-encoder
- Gemini embeddings
- Gemini LLM classification

Each method writes its predictions to `data/mappings/`.

Then the evaluation script compares those predictions against the pilot answer
key, and the weighted-metrics script estimates performance over the whole
balanced reference set. They measure how many true pairs
were found, how many wrong pairs were predicted, and how accurate the
relationship labels were.

Finally, the report script creates a readable Markdown report at:

```text
data/evaluation/report.md
```

That report is the final output of the project.
