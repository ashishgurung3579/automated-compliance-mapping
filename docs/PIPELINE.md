# Automated Compliance Mapping — Project Notes

Two parts. **Part I** explains what the project is and what it found, for someone who
needs the whole picture in fifteen minutes. **Part II** is the operational record: every
stage, every command, every file it touches.

- Part I — [What this is](#part-i--what-this-is)
- Part II — [How to run it](#part-ii--how-to-run-it) (numbered sections 1–26)

---

# Part I — What this is

## The problem

A company shipping an AI-enabled IoT device has to satisfy several security standards at
once. Those standards overlap heavily, the same underlying requirement appearing in two
documents in different words, but nobody publishes a map between them. Building that map
by hand needs an expert in both domains and takes weeks.

- **ETSI EN 303 645** — cybersecurity for consumer IoT devices. 69 provisions extracted.
- **ETSI EN 304 223** — cybersecurity for AI systems. 72 provisions extracted.

69 × 72 = **4,968 candidate pairs**, each of which has to be judged.

## What "related" means

A binary related/unrelated verdict throws away what a compliance officer needs, so the
project uses six labels:

| Label | Meaning |
| --- | --- |
| `EQUIVALENCE` | same requirement, different wording |
| `OVERLAP` | partly the same; neither contains the other |
| `SUBSUMPTION_A_BROADER` | the IoT provision is the broader of the two |
| `SUBSUMPTION_B_BROADER` | the AI provision is the broader |
| `COMPLEMENTARITY` | different aspects of one security goal |
| `NO_RELATION` | no meaningful connection |

For evaluation the two subsumption directions merge into one class, leaving five.

## The reference set

Three annotation rounds produced **1,027 judged pairs**: a 107-pair pilot chosen
purposively and reviewed by the authors, a 650-pair screened round drawn from three bands
of a method-agnostic similarity ranking, and a 300-pair targeted round aimed at the rare
classes. All were annotated with LLM assistance (Claude Opus 5, deliberately a different
model family from every evaluated method) against the written codebook in
`annotation_codebook.md`. This is disclosed in the thesis rather than hidden.

The **200-pair evaluated set** (`data/baseline/reference_set.csv`) is drawn from that pool
under one rule: negatives are half the set, and positives are balanced across the five
positive classes as far as the material allows.

| Label | Judged | In the evaluated set |
| --- | ---: | ---: |
| `NO_RELATION` | 701 | 100 |
| `OVERLAP` | 174 | 44 |
| `COMPLEMENTARITY` | 139 | 43 |
| `SUBSUMPTION_B_BROADER` | 6 | 6 |
| `SUBSUMPTION_A_BROADER` | 4 | 4 |
| `EQUIVALENCE` | 3 | 3 |
| **total** | **1,027** | **200** |

Half negatives fixes the score of the trivial "everything is related" classifier at
**F1 0.667**, which is the bar every method has to clear and is printed as a row in every
detection table. It also means every reported figure is an **upper bound** on corpus-scale
performance, since the full candidate space is far sparser than half positive.

The pilot pairs were mixed unmarked into the screened round, so re-judging them was blind.
They agree with the reviewed originals at Cohen's **κ = 0.51** on the binary call and
**κ = 0.40** on the six-label call. Moderate: good enough to rank methods against a common
target, not good enough to certify anyone's absolute accuracy.

## The eleven methods

Five families, chosen so the comparison says something about kinds of approach and not
about individual models.

| Family | Method | Model |
| --- | --- | --- |
| Lexical | TF–IDF | — |
| Lexical | Security-keyword Jaccard | — |
| Sentence-embedding | SBERT | `all-MiniLM-L6-v2` |
| Sentence-embedding | MPNet | `all-mpnet-base-v2` |
| Sentence-embedding | BGE | `BAAI/bge-base-en-v1.5` |
| Contextual-embedding | BERT | `bert-base-uncased` |
| Contextual-embedding | SecureBERT | `ehsanaghaei/SecureBERT` |
| Contextual-embedding | CySecBERT | `markusbayer/CySecBERT` |
| Cross-encoder | NLI | `cross-encoder/nli-deberta-v3-base` |
| API | Gemini embeddings | `gemini-embedding-2` |
| API | Gemini LLM | `gemini-2.5-flash-lite` |

Nine of the eleven run locally and free. The two Gemini methods are the only paid ones and
their outputs are stored in the repo, so nothing has to be re-queried.

The NLI cross-encoder is the one methodological addition rather than just an eleventh
model. It reads both provisions together and answers a directional question, so running it
in both directions recovers the direction the taxonomy asks for:

```
both directions entail       -> EQUIVALENCE
A entails B only             -> SUBSUMPTION_B_BROADER
B entails A only             -> SUBSUMPTION_A_BROADER
neither, but some entailment -> OVERLAP
nothing                      -> NO_RELATION
```

Every other method reduces a pair to one symmetric number, so it is the only method in the
study that can express subsumption at all.

## What was found

**1. Four methods ship as constant functions.** At their default thresholds SecureBERT,
CySecBERT and Gemini embeddings mark every one of the 200 annotated pairs as related, and
BERT marks all but one. Their F1 of 0.667–0.669 is what the all-positive baseline scores on
the same pairs.

**2. Most of that is calibration, but calibration does not buy much.** Re-selecting each
threshold by repeated cross-validation:

| Method | Threshold | Precision | F1 |
| --- | ---: | ---: | ---: |
| Gemini embeddings | 0.713 | 0.774 | **0.782** |
| MPNet | 0.353 | 0.732 | 0.721 |
| BGE | 0.537 | 0.583 | 0.711 |
| TF–IDF | 0.001 | 0.572 | 0.703 |
| SBERT | 0.246 | 0.569 | 0.700 |
| BERT | 0.769 | 0.542 | 0.693 |
| CySecBERT | 0.804 | 0.521 | 0.685 |
| NLI | 0.000 | 0.513 | 0.673 |
| SecureBERT | 0.932 | 0.533 | 0.667 |
| *all-positive baseline* | — | *0.500* | *0.667* |

Only Gemini embeddings (+0.116), BGE (+0.044) and CySecBERT (+0.018) beat the baseline by a
margin whose paired-bootstrap interval excludes zero.

**3. Cybersecurity pretraining did not help, twice.** SecureBERT lands exactly on the
baseline even recalibrated; CySecBERT, built independently by another group, finishes below
the plain BERT both were adapted from. What a model was trained *to do* predicts its
behaviour here; what it was trained *on* does not.

**4. One free local model matches the paid one.** MPNet is the only method the paired
bootstrap cannot separate from Gemini embeddings (−0.061, [−0.137, 0.015]), and it runs in
under four seconds with no per-query cost and reproduces bit-identically.

**5. Typing the relationship is not solved.** Best accuracy is 0.505 against a
majority-class baseline of 0.500. No method exceeds it.

**6. The LLM classifier does not agree with itself.** Two runs at temperature 0 over
byte-identical prompts agree on 69% of pairs but reach κ = 0.02, because the model assigns
`COMPLEMENTARITY` to 81% of everything.

**7. Equivalence and subsumption are nearly absent from the corpus.** 13 pairs among the
1,027 judged, and the 300-pair targeted search for more returned none. A device-level
standard and a lifecycle-level standard produce overlap and complementarity in quantity and
strict containment almost never, so the six-label taxonomy is effectively a three-label one
here.

**8. Coverage.** At its calibrated threshold the best method flags 1,018 of the 4,968 pairs
(20.5%). Seven EN 303 645 provisions and three EN 304 223 provisions receive no predicted
counterpart. These are predictions, not verified mappings.

## What is missing

- No corpus-base-rate measurement. Every figure is conditioned on a set built to be half
  positive, and the gap to the sparse reality of 4,968 pairs is unquantified.
- The expert survey in `expert_survey.md` was designed and **not run**. No number anywhere
  in this repository comes from it.
- Only 37 of the 200 evaluated pairs were reviewed by the authors pair by pair; the rest
  rest on the blind comparison against the pilot.
- Ranking metrics are computed but not reported: the judged fraction of each ranking is thin
  enough that precision among judged candidates is 1.000 for every method.

---

# Part II — How to run it

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
- `data/baseline/`: the evaluated reference set and the three annotation rounds it draws from.
- `data/annotation/`: the annotation batches and their labels (19 screened, 8 targeted).
- `data/mappings/`: prediction files created by the different mapping methods.
- `data/evaluation/`: evaluation metrics, calibration analysis, error files, comparison files, and the final report.
- `data/validation/`: inter-annotator agreement results.
- `src/methods.py`: the registry where every method is defined once.
- `src/extraction/`: code that reads PDFs and extracts provisions.
- `src/baseline/`: code that runs the three annotation rounds and assembles the reference set from them.
- `src/mapping/`: code for all automatic mapping methods.
- `src/validation/`: code for inter-rater agreement.
- `src/evaluation/`: code that compares predictions against the reference set.
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

## 5. Step 2 - Create the Pilot Reference Set (Ground Truth)

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
evaluation, the calibration analysis, the figures, and the guard that checks the
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

## 15. The Annotation Rounds and the Reference Set

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
claims made in the thesis and in Part I.

---

## 23. Full End-to-End Order

The pipeline is meant to be run one stage at a time. That makes it easier to
check outputs before moving on, especially before using API-based Gemini steps.

If starting from the raw PDFs, the full order is:

```bash
python3 -m src.extraction.provision_extractor
python3 -m src.baseline.create_baseline      # pilot round -> gt.csv
python3 -m src.baseline.build_reference      # evaluated set -> reference_set.csv
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

The reference set is already complete in the repo. To rebuild
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
Calibrated thresholds, ranking metrics, coverage tables and heatmap
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

Metrics are computed only against the 200 annotated pairs of `reference_set.csv`.
Predictions outside that scope are excluded from every calculation. The thresholds below
are the ones each method ships with; the calibrated operating points are in
`analysis.json` and in Part I.

| Method             | Detection Precision | Detection Recall | Detection F1 | Classification Accuracy | Macro F1 |
| ------------------ | ------------------: | ---------------: | -----------: | ----------------------: | -------: |
| Rule-based TF-IDF  |               0.500 |            0.010 |        0.020 |                   0.495 |    0.133 |
| Rule-based Jaccard |               0.545 |            0.060 |        0.108 |                   0.490 |    0.248 |
| SBERT              |               0.882 |            0.150 |        0.256 |                   0.505 |    0.159 |
| MPNet              |               0.857 |            0.120 |        0.210 |                   0.505 |    0.159 |
| BGE                |               0.525 |            0.930 |        0.671 |                   0.280 |    0.133 |
| BERT               |               0.502 |            1.000 |        0.669 |                   0.135 |    0.065 |
| SecureBERT         |               0.500 |            1.000 |        0.667 |                   0.015 |    0.006 |
| CySecBERT          |               0.500 |            1.000 |        0.667 |                   0.045 |    0.031 |
| NLI cross-encoder  |               0.875 |            0.070 |        0.130 |                   0.500 |    0.161 |
| Gemini Embedding   |               0.500 |            1.000 |        0.667 |                   0.270 |    0.238 |
| Gemini LLM         |               0.492 |            0.880 |        0.631 |                   0.225 |    0.098 |
| *Label everything related* |       *0.500* |          *1.000* |      *0.667* |                       - |        - |

Two caveats matter when reading this table:

- Detection precision counts false positives only on the 100 annotated negatives, so a
  method that predicts almost nothing (TF-IDF, Jaccard, SBERT at its default threshold)
  can reach high precision while missing nearly every true pair.
- The set is half positive by construction, so predicting "related" for everything already
  yields detection F1 = 0.667. SecureBERT, CySecBERT and Gemini Embedding land exactly
  there and BERT within one pair, so detection at shipped thresholds separates almost
  nothing. `analysis.json` (confusion matrices, threshold sweeps, calibrated operating
  points, bootstrap intervals) carries the informative differences.

---

## 26. Where to Read Next

- **Part I** of this document summarises what the project is and what it found.
- `data/evaluation/report.md` is the generated results report.
- `thesis/main.pdf` is the degree project report itself.
- `annotation_codebook.md` holds the label definitions the annotation followed.
- `expert_survey.md` holds the expert validation instrument, designed and not run.
