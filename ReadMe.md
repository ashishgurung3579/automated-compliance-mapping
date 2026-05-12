# Automated Compliance Mapping - Project Notes

This project compares two ETSI security standards and checks which requirements are related to each other.

- **ETSI EN 303 645**: cybersecurity requirements for consumer IoT devices.
- **ETSI EN 304 223**: cybersecurity requirements for AI systems.

Main question:

> RQ1: Does this IoT security requirement match, overlap with, or relate to this AI security requirement? To what extent can NLP and AI techniques automati- cally identify semantic relationships (equivalence, overlap, sub Sumption, complementarity) between security requirements in ETSI EN 303 645 and EN 304 223? What are the fundamental challenges and limitations?
> RQ2:How do different automated approaches (Rule-based NLP, Semantic (BERT-based, sentence embedding), LLM models perform for compliance mapping tasks? Which methods achieve the best balance of accuracy, completeness, and interpretability?
> RQ3: How does automated mapping compare to manual expert mapping in terms of: Accuracy, coverage, efficiency and interpretability.

The project tests multiple mapping methods, compares them with a manually prepared baseline, and writes an evaluation report.

---

## 1. Main Folder Structure

```text
automated-compliance-mapping/
+-- requirements.txt
+-- docs/
+-- src/
|   +-- extraction/
|   +-- baseline/
|   +-- mapping/
|   +-- evaluation/
|   +-- report/
+-- data/
    +-- raw/
    +-- extracted/
    +-- baseline/
    +-- mappings/
    +-- evaluation/
```

 Short Description of the each folder containing the file to understand the layout/idea of the project.

- `data/raw/`: It contains original PDF standards.
- `data/extracted/`: It contains provisions extracted from the PDFs as JSON.
- `data/baseline/`: It contains manually created correct answers, also called ground truth.
- `data/mappings/`: It contains prediction files created by different mapping methods.
- `data/evaluation/`: It contains evaluation metrics, error files, comparison files, and final report.
- `src/extraction/`: It contains code that reads PDFs and extracts provisions.
- `src/baseline/`: It contains code that creates the manual baseline CSV.
- `src/mapping/`: It contains code for all automatic mapping methods.
- `src/evaluation/`:  It contains code that compares predictions against the baseline.
- `src/report/`: It contains code that creates the Markdown evaluation report.

---

## 2. Required setup for the project 

Python packages are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

The extraction step also needs `pdftotext`, which comes from Poppler.

The project is done On macOS:

```bash
brew install poppler
```

Gemini-based scripts also need:

```text
GEMINI_API_KEY
```

Usually this is stored in a `.env` file.

---

## 3. Raw Input Files

The project starts with two PDF files from the ETSI reprosotries. 

```text
data/raw/etsi303645v030103p.pdf
data/raw/etsi304223v020101p.pdf
```

These are the original standards. Nothing can be mapped until the provisions are extracted from these PDFs.

---

## 4. Step 1 - Extract Provisions From PDFs

Used Script:

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

## 5. Step 2 - Create Manual Baseline / Ground Truth

Used Script:

```text
src/baseline/create_baseline.py
```

Run command:

```bash
python3 -m src.baseline.create_baseline
```

What happens:

1. The script contains a manually written list of provision pairs.
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

- 107 manually annotated pairs.
- 85 positive relation pairs.
- 22 `NO_RELATION` negative pairs.

This file is the answer key used during evaluation.

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

---

## 7. Mapping Method 1 - Rule-Based Mapping

Used Script:

```text
src/mapping/rule_based.py
```

Run command:

```bash
python3 -m src.mapping.rule_based
```

What happens:

The script runs two rule-based approaches.

First, we used TF-IDF cosine similarity:

- It converts provision text into word vectors.
- It compares provisions based on shared words and phrases.
- It is good for simple lexical similarity.

Second, we used keyword Jaccard similarity:

- To looks only for predefined cybersecurity keywords.
- For Examples: `password`, `authentication`, `encryption`, `vulnerability`, `logging`, `risk`.
- It compares how many security keywords two provisions share within the standards.

Created files:

```text
data/mappings/rule_based_tfidf.csv
data/mappings/rule_based_jaccard.csv
data/mappings/tfidf_similarity_matrix.npy
```

Short Description of the created file.

- `rule_based_tfidf.csv`: It is generated for the predictions from TF-IDF only.
- `rule_based_jaccard.csv`: It is generated for the predictions from cybersecurity keyword matching.
- `tfidf_similarity_matrix.npy`: It is generated for the raw TF-IDF similarity scores for all provision pairs.

---

## 8. Mapping Method 2 - SBERT Mapping

Used Script:

```text
src/mapping/sbert_mapping.py
```

Run command:

```bash
python3 -m src.mapping.sbert_mapping
```

Model used:

```text
all-MiniLM-L6-v2
```

What happens:

1. With this model each provision text is converted into a sentence embedding.
2. An embedding is a numerical meaning representation of the sentence.
3. The script compares every EN 303 645 embedding with every EN 304 223 embedding.
4. If the similarity score is high enough, the pair is saved as a predicted mapping.
5. The script also assigns a relationship label using score thresholds.

Created files:

```text
data/mappings/sbert_output.csv
data/mappings/sbert_similarity_matrix.npy
```

Short Description of the created file.

- `sbert_output.csv`: it is for predicted mappings from SBERT.
- `sbert_similarity_matrix.npy`: It is for raw similarity matrix for all 4,968 pairs.

---

## 9. Mapping Method 3 - BERT Mapping

Used Script:

```text
src/mapping/bert_mapping.py
```

Run command:

```bash
python3 -m src.mapping.bert_mapping
```

Model used:

```text
bert-base-uncased
```

Helper file:

```text
src/mapping/embeddings.py
```

What happens:

1. The script loads the BERT model.
2. It tokenizes each provision text.
3. It creates embeddings using mean pooling over BERT's token outputs.
4. It normalizes the embeddings.
5. It calculates cosine similarity between every pair.
6. Pairs above the threshold are saved.

Created file:

```text
data/mappings/bert_output.csv
```

---

## 10. Mapping Method 4 - SecureBERT Mapping

Used Script:

```text
src/mapping/securebert_mapping.py
```

Run command:

```bash
python3 -m src.mapping.securebert_mapping
```

Model used:

```text
ehsanaghaei/SecureBERT
```

What happens:

1. This is similar to the BERT script.
2. The difference is that SecureBERT is trained on cybersecurity text.
3. The script creates embeddings for all provisions.
4. It compares every possible pair.
5. Pairs above the threshold are saved.

Created file:

```text
data/mappings/securebert_output.csv
```

Note:

The first run may download the SecureBERT model from Hugging Face.

---

## 11. Mapping Method 5 - Gemini Embedding Mapping

Used Script:

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

## 12. Mapping Method 6 - Gemini LLM Mapping

Used Script:

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

---

## 13. Shared Embedding Helper

Used Script:

```text
src/mapping/embeddings.py
```

This is not usually run directly.

It provides shared functions used by SBERT, BERT, SecureBERT, and Gemini embedding scripts.

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

## 14. Step 4 - Evaluate Predictions

Used Script:

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
data/mappings/rule_based_tfidf.csv
data/mappings/rule_based_jaccard.csv
data/mappings/sbert_output.csv
data/mappings/bert_output.csv
data/mappings/securebert_output.csv
data/mappings/gemini_embedding_output.csv
data/mappings/gemini_output.csv
```

What happens:

1. It loads the manual baseline.
2. It loads each method's prediction CSV.
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

Created files:

```text
data/evaluation/rule_based_tfidf_eval.json
data/evaluation/rule_based_jaccard_eval.json
data/evaluation/sbert_eval.json
data/evaluation/bert_eval.json
data/evaluation/securebert_eval.json
data/evaluation/gemini_embedding_eval.json
data/evaluation/gemini_llm_eval.json
data/evaluation/evaluation_summary.json
```

Also created:

```text
data/evaluation/*_errors.csv
data/evaluation/*_predictions_vs_gt.csv
```

What these mean:

- `*_eval.json`: It is a file containing a metrics for one method.
- `*_errors.csv`: It is a file containing mistakes made by one method.
- `*_predictions_vs_gt.csv`: It is a side-by-side comparison between predicted and true labels.
- `evaluation_summary.json`: It combined metrics for all methods.

---

## 15. Step 5 - Generate Final Report

Used Script:

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

Created file:

```text
data/evaluation/report.md
```

This is the final Markdown evaluation report.

---


## 16. Full End-to-End Order

If starting from the raw PDFs, the full order is:

```bash
python3 -m src.extraction.provision_extractor
python3 -m src.baseline.create_baseline
python3 -m src.mapping.rule_based
python3 -m src.mapping.sbert_mapping
python3 -m src.mapping.bert_mapping
python3 -m src.mapping.securebert_mapping
python3 -m src.evaluation.evaluate
python3 -m src.report.generate_report
```

If we want Gemini results, we run these before evaluation:

```bash
python3 -m src.mapping.gemini_embedding_mapping
python3 -m src.mapping.gemini_mapping
```

---

## 18. Complete File Flow

```text
Raw PDFs
  |
  | src/extraction/provision_extractor.py
  v
Extracted provision JSON files
  |
  | src/baseline/create_baseline.py
  v
Manual ground truth CSV
  |
  | src/mapping/*.py
  v
Automatic mapping CSV files
  |
  | src/evaluation/evaluate.py
  v
Evaluation JSON, error CSV, prediction-vs-ground-truth CSV
  |
  | src/report/generate_report.py
  v
Final Markdown report
```

---

## 19. Current Evaluation Summary

Current results from:

```text
data/evaluation/evaluation_summary.json
```

Metrics are computed only against the 107 annotated GT pairs. Predictions outside GT scope are excluded from all calculations.

| Method | Detection Precision | Detection Recall | Detection F1 | Classification Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Rule-based TF-IDF | 1.000 | 0.024 | 0.046 | 0.206 | 0.069 |
| Rule-based Jaccard | 1.000 | 0.071 | 0.132 | 0.224 | 0.179 |
| SBERT | 1.000 | 0.165 | 0.283 | 0.224 | 0.099 |
| BERT | 0.808 | 0.988 | 0.889 | 0.383 | 0.162 |
| SecureBERT | 0.794 | 1.000 | 0.885 | 0.028 | 0.011 |
| Gemini Embedding | 0.794 | 1.000 | 0.885 | 0.458 | 0.278 |
| Gemini LLM | 0.780 | 0.835 | 0.807 | 0.159 | 0.085 |

In the current saved results, **Gemini Embedding** has the highest classification macro-F1.

---

