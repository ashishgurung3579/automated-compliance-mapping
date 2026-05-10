# Automated Compliance Mapping - Evaluation Report

**Generated**: 2026-05-10  
**Standards**: ETSI EN 303 645 (IoT security) x ETSI EN 304 223 (AI security)  
**Ground truth**: 107 annotated pairs (85 positive, 22 negative)  

---

## Method Comparison

| Method | Det. P | Det. R | Det. F1 | Cls. Accuracy | Macro-F1 |
|--------|-------:|-------:|--------:|--------------:|---------:|
| Rule-Based TF-IDF | 100.0% | 2.4% | 4.6% | 20.6% | 6.9% |
| Rule-Based Jaccard | 100.0% | 7.1% | 13.2% | 22.4% | 17.9% |
| SBERT (all-MiniLM-L6-v2) | 100.0% | 16.5% | 28.3% | 22.4% | 9.9% |
| BERT (bert-base-uncased) | 80.8% | 98.8% | 88.9% | 38.3% | 16.2% |
| SecureBERT (ehsanaghaei/SecureBERT) | 79.4% | 100.0% | 88.5% | 2.8% | 1.1% |
| Gemini Embedding (gemini-embedding-2) | 79.4% | 100.0% | 88.5% | 45.8% | 27.8% |
| Gemini LLM (gemini-2.5-flash-lite) | 78.0% | 83.5% | 80.7% | 15.9% | 8.5% |


## Best Performing Method

**Gemini Embedding (gemini-embedding-2)** achieved the highest macro-F1 of **27.8%** on classification, with pair detection precision 79.4% / recall 100.0% / F1 88.5%.

---

## Per-Method Detail

### Rule-Based TF-IDF

**Pair detection**

- Precision: 1.000  
- Recall: 0.024  
- F1: 0.046  
- True positives: 2 / 85  
- False positives (on GT negatives): 0  
- False negatives: 83  
- Predicted positive pairs within GT scope: 2  

**Classification**

- Overall accuracy: 0.206  
- Macro-F1: 0.069  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.209 | 1.000 | 0.346 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### Rule-Based Jaccard

**Pair detection**

- Precision: 1.000  
- Recall: 0.071  
- F1: 0.132  
- True positives: 6 / 85  
- False positives (on GT negatives): 0  
- False negatives: 79  
- Predicted positive pairs within GT scope: 6  

**Classification**

- Overall accuracy: 0.224  
- Macro-F1: 0.179  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 1.000 | 0.333 | 0.500 |
| NO_RELATION | 22 | 0.218 | 1.000 | 0.358 |
| OVERLAP | 51 | 1.000 | 0.020 | 0.038 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### SBERT (all-MiniLM-L6-v2)

**Pair detection**

- Precision: 1.000  
- Recall: 0.165  
- F1: 0.283  
- True positives: 14 / 85  
- False positives (on GT negatives): 0  
- False negatives: 71  
- Predicted positive pairs within GT scope: 14  

**Classification**

- Overall accuracy: 0.224  
- Macro-F1: 0.099  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.154 | 0.091 | 0.114 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.237 | 1.000 | 0.383 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### BERT (bert-base-uncased)

**Pair detection**

- Precision: 0.808  
- Recall: 0.988  
- F1: 0.889  
- True positives: 84 / 85  
- False positives (on GT negatives): 20  
- False negatives: 1  
- Predicted positive pairs within GT scope: 104  

**Classification**

- Overall accuracy: 0.383  
- Macro-F1: 0.162  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.056 | 0.333 | 0.095 |
| NO_RELATION | 22 | 0.667 | 0.091 | 0.160 |
| OVERLAP | 51 | 0.442 | 0.745 | 0.555 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### SecureBERT (ehsanaghaei/SecureBERT)

**Pair detection**

- Precision: 0.794  
- Recall: 1.000  
- F1: 0.885  
- True positives: 85 / 85  
- False positives (on GT negatives): 22  
- False negatives: 0  
- Predicted positive pairs within GT scope: 107  

**Classification**

- Overall accuracy: 0.028  
- Macro-F1: 0.011  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.028 | 1.000 | 0.054 |
| NO_RELATION | 22 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### Gemini Embedding (gemini-embedding-2)

**Pair detection**

- Precision: 0.794  
- Recall: 1.000  
- F1: 0.885  
- True positives: 85 / 85  
- False positives (on GT negatives): 22  
- False negatives: 0  
- Predicted positive pairs within GT scope: 107  

**Classification**

- Overall accuracy: 0.458  
- Macro-F1: 0.278  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.214 | 0.273 | 0.240 |
| EQUIVALENCE | 3 | 1.000 | 0.333 | 0.500 |
| NO_RELATION | 22 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 51 | 0.538 | 0.824 | 0.651 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### Gemini LLM (gemini-2.5-flash-lite)

**Pair detection**

- Precision: 0.780  
- Recall: 0.835  
- F1: 0.807  
- True positives: 71 / 85  
- False positives (on GT negatives): 20  
- False negatives: 14  
- Predicted positive pairs within GT scope: 91  

**Classification**

- Overall accuracy: 0.159  
- Macro-F1: 0.085  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.161 | 0.591 | 0.252 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.125 | 0.091 | 0.105 |
| OVERLAP | 51 | 0.200 | 0.039 | 0.066 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


---

## Limitations and Notes

- All metrics computed against the 107 annotated GT pairs only; predictions outside GT scope are excluded from evaluation.
- SUBSUMPTION variants (A_BROADER / B_BROADER) are merged into a single SUBSUMPTION class for classification metrics.
- Threshold values for each method were set heuristically; systematic threshold search may improve precision/recall balance.
- Gemini Embedding API results may vary across API versions or rate-limit conditions.
