# Automated Compliance Mapping - Evaluation Report

**Generated**: 2026-05-10  
**Standards**: ETSI EN 303 645 (IoT security) x ETSI EN 304 223 (AI security)  
**Ground truth**: 107 annotated pairs (85 positive, 22 negative)  

---

## Method Comparison

| Method | Det. P | Det. R | Det. F1 | Cls. Accuracy | Macro-F1 |
|--------|-------:|-------:|--------:|--------------:|---------:|
| Rule-Based TF-IDF | 66.7% | 2.4% | 4.5% | 20.6% | 6.9% |
| Rule-Based Jaccard | 5.6% | 7.1% | 6.2% | 22.4% | 17.9% |
| SBERT (all-MiniLM-L6-v2) | 25.4% | 16.5% | 20.0% | 22.4% | 9.9% |
| BERT (bert-base-uncased) | 1.8% | 98.8% | 3.5% | 38.3% | 16.2% |
| SecureBERT (ehsanaghaei/SecureBERT) | 1.7% | 100.0% | 3.4% | 2.8% | 1.1% |
| Gemini Embedding (gemini-embedding-2) | 1.7% | 100.0% | 3.4% | 45.8% | 27.8% |
| Gemini LLM (gemini-2.5-flash-lite) | 1.6% | 83.5% | 3.1% | 15.9% | 8.5% |


## Best Performing Method

**Gemini Embedding (gemini-embedding-2)** achieved the highest macro-F1 of **27.8%** on classification, with pair detection precision 1.7% / recall 100.0% / F1 3.4%.

---

## Per-Method Detail

### Rule-Based TF-IDF

**Pair detection**

- Precision: 0.667  
- Recall: 0.024  
- F1: 0.045  
- True positives: 2 / 85  
- False positives (unannotated): 1  
- False positives (on GT negatives): 0  
- False negatives: 83  

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

- Precision: 0.056  
- Recall: 0.071  
- F1: 0.062  
- True positives: 6 / 85  
- False positives (unannotated): 101  
- False positives (on GT negatives): 0  
- False negatives: 79  

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

- Precision: 0.255  
- Recall: 0.165  
- F1: 0.200  
- True positives: 14 / 85  
- False positives (unannotated): 41  
- False positives (on GT negatives): 0  
- False negatives: 71  

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

- Precision: 0.018  
- Recall: 0.988  
- F1: 0.035  
- True positives: 84 / 85  
- False positives (unannotated): 4611  
- False positives (on GT negatives): 20  
- False negatives: 1  

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

- Precision: 0.017  
- Recall: 1.000  
- F1: 0.034  
- True positives: 85 / 85  
- False positives (unannotated): 4861  
- False positives (on GT negatives): 22  
- False negatives: 0  

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

- Precision: 0.017  
- Recall: 1.000  
- F1: 0.034  
- True positives: 85 / 85  
- False positives (unannotated): 4861  
- False positives (on GT negatives): 22  
- False negatives: 0  

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

- Precision: 0.016  
- Recall: 0.835  
- F1: 0.031  
- True positives: 71 / 85  
- False positives (unannotated): 4350  
- False positives (on GT negatives): 20  
- False negatives: 14  

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

- Ground truth covers 107 of 4,968 possible provision pairs; unannotated false positives may include true positives not yet validated.
- SUBSUMPTION variants (A_BROADER / B_BROADER) are merged into a single SUBSUMPTION class for classification metrics.
- Threshold values for each method were set heuristically; systematic threshold search may improve precision/recall balance.
- Gemini Embedding API results may vary across API versions or rate-limit conditions.
