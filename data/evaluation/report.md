# Automated Compliance Mapping - Evaluation Report

**Generated**: 2026-08-11  
**Standards**: ETSI EN 303 645 (IoT security) x ETSI EN 304 223 (AI security)  
**Ground truth**: 200 annotated pairs (100 positive, 100 negative)  

---

## Method Comparison

| Method | Det. P | Det. R | Det. F1 | Cls. Accuracy | Macro-F1 |
|--------|-------:|-------:|--------:|--------------:|---------:|
| TF-IDF cosine similarity | 50.0% | 1.0% | 2.0% | 49.5% | 13.3% |
| Security-keyword Jaccard overlap | 54.5% | 6.0% | 10.8% | 49.0% | 24.8% |
| Sentence-BERT (all-MiniLM-L6-v2) | 88.2% | 15.0% | 25.6% | 50.5% | 15.9% |
| Sentence-BERT (all-mpnet-base-v2) | 85.7% | 12.0% | 21.1% | 50.5% | 15.9% |
| BGE (BAAI/bge-base-en-v1.5) | 52.5% | 93.0% | 67.2% | 28.0% | 13.3% |
| BERT (bert-base-uncased), mean-pooled | 50.2% | 100.0% | 66.9% | 13.5% | 6.5% |
| SecureBERT (ehsanaghaei/SecureBERT) | 50.0% | 100.0% | 66.7% | 1.5% | 0.6% |
| CySecBERT (markusbayer/CySecBERT) | 50.0% | 100.0% | 66.7% | 4.5% | 3.1% |
| NLI cross-encoder (nli-deberta-v3-base) | 87.5% | 7.0% | 13.0% | 50.0% | 16.1% |
| Gemini embeddings (gemini-embedding-2) | 50.0% | 100.0% | 66.7% | 27.0% | 23.8% |
| Gemini LLM classification (gemini-2.5-flash-lite) | 49.2% | 88.0% | 63.1% | 22.5% | 9.8% |


## Best Performing Method

**Security-keyword Jaccard overlap** achieved the highest macro-F1 of **24.8%** on classification, with pair detection precision 54.5% / recall 6.0% / F1 10.8%.

---

## Per-Method Detail

### TF-IDF cosine similarity

**Pair detection**

- Precision: 0.500  
- Recall: 0.010  
- F1: 0.020  
- True positives: 1 / 100  
- False positives (on GT negatives): 1  
- False negatives: 99  
- Predicted positive pairs within GT scope: 2  

**Classification**

- Overall accuracy: 0.495  
- Macro-F1: 0.133  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 100 | 0.500 | 0.990 | 0.664 |
| OVERLAP | 44 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### Security-keyword Jaccard overlap

**Pair detection**

- Precision: 0.545  
- Recall: 0.060  
- F1: 0.108  
- True positives: 6 / 100  
- False positives (on GT negatives): 5  
- False negatives: 94  
- Predicted positive pairs within GT scope: 11  

**Classification**

- Overall accuracy: 0.490  
- Macro-F1: 0.248  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.111 | 0.023 | 0.038 |
| EQUIVALENCE | 3 | 1.000 | 0.333 | 0.500 |
| NO_RELATION | 100 | 0.503 | 0.950 | 0.657 |
| OVERLAP | 44 | 1.000 | 0.023 | 0.044 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### Sentence-BERT (all-MiniLM-L6-v2)

**Pair detection**

- Precision: 0.882  
- Recall: 0.150  
- F1: 0.256  
- True positives: 15 / 100  
- False positives (on GT negatives): 2  
- False negatives: 85  
- Predicted positive pairs within GT scope: 17  

**Classification**

- Overall accuracy: 0.505  
- Macro-F1: 0.159  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.188 | 0.070 | 0.102 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 100 | 0.535 | 0.980 | 0.693 |
| OVERLAP | 44 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### Sentence-BERT (all-mpnet-base-v2)

**Pair detection**

- Precision: 0.857  
- Recall: 0.120  
- F1: 0.210  
- True positives: 12 / 100  
- False positives (on GT negatives): 2  
- False negatives: 88  
- Predicted positive pairs within GT scope: 14  

**Classification**

- Overall accuracy: 0.505  
- Macro-F1: 0.159  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.250 | 0.070 | 0.109 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 100 | 0.527 | 0.980 | 0.685 |
| OVERLAP | 44 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### BGE (BAAI/bge-base-en-v1.5)

**Pair detection**

- Precision: 0.525  
- Recall: 0.930  
- F1: 0.671  
- True positives: 93 / 100  
- False positives (on GT negatives): 84  
- False negatives: 7  
- Predicted positive pairs within GT scope: 177  

**Classification**

- Overall accuracy: 0.280  
- Macro-F1: 0.133  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.227 | 0.907 | 0.363 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 100 | 0.696 | 0.160 | 0.260 |
| OVERLAP | 44 | 0.200 | 0.023 | 0.041 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### BERT (bert-base-uncased), mean-pooled

**Pair detection**

- Precision: 0.502  
- Recall: 1.000  
- F1: 0.669  
- True positives: 100 / 100  
- False positives (on GT negatives): 99  
- False negatives: 0  
- Predicted positive pairs within GT scope: 199  

**Classification**

- Overall accuracy: 0.135  
- Macro-F1: 0.065  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.017 | 0.333 | 0.032 |
| NO_RELATION | 100 | 1.000 | 0.010 | 0.020 |
| OVERLAP | 44 | 0.179 | 0.568 | 0.272 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### SecureBERT (ehsanaghaei/SecureBERT)

**Pair detection**

- Precision: 0.500  
- Recall: 1.000  
- F1: 0.667  
- True positives: 100 / 100  
- False positives (on GT negatives): 100  
- False negatives: 0  
- Predicted positive pairs within GT scope: 200  

**Classification**

- Overall accuracy: 0.015  
- Macro-F1: 0.006  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.015 | 1.000 | 0.030 |
| NO_RELATION | 100 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 44 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### CySecBERT (markusbayer/CySecBERT)

**Pair detection**

- Precision: 0.500  
- Recall: 1.000  
- F1: 0.667  
- True positives: 100 / 100  
- False positives (on GT negatives): 100  
- False negatives: 0  
- Predicted positive pairs within GT scope: 200  

**Classification**

- Overall accuracy: 0.045  
- Macro-F1: 0.031  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.015 | 0.667 | 0.030 |
| NO_RELATION | 100 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 44 | 0.100 | 0.159 | 0.123 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### NLI cross-encoder (nli-deberta-v3-base)

**Pair detection**

- Precision: 0.875  
- Recall: 0.070  
- F1: 0.130  
- True positives: 7 / 100  
- False positives (on GT negatives): 1  
- False negatives: 93  
- Predicted positive pairs within GT scope: 8  

**Classification**

- Overall accuracy: 0.500  
- Macro-F1: 0.161  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 100 | 0.516 | 0.990 | 0.678 |
| OVERLAP | 44 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 10 | 0.167 | 0.100 | 0.125 |


### Gemini embeddings (gemini-embedding-2)

**Pair detection**

- Precision: 0.500  
- Recall: 1.000  
- F1: 0.667  
- True positives: 100 / 100  
- False positives (on GT negatives): 100  
- False negatives: 0  
- Predicted positive pairs within GT scope: 200  

**Classification**

- Overall accuracy: 0.270  
- Macro-F1: 0.238  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.145 | 0.256 | 0.185 |
| EQUIVALENCE | 3 | 1.000 | 0.333 | 0.500 |
| NO_RELATION | 100 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 44 | 0.342 | 0.955 | 0.503 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


### Gemini LLM classification (gemini-2.5-flash-lite)

**Pair detection**

- Precision: 0.492  
- Recall: 0.880  
- F1: 0.631  
- True positives: 88 / 100  
- False positives (on GT negatives): 91  
- False negatives: 12  
- Predicted positive pairs within GT scope: 179  

**Classification**

- Overall accuracy: 0.225  
- Macro-F1: 0.098  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 43 | 0.216 | 0.837 | 0.343 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 100 | 0.429 | 0.090 | 0.149 |
| OVERLAP | 44 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 10 | 0.000 | 0.000 | 0.000 |


---

## Limitations and Notes

- All metrics computed against the 200 annotated GT pairs only; predictions outside GT scope are excluded from evaluation.
- The reference set is balanced at half negatives by design, so every figure here is an upper bound on corpus-scale performance.
- SUBSUMPTION variants (A_BROADER / B_BROADER) are merged into a single SUBSUMPTION class for classification metrics.
- Methods run at their shipped thresholds here; the cross-validated recalibration is in data/evaluation/analysis.json.
- Gemini Embedding API results may vary across API versions or rate-limit conditions.
