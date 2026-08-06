# Automated Compliance Mapping - Evaluation Report

**Generated**: 2026-08-07  
**Standards**: ETSI EN 303 645 (IoT security) x ETSI EN 304 223 (AI security)  
**Ground truth**: 107 annotated pairs (85 positive, 22 negative)  

---

## Method Comparison

| Method | Det. P | Det. R | Det. F1 | Cls. Accuracy | Macro-F1 |
|--------|-------:|-------:|--------:|--------------:|---------:|
| TF-IDF cosine similarity | 100.0% | 3.5% | 6.8% | 21.5% | 8.6% |
| Security-keyword Jaccard overlap | 100.0% | 7.1% | 13.2% | 22.4% | 17.9% |
| Sentence-BERT (all-MiniLM-L6-v2) | 100.0% | 17.6% | 30.0% | 22.4% | 9.9% |
| Sentence-BERT (all-mpnet-base-v2) | 100.0% | 16.5% | 28.3% | 22.4% | 10.0% |
| BGE (BAAI/bge-base-en-v1.5) | 83.3% | 88.2% | 85.7% | 24.3% | 14.9% |
| BERT (bert-base-uncased), mean-pooled | 81.0% | 100.0% | 89.5% | 41.1% | 17.2% |
| SecureBERT (ehsanaghaei/SecureBERT) | 79.4% | 100.0% | 88.5% | 2.8% | 1.1% |
| CySecBERT (markusbayer/CySecBERT) | 79.4% | 100.0% | 88.5% | 19.6% | 8.7% |
| NLI cross-encoder (nli-deberta-v3-base) | 100.0% | 5.9% | 11.1% | 21.5% | 10.2% |
| Gemini embeddings (gemini-embedding-2) | 79.4% | 100.0% | 88.5% | 47.7% | 27.9% |
| Gemini LLM classification (gemini-2.5-flash-lite) | 77.7% | 85.9% | 81.6% | 18.7% | 8.8% |


## Best Performing Method

**Gemini embeddings (gemini-embedding-2)** achieved the highest macro-F1 of **27.9%** on classification, with pair detection precision 79.4% / recall 100.0% / F1 88.5%.

---

## Per-Method Detail

### TF-IDF cosine similarity

**Pair detection**

- Precision: 1.000  
- Recall: 0.035  
- F1: 0.068  
- True positives: 3 / 85  
- False positives (on GT negatives): 0  
- False negatives: 82  
- Predicted positive pairs within GT scope: 3  

**Classification**

- Overall accuracy: 0.215  
- Macro-F1: 0.086  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.333 | 0.045 | 0.080 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.211 | 1.000 | 0.349 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### Security-keyword Jaccard overlap

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


### Sentence-BERT (all-MiniLM-L6-v2)

**Pair detection**

- Precision: 1.000  
- Recall: 0.176  
- F1: 0.300  
- True positives: 15 / 85  
- False positives (on GT negatives): 0  
- False negatives: 70  
- Predicted positive pairs within GT scope: 15  

**Classification**

- Overall accuracy: 0.224  
- Macro-F1: 0.099  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.143 | 0.091 | 0.111 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.239 | 1.000 | 0.386 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### Sentence-BERT (all-mpnet-base-v2)

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
- Macro-F1: 0.100  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.167 | 0.091 | 0.118 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.237 | 1.000 | 0.383 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### BGE (BAAI/bge-base-en-v1.5)

**Pair detection**

- Precision: 0.833  
- Recall: 0.882  
- F1: 0.857  
- True positives: 75 / 85  
- False positives (on GT negatives): 15  
- False negatives: 10  
- Predicted positive pairs within GT scope: 90  

**Classification**

- Overall accuracy: 0.243  
- Macro-F1: 0.149  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.198 | 0.773 | 0.315 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.412 | 0.318 | 0.359 |
| OVERLAP | 51 | 0.500 | 0.039 | 0.073 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### BERT (bert-base-uncased), mean-pooled

**Pair detection**

- Precision: 0.809  
- Recall: 1.000  
- F1: 0.895  
- True positives: 85 / 85  
- False positives (on GT negatives): 20  
- False negatives: 0  
- Predicted positive pairs within GT scope: 105  

**Classification**

- Overall accuracy: 0.411  
- Macro-F1: 0.172  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.062 | 0.333 | 0.105 |
| NO_RELATION | 22 | 1.000 | 0.091 | 0.167 |
| OVERLAP | 51 | 0.461 | 0.804 | 0.586 |
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


### CySecBERT (markusbayer/CySecBERT)

**Pair detection**

- Precision: 0.794  
- Recall: 1.000  
- F1: 0.885  
- True positives: 85 / 85  
- False positives (on GT negatives): 22  
- False negatives: 0  
- Predicted positive pairs within GT scope: 107  

**Classification**

- Overall accuracy: 0.196  
- Macro-F1: 0.087  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.037 | 0.667 | 0.070 |
| NO_RELATION | 22 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 51 | 0.358 | 0.372 | 0.365 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### NLI cross-encoder (nli-deberta-v3-base)

**Pair detection**

- Precision: 1.000  
- Recall: 0.059  
- F1: 0.111  
- True positives: 5 / 85  
- False positives (on GT negatives): 0  
- False negatives: 80  
- Predicted positive pairs within GT scope: 5  

**Classification**

- Overall accuracy: 0.215  
- Macro-F1: 0.102  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.000 | 0.000 | 0.000 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.216 | 1.000 | 0.355 |
| OVERLAP | 51 | 0.000 | 0.000 | 0.000 |
| SUBSUMPTION | 9 | 0.250 | 0.111 | 0.154 |


### Gemini embeddings (gemini-embedding-2)

**Pair detection**

- Precision: 0.794  
- Recall: 1.000  
- F1: 0.885  
- True positives: 85 / 85  
- False positives (on GT negatives): 22  
- False negatives: 0  
- Predicted positive pairs within GT scope: 107  

**Classification**

- Overall accuracy: 0.477  
- Macro-F1: 0.279  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.217 | 0.227 | 0.222 |
| EQUIVALENCE | 3 | 1.000 | 0.333 | 0.500 |
| NO_RELATION | 22 | 0.000 | 0.000 | 0.000 |
| OVERLAP | 51 | 0.542 | 0.882 | 0.672 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


### Gemini LLM classification (gemini-2.5-flash-lite)

**Pair detection**

- Precision: 0.777  
- Recall: 0.859  
- F1: 0.816  
- True positives: 73 / 85  
- False positives (on GT negatives): 21  
- False negatives: 12  
- Predicted positive pairs within GT scope: 94  

**Classification**

- Overall accuracy: 0.187  
- Macro-F1: 0.088  

**Per-class F1**

| Class | Support | Precision | Recall | F1 |
|-------|--------:|----------:|-------:|---:|
| COMPLEMENTARITY | 22 | 0.198 | 0.773 | 0.315 |
| EQUIVALENCE | 3 | 0.000 | 0.000 | 0.000 |
| NO_RELATION | 22 | 0.077 | 0.045 | 0.057 |
| OVERLAP | 51 | 0.286 | 0.039 | 0.069 |
| SUBSUMPTION | 9 | 0.000 | 0.000 | 0.000 |


---

## Limitations and Notes

- All metrics computed against the 107 annotated GT pairs only; predictions outside GT scope are excluded from evaluation.
- SUBSUMPTION variants (A_BROADER / B_BROADER) are merged into a single SUBSUMPTION class for classification metrics.
- Threshold values for each method were set heuristically; systematic threshold search may improve precision/recall balance.
- Gemini Embedding API results may vary across API versions or rate-limit conditions.
