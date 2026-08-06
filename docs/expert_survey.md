# Expert validation survey — instrument and analysis plan

Proposed protocol for validating the ground-truth mapping between ETSI EN 303 645 and
ETSI EN 304 223 with practising cybersecurity professionals. This document is the
distribution-ready version of the instrument; the thesis appendix contains the same
content in condensed form. The study is designed but has not been executed within the
degree project; the author re-annotation study (`src/validation/`) serves as the
in-project validation.

## 1. Purpose

Measure whether independent domain experts assign the same relationship labels to
provision pairs as the project's ground truth, and quantify how consistent experts are
with each other. High expert–ground-truth agreement supports using the ground truth as
an evaluation reference; low agreement localises pairs where the annotation is
contestable.

## 2. Participants

- **Target group:** professionals with ≥ 2 years in cybersecurity engineering,
  compliance, or security assurance; familiarity with at least one of the two
  standards (or equivalent frameworks such as ISO/IEC 27001, IEC 62443).
- **Target sample:** 5–10 respondents. With Fleiss' kappa on 30 items and 6
  categories, 5 raters already gives a usable estimate; more raters narrow the CI.
- **Recruitment:** professional networks, university industry contacts, LinkedIn
  groups on IoT/AI security. No compensation; participation anonymous.

## 3. Ethics and data handling

- No personal data beyond the screening variables (role category, years of
  experience, self-rated familiarity). No names or employers recorded.
- Responses stored pseudonymously (respondent R1, R2, …).
- Participants are informed of the study purpose and that aggregated results will be
  published in a master's thesis. Consent obtained via a checkbox before the task.
- Under Swedish rules for student projects, this design (anonymous professional
  judgements, no sensitive data) does not require ethics-board approval; supervisor
  confirmation is obtained before deployment.

## 4. Screening questions

| # | Question | Type |
|---|---|---|
| S1 | Current role (security engineer / compliance-GRC / auditor / researcher / other) | single choice |
| S2 | Years of professional experience in cybersecurity | number |
| S3 | Familiarity with ETSI EN 303 645 | 1–5 Likert |
| S4 | Familiarity with ETSI EN 304 223 or other AI-security guidance | 1–5 Likert |
| S5 | Familiarity with compliance/standards mapping as an activity | 1–5 Likert |

Respondents with S2 < 2 years are excluded from the main analysis but retained in a
sensitivity check.

## 5. Main task

Each respondent rates the **same 30 provision pairs**, drawn from
`data/baseline/gt_sample.csv` and stratified over all six labels. Item order is
randomised per respondent.

For each pair the respondent sees the full text of both provisions and answers:

- **Q1 — Relationship** (single choice, definitions shown on every page):
  - *Equivalence* — the two requirements express the same security objective with
    comparable scope.
  - *Subsumption (IoT broader)* — the EN 303 645 requirement covers the EN 304 223
    requirement and more.
  - *Subsumption (AI broader)* — the EN 304 223 requirement covers the EN 303 645
    requirement and more.
  - *Overlap* — the requirements share part of their objective but each has
    uncovered aspects.
  - *Complementarity* — the requirements address different aspects that jointly
    support a common security goal.
  - *No relation* — no meaningful connection.
- **Q2 — Confidence** in the chosen label (1 = guessing, 5 = certain).
- **Q3 — Optional free-text justification.**

Estimated completion time: 35–45 minutes. An attention check (one duplicated pair
with swapped order) is included; inconsistent answers flag the respondent for review.

## 6. Post-task questions

| # | Question | Type |
|---|---|---|
| P1 | How clear were the six relationship definitions? | 1–5 Likert |
| P2 | Which label pair was hardest to distinguish? | single choice among label pairs |
| P3 | Would a tool proposing candidate mappings with these labels help your compliance work? | 1–5 Likert + free text |

## 7. Analysis plan

1. **Inter-expert agreement:** Fleiss' kappa over all respondents on the six-label
   scheme and on the merged five-label scheme (subsumption directions collapsed).
   Interpretation bands: < 0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate,
   0.61–0.80 substantial, > 0.80 almost perfect (Landis & Koch).
2. **Expert–ground-truth agreement:** majority vote per item vs. ground-truth label;
   report percent agreement and Cohen's kappa. Items with no majority are reported
   separately as "contested".
3. **Confidence analysis:** mean confidence per label; correlation between low
   confidence and disagreement (point-biserial).
4. **Error localisation:** items where majority vote differs from ground truth are
   re-examined and, where the experts are convincing, the ground truth is revised in
   a documented change log before final evaluation runs.
5. **Method ranking stability:** re-run the full evaluation with the expert-revised
   ground truth and report whether the ranking of the eleven automated methods
   changes.

## 8. Expected outcomes

Based on published compliance-mapping studies, moderate inter-expert agreement
(κ ≈ 0.4–0.6) is expected on the six-label scheme, higher on the merged scheme; the
overlap/complementarity boundary is expected to be the main source of disagreement.
The ground truth is considered validated for its purpose if expert majority
agreement reaches ≥ 70 % on the merged scheme.
