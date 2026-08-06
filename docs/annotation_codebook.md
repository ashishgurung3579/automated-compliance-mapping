# Annotation codebook — provision relationship labelling

Rules applied when labelling pairs drawn from `data/baseline/reference_pool.csv`.
Written before annotation and unchanged during it, so that labels stay comparable
across batches and the protocol can be audited.

Requirement A always comes from ETSI EN 303 645 (consumer IoT), requirement B from
ETSI EN 304 223 (AI systems).

## Decision procedure

Applied in order. The first rule that fires wins.

1. **Do both requirements pursue the same security objective?**
   If not, the answer is `COMPLEMENTARITY` or `NO_RELATION` — go to rule 4.
2. **Same objective, and does one requirement fully contain the other?**
   - Neither is broader, scope is essentially the same → `EQUIVALENCE`
   - A contains B and demands more → `SUBSUMPTION_A_BROADER`
   - B contains A and demands more → `SUBSUMPTION_B_BROADER`
3. **Same objective, but each mandates something the other does not** → `OVERLAP`
4. **Different objectives that must both hold for one larger control to work**
   → `COMPLEMENTARITY`
5. **Otherwise** → `NO_RELATION`

## The distinction that carries most of the difficulty

`OVERLAP` is the same objective pursued by partly different means.
`COMPLEMENTARITY` is different objectives that combine into a larger control.

Two requirements are **not** complementary merely because both concern security. The
test: name the single larger control they jointly serve. If that cannot be named in
one clause, the label is `NO_RELATION`.

This test exists because it is the failure mode observed in the automated methods —
the zero-shot LLM assigned `COMPLEMENTARITY` to 4,045 of 4,968 pairs, which makes the
label meaningless.

**Amendment made during batch 4, applied to batches 4 onward.** Pairing a specific
control against a general process requirement — threat modelling, risk review, staff
training — was initially labelled `COMPLEMENTARITY` on the reasoning that the process
selects the control. That reasoning relates such a process provision to *every*
control provision in the other standard, reproducing exactly the degenerate behaviour
criticised above. The rule is now: a general process provision is `COMPLEMENTARITY`
with a specific control only when the process text names that control's risk domain.
EN 304 223 provision 5.1.3-1 pairs with input validation, because its text names data
poisoning; it does not pair with secure boot or brute-force resistance.

Batches 0–3 were labelled before this amendment and contain four pairs that the rule
would now move to `NO_RELATION`. They are left as recorded rather than retrofitted,
and the affected pairs are re-examined in the test–retest pass, where the amendment
means disagreement is expected and informative.

## Calibration notes

- **Expect `NO_RELATION` to dominate.** The two standards address different layers:
  EN 303 645 governs a physical consumer device, EN 304 223 governs an AI system
  lifecycle and the organizations around it. Most pairings are genuinely unrelated,
  and a reference set that is mostly positive is a sign of selection bias, not of
  two closely aligned standards.
- **Shared vocabulary is not a relationship.** "Security", "risk", "access",
  "update" and "data" appear throughout both documents. Judge the obligation, not
  the wording.
- **Different actors weaken the case for `OVERLAP`.** EN 303 645 obligates the
  manufacturer or the device; EN 304 223 obligates Developers, System Operators,
  Data Custodians or End-users. Where the obligation falls on parties at different
  points in the supply chain, `COMPLEMENTARITY` usually fits better.
- **`EQUIVALENCE` should be rare.** It requires substantially the same scope, not
  merely the same topic. A device-level control and an organization-level process
  are not equivalent even when both concern, say, authentication.
- **Subsumption is directional and asymmetric.** Use it only when one requirement's
  obligations are a strict superset. When both sides carry something unique, the
  label is `OVERLAP`.

## Confidence

- **3** — the label follows directly from the requirement texts.
- **2** — defensible, but a second annotator could reasonably choose an adjacent
  label (usually `OVERLAP` versus `COMPLEMENTARITY`, or `COMPLEMENTARITY` versus
  `NO_RELATION`).
- **1** — the provision text is too fragmentary or generic to judge confidently.

Confidence is recorded per pair so that agreement can be analysed against it; if
low-confidence pairs are not where re-annotation disagrees, the field carries no
information and is reported as such.

## Procedure

Pairs are shuffled under a fixed seed and presented in batches, so neighbouring
provisions are not judged together and ordering cannot induce anchoring. Each pair
receives a label, a one-sentence justification, and a confidence score. The pilot
set's 107 pairs are mixed into the same stream without being marked, which makes the
comparison against their earlier human-reviewed labels blind.
