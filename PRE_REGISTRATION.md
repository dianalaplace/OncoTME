# Pre-registration — OncoTME honest hypothesis test

**Date locked:** 2026-04-21
**Author:** D. Lysenko + OncoTME pipeline
**Status:** LOCKED before any new analysis. Formulas cannot be changed.

## Goal

To honestly evaluate whether TME plays a clinically significant role in
targeted therapy response in breast cancer, without relying on publication-grade
external validation (which is not currently achievable in this project scope).

## Why pre-registration here

Previous analyses in OncoTME were partially informed by exploratory per-signature
testing on the same I-SPY2 cohort (GSE194040). This creates a post-hoc bias that
inflates p-values and produces overconfident composites. To honestly answer the
hypothesis, we freeze formulas and tests **before** any new computation.

## Locked composite formula

### Primary: `stromal_score`

```
stromal_score = z(CAF_proxy) + z(TGFb_activity)
```

**Interpretation**: HIGH stromal_score = CAF/TGFβ-rich tumor microenvironment,
predicted to be associated with resistance to targeted therapy.

**Biological rationale** (pre-specified, not informed by I-SPY2):
- Mariathasan et al. Nature 2018 — IMvigor210, TGFβ signaling in fibroblasts →
  immune exclusion → resistance to checkpoint blockade
- Calon et al. Nat Genet 2015 — TGFβ-driven stromal signature → poor response
  in colorectal cancer
- Dominguez et al. Cancer Discov 2020 — LRRC15+ CAF subtype → pan-cancer
  resistance to checkpoint + targeted therapies
- Chen & Song JCI 2019 — TGFβ-activated stroma universally dampens drug efficacy

This composite intentionally uses only TWO signatures (both stromal) and
no immune signatures. This is the simplest testable anti-stromal hypothesis
and ONLY uses signatures whose inclusion is pre-specified by the biology
citations above.

## Hypotheses (locked)

### H1 — Cross-arm consistency (PRIMARY, single main claim)

**Claim**: In I-SPY2 (GSE194040), stromal_score negatively modifies benefit
from targeted therapy across multiple drug classes. Specifically:
OR_interaction < 1 in a majority of arms.

**Statistical test**: Across all 9 I-SPY2 targeted arms vs Paclitaxel control,
fit logistic regression:
```
pCR ~ stromal_score_z + arm + stromal_score_z : arm + HR + MP + HER2
```
Extract OR_interaction per arm. **Primary test = sign test**: what fraction
of 9 arms shows OR_interaction < 1?

- 9/9 or 0/9: p = 0.004 (one-sided binomial 9, 0.5)
- 8/9 or 1/9: p = 0.020
- 7/9 or 2/9: p = 0.090
- 6/9 or 3/9: p = 0.254
- 5/9 or 4/9: p = 0.500

**Success criterion for H1**: ≥7/9 arms with OR_interaction < 1 (one-sided
binomial p ≤ 0.090). This is direction-only, before effect size magnitude.

**Secondary**: random-effects meta-analysis of log(OR_interaction) across 9
arms (DerSimonian-Laird), 95% CI must fully lie below 1 for strong support.

### H2 — Subtype-stratified within-HER2- arms (SECONDARY)

**Claim**: stromal_score × arm interaction for ≥1 of the individual 9 arms
shows consistent direction (OR_int < 1) in BOTH HR+ and HR- (TNBC) subsets
when analyzed separately.

**Rationale**: HR+ and TNBC have fundamentally different biology. If stromal
effect replicates in both subtypes, it's a robust biology-level effect. If
only one — narrower applicability.

**Pre-specified focus arms**: Ganitumab (anti-IGF-1R), Pembrolizumab (ICI),
Neratinib (pan-HER). These are chosen because their interactions were
previously suggested — subtype stratification is the next step to test
robustness, not to discover.

**Success criterion**: ≥1 arm where OR_int < 1 (point estimate) in BOTH
HR+ and HR- subsets, with total N_trt ≥ 20 per subset.

### H3 — Negative control (NULL CHECK)

**Claim**: stromal_score should NOT show significant interaction with
pure chemotherapy (no added targeted agent) in GSE25066 (Hatzis chemo
cohort, N=488, T-FAC/FEC).

**Statistical test**: GSE25066 has only one treatment arm (neoadjuvant
chemo) — there is no control arm. So the test is:
**stromal_score × pCR association** via logistic: pCR ~ stromal_score_z +
HR + subtype_her2 + grade. Expected: stromal_score may correlate with pCR
(prognostic for chemo) BUT this is separate from targeted modulation.

Actually correction: H3 is poorly defined in GSE25066 because there is
no treatment comparison. Replace with:

**H3 (revised)**: In I-SPY2, the SAME stromal_score should NOT show
significant interaction with "soft" arms where adding the agent is known
not to modulate stromal biology: **Ganetespib (HSP90i)** and **AMG-386
(anti-angiopoietin)** are least biologically linked to CAF/TGFβ. If
stromal_score shows **stronger** interaction with these than with
biologically-plausible arms (Ganitumab, Neratinib), that suggests the
"anti-stromal axis" hypothesis is likely artifactual.

**Success criterion for H3**: biologically-plausible arms (Ganitumab,
Neratinib, anti-HER2 if testable) should show numerically stronger
interactions than "stromal-unrelated" arms (Ganetespib, AMG-386,
Ganitumab-as-control for anti-stromal).

## Robustness checks (also locked)

### R1 — Split-half stability
1000 random 50/50 splits of I-SPY2 (stratified by arm + pCR). Fit composite
on half-A, test on half-B. **Report**: fraction of splits where p_int < 0.05
in held-out half.

### R2 — Leave-one-component-out
For the primary composite (CAF + TGFβ), fit WITH only CAF, and WITH only TGFβ.
If effect survives each single component, it's not an artifact of one bad
signature.

### R3 — Permutation null
1000 permutations of pCR label within each arm-vs-control comparison. Record
null distribution of OR_int. Compute empirical p_interaction.

## Analysis plan (chronological)

1. Load I-SPY2 (already computed ssGSEA scores). DO NOT re-examine individual
   signatures.
2. Compute stromal_score per sample.
3. For each of 9 arms in I-SPY2: fit H1 model. Record OR_int, p_Wald,
   covariate balance.
4. Apply sign test and meta-analysis (H1).
5. Restrict to Ganitumab, Pembro, Neratinib arms and run HR+ vs HR- analyses (H2).
6. Compare stromal-related arms vs stromal-unrelated arms (H3).
7. Run R1 split-half on the arm with strongest H1 result.
8. Run R2 leave-one-out on strongest arm.
9. Run R3 permutation on all 9 arms.
10. Write HONEST_VERDICT.md with one paragraph per hypothesis.

## What this analysis DOES NOT claim

- Clinical use of this composite as a biomarker
- External reproducibility in any cohort beyond I-SPY2
- Publication-grade evidence level
- Mechanism proof

## What this analysis CAN claim (if positive)

- Suggestive evidence that TME anti-stromal axis modifies targeted therapy
  response in a direction-consistent way across multiple drug classes
- Hypothesis for external validation in future cohorts
- Reasonable grounds for combination trials (anti-stromal + targeted)

## What this analysis CAN claim (if negative)

- That anti-stromal axis in I-SPY2 does NOT show consistent direction
- Previous nominal positive findings (Ganitumab, Neratinib) were likely
  post-hoc false-positives
- The project's "TME plays clinically significant role in targeted therapy"
  hypothesis is not supported at this evidence level and requires different
  approach or data
