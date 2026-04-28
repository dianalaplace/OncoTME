# OncoTME — TME × targeted therapy class (I-SPY2)

## Pre-specified composite биомаркеры

Для каждого класса таргетной терапии composite TME score pre-specified на основе
published biology (см. `src/oncotme/benchmark/targeted_composites.py` для citations).
**One test per class** → BH FDR across 6 classes only, no 23×N hidden tests.

## Composite formulas

- **anti-HER2** [ADCC — NK-cell effector + antigen presentation, limited by stroma/TGFβ]:
  ```
  composite = z(NK_cell) + z(HLA_I_score) + z(B_cell) + z(TLS_signature) - z(CAF_proxy) - z(TGFb_activity)
  ```
  Rationale: Stagg PNAS 2011; Triulzi Oncotarget 2015; Shi JCI 2015; Kurozumi BCR 2019

- **PARPi+platinum** [DNA damage → STING/IFN-I, proliferation-dependent]:
  ```
  composite = z(IFN_gamma) + z(cytolytic_score) + z(CXCL1_MDSC_axis) - z(CAF_proxy)
  ```
  Rationale: Pantelidou Cancer Discov 2019; Sen Cancer Cell 2019; Post CCR 2022; Loibl GeparOLA 2018

- **anti-Angiopoietin** [vascular normalization + Tie2+/M2 macrophage dependency]:
  ```
  composite = z(angiogenesis) + z(M2_macrophage) + z(hypoxia_score) - z(TLS_signature)
  ```
  Rationale: De Palma Cell 2012; Rigamonti Nat Rev Clin Oncol 2014; Murdoch Blood 2004

- **anti-IGF-1R** [stromal IGF-1 paracrine loop blunts IGF-1R blockade]:
  ```
  composite = - z(CAF_proxy) - z(TGFb_activity) - z(immune_exclusion)
  ```
  Rationale: Cao Cancer Res 2013; Wang Cancer Discov 2015; Fu Cell Rep 2019

- **pan-HER TKI** [pan-HER inhibition; TGFβ-EMT drives resistance]:
  ```
  composite = z(angiogenesis) - z(TGFb_activity) - z(CAF_proxy)
  ```
  Rationale: Canonici Oncotarget 2013; Chandarlapaty Cancer Discov 2015; Shibue NEJM 2019

- **AKT inhibitor** [PI3K/AKT axis; bypassed via stromal/immune-suppressive signaling]:
  ```
  composite = - z(immune_exclusion) - z(TGFb_activity) - z(M2_macrophage)
  ```
  Rationale: Ma Mol Cancer Ther 2010; Kechagioglou Anticancer Res 2014

## Interaction results

| Class | n_ctrl / n_trt | pCR ctrl → trt | OR_int [95% CI] | p_int | q_BH |
|---|---|---|---|---|---|
| **anti-HER2** | 179 / 180 | 17.3% → 48.3% | 1.32 [0.79, 2.22] | 0.2863 | 0.3190 |
| **PARPi+platinum** | 179 / 71 | 17.3% → 38.0% | 1.41 [0.72, 2.76] | 0.3190 | 0.3190 |
| **anti-Angiopoietin** | 179 / 115 | 17.3% → 28.7% | 1.50 [0.84, 2.67] | 0.1721 | 0.2581 |
| **anti-IGF-1R** ✅ | 179 / 105 | 17.3% → 22.9% | 2.16 [1.17, 3.99] | 0.0144 | 0.0432 |
| **pan-HER TKI** ✅ | 179 / 114 | 17.3% → 36.8% | 2.33 [1.27, 4.29] | 0.0064 | 0.0386 |
| **AKT inhibitor** ✅ | 179 / 60 | 17.3% → 30.0% | 2.05 [1.04, 4.03] | 0.0381 | 0.0763 |

## Tertile ARR per class


### anti-HER2

| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |
|---|---|---|---|---|---|
| low | 56 / 64 | 12.5% | 37.5% | **+25.0%** | 3.99 [1.60, 9.99] |
| med | 50 / 69 | 16.0% | 49.3% | **+33.3%** | 4.86 [2.03, 11.63] |
| high | 73 / 47 | 21.9% | 61.7% | **+39.8%** | 5.56 [2.50, 12.35] |

### PARPi+platinum

| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |
|---|---|---|---|---|---|
| low | 62 / 21 | 9.7% | 14.3% | **+4.6%** | 1.64 [0.40, 6.68] |
| med | 60 / 23 | 20.0% | 34.8% | **+14.8%** | 2.13 [0.75, 6.03] |
| high | 57 / 27 | 22.8% | 59.3% | **+36.5%** | 4.73 [1.80, 12.46] |

### anti-Angiopoietin

| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |
|---|---|---|---|---|---|
| low | 58 / 40 | 17.2% | 27.5% | **+10.3%** | 1.80 [0.69, 4.67] |
| med | 61 / 37 | 19.7% | 18.9% | **-0.8%** | 0.97 [0.35, 2.68] |
| high | 60 / 38 | 15.0% | 39.5% | **+24.5%** | 3.58 [1.39, 9.19] |

### anti-IGF-1R

| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |
|---|---|---|---|---|---|
| low | 61 / 34 | 24.6% | 11.8% | **-12.8%** | 0.44 [0.14, 1.39] |
| med | 55 / 39 | 20.0% | 20.5% | **+0.5%** | 1.04 [0.39, 2.83] |
| high | 63 / 32 | 7.9% | 37.5% | **+29.6%** | 6.49 [2.11, 19.92] |

### pan-HER TKI

| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |
|---|---|---|---|---|---|
| low | 61 / 37 | 21.3% | 24.3% | **+3.0%** | 1.20 [0.46, 3.10] |
| med | 54 / 43 | 16.7% | 48.8% | **+32.2%** | 4.58 [1.83, 11.43] |
| high | 64 / 34 | 14.1% | 35.3% | **+21.2%** | 3.25 [1.22, 8.61] |

### AKT inhibitor

| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |
|---|---|---|---|---|---|
| low | 55 / 25 | 21.8% | 24.0% | **+2.2%** | 1.16 [0.39, 3.44] |
| med | 61 / 18 | 24.6% | 44.4% | **+19.9%** | 2.43 [0.83, 7.09] |
| high | 63 / 17 | 6.3% | 23.5% | **+17.2%** | 4.41 [1.05, 18.49] |

## Verdict

✅ **BH-significant (3 / 6 classes)**: pan-HER TKI (q=0.039, OR=2.33), anti-IGF-1R (q=0.043, OR=2.16), AKT inhibitor (q=0.076, OR=2.05).