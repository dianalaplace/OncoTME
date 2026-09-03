# OncoTME

**Tumor microenvironment (TME) as a prognostic and predictive biomarker in breast cancer.**

An ML framework that uses molecular TME profiling (immune infiltration, stroma, antigen presentation) for two parallel tasks:

1. **Prognostic** — predicting relapse-free / overall survival (RFS / DFS / OS)
2. **Predictive** — predicting response to therapy (neoadjuvant pCR, ORR) across treatment classes: chemo, endocrine, anti-HER2, PARP, CDK4/6, ICI

## Hypotheses

| # | |
|---|---|
| **H1** | TME archetype is an independent predictor of RFS/OS after controlling for clinical variables and subtype |
| **H2** | The role of TME components depends on treatment class (chemo / anti-HER2 / PARP / ICI / endocrine) |
| **H3** | The TME effect is subtype-specific (TNBC / HER2+ / HR+) |
| **H4** | Antigen-presentation defects (APM/HLA-low) mark poor RFS independent of infiltration |
| **H5** | A "TME-hot" profile confers greater relative benefit from adding targeted therapy / ICI on top of chemo |

## Data (public)

- **Discovery / prognostic**: TCGA-BRCA, METABRIC, SCAN-B
- **Predictive / response**: GSE25066 (MDACC), I-SPY1/2 (GSE22226, GSE194040), GeparSixto (GSE87455), GeparNuevo (GSE173839), CALGB 40601 (GSE181574), NeoALTTO (GSE50948), TransATAC (GSE59515), Bassez 2021 (GSE169246)

## Structure

```
src/oncotme/
├── cohorts/ # per-cohort loaders → CohortBundle
├── preprocess/ # normalization, gene harmonization, ComBat, QC
├── features/ # ssGSEA signatures, TME panel, baselines (TIS, CYT, ESTIMATE)
├── models/ # response.py (clf), survival.py (Cox/RSF), blocks.py (ablation)
├── explain/ # SHAP, PD, interactions
├── stats/ # DeLong, compareC, bootstrap, calibration, DCA
└── report/ # Quarto integration
```

## Key results (flagship figures)

- **TME × treatment interaction heatmap** — `feature × treatment → {pCR-OR, RFS-HR}`
- **Subtype-stratified SHAP** — separate rankings for TNBC / HER2+ / HR+
- **Block importance** — % |SHAP| by module (immune / stromal / APM / clinical)
- **Survival by TME archetype** — KM curves with multivariable Cox models

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# build per-cohort artifacts
python scripts/build_features.py --config config/cohorts.yaml

# train models
python scripts/train_survival.py --cohort tcga_brca
python scripts/train_response.py --cohort pooled_neoadjuvant

# render report
quarto render report/
```

## Shared components

This project reuses several modules from the sibling `DrugResponse/` repo (ssGSEA wrapper, ComBat, nested-CV scaffold, TME signature GMTs).

## License / status

Research use. In development.
