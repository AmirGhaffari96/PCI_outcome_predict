# One-year MACE prediction after PCI in elderly patients

[![Article DOI](https://img.shields.io/badge/DOI-10.1186%2Fs12911--025--03238--7-7b1fa2)](https://doi.org/10.1186/s12911-025-03238-7)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

Reproducible code, original analysis notebooks, and publication assets for:

> Jolfayi AG, Nasrollahizadeh A, Nasrollahizadeh A, et al. **Applied machine learning to predict 1-year major adverse cardiovascular events in elderly patients after percutaneous coronary intervention.** *BMC Medical Informatics and Decision Making*. 2025;25:401. [doi:10.1186/s12911-025-03238-7](https://doi.org/10.1186/s12911-025-03238-7)

The retrospective cohort included 1,358 patients aged 65 years or older who underwent PCI for STEMI. The endpoint was one-year MACE: cardiovascular death, myocardial infarction, stroke, or repeat revascularization. The article compares XGBoost, random forest, logistic regression, a neural network, SVM, naive Bayes, KNN, and a decision tree. In the published test results, random forest achieved AUC 0.953 (95% CI 0.943-0.964) and XGBoost achieved AUC 0.941 (95% CI 0.930-0.950).

> [!IMPORTANT]
> This repository is for research reproducibility, not clinical decision-making. The patient-level dataset is not public, so the included synthetic data can validate the software only; it cannot reproduce or validate the reported clinical performance.

## Reproducible workflow

The cleaned implementation in `src/pci_mace/` follows the article while preventing evaluation leakage:

1. Split the original cohort into stratified training and test sets.
2. Within the training pipeline, median-impute, standardize, and apply SMOTE.
3. Optionally tune hyperparameters with stratified cross-validation on the training set only.
4. Evaluate the final models once on the untouched test set.
5. Export metrics, bootstrap AUC confidence intervals, ROC curves, run metadata, and optional XGBoost SHAP plots.

This ordering is intentional. The historical notebooks are preserved under `notebooks/original/`, but some of them scale or apply SMOTE before splitting. Use the packaged pipeline for new analyses.

### Installation

```bash
git clone https://github.com/AmirGhaffari96/PCI_outcome_predict.git
cd PCI_outcome_predict
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### Smoke test with synthetic data

```bash
python scripts/generate_demo_data.py --rows 600 --output data/demo.csv
pci-mace train \
  --data data/demo.csv \
  --output outputs/demo \
  --models xgboost random_forest logistic_regression decision_tree \
  --bootstrap 100
pytest
```

Synthetic results must not be compared with the article's clinical results.

### Run on authorized study data

The expected columns are documented in [`data/README.md`](data/README.md). To run all eight models with training-fold grid search and SHAP explanations:

```bash
pci-mace train \
  --data data/Data.xlsx \
  --output outputs/paper-run \
  --models all \
  --tune \
  --shap
```

The command writes:

- `metrics.csv` - AUC, 95% bootstrap CI, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, and F1.
- `roc-curves.png` - test-set ROC comparison.
- `run-metadata.json` - features, split settings, seed, SMOTE ratio, and best hyperparameters.
- `shap-feature-importance.png` and `shap-summary.png` - generated when `--shap` is requested.

## Repository layout

```text
.
├── src/pci_mace/           # Leakage-resistant reusable pipeline and CLI
├── scripts/                # Synthetic-data generator
├── tests/                  # Automated smoke tests
├── data/README.md          # Data-access note and required schema
├── notebooks/original/     # Author-provided historical notebooks
├── paper/                  # Published PDF and unmodified figures
├── CITATION.cff            # Machine-readable citation
└── THIRD_PARTY_NOTICES.md  # Article/figure license notice
```

The older root-level scripts and notebooks remain available for provenance and comparison.

## Article figures

All images below are unmodified extractions from the published article PDF.

### Graphical abstract

![Graphical abstract summarizing the population, endpoint, preprocessing, model performance, and feature importance](paper/figures/graphical-abstract.png)

### Figure 1 - Summary of methods and results

![Figure 1: summary of the study population, outcome, data preparation, model performance, and SHAP feature importance](paper/figures/figure-1.png)

### Figure 2 - Correlation heatmaps and feature selection

![Figure 2: heatmaps before and after removal of highly correlated variables](paper/figures/figure-2.png)

### Figure 3 - ROC curves for all models

![Figure 3: ROC curves comparing the eight machine-learning models](paper/figures/figure-3.jpg)

### Figure 4 - XGBoost SHAP interpretation

![Figure 4: SHAP feature importance and summary plots for MACE prediction](paper/figures/figure-4.jpg)

## Data availability and privacy

The article states that the supporting data are available from the corresponding author upon reasonable request. No patient-level data are committed here. The `.gitignore` excludes CSV and Excel files under `data/` to reduce the risk of accidentally publishing protected clinical information.

## Citation and article license

Citation metadata are available in [`CITATION.cff`](CITATION.cff). The article PDF and its figures are distributed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). They are included unmodified with attribution. The article license does not automatically license the source code.
