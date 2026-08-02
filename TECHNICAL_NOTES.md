# Technical issues and reproducibility audit

This audit distinguishes the unmodified historical notebooks and published manuscript from the clean implementation in `src/pci_mace/`. It documents observable issues without rewriting the publication record. Severity reflects potential impact on reproducibility or estimated model performance, not intent.

## Historical notebook issues

| Severity | Issue | Evidence | Likely impact | Repository response |
|---|---|---|---|---|
| High | Preprocessing and SMOTE occur before the train/test split | `machine-3-pci.ipynb` fits SMOTE and `StandardScaler` to the full dataset before `train_test_split`; `xgb-shap.ipynb` encodes, scales, and oversamples before splitting. | Synthetic or transformed test observations influence training and can produce optimistic performance estimates. | The clean implementation splits first and places imputation, scaling, and SMOTE inside an `imblearn` pipeline fitted only on training folds. |
| High | Feature-selection dataframe is not used | `machine-3-pci.ipynb` creates `data2` by dropping correlated/redundant variables, but then constructs `X` from `data` rather than `data2`. | The evaluated model may use a different predictor set from the seven-feature model described in the manuscript. | `DEFAULT_FEATURES` explicitly contains pre-PCI EF, age, BMI, LDL/HDL ratio, FBS, hemoglobin, and creatinine. |
| Medium | Splits are not stratified | The three historical notebooks call `train_test_split` without `stratify=y`. | With an 11.2% event rate, class prevalence and metrics can vary unnecessarily across splits. | The clean split uses `stratify=y`; cross-validation uses shuffled `StratifiedKFold`. |
| Medium | Feature encoding is fitted before splitting | `xgb-shap.ipynb` applies each `LabelEncoder` to the full dataset. | Test-set category information enters preprocessing. | The clean seven-feature model uses numeric predictors and fits all learned transformations inside training folds. |
| Medium | A continuous variable is treated as categorical | `xgb-shap.ipynb` includes `Door to Device Distance` in `categorical_cols`. | Label encoding imposes arbitrary integer category structure on a duration variable. | Door-to-device time is not part of the article's final seven predictors. |
| Medium | Hyperparameter search is not present in the supplied notebooks | The manuscript reports grid-search cross-validation, while the preserved notebooks instantiate mostly default estimators and do not execute `GridSearchCV`. | Exact manuscript models and results cannot be reconstructed from the supplied notebooks alone. | The clean CLI provides `--tune` and records selected parameters in `run-metadata.json`. |
| Medium | Reported AUC confidence intervals are not generated | The historical notebooks compute point AUCs but do not show confidence-interval code. | The provenance and method for Table 2 confidence intervals are unavailable. | The clean pipeline exports percentile bootstrap 95% AUC intervals and records the bootstrap iteration count and seed. |
| Medium | Randomness is only partly controlled | Split/SMOTE seeds are sometimes fixed, but random forest, MLP, decision tree, and some XGBoost instances lack `random_state`. | Repeated executions may not reproduce the same metrics. | The clean implementation passes one recorded seed to every stochastic component. |
| Low | Duplicate feature entry | `machine-2-pci.ipynb` includes `BMI` twice in `columns_to_scale`. | Usually redundant, but it obscures the intended feature specification and can complicate maintenance. | The clean feature tuple contains each predictor once. |
| Low | Class labels and probabilities are conflated | `xgb-shap.ipynb` assigns `model.predict(X_test)` to `y_pred_proba`, although `predict` returns class labels. | Using that variable for AUC or calibration would be incorrect; the same notebook separately uses `predict_proba` correctly for AUC. | The clean pipeline consistently evaluates continuous `predict_proba` output and derives labels at the decision threshold. |
| Low | SMOTE settings differ between notebooks | `machine-3-pci.ipynb` uses `sampling_strategy=0.5`; `xgb-shap.ipynb` uses the default strategy, which balances the classes fully. | Results depend on which notebook/settings are treated as authoritative. | The CLI defaults to 0.5 and records the chosen ratio; users can override it with `--smote-ratio`. |

## Manuscript and figure reporting issues

| Severity | Issue | Published evidence | Interpretation |
|---|---|---|---|
| Medium | XGBoost AUC differs across the article | Table 2 and the results text report XGBoost AUC 0.941, while the graphical abstract, Figure 1 artwork, and discussion report 95.1%. Random forest is consistently reported as 0.953/95.3%. | The locked prediction file and analysis script are needed to determine whether 0.941 or 0.951 is correct. The README uses Table 2 as the primary numeric source and flags the discrepancy beside the figures. |
| Medium | Boosted-model naming is inconsistent | Figure 3 labels the leading curve "Gradient Boosting," while the methods, results, and Table 2 use XGB/XGBoost. | Standard gradient boosting and XGBoost are different estimators. The curve should be tied to a saved model identifier. |
| Medium | SHAP direction statements conflict | The results text says lower pre-PCI EF and age increase risk but calls them red, although the Figure 4 legend defines red as high. The Figure 4 caption says high pre-PCI EF and age increase risk. Table 1 shows lower EF but higher age among MACE patients. | Feature rank is clear, but the direction of effect for EF and age should be verified against the original SHAP values before clinical interpretation. |
| Low | Terminology varies | The article uses "XGradient Boosting," "XGB," "XGBoost," and Figure 3's "Gradient Boosting." | Use "XGBoost (extreme gradient boosting)" consistently and record the estimator class/version. |

## Study-level technical limitations

These are limitations of what can be concluded, not necessarily implementation defects:

- The cohort is retrospective and single-center; transportability to other hospitals, countries, treatment eras, and elderly subgroups requires external validation.
- The reduction from 13,682 PCI procedures to 1,358 analyzed patients can introduce selection bias. A reproducible cohort-flow table and exclusion counts would help assess it.
- The event rate is 11.2%. Oversampling can help fitting but does not replace evaluation on the original prevalence distribution.
- Univariate screening performed before data splitting can leak outcome information. Feature selection should be nested inside cross-validation when it is learned from the outcome.
- The composite MACE endpoint combines cardiovascular death, myocardial infarction, stroke, and revascularization. Performance for individual components may differ.
- AUC measures discrimination, not calibration or clinical utility. The clean pipeline now exports Brier scores and calibration curves, but decision-curve analysis and prospective impact evaluation remain future work.
- The manuscript describes mean-versus-median imputation based on distribution shape, but the supplied materials do not specify the variable-by-variable assignment. The clean implementation uses a deterministic median imputer and records that choice.
- The patient-level dataset, fitted models, prediction vectors, exact hyperparameter grid, selected thresholds, and software environment used for the article are not public. Exact numerical reproduction is therefore not currently possible.

## What the clean implementation resolves

The implementation in [`src/pci_mace/pipeline.py`](src/pci_mace/pipeline.py):

1. creates an untouched, stratified test set before fitting any preprocessing;
2. nests imputation, standardization, and SMOTE inside training and cross-validation;
3. fixes the final seven-feature schema explicitly;
4. supports training-only grid search for all eight algorithms;
5. controls and records stochastic seeds;
6. evaluates probability scores rather than class labels;
7. exports discrimination, calibration, likelihood ratios, bootstrap uncertainty, and SHAP explanations; and
8. records run configuration and best hyperparameters in machine-readable JSON.

These changes improve the reliability of future runs but cannot retroactively verify the article's numeric results without the original authorized data and locked analysis artifacts.

## Recommended manuscript corrections

Before a revised manuscript, supplement, or model card is released:

1. reconcile the XGBoost AUC across Table 2, prose, graphical abstract, and Figure 1;
2. confirm whether Figure 3 represents `XGBClassifier` or `GradientBoostingClassifier`;
3. regenerate the Figure 4 direction statements directly from saved SHAP values;
4. publish the exact cohort split, seed, preprocessing pipeline, hyperparameter grids, best parameters, threshold-selection rule, and CI method;
5. report calibration with confidence intervals and assess clinical utility with decision-curve analysis; and
6. perform temporal or external validation before clinical deployment.
