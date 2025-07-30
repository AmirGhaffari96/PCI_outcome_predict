# Applied Machine Learning to Predict 1-Year Major Adverse Cardiovascular Events in the Elderly After PCI

This repository contains the codebase for our study:  
**"Applied Machine Learning to Predict 1-Year Major Adverse Cardiovascular Events in the Elderly After Percutaneous Coronary Intervention (PCI)"**

We developed machine learning models—primarily using XGBoost—to predict the risk of MACE (Major Adverse Cardiovascular Events) within one year after PCI in elderly patients. The model is trained on clinical, demographic, and laboratory data.

---

## 📁 Contents

- `XGB.ipynb` – Main notebook with end-to-end implementation using XGBoost
- `figures/` – Stores generated ROC, calibration, and SHAP plots
- `data/` – Expected directory for anonymized dataset (not provided here)
- `README.md` – Documentation file (this file)

---

## 🧠 Key Features

- End-to-end ML pipeline for tabular medical data
- Stratified 5-fold cross-validation
- SMOTE for handling class imbalance
- ROC curve, AUC calculation, and confidence intervals
- Permutation test for comparing AUCs
- SHAP for model interpretability
- Calibration curve for prediction reliability

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/mace-prediction.git
cd mace-prediction
```

### 2. Create Environment and Install Dependencies
```bash
conda create -n mace-pred python=3.9
conda activate mace-pred
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the main libraries manually:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib shap seaborn scipy
```

---

## 🚀 Running the Code

### Option 1: Run Jupyter Notebook
```bash
jupyter notebook XGB.ipynb
```

Follow the notebook steps:
1. Load your dataset (CSV, Excel, etc.)
2. Preprocess data: normalization, encoding, SMOTE
3. Train XGBoost using stratified K-fold CV
4. Evaluate using AUC with CI, calibration, and SHAP
5. Visualize model results and interpret key features

### Option 2: Adapt Code for Scripted Training
Modularize the notebook for use in `.py` scripts (e.g., `train_xgb.py`, `plot_results.py`).

---

## 📊 Output Examples

After running the notebook, you will get:

- `figures/roc_curve.png` – ROC curve of the model
- `figures/calibration_curve.png` – Calibration plot
- `figures/shap_summary.png` – SHAP summary of top features

---

## 🧪 Example Code Snippets

**Train Model with Cross-Validation**
```python
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print("Mean AUC:", scores.mean())
```

**SHAP Interpretability**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## 🔒 Data Privacy Notice

This repository does **not** contain any patient-level data. You must use your own anonymized dataset with the following structure:

- Demographics (e.g., Age, Gender)
- Clinical features (e.g., EF, DM, HTN)
- Labs (e.g., FBS, Creatinine)
- Target variable (e.g., `MACE_1y`: 0 or 1)

---

## 🧑‍🔬 Citation

If you use this code in academic work, please cite:

```bibtex
@article{your_citation_key,
  title={Applied Machine Learning to Predict 1-Year Major Adverse Cardiovascular Events in the Elderly After Percutaneous Coronary Intervention},
  author={Author1, Author2, ...},
  journal={Journal Name},
  year={2025}
}
```

---

## 📧 Contact

For questions or collaboration:

**Lead Author:** your_email@example.com  
**GitHub:** [github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

*This project aims to assist the medical and data science communities in developing reproducible and interpretable cardiovascular risk models.*
