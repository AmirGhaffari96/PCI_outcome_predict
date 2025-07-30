# Applied Machine Learning to Predict 1-Year Major Adverse Cardiovascular Events (MACE) After PCI in Elderly Patients

This repository contains the source code and workflows for our study applying machine learning (ML) techniques to predict 1-year MACE outcomes in elderly patients following Percutaneous Coronary Intervention (PCI). The project involves data preprocessing, model training, evaluation, SHAP interpretability, and model calibration.

## 📂 Repository Structure

| File Name | Description |
|-----------|-------------|
| `Predict PCI outcome.py` | Main script for data preprocessing, feature selection, model training, and evaluation of multiple classifiers (XGBoost, Random Forest, Logistic Regression, Neural Network). |
| `Machine Learning Models Performance for PCI predict.ipynb` | Jupyter notebook for visualization and comparison of ML model performance using ROC curves, AUC, accuracy, precision, recall, and F1-score. |
| `ML model Calibration Plots for PCI predict.py` | Python script for calibration analysis using reliability curves and Brier scores, assessing probabilistic output quality of each model. |
| `XGBoost for PCI.ipynb` | Jupyter notebook for standalone implementation of XGBoost model with SHAP (Shapley Additive Explanations) visualization for interpretability. |

## 💻 How to Use

### 1. Clone the repository
```bash
git clone https://github.com/AmirGhaffari96/PCI_outcome_predict.git
cd PCI_outcome_predict
```

### 2. Install requirements
Make sure you have Python 3.8+ and run:
```bash
pip install -r requirements.txt
```
Dependencies include:
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`

## 🚀 Execution Guide

### A. Full Pipeline
Run the following for complete data loading, model training, and evaluation:
```bash
python Predict\ PCI\ outcome.py
```

### B. Visualize Model Performance
Open:
```
Machine Learning Models Performance for PCI predict.ipynb
```
This notebook compares the ROC, AUC, and other metrics across models.

### C. Model Calibration
Run:
```bash
python ML\ model\ Calibration\ Plots\ for\ PCI\ predict.py
```
This script evaluates the calibration of predicted probabilities using plots and Brier scores.

### D. Interpretability with SHAP (XGBoost)
Open:
```
XGBoost for PCI.ipynb
```
This notebook provides SHAP visualizations to explain how input features affect the XGBoost model's predictions.

## 📊 Dataset
The dataset used in this study contains anonymized clinical, laboratory, and procedural data from 1,358 elderly patients undergoing PCI. It includes:
- Demographic features
- Lab values
- Medical history
- PCI procedural data
- 1-year MACE outcome labels

**Note**: The dataset is not publicly available due to privacy restrictions.

## 📈 Evaluation Metrics

- AUC-ROC
- Accuracy
- Precision, Recall, F1-score
- Calibration (Brier Score, Calibration Curve)
- SHAP values for feature interpretability

## 📄 License
No need to License

## 📬 Contact
For questions or collaborations, contact:
**[Your Name]**  
Email: [ghaffari.amr@gmail.com]  
Institution: Rajaie Cardiovascular Institue / Tehran Heart Center 
