Applied Machine Learning for Predicting 1-Year MACE After PCI

This repository contains Python code and notebooks used to develop and evaluate machine learning models for predicting 1-year Major Adverse Cardiovascular Events (MACE) in elderly patients (≥65 years) after Percutaneous Coronary Intervention (PCI).

The repository includes:

    Model training pipeline (preprocessing, correlation matirx and normalization).

    Implementations of XGBoost as explainable moldel.

    SHAP explainability for feature interpretation.

📦 Code Files

    XGB.ipynb
        End-to-end workflow for training and evaluating XGBoost models, with:

        Train-test split 


        AUC and ROC plotting

        SHAP-based feature importance and summary plots.

        Pairwise AUC comparison using permutation tests.

    utils/ (optional)
    For helper scripts such as plotting ROC curves, computing calibration metrics, or formatting datasets. (You can modularize the code from XGB.ipynb here.)

    figures/
    Stores generated plots (ROC, calibration curves, SHAP).

⚙️ Installation & Setup
1. Clone the Repo

git clone https://github.com/YOUR_USERNAME/mace-prediction.git
cd mace-prediction

2. Install Requirements

pip install -r requirements.txt

Key packages:

    xgboost

    scikit-learn

    shap

    matplotlib

    numpy, pandas

    scipy

🧩 Running the Code
Option A – Run Entire Workflow (XGBoost Example)

    Open the notebook:

    jupyter notebook XGB.ipynb

    Execute cells sequentially:

        Data loading: Provide your dataset (.csv or .xlsx).

        Preprocessing: Normalization, one-hot encoding, and SMOTE for class imbalance.

        Model Training: XGBoost with 5-fold cross-validation.

        Evaluation: AUC, ROC curve, calibration plots.

        Interpretation: Run SHAP to visualize feature importance.

Option B – Run Model Training from Script

If modularized as scripts (recommended):

python train_xgb.py --data data/your_dataset.csv --cv 5 --save-model models/xgb_model.pkl

🧪 Example Code Snippets

Training an XGBoost Model with Cross-Validation

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.3f} ± {scores.std():.3f}")

Plotting ROC Curve

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.legend()
plt.show()

Feature Importance (SHAP)

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

📊 Output Examples

    figures/roc_curve.png – ROC curve with AUC values.

    figures/calibration_curve.png – Model calibration performance.

    figures/shap_summary.png – Feature importance visualization.

🔒 Data Notice

The dataset is not included due to privacy restrictions. To run the code:

    Use a synthetic dataset (you can generate with sklearn.datasets.make_classification) for testing.

    Place your preprocessed dataset in the data/ directory.

🔮 Next Steps

    Add a train_all_models.ipynb notebook to compare all models (RF, LR, NN, XGBoost) side by side.

    Deploy a Streamlit web app for interactive risk prediction (codes can be adapted directly from the notebooks).
