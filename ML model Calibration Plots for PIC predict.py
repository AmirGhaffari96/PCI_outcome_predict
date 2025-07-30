import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.calibration import CalibrationDisplay
from sklearn.utils import resample

# Load data
file_path = 'Data.xlsx'
data = pd.read_excel(file_path)
data2 = data.drop(['Hyperlipidemia','Hypertension', 'Gender','Diabetes' , 'LDL', 'HDL','waist_circ', 'Triglyceride' , 'PreviousCABG','Smoking','HDL/Chl','Total Cholesterol',
             'PreviousPCI', 'Door to Device Distance',  'LDL/Chl' ], axis=1)

X = data.drop(['MACE'], axis=1)
y = data['MACE']

# Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)



columns_to_scale = ['Pre PCI EF', 'Age', 'BMI', 'FBS','LDLtoHDL','Creatinine','Hemoglobin']
scaler = StandardScaler()
X_scaled = X_resampled.copy()
X_scaled[columns_to_scale] = scaler.fit_transform(X_scaled[columns_to_scale])
y_true = y_resampled.reset_index(drop=True)
y_true_array = np.array(y_true)


# Define models
models = {
    #'XGB': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Neural Net': MLPClassifier(max_iter=1000),
    'SVM': SVC(kernel='linear', probability=True),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier()
}


##Calibration Plots for Various ML models 

v = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
auc_dict = {}

plt.figure(figsize=(14, 10))

for name, model in models.items():
    y_probs_cv = cross_val_predict(model, X_scaled, y_resampled, cv=cv, method='predict_proba')[:, 1]
    y_preds_cv = (y_probs_cv >= 0.5).astype(int)

    auc = roc_auc_score(y_resampled, y_probs_cv)

    # Confidence interval with bootstrapping
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(1000):
        indices = rng.randint(0, len(y_resampled), len(y_resampled))
        if len(np.unique(y_resampled.iloc[indices])) < 2:
            continue
        score = roc_auc_score(y_resampled.iloc[indices], y_probs_cv[indices])
        bootstrapped_scores.append(score)
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)

    acc = accuracy_score(y_resampled, y_preds_cv)
    bal_acc = balanced_accuracy_score(y_resampled, y_preds_cv)
    recall = recall_score(y_resampled, y_preds_cv)
    precision = precision_score(y_resampled, y_preds_cv)
    f1 = f1_score(y_resampled, y_preds_cv)

    tn, fp, fn, tp = confusion_matrix(y_resampled, y_preds_cv).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    brier = brier_score_loss(y_resampled, y_probs_cv)

    fpr, tpr, _ = roc_curve(y_resampled, y_probs_cv)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f}, 95% CI [{ci_lower:.2f}-{ci_upper:.2f}])")

    results.append([name, auc, ci_lower, ci_upper, acc, bal_acc, recall, precision, f1, npv, specificity, brier])
    auc_dict[name] = y_probs_cv

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_resampled, y_probs_cv, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"Calibration Curve - {name}")
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    plt.grid()
    plt.show()







# --- Final Combined ROC Curve for All Models ---
plt.figure(figsize=(12, 8))

for name, model in models.items():
    try:
        # Get predicted probabilities via cross_val_predict
        y_probs_cv = cross_val_predict(model, X_scaled, y_resampled, cv=cv, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y_resampled, y_probs_cv)
        auc_score = roc_auc_score(y_resampled, y_probs_cv)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
    except Exception as e:
        print(f"Skipping {name} in ROC plot due to error: {e}")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title("Combined ROC Curve for All Models (Cross-Validated)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()


#Premutation test for comparing teh AUCs
from sklearn.metrics import roc_auc_score
from itertools import combinations

#  Ensure all models (including XGBoost) use the same y_true_array
# Replace y_true_array with XGBoost's targets
y_true_array = np.array(all_targets)  # This ensures consistency!

# Permutation Test
n_permutations = 1000
model_names = list(auc_dict.keys())  # Should now include 'XGBoost'
pval_matrix_perm = pd.DataFrame(np.ones((len(model_names), len(model_names))),
                                index=model_names, columns=model_names)

print("\n=== Pairwise AUC Comparison (Permutation Test p-values, including XGBoost) ===")

for model1, model2 in combinations(model_names, 2):
    probs1 = np.array(auc_dict[model1])
    probs2 = np.array(auc_dict[model2])

    # Ensure all prob arrays match y_true_array length
    assert len(probs1) == len(y_true_array), f"{model1} length mismatch"
    assert len(probs2) == len(y_true_array), f"{model2} length mismatch"

    true_diff = roc_auc_score(y_true_array, probs1) - roc_auc_score(y_true_array, probs2)

    combined = np.vstack((probs1, probs2)).T
    perm_diffs = []

    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        flip = rng.randint(0, 2, size=combined.shape[0])
        perm1 = np.where(flip, combined[:, 0], combined[:, 1])
        perm2 = np.where(flip, combined[:, 1], combined[:, 0])
        auc1 = roc_auc_score(y_true_array, perm1)
        auc2 = roc_auc_score(y_true_array, perm2)
        perm_diffs.append(auc1 - auc2)

    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(true_diff))  # two-sided test

    # Fill symmetric matrix
    pval_matrix_perm.loc[model1, model2] = p_value
    pval_matrix_perm.loc[model2, model1] = p_value

# Show final matrix
print(pval_matrix_perm.round(4))
