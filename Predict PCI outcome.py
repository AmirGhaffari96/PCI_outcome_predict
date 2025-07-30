# Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score




# Step 1: Load your dataset
# Replace with your actual dataset
data = pd.read_excel("Data.xlsx")


#In you need drop some columns 
data2=data.drop([], axis=1)

# Step 2: Separate features and target
X = data2.drop(['MACE'], axis=1)
y = data2['MACE']

# Step 3: defining the categorical and continious parameters 

categorical_cols = ['Gender', 'Hyperlipidemia', 'Diabetes', 'Hypertension', 'Smoking',
       'PreviousCABG', 'PreviousPCI', 'Door to Device Distance',]  
continuous_cols = ['Pre PCI EF', 'Age', 'BMI', 'waist_circ', 'Total Cholesterol',
       'Triglyceride', 'LDL', 'HDL', 'LDLtoHDL', 'FBS', 'Creatinine',
       'Hemoglobin']

#Step 4: Correlation 
#Correlation Heatmap to identify and remove Highly correlated parammeters
plt.figure(figsize=(20, 16))
sns.heatmap(pd.DataFrame(X, columns=X.columns).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Step 5: Pre-processing 
# Encode categorical columns using LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Step 5: Scale continuous columns
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# Step 6: Handle class imbalance using SMOTE
# Use cautiously and consider limiting the SMOTE ratio (preferably below 20%).
# Excessive use of SMOTE may lead to synthetic data hallucination and introduce bias into the model.

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)



# Step 7: Train-test split on resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# Step 8: Build and train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 9: Predict probabilities for class '1' on test data
predicted_probs = model.predict_proba(X_test)[:, 1]  # Probability for class '1'

# Step 10: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype(int)
y_pred_proba=model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Sensitivity (Recall), Specificity, PPV, NPV
sensitivity = tp / (tp + fn)  # Sensitivity or Recall
specificity = tn / (tn + fp)  # Specificity
ppv = tp / (tp + fp)          # Positive Predictive Value (PPV)
npv = tn / (tn + fn)          # Negative Predictive Value (NPV)

# Likelihood Ratios
lr_positive = sensitivity / (1 - specificity)  # Positive Likelihood Ratio
lr_negative = (1 - sensitivity) / specificity  # Negative Likelihood Ratio

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Balanced Accuracy: {balanced_acc}')
print(f'Sensitivity (Recall): {sensitivity}')
print(f'Specificity: {specificity}')
print(f'PPV: {ppv}')
print(f'NPV: {npv}')
print(f'Likelihood Ratio Positive: {lr_positive}')
print(f'Likelihood Ratio Negative: {lr_negative}')


# Step 11: Plot the ROC curve
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, predicted_probs)

from sklearn.metrics import roc_curve, auc
# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, predicted_probs)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Step 10: Create SHAP explainer and generate SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Instantiate the XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',  # For binary classification (modify if multi-class)
    scale_pos_weight=len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1]),  # Handling imbalance
    eval_metric='auc',
    use_label_encoder=False
)

# Train the model
xgb_clf.fit(X_train, y_train)

# Predictions
y_pred = xgb_clf.predict(X_test)
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]  # For ROC-AUC

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Detailed classification report
print(classification_report(y_test, y_pred))

import shap

# Step 12: SHAP requires a small subset of data for performance reasons
explainer = shap.Explainer(xgb_clf)
shap_values = explainer(X_test)

# Summary plot - global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")


# Step 13: SHAP Beeswarm Plot (P Swarm Plot)
# Shows distribution of SHAP values for each feature
shap.summary_plot(shap_values, X_test)

# Bar Absolute Mean SHAP Plot (Feature Importance by Contribution)
# Bar plot showing the absolute mean SHAP values (to highlight feature importance)
shap.plots.bar(shap_values, max_display=15)