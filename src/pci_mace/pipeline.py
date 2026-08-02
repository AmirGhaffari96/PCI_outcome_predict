"""Leakage-resistant training and evaluation pipeline for the article models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DEFAULT_FEATURES = (
    "Pre PCI EF",
    "Age",
    "BMI",
    "LDLtoHDL",
    "FBS",
    "Hemoglobin",
    "Creatinine",
)

MODEL_NAMES = (
    "xgboost",
    "random_forest",
    "logistic_regression",
    "neural_network",
    "svm",
    "naive_bayes",
    "knn",
    "decision_tree",
)

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "neural_network": "Neural Network",
    "svm": "SVM",
    "naive_bayes": "Naive Bayes",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
}


@dataclass(frozen=True)
class ModelSpec:
    estimator: BaseEstimator
    grid: dict[str, list[Any]]


def model_specs(random_state: int) -> dict[str, ModelSpec]:
    """Return the eight classifiers described in the article and compact tuning grids."""
    return {
        "xgboost": ModelSpec(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 6],
            },
        ),
        "random_forest": ModelSpec(
            RandomForestClassifier(
                n_estimators=100,
                criterion="entropy",
                random_state=random_state,
                n_jobs=1,
            ),
            {
                "model__n_estimators": [100, 300],
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 10],
            },
        ),
        "logistic_regression": ModelSpec(
            LogisticRegression(max_iter=1000, solver="lbfgs"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "neural_network": ModelSpec(
            MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=random_state,
            ),
            {
                "model__hidden_layer_sizes": [(50,), (100,)],
                "model__alpha": [0.0001, 0.001],
            },
        ),
        "svm": ModelSpec(
            CalibratedClassifierCV(
                estimator=SVC(kernel="linear"),
                method="sigmoid",
                cv=5,
                ensemble=False,
            ),
            {"model__estimator__C": [0.1, 1.0, 10.0]},
        ),
        "naive_bayes": ModelSpec(
            GaussianNB(),
            {"model__var_smoothing": [1e-10, 1e-9, 1e-8]},
        ),
        "knn": ModelSpec(
            KNeighborsClassifier(n_neighbors=5),
            {"model__n_neighbors": [3, 5, 7, 11]},
        ),
        "decision_tree": ModelSpec(
            DecisionTreeClassifier(random_state=random_state),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 3, 6, 10],
            },
        ),
    }


def load_dataset(path: str | Path, target: str = "MACE") -> pd.DataFrame:
    """Load CSV/XLSX data and validate the article's seven selected predictors."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Dataset must be CSV, XLSX, or XLS.")

    required = [*DEFAULT_FEATURES, target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    clean = frame.loc[:, required].copy()
    for column in DEFAULT_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean[target] = pd.to_numeric(clean[target], errors="raise").astype(int)
    values = set(clean[target].dropna().unique())
    if not values <= {0, 1} or len(values) != 2:
        raise ValueError(f"{target} must contain both binary labels 0 and 1.")
    return clean


def make_pipeline(estimator: BaseEstimator, smote_ratio: float, random_state: int) -> Pipeline:
    """Put every learned preprocessing step inside the training pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(sampling_strategy=smote_ratio, random_state=random_state),
            ),
            ("model", estimator),
        ]
    )


def _bootstrap_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    random_state: int,
) -> tuple[float, float]:
    if iterations <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        scores.append(roc_auc_score(sampled_y, probabilities[indices]))
    if not scores:
        return (float("nan"), float("nan"))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return float(lower), float(upper)


def _evaluate(
    name: str,
    estimator: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    bootstrap_iterations: int,
    random_state: int,
) -> tuple[
    dict[str, Any],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    probabilities = estimator.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, _tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    sensitivity = recall_score(y_test, predictions, zero_division=0)
    lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float("inf")
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float("inf")
    auc = roc_auc_score(y_test, probabilities)
    low, high = _bootstrap_auc(
        y_test.to_numpy(), probabilities, bootstrap_iterations, random_state
    )
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    row = {
        "model": DISPLAY_NAMES[name],
        "auc": auc,
        "auc_ci_low": low,
        "auc_ci_high": high,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision_score(y_test, predictions, zero_division=0),
        "npv": npv,
        "f1": f1_score(y_test, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_test, probabilities),
        "lr_positive": lr_positive,
        "lr_negative": lr_negative,
        "test_n": len(y_test),
        "test_events": int(y_test.sum()),
    }
    calibration_true, calibration_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )
    return row, (fpr, tpr), (calibration_predicted, calibration_true)


def _plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], metrics: pd.DataFrame, path: Path) -> None:
    auc_by_name = dict(zip(metrics["model"], metrics["auc"]))
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (fpr, tpr) in curves.items():
        display = DISPLAY_NAMES[name]
        ax.plot(fpr, tpr, linewidth=2, label=f"{display} (AUC={auc_by_name[display]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves on untouched test data")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    brier_by_name = dict(zip(metrics["model"], metrics["brier_score"]))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for name, (mean_predicted, fraction_positive) in curves.items():
        display = DISPLAY_NAMES[name]
        ax.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            linewidth=1.5,
            label=f"{display} (Brier={brier_by_name[display]:.3f})",
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event fraction",
        title="Calibration on untouched test data",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _export_shap(estimator: Pipeline, x_test: pd.DataFrame, output_dir: Path) -> None:
    """Export article-style global SHAP plots for a fitted XGBoost pipeline."""
    import shap

    transformed = estimator.named_steps["imputer"].transform(x_test)
    transformed = estimator.named_steps["scaler"].transform(transformed)
    transformed_frame = pd.DataFrame(transformed, columns=DEFAULT_FEATURES, index=x_test.index)
    model = estimator.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer(transformed_frame)

    shap.plots.bar(values, max_display=len(DEFAULT_FEATURES), show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap-feature-importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(values, max_display=len(DEFAULT_FEATURES), show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap-summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_benchmark(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    target: str = "MACE",
    model_names: Iterable[str] = MODEL_NAMES,
    test_size: float = 0.20,
    smote_ratio: float = 0.50,
    cv: int = 5,
    bootstrap_iterations: int = 1000,
    random_state: int = 42,
    tune: bool = False,
    export_shap: bool = False,
) -> pd.DataFrame:
    """Train, evaluate, and save model-comparison artifacts."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < smote_ratio <= 1:
        raise ValueError("smote_ratio must be between 0 and 1.")

    selected = list(dict.fromkeys(model_names))
    unknown = sorted(set(selected) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model names: {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one model must be selected.")

    frame = load_dataset(data_path, target)
    x = frame.loc[:, DEFAULT_FEATURES]
    y = frame[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs(random_state)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    calibration_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, Pipeline] = {}
    best_params: dict[str, dict[str, Any]] = {}

    for name in selected:
        spec = specs[name]
        pipeline = make_pipeline(spec.estimator, smote_ratio, random_state)
        if tune:
            search = GridSearchCV(
                pipeline,
                spec.grid,
                scoring="roc_auc",
                cv=splitter,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train)
            estimator = search.best_estimator_
            best_params[name] = search.best_params_
        else:
            estimator = pipeline.fit(x_train, y_train)
            best_params[name] = {}
        row, curve, calibration = _evaluate(
            name,
            estimator,
            x_test,
            y_test,
            bootstrap_iterations,
            random_state,
        )
        rows.append(row)
        curves[name] = curve
        calibration_curves[name] = calibration
        fitted[name] = estimator

    metrics = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _plot_roc(curves, metrics, output_dir / "roc-curves.png")
    _plot_calibration(
        calibration_curves,
        metrics,
        output_dir / "calibration-curves.png",
    )
    metadata = {
        "data": str(Path(data_path).resolve()),
        "features": list(DEFAULT_FEATURES),
        "target": target,
        "train_n": len(x_train),
        "test_n": len(x_test),
        "test_size": test_size,
        "smote_ratio": smote_ratio,
        "cross_validation_folds": cv,
        "bootstrap_iterations": bootstrap_iterations,
        "random_state": random_state,
        "tuned": tune,
        "best_parameters": best_params,
    }
    (output_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if export_shap:
        if "xgboost" not in fitted:
            raise ValueError("--shap requires xgboost in the selected models.")
        _export_shap(fitted["xgboost"], x_test, output_dir)
    return metrics


def generate_demo_data(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Create non-clinical synthetic data with the article's schema for smoke tests."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100.")
    rng = np.random.default_rng(random_state)
    age = np.clip(rng.normal(74.1, 6.7, n_samples), 65, 98)
    ef = np.clip(rng.normal(40.4, 8.3, n_samples), 10, 70)
    bmi = np.clip(rng.normal(27.4, 4.5, n_samples), 15, 48)
    ratio = np.clip(rng.normal(2.56, 0.98, n_samples), 0.3, 8)
    fbs = np.clip(rng.lognormal(np.log(130), 0.38, n_samples), 55, 500)
    hemoglobin = np.clip(rng.normal(14.4, 1.9, n_samples), 7, 20)
    creatinine = np.clip(rng.lognormal(np.log(0.95), 0.35, n_samples), 0.3, 5)
    log_odds = (
        -2.7
        + 0.045 * (age - 74)
        - 0.060 * (ef - 40)
        - 0.035 * (bmi - 27)
        - 0.12 * (ratio - 2.5)
        + 0.006 * (fbs - 140)
        - 0.10 * (hemoglobin - 14)
        + 0.85 * (creatinine - 1)
    )
    risk = 1 / (1 + np.exp(-log_odds))
    mace = rng.binomial(1, risk)
    return pd.DataFrame(
        {
            "Pre PCI EF": ef,
            "Age": age,
            "BMI": bmi,
            "LDLtoHDL": ratio,
            "FBS": fbs,
            "Hemoglobin": hemoglobin,
            "Creatinine": creatinine,
            "MACE": mace,
        }
    )
