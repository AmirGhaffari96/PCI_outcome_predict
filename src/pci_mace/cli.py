"""Command-line interface for the PCI MACE benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import MODEL_NAMES, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pci-mace",
        description="Benchmark models for one-year MACE prediction after PCI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train and evaluate one or more models.")
    train.add_argument("--data", type=Path, required=True, help="CSV or Excel dataset.")
    train.add_argument("--output", type=Path, default=Path("outputs"))
    train.add_argument(
        "--models",
        nargs="+",
        choices=["all", *MODEL_NAMES],
        default=["all"],
        help="Model identifiers; defaults to all eight models.",
    )
    train.add_argument("--target", default="MACE")
    train.add_argument("--test-size", type=float, default=0.20)
    train.add_argument("--smote-ratio", type=float, default=0.50)
    train.add_argument("--cv", type=int, default=5)
    train.add_argument("--bootstrap", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--tune", action="store_true", help="Run training-fold grid search.")
    train.add_argument("--shap", action="store_true", help="Export XGBoost SHAP plots.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        models = list(MODEL_NAMES) if args.models == ["all"] else args.models
        metrics = run_benchmark(
            data_path=args.data,
            output_dir=args.output,
            target=args.target,
            model_names=models,
            test_size=args.test_size,
            smote_ratio=args.smote_ratio,
            cv=args.cv,
            bootstrap_iterations=args.bootstrap,
            random_state=args.seed,
            tune=args.tune,
            export_shap=args.shap,
        )
        print(metrics.to_string(index=False))
        print(f"\nArtifacts written to {args.output.resolve()}")
