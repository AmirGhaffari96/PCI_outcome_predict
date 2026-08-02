"""Reproducible machine-learning workflow for one-year MACE after PCI."""

from .pipeline import DEFAULT_FEATURES, generate_demo_data, run_benchmark

__all__ = ["DEFAULT_FEATURES", "generate_demo_data", "run_benchmark"]
__version__ = "1.0.0"
