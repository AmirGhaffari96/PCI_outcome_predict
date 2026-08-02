#!/usr/bin/env python3
"""Generate synthetic, non-clinical data for a pipeline smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from pci_mace import generate_demo_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/demo.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Wrote synthetic demo data to {args.output}")


if __name__ == "__main__":
    main()
