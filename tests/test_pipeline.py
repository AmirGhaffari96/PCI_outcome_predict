from pathlib import Path

import pandas as pd

from pci_mace import DEFAULT_FEATURES, generate_demo_data, run_benchmark


def test_demo_data_has_expected_schema() -> None:
    frame = generate_demo_data(200, random_state=7)
    assert list(frame.columns) == [*DEFAULT_FEATURES, "MACE"]
    assert set(frame["MACE"].unique()) == {0, 1}


def test_decision_tree_smoke_run(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.csv"
    output_path = tmp_path / "outputs"
    generate_demo_data(300, random_state=8).to_csv(data_path, index=False)
    metrics = run_benchmark(
        data_path,
        output_path,
        model_names=["decision_tree"],
        bootstrap_iterations=20,
        random_state=8,
    )
    assert metrics.loc[0, "model"] == "Decision Tree"
    assert 0 <= metrics.loc[0, "auc"] <= 1
    assert (output_path / "metrics.csv").exists()
    assert (output_path / "roc-curves.png").exists()
    assert (output_path / "run-metadata.json").exists()
    persisted = pd.read_csv(output_path / "metrics.csv")
    assert len(persisted) == 1
