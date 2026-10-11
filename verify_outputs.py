#!/usr/bin/env python3
# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Ruihan Sun.
# Initial student coding was developed with supervisory teaching and improvement guidance.
"""Fail fast if final analysis artifacts violate core integrity conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tables = args.output / "tables"

    quality = pd.read_csv(tables / "data_quality.csv").iloc[0]
    splits = pd.read_csv(tables / "split_summary.csv")
    primary = pd.read_csv(tables / "primary_split.csv")
    performance = pd.read_csv(tables / "calibration_threshold_performance.csv")
    summary = json.loads((args.output / "analysis_summary.json").read_text(encoding="utf-8"))

    assert int(quality["original_rows"]) == (
        int(quality["clean_rows"])
        + int(quality["exact_duplicate_rows_removed"])
        + int(quality["ambiguous_rows_removed"])
    )
    spatial_rows = splits[
        splits["validation_scheme"].str.contains("Spatial-block|spatiotemporal", regex=True)
    ]
    assert (spatial_rows["overlapping_blocks"] == 0).all()
    assert set(primary["partition"]) == {
        "model training", "calibration", "threshold/model selection", "final test"
    }
    final = performance[performance["evaluation"].eq("calibrated_validation_selected")].iloc[0]
    assert int(final["tn"] + final["fp"] + final["fn"] + final["tp"]) == int(final["n"])
    assert abs(float(final["auroc"]) - summary["primary_test_metrics"]["auroc"]) < 1e-12
    print("Output verification passed: cleaning arithmetic, split isolation, and metrics are consistent.")


if __name__ == "__main__":
    main()
