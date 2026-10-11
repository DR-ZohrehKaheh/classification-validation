#!/usr/bin/env python3
# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Ruihan Sun.
# Initial student coding was developed with supervisory teaching and improvement guidance.
"""Technical implementation for the validation code portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "latitude", "longitude", "datetime", "pr", "rmax", "rmin", "sph",
        "srad", "tmmn", "tmmx", "vs", "bi", "fm100", "fm1000", "erc",
        "etr", "pet", "vpd",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = data.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="raise")
    day_of_year = result["datetime"].dt.dayofyear
    result["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["tmmn_c"] = result["tmmn"] - 273.15
    result["tmmx_c"] = result["tmmx"] - 273.15
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV containing predictor rows")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory created by analysis_pipeline.py")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV")
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(args.model_dir / "selected_model.joblib")
    calibrator = joblib.load(args.model_dir / "platt_logit_model.joblib")
    data = engineer_features(pd.read_csv(args.input, low_memory=False))
    raw_probability = model.predict_proba(data[metadata["features"]])[:, 1]
    clipped = np.clip(raw_probability, *metadata["probability_clip"])
    logit_probability = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrated_probability = calibrator.predict_proba(logit_probability)[:, 1]
    scored = data.copy()
    scored["wildfire_probability"] = calibrated_probability
    scored["wildfire_predicted"] = (
        calibrated_probability >= metadata["selected_probability_threshold"]
    ).astype(np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
