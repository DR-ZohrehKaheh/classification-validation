#!/usr/bin/env python3
# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Ruihan Sun.
# Initial student coding was developed with supervisory teaching and improvement guidance.
"""Technical implementation for the validation code portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
BLOCK_SIZE_DEGREES = 0.5
BOOTSTRAP_REPLICATES = 200

REQUIRED_COLUMNS = [
    "latitude", "longitude", "datetime", "Wildfire", "pr", "rmax", "rmin",
    "sph", "srad", "tmmn", "tmmx", "vs", "bi", "fm100", "fm1000",
    "erc", "etr", "pet", "vpd",
]

WEATHER_FEATURES = [
    "pr", "rmax", "rmin", "sph", "srad", "tmmn_c", "tmmx_c", "vs",
    "etr", "pet", "vpd",
]
FIRE_INDEX_FEATURES = WEATHER_FEATURES + ["bi", "fm100", "fm1000", "erc"]
FULL_FEATURES = FIRE_INDEX_FEATURES + [
    "latitude", "longitude", "doy_sin", "doy_cos",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, bins - 1)
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        mask = ids == index
        if mask.any():
            ece += mask.mean() * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def calibration_points(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = ids == index
        if mask.any():
            rows.append({
                "bin": index + 1,
                "n": int(mask.sum()),
                "mean_predicted": float(prob[mask].mean()),
                "observed_prevalence": float(y_true[mask].mean()),
            })
    return pd.DataFrame(rows)


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_wildfire": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_wildfire": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1_wildfire": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": expected_calibration_error(y_true, prob, bins=10),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def optimal_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1], denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


class PlattCalibrator:
    """One-dimensional Platt scaling fit on held-out observations."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    @staticmethod
    def _logit(prob: np.ndarray) -> np.ndarray:
        clipped = np.clip(prob, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._logit(prob), y_true)
        return self

    def predict(self, prob: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(prob))[:, 1]


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(path, parse_dates=["datetime"], low_memory=False)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if set(data["Wildfire"].dropna().unique()) - {"No", "Yes"}:
        raise ValueError("Wildfire must contain only 'No' and 'Yes'.")

    original_rows = len(data)
    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    key = ["latitude", "longitude", "datetime"]
    label_count = data.groupby(key, sort=False)["Wildfire"].transform("nunique")
    ambiguous_rows = int((label_count > 1).sum())
    ambiguous_groups = int(
        data.loc[label_count > 1, key].drop_duplicates().shape[0]
    )
    data = data.loc[label_count == 1].copy()

    data["y"] = (data["Wildfire"] == "Yes").astype(np.int8)
    data["year"] = data["datetime"].dt.year.astype(np.int16)
    data["month"] = data["datetime"].dt.month.astype(np.int8)
    data["day_of_year"] = data["datetime"].dt.dayofyear.astype(np.int16)
    data["doy_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)
    data["tmmn_c"] = data["tmmn"] - 273.15
    data["tmmx_c"] = data["tmmx"] - 273.15
    data["spatial_block"] = (
        np.floor(data["latitude"] / BLOCK_SIZE_DEGREES).astype(int).astype(str)
        + "_"
        + np.floor(data["longitude"] / BLOCK_SIZE_DEGREES).astype(int).astype(str)
    )

    invalid_numeric = int((~np.isfinite(data[FULL_FEATURES].to_numpy())).sum())
    if invalid_numeric:
        raise ValueError(f"Found {invalid_numeric} non-finite feature values after cleaning.")

    location_summary = (
        data.groupby(["latitude", "longitude"], sort=False)
        .agg(n=("y", "size"), positives=("y", "sum"), dates=("datetime", "nunique"))
    )
    canonical_mask = (
        (location_summary["n"] == 75)
        & (location_summary["positives"] == 15)
        & (location_summary["dates"] == 75)
    )
    canonical_locations = location_summary.index[canonical_mask]
    canonical_windows = int(canonical_mask.sum())
    deterministic_windows = 0
    consecutive_date_windows = 0
    if canonical_windows:
        canonical_rows = (
            data.set_index(["latitude", "longitude"])
            .loc[canonical_locations]
            .reset_index()
            .sort_values(["latitude", "longitude", "datetime"])
        )
        expected_labels = np.r_[np.zeros(60, dtype=np.int8), np.ones(15, dtype=np.int8)]
        for _, group in canonical_rows.groupby(["latitude", "longitude"], sort=False):
            labels = group["y"].to_numpy(dtype=np.int8)
            deterministic_windows += int(np.array_equal(labels, expected_labels))
            dates = group["datetime"].to_numpy()
            day_steps = np.diff(dates).astype("timedelta64[D]").astype(int)
            consecutive_date_windows += int(np.all(day_steps == 1))
    quality = {
        "source_file": path.name,
        "source_sha256": sha256(path),
        "original_rows": original_rows,
        "exact_duplicate_rows_removed": exact_duplicates,
        "ambiguous_location_date_groups_removed": ambiguous_groups,
        "ambiguous_rows_removed": ambiguous_rows,
        "clean_rows": int(len(data)),
        "clean_positive_rows": int(data["y"].sum()),
        "clean_positive_prevalence": float(data["y"].mean()),
        "unique_dates": int(data["datetime"].nunique()),
        "unique_exact_locations": int(data[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "spatial_blocks_0_5_degree": int(data["spatial_block"].nunique()),
        "locations_with_75_rows_and_15_positive_days": canonical_windows,
        "locations_with_ordered_60_no_then_15_yes_pattern": deterministic_windows,
        "pattern_locations_with_75_consecutive_dates": consecutive_date_windows,
        "rows_in_ordered_60_no_then_15_yes_pattern": int(75 * deterministic_windows),
        "share_rows_in_ordered_60_no_then_15_yes_pattern": float(
            75 * deterministic_windows / len(data)
        ),
    }
    return data.reset_index(drop=True), quality


def assign_spatial_folds(data: pd.DataFrame) -> pd.Series:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = np.full(len(data), -1, dtype=np.int8)
    placeholder = np.zeros(len(data), dtype=np.int8)
    for fold, (_, held) in enumerate(splitter.split(placeholder, data["y"], data["spatial_block"])):
        folds[held] = fold
    if (folds < 0).any():
        raise RuntimeError("Spatial fold assignment failed.")
    return pd.Series(folds, index=data.index, name="spatial_fold")


def build_models() -> dict[str, object]:
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])
    forest = RandomForestClassifier(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        max_samples=200_000,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    histogram_boosting = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return {
        "Prevalence baseline": DummyClassifier(strategy="prior", random_state=RANDOM_SEED),
        "Balanced logistic regression": logistic,
        "Random forest": forest,
        "Histogram gradient boosting": histogram_boosting,
    }


def probability(model: object, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def split_definitions(data: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    all_idx = np.arange(len(data))
    random_train, random_test = train_test_split(
        all_idx, test_size=0.20, stratify=data["y"], random_state=RANDOM_SEED,
    )
    return {
        "Random row holdout": (random_train, random_test),
        "Temporal holdout (2022)": (
            data.index[data["year"] <= 2021].to_numpy(),
            data.index[data["year"] == 2022].to_numpy(),
        ),
        "Spatial-block holdout": (
            data.index[data["spatial_fold"] != 0].to_numpy(),
            data.index[data["spatial_fold"] == 0].to_numpy(),
        ),
        "Combined spatiotemporal holdout": (
            data.index[(data["year"] <= 2021) & data["spatial_fold"].isin([2, 3, 4])].to_numpy(),
            data.index[(data["year"] == 2022) & (data["spatial_fold"] == 0)].to_numpy(),
        ),
    }


def fit_and_score(model: object, data: pd.DataFrame, train_idx: np.ndarray,
                  test_idx: np.ndarray, features: list[str], threshold: float = 0.5):
    fitted = clone(model)
    fitted.fit(data.loc[train_idx, features], data.loc[train_idx, "y"])
    prob = probability(fitted, data.loc[test_idx, features])
    return fitted, prob, metrics(data.loc[test_idx, "y"].to_numpy(), prob, threshold)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def cluster_bootstrap_ci(y: np.ndarray, prob: np.ndarray, blocks: np.ndarray,
                         threshold: float, replicates: int = BOOTSTRAP_REPLICATES) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    unique = np.unique(blocks)
    positions = {block: np.flatnonzero(blocks == block) for block in unique}
    names = ["auroc", "average_precision", "precision_wildfire", "recall_wildfire", "f1_wildfire", "brier"]
    samples = {name: [] for name in names}
    for _ in range(replicates):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([positions[block] for block in picked])
        if np.unique(y[idx]).size < 2:
            continue
        result = metrics(y[idx], prob[idx], threshold)
        for name in names:
            samples[name].append(result[name])
    estimate = metrics(y, prob, threshold)
    rows = []
    for name in names:
        values = np.asarray(samples[name])
        rows.append({
            "metric": name,
            "estimate": estimate[name],
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "bootstrap_replicates": int(len(values)),
        })
    return pd.DataFrame(rows)


def plot_descriptive(annual: pd.DataFrame, monthly: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(annual["year"], annual["prevalence"], marker="o", color="#1f4e79", linewidth=2)
    axes[0].set_xticks(annual["year"])
    axes[0].set_ylabel("Wildfire-labelled prevalence")
    axes[0].set_xlabel("Year")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.25)
    axes[1].plot(monthly["month"], monthly["prevalence"], marker="o", color="#a23b3b", linewidth=2)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xlabel("Calendar month")
    axes[1].set_ylabel("Wildfire-labelled prevalence")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_validation(performance: pd.DataFrame, path: Path) -> None:
    display = performance.set_index("validation_scheme")[["auroc", "average_precision"]]
    ax = display.plot(kind="bar", figsize=(10, 5), color=["#376a8f", "#c55a3d"])
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(["AUROC", "Average precision"], frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    prevalence = float(y.mean())
    for name, (_, prob) in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        precision, recall, _ = precision_recall_curve(y, prob)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, prob):.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(y, prob):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].axhline(prevalence, linestyle="--", color="gray", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, path: Path) -> None:
    raw_points = calibration_points(y, raw)
    cal_points = calibration_points(y, calibrated)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.plot(raw_points["mean_predicted"], raw_points["observed_prevalence"], marker="o", label="Uncalibrated")
    ax.plot(cal_points["mean_predicted"], cal_points["observed_prevalence"], marker="o", label="Platt calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed prevalence", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_threshold(y: np.ndarray, prob: np.ndarray, chosen: float, path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.plot(thresholds, f1, label="F1")
    ax.axvline(chosen, color="black", linestyle="--", label=f"Selected threshold = {chosen:.3f}")
    ax.set(xlabel="Probability threshold", ylabel="Validation score", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    show = importance.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(show["feature"], show["importance_mean"], xerr=show["importance_sd"], color="#3b75a1", alpha=0.9)
    ax.set_xlabel("Decrease in average precision after permutation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(data_path: Path, output_dir: Path, random_seed: int = RANDOM_SEED) -> None:
    global RANDOM_SEED
    RANDOM_SEED = int(random_seed)
    start = time.time()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    for directory in (table_dir, figure_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...", flush=True)
    data, quality = load_and_clean(data_path)
    data["spatial_fold"] = assign_spatial_folds(data)

    annual = data.groupby("year").agg(rows=("y", "size"), positives=("y", "sum")).reset_index()
    annual["prevalence"] = annual["positives"] / annual["rows"]
    monthly = data.groupby("month").agg(rows=("y", "size"), positives=("y", "sum")).reset_index()
    monthly["prevalence"] = monthly["positives"] / monthly["rows"]
    save_table(annual, table_dir / "annual_prevalence.csv")
    save_table(monthly, table_dir / "monthly_prevalence.csv")
    save_table(pd.DataFrame([quality]), table_dir / "data_quality.csv")
    plot_descriptive(annual, monthly, figure_dir / "descriptive_prevalence.png")

    models = build_models()
    validation_rows = []
    print("Evaluating validation designs with a fixed classifier...", flush=True)
    fixed_model = models["Histogram gradient boosting"]
    definitions = split_definitions(data)
    split_rows = []
    for name, (train_idx, test_idx) in definitions.items():
        split_rows.append({
            "validation_scheme": name,
            "train_n": len(train_idx),
            "train_prevalence": float(data.loc[train_idx, "y"].mean()),
            "test_n": len(test_idx),
            "test_prevalence": float(data.loc[test_idx, "y"].mean()),
            "train_blocks": int(data.loc[train_idx, "spatial_block"].nunique()),
            "test_blocks": int(data.loc[test_idx, "spatial_block"].nunique()),
            "overlapping_blocks": int(len(set(data.loc[train_idx, "spatial_block"]) & set(data.loc[test_idx, "spatial_block"]))),
        })
        _, _, result = fit_and_score(fixed_model, data, train_idx, test_idx, FULL_FEATURES)
        result.update({"validation_scheme": name, "model": "Histogram gradient boosting"})
        validation_rows.append(result)
        print(f"  {name}: AUROC={result['auroc']:.3f}, AP={result['average_precision']:.3f}", flush=True)
    validation_performance = pd.DataFrame(validation_rows)
    save_table(pd.DataFrame(split_rows), table_dir / "split_summary.csv")
    save_table(validation_performance, table_dir / "validation_performance.csv")
    plot_validation(validation_performance, figure_dir / "validation_comparison.png")

    primary_train = data.index[(data["year"] <= 2021) & data["spatial_fold"].isin([2, 3, 4])].to_numpy()
    validation_all = data.index[(data["year"] == 2021) & (data["spatial_fold"] == 1)].to_numpy()
    test_idx = data.index[(data["year"] == 2022) & (data["spatial_fold"] == 0)].to_numpy()

    validation_blocks = np.array(sorted(data.loc[validation_all, "spatial_block"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(validation_blocks)
    midpoint = max(1, len(validation_blocks) // 2)
    calibration_blocks = set(validation_blocks[:midpoint])
    threshold_blocks = set(validation_blocks[midpoint:])
    calibration_idx = validation_all[data.loc[validation_all, "spatial_block"].isin(calibration_blocks).to_numpy()]
    threshold_idx = validation_all[data.loc[validation_all, "spatial_block"].isin(threshold_blocks).to_numpy()]
    if not len(threshold_idx):
        raise RuntimeError("No threshold-validation observations were assigned.")

    primary_split = pd.DataFrame([
        {"partition": "model training", "n": len(primary_train), "prevalence": data.loc[primary_train, "y"].mean(), "year_rule": "2020-2021", "spatial_fold_rule": "2,3,4"},
        {"partition": "calibration", "n": len(calibration_idx), "prevalence": data.loc[calibration_idx, "y"].mean(), "year_rule": "2021", "spatial_fold_rule": "half of fold 1 blocks"},
        {"partition": "threshold/model selection", "n": len(threshold_idx), "prevalence": data.loc[threshold_idx, "y"].mean(), "year_rule": "2021", "spatial_fold_rule": "remaining fold 1 blocks"},
        {"partition": "final test", "n": len(test_idx), "prevalence": data.loc[test_idx, "y"].mean(), "year_rule": "2022", "spatial_fold_rule": "fold 0"},
    ])
    save_table(primary_split, table_dir / "primary_split.csv")

    print("Comparing models on the strict primary design...", flush=True)
    model_rows = []
    fitted_models = {}
    test_curves = {}
    validation_ap = {}
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(data.loc[primary_train, FULL_FEATURES], data.loc[primary_train, "y"])
        val_prob = probability(fitted, data.loc[threshold_idx, FULL_FEATURES])
        test_prob = probability(fitted, data.loc[test_idx, FULL_FEATURES])
        validation_ap[name] = average_precision_score(data.loc[threshold_idx, "y"], val_prob)
        result = metrics(data.loc[test_idx, "y"].to_numpy(), test_prob, 0.5)
        result.update({"model": name, "validation_average_precision": validation_ap[name]})
        model_rows.append(result)
        fitted_models[name] = fitted
        test_curves[name] = (data.loc[test_idx, "y"].to_numpy(), test_prob)
        print(f"  {name}: validation AP={validation_ap[name]:.3f}, test AP={result['average_precision']:.3f}", flush=True)
    model_performance = pd.DataFrame(model_rows).sort_values("validation_average_precision", ascending=False)
    save_table(model_performance, table_dir / "strict_model_performance.csv")
    plot_roc_pr(test_curves, data.loc[test_idx, "y"].to_numpy(), figure_dir / "strict_roc_pr.png")

    selected_name = max(validation_ap, key=validation_ap.get)
    selected_model = fitted_models[selected_name]
    raw_calibration_prob = probability(selected_model, data.loc[calibration_idx, FULL_FEATURES])
    calibrator = PlattCalibrator().fit(raw_calibration_prob, data.loc[calibration_idx, "y"].to_numpy())
    raw_threshold_prob = probability(selected_model, data.loc[threshold_idx, FULL_FEATURES])
    calibrated_threshold_prob = calibrator.predict(raw_threshold_prob)
    chosen_threshold, validation_f1 = optimal_f1_threshold(
        data.loc[threshold_idx, "y"].to_numpy(), calibrated_threshold_prob
    )
    raw_test_prob = probability(selected_model, data.loc[test_idx, FULL_FEATURES])
    calibrated_test_prob = calibrator.predict(raw_test_prob)
    test_y = data.loc[test_idx, "y"].to_numpy()

    threshold_results = []
    for label, prob, threshold in [
        ("uncalibrated_at_0.5", raw_test_prob, 0.5),
        ("calibrated_at_0.5", calibrated_test_prob, 0.5),
        ("calibrated_validation_selected", calibrated_test_prob, chosen_threshold),
    ]:
        row = metrics(test_y, prob, threshold)
        row.update({"model": selected_name, "evaluation": label})
        threshold_results.append(row)
    threshold_frame = pd.DataFrame(threshold_results)
    save_table(threshold_frame, table_dir / "calibration_threshold_performance.csv")
    plot_calibration(test_y, raw_test_prob, calibrated_test_prob, figure_dir / "calibration.png")
    plot_threshold(
        data.loc[threshold_idx, "y"].to_numpy(), calibrated_threshold_prob,
        chosen_threshold, figure_dir / "threshold_selection.png",
    )

    cal_raw_points = calibration_points(test_y, raw_test_prob)
    cal_raw_points["probability_type"] = "uncalibrated"
    cal_adj_points = calibration_points(test_y, calibrated_test_prob)
    cal_adj_points["probability_type"] = "Platt calibrated"
    save_table(pd.concat([cal_raw_points, cal_adj_points], ignore_index=True), table_dir / "calibration_points.csv")

    print("Running feature-set ablation...", flush=True)
    ablation_rows = []
    ablation_model = models["Histogram gradient boosting"]
    for feature_set, features in [
        ("weather_only", WEATHER_FEATURES),
        ("weather_plus_fire_indices", FIRE_INDEX_FEATURES),
        ("weather_indices_space_time", FULL_FEATURES),
    ]:
        _, prob, result = fit_and_score(ablation_model, data, primary_train, test_idx, features)
        result.update({"model": "Histogram gradient boosting", "feature_set": feature_set, "feature_count": len(features)})
        ablation_rows.append(result)
    save_table(pd.DataFrame(ablation_rows), table_dir / "feature_ablation.csv")

    print("Calculating permutation importance and block-bootstrap intervals...", flush=True)
    importance_rng = np.random.default_rng(RANDOM_SEED)
    importance_n = min(20_000, len(test_idx))
    importance_idx = importance_rng.choice(test_idx, size=importance_n, replace=False)
    perm = permutation_importance(
        selected_model,
        data.loc[importance_idx, FULL_FEATURES],
        data.loc[importance_idx, "y"],
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame({
        "feature": FULL_FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_sd": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    save_table(importance_frame, table_dir / "permutation_importance.csv")
    plot_importance(importance_frame, figure_dir / "permutation_importance.png")

    ci = cluster_bootstrap_ci(
        test_y, calibrated_test_prob,
        data.loc[test_idx, "spatial_block"].to_numpy(),
        chosen_threshold,
    )
    save_table(ci, table_dir / "strict_test_cluster_bootstrap_ci.csv")

    joblib.dump(selected_model, model_dir / "selected_model.joblib", compress=3)
    # Save the sklearn estimator rather than this module's wrapper class so the
    # artifact stays loadable after this script has run as ``__main__``.
    joblib.dump(calibrator.model, model_dir / "platt_logit_model.joblib", compress=3)
    model_metadata = {
        "selected_model": selected_name,
        "selection_metric": "average precision on the held-out threshold/model-selection partition",
        "validation_average_precision": validation_ap[selected_name],
        "selected_probability_threshold": chosen_threshold,
        "validation_f1_at_selected_threshold": validation_f1,
        "features": FULL_FEATURES,
        "calibration": "LogisticRegression applied to logit-clipped raw probabilities",
        "probability_clip": [1e-6, 1 - 1e-6],
        "random_seed": RANDOM_SEED,
        "spatial_block_size_degrees": BLOCK_SIZE_DEGREES,
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    summary = {
        "data_quality": quality,
        "selected_model": selected_name,
        "selected_threshold": chosen_threshold,
        "primary_test_metrics": metrics(test_y, calibrated_test_prob, chosen_threshold),
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to CA_2020_2022_valid_data.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.output, args.random_seed)
