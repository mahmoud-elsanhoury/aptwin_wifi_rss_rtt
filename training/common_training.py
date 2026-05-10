# ============================================================
# training/common_training.py
# ------------------------------------------------------------
# Purpose:
# - Shared utilities for standalone AP Digital Twin training scripts.
# - Keep repeated metric, path, feature, seed, and prediction-saving logic
#   in one trusted place instead of duplicating fragile notebook state.
#
# Scientific protocol:
# - Helpers do not train or tune models.
# - Helpers do not change the official train/validation/test splits.
# - Random seed is treated as an experimental source of stochastic variation,
#   not as a hyperparameter to optimize.
# - Scenario-macro metrics are computed per scenario, then averaged.
# - The primary metric is scenario-macro RMSE.
# - Secondary metrics are scenario-macro MAE and scenario-macro P95.
# - Bias is intentionally not used in the final reporting protocol.
# - Multi-seed main results should be reported as mean ± standard deviation
#   across fixed seeds.
# - For CDF visualization only, a representative seed can be selected using
#   the validation-median RMSE rule. No test metric is used for seed selection.
# - Prediction CSVs use a consistent schema so final metric/plot scripts
#   can consume all model families uniformly.
# ============================================================

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ============================================================
# Section 1 — Repository paths
# ------------------------------------------------------------
# Purpose:
# - Resolve all repository paths from this file location.
# - Avoid notebook-dependent or machine-specific paths.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "output_csvs" / "processed_features"
PREDICTION_DIR = REPO_ROOT / "output_csvs" / "predictions"
MODEL_DIR = REPO_ROOT / "models" / "saved_models"
RESULTS_TABLE_DIR = REPO_ROOT / "results" / "tables"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

for _p in [PREDICTION_DIR, MODEL_DIR, RESULTS_TABLE_DIR, FIGURE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Section 2 — Reproducibility and multi-seed protocol
# ------------------------------------------------------------
# Purpose:
# - Define the fixed repeated-seed protocol used by training scripts.
# - Provide one shared seed-setting function.
#
# Scientific protocol:
# - The seed is not optimized.
# - Main results should be summarized over all seeds.
# - The representative seed is only for readable CDF visualization.
# ============================================================

SEED = 42
SEEDS = [11, 22, 33, 44, 55]

PRIMARY_METRIC = "RMSE_macro"
REPORT_METRICS = ["RMSE_macro", "MAE_macro", "P95_macro"]
SECONDARY_METRICS = ["MAE_macro", "P95_macro"]


def set_all_seeds(seed: int) -> None:
    """
    Set common random seeds for Python, NumPy, and optional DL libraries.

    This function is intentionally lightweight. It does not force expensive
    deterministic modes unless the installed library exposes safe flags.
    Training scripts may still set model-specific random_state=seed.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Prefer reproducibility where available, without making this helper
        # fail on older PyTorch versions.
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    except ImportError:
        pass


# ============================================================
# Section 3 — Dataset schema
# ------------------------------------------------------------
# Purpose:
# - Centralize feature, target, and ID-column definitions.
# ============================================================

TARGETS = ["RSS_dBm", "RTT_m"]

FEATURES_GEOM_ONLY = [
    "RX_x", "RX_y",
    "AP_x", "AP_y",
    "dx", "dy",
    "distance",
    "LOS_flag", "LOS_known",
]

WALL_OBSTRUCTION_FEATURES = [
    "wall_cross_count_total",
    "wall_cross_count_internal",
    "wall_cross_count_external",
    "obstructed_path",
]

ENDPOINT_WALL_CONTEXT_FEATURES = [
    "AP_dist_nearest_wall",
    "AP_dist_nearest_external_wall",
    "AP_dist_nearest_internal_wall",
    "RP_dist_nearest_wall",
    "RP_dist_nearest_external_wall",
    "RP_dist_nearest_internal_wall",
    "AP_near_wall_1m",
    "RP_near_wall_1m",
]

FEATURES_WALL_OBS = FEATURES_GEOM_ONLY + WALL_OBSTRUCTION_FEATURES
FEATURES_WALL_OBS_CONTEXT = FEATURES_WALL_OBS + ENDPOINT_WALL_CONTEXT_FEATURES

ID_COLUMNS_KEEP = [
    "scenario", "batch", "area_type",
    "RX_x", "RX_y", "AP_x", "AP_y",
    "AP_id_raw", "AP_index_usable",
    "distance", "LOS_flag", "LOS_known",
    "row_pos_within_split", "lams_id",
]


# ============================================================
# Section 4 — Logging and file checks
# ------------------------------------------------------------
# Purpose:
# - Keep script progress messages clear in CMD/PowerShell.
# - Fail clearly when required inputs are missing.
# ============================================================

def log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}", flush=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_csv(path: Path, name: str, prefix: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    log(prefix, f"Loaded {name}: {len(df):,} rows from {path}")
    return df.reset_index(drop=True)


# ============================================================
# Section 5 — Split loading
# ------------------------------------------------------------
# Purpose:
# - Load prepared row-level and geometry-level train/val/test splits.
#
# Scientific protocol:
# - These functions only read already prepared files.
# - They do not rebuild, reshuffle, or alter the official splits.
# ============================================================

def load_row_splits(kind: str = "base", prefix: str = "train") -> Dict[str, pd.DataFrame]:
    """
    Load row-level splits.

    kind='base' uses row_train/row_val/row_test.
    kind='wall_context' uses row_*_wall_context.
    kind='wall_obstruction' uses row_*_wall_obstruction.
    """
    suffix = "" if kind == "base" else f"_{kind}"
    return {
        "subtrain": load_csv(PROCESSED_DIR / f"row_train{suffix}.csv", f"row_train{suffix}", prefix),
        "val": load_csv(PROCESSED_DIR / f"row_val{suffix}.csv", f"row_val{suffix}", prefix),
        "test": load_csv(PROCESSED_DIR / f"row_test{suffix}.csv", f"row_test{suffix}", prefix),
    }


def load_geometry_splits(kind: str = "base", prefix: str = "train") -> Dict[str, pd.DataFrame]:
    suffix = "" if kind == "base" else f"_{kind}"
    return {
        "subtrain": load_csv(PROCESSED_DIR / f"geometry_train{suffix}.csv", f"geometry_train{suffix}", prefix),
        "val": load_csv(PROCESSED_DIR / f"geometry_val{suffix}.csv", f"geometry_val{suffix}", prefix),
        "test": load_csv(PROCESSED_DIR / f"geometry_test{suffix}.csv", f"geometry_test{suffix}", prefix),
    }


# ============================================================
# Section 6 — Data validation and scaling
# ------------------------------------------------------------
# Purpose:
# - Provide common feature/target checks and standard scaling.
# ============================================================

def assert_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def build_standard_scalers(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[StandardScaler, StandardScaler]:
    assert_columns(train_df, list(feature_cols) + TARGETS, "training dataframe")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(train_df[list(feature_cols)].to_numpy(dtype=np.float32))
    y_scaler.fit(train_df[TARGETS].to_numpy(dtype=np.float32))
    return x_scaler, y_scaler


def scenario_balanced_weights(df: pd.DataFrame, scenario_col: str = "scenario") -> np.ndarray:
    counts = df[scenario_col].value_counts()
    weights = df[scenario_col].map(lambda s: 1.0 / counts.loc[s]).astype(float).to_numpy()
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


# ============================================================
# Section 7 — Metrics
# ------------------------------------------------------------
# Purpose:
# - Compute the agreed metrics consistently across all model families.
#
# Scientific protocol:
# - RMSE is the primary metric.
# - MAE and P95 are secondary metrics.
# - Bias is intentionally excluded from this reporting layer.
# - Scenario-macro metrics are computed per scenario, then averaged.
# ============================================================

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "P95": float(np.percentile(abs_err, 95)),
        "n": int(len(y_true)),
    }


def compute_micro_macro_metrics(pred_df: pd.DataFrame, model: str, level: str, family: str) -> pd.DataFrame:
    """
    Compute micro, per-scenario, and scenario-macro metrics.

    Input prediction table must use the wide prediction schema produced by
    make_row_prediction_df() or geometry_average_from_rows().
    """
    rows = []
    for target, true_col, pred_col in [
        ("RSS", "RSS_true", "RSS_pred"),
        ("RTT", "RTT_true", "RTT_pred"),
    ]:
        if true_col not in pred_df.columns or pred_col not in pred_df.columns:
            continue

        valid_all = pred_df.dropna(subset=[true_col, pred_col]).copy()
        if valid_all.empty:
            continue

        micro = metric_dict(valid_all[true_col], valid_all[pred_col])
        rows.append({
            "family": family,
            "model": model,
            "target": target,
            "level": level,
            "aggregation": "micro",
            "scenario": "ALL",
            **micro,
        })

        scenario_metrics = []
        for scenario, g in valid_all.groupby("scenario", sort=True):
            m = metric_dict(g[true_col], g[pred_col])
            scenario_metrics.append(m)
            rows.append({
                "family": family,
                "model": model,
                "target": target,
                "level": level,
                "aggregation": "scenario",
                "scenario": scenario,
                **m,
            })

        scenario_df = pd.DataFrame(scenario_metrics)
        rows.append({
            "family": family,
            "model": model,
            "target": target,
            "level": level,
            "aggregation": "scenario_macro",
            "scenario": "MACRO",
            "MAE": float(scenario_df["MAE"].mean()),
            "RMSE": float(scenario_df["RMSE"].mean()),
            "P95": float(scenario_df["P95"].mean()),
            "n": int(valid_all["scenario"].nunique()),
        })

    return pd.DataFrame(rows)


def extract_scenario_macro_metrics(metrics_df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """
    Convert long metrics into one row per family/model/level/target macro metric.

    Output columns:
        family, model, level, target, seed, RMSE_macro, MAE_macro, P95_macro,
        n_scenarios
    """
    if metrics_df.empty:
        return pd.DataFrame()

    macro = metrics_df[metrics_df["aggregation"] == "scenario_macro"].copy()
    if macro.empty:
        return pd.DataFrame()

    out = macro[["family", "model", "level", "target", "RMSE", "MAE", "P95", "n"]].copy()
    out = out.rename(columns={
        "RMSE": "RMSE_macro",
        "MAE": "MAE_macro",
        "P95": "P95_macro",
        "n": "n_scenarios",
    })
    if seed is not None:
        out.insert(2, "seed", int(seed))
    return out.reset_index(drop=True)


# ============================================================
# Section 8 — Prediction table construction
# ------------------------------------------------------------
# Purpose:
# - Build consistent wide prediction CSVs for row and geometry levels.
# ============================================================

def make_row_prediction_df(df: pd.DataFrame, y_pred: np.ndarray, model: str, family: str) -> pd.DataFrame:
    out_cols = [c for c in ID_COLUMNS_KEEP if c in df.columns]
    out = df[out_cols].copy()
    out["family"] = family
    out["model"] = model
    out["level"] = "row"
    out["RSS_true"] = df["RSS_dBm"].to_numpy(dtype=float)
    out["RTT_true"] = df["RTT_m"].to_numpy(dtype=float)
    out["RSS_pred"] = np.asarray(y_pred)[:, 0].astype(float)
    out["RTT_pred"] = np.asarray(y_pred)[:, 1].astype(float)
    out["RSS_abs_error"] = np.abs(out["RSS_true"] - out["RSS_pred"])
    out["RTT_abs_error"] = np.abs(out["RTT_true"] - out["RTT_pred"])
    return out


def geometry_average_from_rows(row_pred_df: pd.DataFrame, model: str, family: str) -> pd.DataFrame:
    key_cols = ["scenario", "AP_x", "AP_y", "RX_x", "RX_y"]
    if "AP_id_raw" in row_pred_df.columns:
        key_cols = ["scenario", "AP_id_raw", "AP_x", "AP_y", "RX_x", "RX_y"]

    agg = row_pred_df.groupby(key_cols, as_index=False).agg(
        RSS_true=("RSS_true", "mean"),
        RTT_true=("RTT_true", "mean"),
        RSS_pred=("RSS_pred", "mean"),
        RTT_pred=("RTT_pred", "mean"),
        n_repeat=("RSS_true", "size"),
    )
    agg["family"] = family
    agg["model"] = model
    agg["level"] = "geometry"
    agg["RSS_abs_error"] = np.abs(agg["RSS_true"] - agg["RSS_pred"])
    agg["RTT_abs_error"] = np.abs(agg["RTT_true"] - agg["RTT_pred"])
    return agg


def add_seed_column(df: pd.DataFrame, seed: int | None) -> pd.DataFrame:
    out = df.copy()
    if seed is not None:
        out["seed"] = int(seed)
    return out


def save_prediction_bundle(
    row_df: pd.DataFrame,
    y_pred: np.ndarray,
    model: str,
    family: str,
    prefix: str,
    split_name: str = "test",
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Save row-level and geometry-level predictions plus metrics.

    Backward-compatible default:
        <prefix>_row_test_predictions.csv
        <prefix>_geometry_test_predictions.csv
        <prefix>_metrics.csv

    Multi-seed usage:
        pass prefix='<model>_seed11' and seed=11.
        pass prefix='<model>_representative' for the CDF representative seed.
    """
    row_pred = make_row_prediction_df(row_df, y_pred, model=model, family=family)
    row_pred = add_seed_column(row_pred, seed)

    geo_pred = geometry_average_from_rows(row_pred, model=model, family=family)
    geo_pred = add_seed_column(geo_pred, seed)

    row_path = PREDICTION_DIR / f"{prefix}_row_{split_name}_predictions.csv"
    geo_path = PREDICTION_DIR / f"{prefix}_geometry_{split_name}_predictions.csv"

    row_pred.to_csv(row_path, index=False)
    geo_pred.to_csv(geo_path, index=False)

    metrics = pd.concat([
        compute_micro_macro_metrics(row_pred, model=model, level="row", family=family),
        compute_micro_macro_metrics(geo_pred, model=model, level="geometry", family=family),
    ], ignore_index=True)

    if seed is not None and not metrics.empty:
        metrics.insert(2, "seed", int(seed))

    metric_path = RESULTS_TABLE_DIR / f"{prefix}_{split_name}_metrics.csv"
    metrics.to_csv(metric_path, index=False)

    return row_pred, geo_pred, metrics


# ============================================================
# Section 9 — Multi-seed selection and summaries
# ------------------------------------------------------------
# Purpose:
# - Provide the minimal shared logic needed for seed reporting.
#
# Scientific protocol:
# - Main reporting should use mean ± std across seeds.
# - Representative seed selection uses validation metrics only.
# - The representative seed is selected by median validation RMSE, not best RMSE.
# ============================================================

def select_validation_median_seed(validation_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Select one representative seed per family/model/level/target.

    Rule:
        Choose the seed whose validation RMSE_macro is closest to the median
        validation RMSE_macro across seeds.

    Required columns:
        family, model, seed, level, target, RMSE_macro
    """
    required = {"family", "model", "seed", "level", "target", PRIMARY_METRIC}
    missing = required - set(validation_metrics.columns)
    if missing:
        raise ValueError(
            "validation_metrics is missing required columns for representative "
            f"seed selection: {sorted(missing)}"
        )

    rows = []
    group_cols = ["family", "model", "level", "target"]

    for keys, g in validation_metrics.groupby(group_cols, sort=True):
        family, model, level, target = keys
        gg = g.dropna(subset=[PRIMARY_METRIC]).copy()

        if gg.empty:
            continue

        median_value = float(gg[PRIMARY_METRIC].median())
        gg["distance_to_validation_median_RMSE"] = (gg[PRIMARY_METRIC] - median_value).abs()
        gg = gg.sort_values(
            ["distance_to_validation_median_RMSE", PRIMARY_METRIC, "seed"],
            ascending=[True, True, True],
        )
        chosen = gg.iloc[0]

        row = {
            "family": family,
            "model": model,
            "level": level,
            "target": target,
            "representative_seed": int(chosen["seed"]),
            "selection_rule": "validation_median_RMSE_macro",
            "validation_RMSE_macro": float(chosen[PRIMARY_METRIC]),
            "validation_RMSE_median": median_value,
            "distance_to_validation_median_RMSE": float(chosen["distance_to_validation_median_RMSE"]),
        }

        for metric in SECONDARY_METRICS:
            if metric in chosen.index:
                row[f"validation_{metric}"] = float(chosen[metric])

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize seed-level metrics as mean ± std.

    Required columns:
        family, model, seed, level, target, RMSE_macro, MAE_macro, P95_macro
    """
    required = {"family", "model", "seed", "level", "target", *REPORT_METRICS}
    missing = required - set(seed_metrics.columns)
    if missing:
        raise ValueError(
            f"seed_metrics is missing required columns: {sorted(missing)}"
        )

    rows = []
    group_cols = ["family", "model", "level", "target"]

    for keys, g in seed_metrics.groupby(group_cols, sort=True):
        family, model, level, target = keys
        row = {
            "family": family,
            "model": model,
            "level": level,
            "target": target,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": ",".join(str(int(s)) for s in sorted(g["seed"].dropna().unique())),
        }

        for metric in REPORT_METRICS:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def save_multiseed_metric_tables(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    family_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Save the agreed multi-seed metric tables for one training script.

    Outputs:
        <family_prefix>_multiseed_validation_metrics.csv
        <family_prefix>_multiseed_test_metrics.csv
        <family_prefix>_multiseed_test_summary_mean_std.csv
        <family_prefix>_representative_seed_for_cdf.csv
    """
    validation_metrics = validation_metrics.copy().reset_index(drop=True)
    test_metrics = test_metrics.copy().reset_index(drop=True)

    representative = select_validation_median_seed(validation_metrics)
    test_summary = summarize_seed_metrics(test_metrics)

    validation_path = RESULTS_TABLE_DIR / f"{family_prefix}_multiseed_validation_metrics.csv"
    test_path = RESULTS_TABLE_DIR / f"{family_prefix}_multiseed_test_metrics.csv"
    summary_path = RESULTS_TABLE_DIR / f"{family_prefix}_multiseed_test_summary_mean_std.csv"
    representative_path = RESULTS_TABLE_DIR / f"{family_prefix}_representative_seed_for_cdf.csv"

    validation_metrics.to_csv(validation_path, index=False)
    test_metrics.to_csv(test_path, index=False)
    test_summary.to_csv(summary_path, index=False)
    representative.to_csv(representative_path, index=False)

    return validation_metrics, test_metrics, test_summary, representative


def copy_representative_prediction_files(
    model_prefix: str,
    representative_seed: int,
    split_name: str = "test",
) -> dict:
    """
    Copy one seed-specific prediction pair into representative prediction files.

    Example source files:
        xgb_wall_context_seed11_row_test_predictions.csv
        xgb_wall_context_seed11_geometry_test_predictions.csv

    Example output files:
        xgb_wall_context_representative_row_test_predictions.csv
        xgb_wall_context_representative_geometry_test_predictions.csv
    """
    representative_seed = int(representative_seed)
    copied = {}

    for level in ["row", "geometry"]:
        src = PREDICTION_DIR / f"{model_prefix}_seed{representative_seed}_{level}_{split_name}_predictions.csv"
        dst = PREDICTION_DIR / f"{model_prefix}_representative_{level}_{split_name}_predictions.csv"
        require_file(src)
        shutil.copy2(src, dst)
        copied[level] = dst

    return copied


# ============================================================
# Section 10 — Miscellaneous persistence helpers
# ------------------------------------------------------------
# Purpose:
# - Save small metadata objects reproducibly.
# ============================================================

def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
