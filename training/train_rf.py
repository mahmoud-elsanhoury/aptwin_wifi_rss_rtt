# ============================================================
# training/train_rf.py
# ------------------------------------------------------------
# Purpose:
# - Train and evaluate a trusted Random Forest baseline as a standalone
#   repository script.
# - Remove notebook-memory dependencies such as test_df, X_sub_s,
#   Y_sub_s, X_test_s, y_scaler, and OUTPUT_DIR.
# - Add the agreed fixed multi-seed reporting protocol without changing
#   the model family, features, hyperparameters, or data splits.
#
# Scientific protocol:
# - Uses scikit-learn RandomForestRegressor through MultiOutputRegressor.
# - Trains on the saved RP-level subtrain split only.
# - Evaluates validation and official test splits without changing the model.
# - Uses a fixed configuration, not a hyperparameter search.
# - Repeats training over fixed seeds to quantify stochastic variation.
# - Main multi-seed reporting uses mean ± std over seeds.
# - The representative seed is used only for CDF plotting and is selected
#   from validation scenario-macro RMSE using the validation-median rule.
# - No test metric is used for seed selection.
# - Reports row-level and geometry-level predictions.
# - No target values are used to create features.
# Random Forest and XGBoost are treated as deterministic tabular regressors optimized for point prediction. 
# In contrast, the Gaussian MLP is formulated as a probabilistic regressor that predicts both mean and uncertainty; 
# therefore, it is trained using Gaussian negative log-likelihood and calibrated on the validation split.
# ============================================================

from __future__ import annotations

import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from common_training import (
    FEATURES_GEOM_ONLY,
    MODEL_DIR,
    PREDICTION_DIR,
    RESULTS_TABLE_DIR,
    SEED,
    SEEDS,
    TARGETS,
    build_standard_scalers,
    extract_scenario_macro_metrics,
    load_row_splits,
    log,
    save_json,
    save_prediction_bundle,
    set_all_seeds,
    summarize_seed_metrics,
)

PREFIX = "train_rf"
FAMILY = "RF"
MODEL_NAME = "RandomForest"
MODEL_PREFIX = "rf"

RF_CONFIG = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 1,
}


def select_representative_seed_for_model(validation_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Select one representative seed for the RF model candidate.

    Scientific rule:
    - Selection uses validation metrics only.
    - RMSE_macro is the primary metric.
    - Because one prediction file contains both RSS and RTT, the selected
      seed is shared across all targets and levels for this model.
    - The selected seed is the seed whose mean validation RMSE_macro across
      available level-target pairs is closest to the median across seeds.
    """
    summary = (
        validation_macro
        .groupby(["family", "model", "seed"], as_index=False)
        .agg(validation_RMSE_macro_mean=("RMSE_macro", "mean"))
    )

    rows = []
    for (family, model), g in summary.groupby(["family", "model"], sort=True):
        median_rmse = float(g["validation_RMSE_macro_mean"].median())
        gg = g.copy()
        gg["distance_to_validation_median_RMSE"] = (
            gg["validation_RMSE_macro_mean"] - median_rmse
        ).abs()
        chosen = gg.sort_values(
            ["distance_to_validation_median_RMSE", "validation_RMSE_macro_mean", "seed"],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append({
            "family": family,
            "model": model,
            "representative_seed": int(chosen["seed"]),
            "selection_rule": "validation_median_RMSE_macro_mean_over_level_target_pairs",
            "validation_RMSE_macro_mean": float(chosen["validation_RMSE_macro_mean"]),
            "validation_RMSE_macro_median": median_rmse,
            "distance_to_validation_median_RMSE": float(chosen["distance_to_validation_median_RMSE"]),
        })

    return pd.DataFrame(rows)


def copy_representative_predictions(model_prefix: str, representative_seed: int) -> None:
    """Copy the representative seed prediction files used by final CDF plotting."""
    for level in ["row", "geometry"]:
        src = PREDICTION_DIR / f"{model_prefix}_seed{representative_seed}_{level}_test_predictions.csv"
        dst = PREDICTION_DIR / f"{model_prefix}_representative_{level}_test_predictions.csv"
        if not src.exists():
            raise FileNotFoundError(f"Representative source prediction file not found: {src}")
        shutil.copy2(src, dst)
        log(PREFIX, f"Saved representative {level} predictions: {dst}")


def main() -> None:
    log(PREFIX, "Starting standalone Random Forest multi-seed training.")
    log(PREFIX, f"Fixed seed set: {SEEDS}")

    splits = load_row_splits(kind="base", prefix=PREFIX)

    train_df = splits["subtrain"]
    val_df = splits["val"]
    test_df = splits["test"]

    x_scaler, y_scaler = build_standard_scalers(train_df, FEATURES_GEOM_ONLY)

    X_train = x_scaler.transform(train_df[FEATURES_GEOM_ONLY].to_numpy(dtype=np.float32))
    Y_train = y_scaler.transform(train_df[TARGETS].to_numpy(dtype=np.float32))
    X_val = x_scaler.transform(val_df[FEATURES_GEOM_ONLY].to_numpy(dtype=np.float32))
    X_test = x_scaler.transform(test_df[FEATURES_GEOM_ONLY].to_numpy(dtype=np.float32))

    log(PREFIX, f"Feature count: {len(FEATURES_GEOM_ONLY)}")
    log(PREFIX, f"Subtrain rows: {len(train_df):,}")
    log(PREFIX, f"Validation rows: {len(val_df):,}")
    log(PREFIX, f"Test rows: {len(test_df):,}")

    model_dir = MODEL_DIR / "rf"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "y_scaler.joblib")

    validation_macro_all = []
    test_macro_all = []
    seed_model_paths = []

    for seed in SEEDS:
        set_all_seeds(seed)
        log(PREFIX, "=" * 78)
        log(PREFIX, f"Training seed {seed}.")

        rf_config_seed = dict(RF_CONFIG)
        rf_config_seed["random_state"] = int(seed)

        model = MultiOutputRegressor(
            RandomForestRegressor(**rf_config_seed)
        )

        t0 = time.time()
        log(PREFIX, f"Random Forest training started for seed {seed}.")
        model.fit(X_train, Y_train)
        log(PREFIX, f"Random Forest seed {seed} finished in {(time.time() - t0) / 60.0:.2f} min.")

        val_pred_scaled = model.predict(X_val)
        val_pred = y_scaler.inverse_transform(val_pred_scaled)
        _, _, val_metrics = save_prediction_bundle(
            row_df=val_df,
            y_pred=val_pred,
            model=MODEL_NAME,
            family=FAMILY,
            prefix=f"{MODEL_PREFIX}_seed{seed}",
            split_name="val",
            seed=seed,
        )
        validation_macro_all.append(extract_scenario_macro_metrics(val_metrics, seed=seed))

        test_pred_scaled = model.predict(X_test)
        test_pred = y_scaler.inverse_transform(test_pred_scaled)
        _, _, test_metrics = save_prediction_bundle(
            row_df=test_df,
            y_pred=test_pred,
            model=MODEL_NAME,
            family=FAMILY,
            prefix=f"{MODEL_PREFIX}_seed{seed}",
            split_name="test",
            seed=seed,
        )
        test_macro_all.append(extract_scenario_macro_metrics(test_metrics, seed=seed))

        model_path = model_dir / f"random_forest_multioutput_seed{seed}.joblib"
        joblib.dump(model, model_path)
        seed_model_paths.append(str(model_path))
        log(PREFIX, f"Saved RF seed {seed} model: {model_path}")

    validation_macro_all = pd.concat(validation_macro_all, ignore_index=True)
    test_macro_all = pd.concat(test_macro_all, ignore_index=True)

    validation_macro_all.to_csv(RESULTS_TABLE_DIR / "rf_multiseed_validation_metrics.csv", index=False)
    test_macro_all.to_csv(RESULTS_TABLE_DIR / "rf_multiseed_test_metrics.csv", index=False)

    test_summary = summarize_seed_metrics(test_macro_all)
    test_summary.to_csv(RESULTS_TABLE_DIR / "rf_multiseed_test_summary_mean_std.csv", index=False)

    representative = select_representative_seed_for_model(validation_macro_all)
    representative.to_csv(RESULTS_TABLE_DIR / "rf_representative_seed_for_cdf.csv", index=False)

    representative_seed = int(representative.iloc[0]["representative_seed"])
    copy_representative_predictions(MODEL_PREFIX, representative_seed)

    save_json(
        {
            "model": MODEL_NAME,
            "family": FAMILY,
            "features": FEATURES_GEOM_ONLY,
            "config": {**RF_CONFIG, "random_state": "varied_by_SEEDS"},
            "seeds": SEEDS,
            "primary_metric": "scenario-macro RMSE",
            "secondary_metrics": ["scenario-macro MAE", "scenario-macro P95"],
            "representative_seed_rule": "validation-median RMSE_macro mean across available level-target pairs",
            "representative_seed": representative_seed,
            "seed_model_paths": seed_model_paths,
        },
        model_dir / "rf_metadata.json",
    )

    log(PREFIX, "Saved RF multi-seed models, predictions, and metric tables.")
    log(PREFIX, "training/train_rf.py completed successfully.")


if __name__ == "__main__":
    main()
