# ============================================================
# training/train_xgboost.py
# ------------------------------------------------------------
# Purpose:
# - Train standalone XGBoost models for the controlled AP Digital Twin
#   tabular feature ladder.
# - Remove notebook-memory dependencies such as test_df_wall,
#   subtrain_df_wall_context, wall_xgb_predictions, and WALL_FEATURE_DIR.
# - Add the agreed fixed multi-seed reporting protocol without changing
#   the feature ladder, model candidates, or hyperparameter settings.
#
# Scientific protocol:
# - Uses the trusted xgboost.XGBRegressor library.
# - Trains one RSS regressor and one RTT regressor per feature set.
# - Uses fixed model settings, not broad hyperparameter search.
# - Uses validation logging and early stopping where supported.
# - Keeps the original feature-ladder logic intact:
#     1) geometry only
#     2) geometry + wall obstruction
#     3) geometry + wall obstruction + endpoint context
#     4) scenario-weighted versions of the same three feature sets
# - Repeats the same fixed protocol over fixed seeds.
# - Main multi-seed reporting uses mean ± std over seeds.
# - The representative seed is used only for CDF plotting and is selected
#   from validation scenario-macro RMSE using the validation-median rule.
# - No test metric is used for seed selection.
# - The official test set is evaluated only after model training.
# Random Forest and XGBoost are treated as deterministic tabular regressors optimized for point prediction. 
# In contrast, the Gaussian MLP is formulated as a probabilistic regressor that predicts both mean and uncertainty; 
# therefore, it is trained using Gaussian negative log-likelihood and calibrated on the validation split.
# ============================================================

from __future__ import annotations

import shutil
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise ImportError(
        "XGBoost is required for training/train_xgboost.py. Install it with: pip install xgboost"
    ) from exc

from common_training import (
    FEATURES_GEOM_ONLY,
    FEATURES_WALL_OBS,
    FEATURES_WALL_OBS_CONTEXT,
    MODEL_DIR,
    PREDICTION_DIR,
    RESULTS_TABLE_DIR,
    SEED,
    SEEDS,
    TARGETS,
    extract_scenario_macro_metrics,
    load_row_splits,
    log,
    save_json,
    save_prediction_bundle,
    scenario_balanced_weights,
    set_all_seeds,
    summarize_seed_metrics,
)

PREFIX = "train_xgboost"
FAMILY = "XGBoost"

XGB_BASE_CONFIG = {
    "n_estimators": 700,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "rmse",
}


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def make_xgb(seed: int, use_constructor_early_stopping: bool = True) -> XGBRegressor:
    """Construct one XGBoost regressor with the same fixed settings and a seed."""
    params = dict(XGB_BASE_CONFIG)
    params["random_state"] = int(seed)
    if use_constructor_early_stopping:
        params["early_stopping_rounds"] = 30
    return XGBRegressor(**params)


def fit_xgb_safely(
    X_tr,
    y_tr,
    X_va,
    y_va,
    model_name: str,
    seed: int,
    sample_weight=None,
    sample_weight_val=None,
) -> XGBRegressor:
    """
    Fit one XGBoost model with version-compatible early stopping.

    Scientific note:
    - This is not a hyperparameter search.
    - The fallback logic only handles differences between installed XGBoost APIs.
    - The model settings remain fixed; only the random seed changes.
    """
    log(
        PREFIX,
        f"[{now()}] Training {model_name} | seed={seed} | features={X_tr.shape[1]} | "
        f"subtrain={X_tr.shape[0]:,} | val={X_va.shape[0]:,}",
    )
    t0 = time.time()
    model = make_xgb(seed=seed, use_constructor_early_stopping=True)

    try:
        fit_kwargs = dict(eval_set=[(X_va, y_va)], verbose=25)
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if sample_weight_val is not None:
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
        model.fit(X_tr, y_tr, **fit_kwargs)
    except TypeError:
        log(PREFIX, f"{model_name}: retrying with fit-level early stopping.")
        model = make_xgb(seed=seed, use_constructor_early_stopping=False)
        try:
            fit_kwargs = dict(eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=25)
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            if sample_weight_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
            model.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError:
            log(PREFIX, f"{model_name}: retrying without early stopping due to installed XGBoost API.")
            params = dict(XGB_BASE_CONFIG)
            params["random_state"] = int(seed)
            params["n_estimators"] = 500
            model = XGBRegressor(**params)
            if sample_weight is not None:
                try:
                    model.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
                except TypeError:
                    model.fit(X_tr, y_tr, sample_weight=sample_weight)
            else:
                try:
                    model.fit(X_tr, y_tr, verbose=False)
                except TypeError:
                    model.fit(X_tr, y_tr)

    elapsed = (time.time() - t0) / 60.0
    best_iter = getattr(model, "best_iteration", None)
    log(PREFIX, f"[{now()}] Finished {model_name} in {elapsed:.2f} min | best_iteration={best_iter}")
    return model


def train_pair(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_name: str,
    seed: int,
    weighted: bool = False,
):
    """Train RSS and RTT XGBoost regressors for one feature-set candidate."""
    for label, df in [("subtrain", train_df), ("val", val_df), ("test", test_df)]:
        missing = [c for c in features + TARGETS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{model_name}: {label} missing columns: {missing}")

    X_tr = train_df[features].to_numpy(dtype=np.float32)
    X_va = val_df[features].to_numpy(dtype=np.float32)
    X_te = test_df[features].to_numpy(dtype=np.float32)
    y_rss_tr = train_df["RSS_dBm"].to_numpy(dtype=np.float32)
    y_rtt_tr = train_df["RTT_m"].to_numpy(dtype=np.float32)
    y_rss_va = val_df["RSS_dBm"].to_numpy(dtype=np.float32)
    y_rtt_va = val_df["RTT_m"].to_numpy(dtype=np.float32)

    w_tr = scenario_balanced_weights(train_df) if weighted else None
    w_va = scenario_balanced_weights(val_df) if weighted else None

    rss_model = fit_xgb_safely(
        X_tr,
        y_rss_tr,
        X_va,
        y_rss_va,
        f"{model_name}_RSS",
        seed=seed,
        sample_weight=w_tr,
        sample_weight_val=w_va,
    )
    rtt_model = fit_xgb_safely(
        X_tr,
        y_rtt_tr,
        X_va,
        y_rtt_va,
        f"{model_name}_RTT",
        seed=seed,
        sample_weight=w_tr,
        sample_weight_val=w_va,
    )

    val_pred = np.column_stack([rss_model.predict(X_va), rtt_model.predict(X_va)])
    test_pred = np.column_stack([rss_model.predict(X_te), rtt_model.predict(X_te)])
    return rss_model, rtt_model, val_pred, test_pred


def select_representative_seeds_for_models(validation_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Select one representative seed per XGBoost model candidate.

    Rule:
    - Use validation metrics only.
    - Compute the mean validation RMSE_macro across available level-target pairs.
    - Select the seed closest to the median of that mean RMSE across seeds.
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
    """Copy seed-specific prediction files into representative filenames."""
    for level in ["row", "geometry"]:
        src = PREDICTION_DIR / f"{model_prefix}_seed{representative_seed}_{level}_test_predictions.csv"
        dst = PREDICTION_DIR / f"{model_prefix}_representative_{level}_test_predictions.csv"
        if not src.exists():
            raise FileNotFoundError(f"Representative source prediction file not found: {src}")
        shutil.copy2(src, dst)
        log(PREFIX, f"Saved representative {level} predictions: {dst}")


def main() -> None:
    log(PREFIX, "Starting standalone XGBoost multi-seed feature-ladder training.")
    log(PREFIX, f"Fixed seed set: {SEEDS}")

    base = load_row_splits(kind="base", prefix=PREFIX)
    wall_obs = load_row_splits(kind="wall_obstruction", prefix=PREFIX)
    wall_ctx = load_row_splits(kind="wall_context", prefix=PREFIX)

    model_dir = MODEL_DIR / "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_ladder = [
        ("XGB_GEOM_ONLY", base, FEATURES_GEOM_ONLY, False, "xgb_geom_only"),
        ("XGB_WALL_OBS", wall_obs, FEATURES_WALL_OBS, False, "xgb_wall_obs"),
        ("XGB_WALL_CONTEXT", wall_ctx, FEATURES_WALL_OBS_CONTEXT, False, "xgb_wall_context"),
        ("XGB_GEOM_ONLY_SW", base, FEATURES_GEOM_ONLY, True, "xgb_geom_only_sw"),
        ("XGB_WALL_OBS_SW", wall_obs, FEATURES_WALL_OBS, True, "xgb_wall_obs_sw"),
        ("XGB_WALL_CONTEXT_SW", wall_ctx, FEATURES_WALL_OBS_CONTEXT, True, "xgb_wall_context_sw"),
    ]

    validation_macro_all = []
    test_macro_all = []
    model_metadata = []

    for seed in SEEDS:
        set_all_seeds(seed)
        log(PREFIX, "=" * 78)
        log(PREFIX, f"Starting XGBoost seed {seed}.")

        for model_name, splits, features, weighted, out_prefix in feature_ladder:
            log(PREFIX, "-" * 78)
            log(PREFIX, f"Training {model_name} | seed={seed} | weighted={weighted} | features={len(features)}")

            rss_model, rtt_model, val_pred, test_pred = train_pair(
                train_df=splits["subtrain"],
                val_df=splits["val"],
                test_df=splits["test"],
                features=features,
                model_name=model_name,
                seed=seed,
                weighted=weighted,
            )

            joblib.dump(rss_model, model_dir / f"{model_name}_RSS_seed{seed}.joblib")
            joblib.dump(rtt_model, model_dir / f"{model_name}_RTT_seed{seed}.joblib")

            _, _, val_metrics = save_prediction_bundle(
                row_df=splits["val"],
                y_pred=val_pred,
                model=model_name,
                family=FAMILY,
                prefix=f"{out_prefix}_seed{seed}",
                split_name="val",
                seed=seed,
            )
            validation_macro_all.append(extract_scenario_macro_metrics(val_metrics, seed=seed))

            _, _, test_metrics = save_prediction_bundle(
                row_df=splits["test"],
                y_pred=test_pred,
                model=model_name,
                family=FAMILY,
                prefix=f"{out_prefix}_seed{seed}",
                split_name="test",
                seed=seed,
            )
            test_macro_all.append(extract_scenario_macro_metrics(test_metrics, seed=seed))

            model_metadata.append({
                "model": model_name,
                "features": features,
                "weighted": weighted,
                "prefix": out_prefix,
                "seed": int(seed),
            })

            if weighted:
                audit = splits["subtrain"].copy()
                audit["sample_weight"] = scenario_balanced_weights(splits["subtrain"])
                weight_audit = audit.groupby("scenario", as_index=False).agg(
                    n_subtrain=("sample_weight", "size"),
                    mean_weight=("sample_weight", "mean"),
                    total_weight=("sample_weight", "sum"),
                )
                weight_audit.to_csv(
                    RESULTS_TABLE_DIR / f"{out_prefix}_seed{seed}_scenario_weight_audit.csv",
                    index=False,
                )

    validation_macro_all = pd.concat(validation_macro_all, ignore_index=True)
    test_macro_all = pd.concat(test_macro_all, ignore_index=True)

    validation_macro_all.to_csv(RESULTS_TABLE_DIR / "xgboost_multiseed_validation_metrics.csv", index=False)
    test_macro_all.to_csv(RESULTS_TABLE_DIR / "xgboost_multiseed_test_metrics.csv", index=False)

    test_summary = summarize_seed_metrics(test_macro_all)
    test_summary.to_csv(RESULTS_TABLE_DIR / "xgboost_multiseed_test_summary_mean_std.csv", index=False)

    representative = select_representative_seeds_for_models(validation_macro_all)
    representative.to_csv(RESULTS_TABLE_DIR / "xgboost_representative_seed_for_cdf.csv", index=False)

    prefix_lookup = {model_name: out_prefix for model_name, _, _, _, out_prefix in feature_ladder}
    for _, row in representative.iterrows():
        copy_representative_predictions(
            model_prefix=prefix_lookup[row["model"]],
            representative_seed=int(row["representative_seed"]),
        )

    save_json(
        {
            "family": FAMILY,
            "config": {**XGB_BASE_CONFIG, "random_state": "varied_by_SEEDS"},
            "seeds": SEEDS,
            "primary_metric": "scenario-macro RMSE",
            "secondary_metrics": ["scenario-macro MAE", "scenario-macro P95"],
            "representative_seed_rule": "validation-median RMSE_macro mean across available level-target pairs",
            "models": model_metadata,
        },
        model_dir / "xgboost_metadata.json",
    )

    log(PREFIX, "training/train_xgboost.py completed successfully.")


if __name__ == "__main__":
    main()
