# ============================================================
# performance_metrics/compute_metrics.py
# ------------------------------------------------------------
# Purpose:
# - Compute final paper-grade metrics from saved prediction CSV files.
# - Replace notebook-memory dependencies such as rf_metrics,
#   xgb_scenario_metrics, gaussian_scenario_metrics_table,
#   wall_xgb_predictions, and test_df.
# - Support the agreed multi-seed protocol without changing the scientific
#   meaning of the saved predictions.
#
# Scientific protocol:
# - This script does not train, tune, or select hyperparameters.
# - It reads already-saved TEST prediction CSV files only.
# - It never fabricates missing predictions.
# - It does not downsample or alter the official test data.
# - Micro metrics are computed over all available test samples.
# - Scenario-macro metrics are computed per scenario, then averaged.
# - Scenario-macro RMSE is the primary comparison metric because the
#   test set is strongly scenario-imbalanced.
# - Scenario-macro MAE and P95 are reported as secondary metrics.
# - Bias and R² are intentionally excluded from the final reporting layer.
# - Final comparison/CDF registry uses representative-seed prediction files
#   only. The representative seed is selected upstream from validation-median
#   RMSE and is used only to keep CDF figures readable.
# - Multi-seed stability is reported separately as mean ± standard deviation
#   over all saved seed-specific test prediction files.
#
# Expected prediction sources:
# - Main final comparison:
#       output_csvs/predictions/*representative*test_predictions.csv
# - Multi-seed stability reporting:
#       output_csvs/predictions/*_seed*_test_predictions.csv
#
# Supported prediction schemas:
# 1) Wide multi-target schema:
#       scenario, family, model, level,
#       RSS_true, RSS_pred, RTT_true, RTT_pred
#
# 2) Wide single-target multi-model schema, used by CNN/LAMS:
#       scenario, RSS_true, RSS_pred_MODEL_A, RSS_pred_MODEL_B
#       scenario, RTT_true, RTT_pred_MODEL_A, RTT_pred_MODEL_B
#
# 3) Long final-reporting schema:
#       level, target, family, model, scenario, unit_id, y_true, y_pred
#
# Outputs:
# - output_csvs/predictions/_prediction_registry_long.csv
# - results/tables/prediction_file_audit.csv
# - results/tables/detailed_micro_scenario_macro_metrics_test.csv
# - results/tables/all_candidate_metrics_test.csv
# - results/tables/selected_best_model_per_family_test.csv
# - results/tables/multiseed_prediction_file_audit.csv
# - results/tables/multiseed_test_metrics_all_seeds.csv
# - results/tables/multiseed_test_summary_mean_std.csv
# ============================================================

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Section 1 — Repository paths and configuration
# ------------------------------------------------------------
# Purpose:
# - Resolve all paths relative to the repository root.
# - Avoid fragile absolute Windows paths.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = REPO_ROOT / "output_csvs" / "predictions"
RESULT_TABLE_DIR = REPO_ROOT / "results" / "tables"

RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_LONG_COLUMNS = {
    "level",
    "target",
    "family",
    "model",
    "scenario",
    "unit_id",
    "y_true",
    "y_pred",
}

UNIT_ID_CANDIDATES = [
    "lams_id",
    "AP_id_raw",
    "AP_index_usable",
    "AP_x",
    "AP_y",
    "RX_x",
    "RX_y",
    "row_pos_within_split",
]

SEED_RE = re.compile(r"(?:^|_)seed(\d+)(?:_|\.)", flags=re.IGNORECASE)
REPORT_METRICS = ["RMSE", "MAE", "P95"]
REPORT_MACRO_METRICS = ["RMSE_macro", "MAE_macro", "P95_macro"]


# ============================================================
# Section 2 — Logging
# ============================================================

def log(message: str) -> None:
    print(f"[compute_metrics] {message}", flush=True)


# ============================================================
# Section 3 — Prediction-file discovery
# ------------------------------------------------------------
# Purpose:
# - Separate representative files from seed-specific files.
# - Prevent old single-seed or helper CSVs from contaminating final tables.
# ============================================================

def is_helper_csv(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.startswith("_")
        or "audit" in name
        or "metric" in name
        or "summary" in name
        or "history" in name
        or "selected" in name
    )


def is_seed_prediction_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        "test_predictions" in name
        and SEED_RE.search(name) is not None
        and "representative" not in name
        and not is_helper_csv(path)
    )


def is_representative_prediction_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        "representative" in name
        and "test_predictions" in name
        and not is_helper_csv(path)
    )


def list_prediction_files(kind: str) -> List[Path]:
    all_files = sorted(PREDICTION_DIR.glob("*.csv"))

    if kind == "representative":
        files = [p for p in all_files if is_representative_prediction_file(p)]
    elif kind == "seed":
        files = [p for p in all_files if is_seed_prediction_file(p)]
    else:
        raise ValueError(f"Unsupported prediction file kind: {kind}")

    if not files:
        expected = "*representative*test_predictions.csv" if kind == "representative" else "*_seed*_test_predictions.csv"
        raise FileNotFoundError(
            f"No {kind} prediction files found in: {PREDICTION_DIR}\n"
            f"Expected pattern: {expected}\n"
            "Run the updated multi-seed training scripts first."
        )

    return files


def extract_seed_from_source(source_name: str) -> Optional[int]:
    match = SEED_RE.search(source_name)
    if match:
        return int(match.group(1))
    return None


# ============================================================
# Section 4 — Prediction-file audit helpers
# ------------------------------------------------------------
# Purpose:
# - Keep track of which files were read, accepted, partially accepted,
#   or skipped.
#
# Scientific protocol:
# - Files with unclear schemas are not forced into the registry.
# - Skipped files are listed in the audit table for transparency.
# ============================================================

def infer_family_from_model_or_file(model_name: str, source_name: str) -> str:
    text = f"{model_name} {source_name}".lower()

    if "randomforest" in text or "random_forest" in text or re.search(r"\brf\b", text):
        return "RF"
    if "xgb" in text or "xgboost" in text:
        return "XGBoost"
    if "mlp" in text or "gaussian" in text:
        return "MLP"
    if "lams" in text or "cnn" in text:
        return "CNN_LAMS"
    if "gnn" in text or "graphsage" in text:
        return "GNN"

    return "Unknown"


def infer_level_from_file_or_df(source_name: str, df: pd.DataFrame) -> str:
    if "level" in df.columns:
        values = df["level"].dropna().astype(str).str.lower().unique().tolist()
        if len(values) == 1 and values[0] in {"row", "geometry"}:
            return values[0]

    name = source_name.lower()

    if "geometry" in name or "geo" in name:
        return "geometry"
    if "row" in name:
        return "row"
    if "lams" in name or "gnn" in name or "graphsage" in name:
        return "geometry"

    raise ValueError(
        f"Could not infer prediction level for {source_name}. "
        "The file should contain a level column or include row/geometry in its filename."
    )


def make_unit_ids(df: pd.DataFrame, source_name: str) -> pd.Series:
    available = [c for c in UNIT_ID_CANDIDATES if c in df.columns]

    if available:
        return df[available].astype(str).agg("|".join, axis=1)

    return pd.Series(
        [f"{source_name}:row_{i}" for i in range(len(df))],
        index=df.index,
    )


def normalize_target_name(raw: str) -> str:
    text = str(raw).strip().upper()
    if text.startswith("RSS"):
        return "RSS"
    if text.startswith("RTT"):
        return "RTT"
    raise ValueError(f"Unsupported target name: {raw}")


# ============================================================
# Section 5 — Schema converters
# ------------------------------------------------------------
# Purpose:
# - Convert different valid saved-prediction formats into one strict
#   long registry:
#       level,target,family,model,scenario,unit_id,y_true,y_pred
#
# Scientific protocol:
# - Conversion only changes table shape, not prediction values.
# - Unknown schemas are rejected rather than guessed aggressively.
# ============================================================

def attach_source_metadata(out: pd.DataFrame, source_name: str, source_kind: str) -> pd.DataFrame:
    out = out.copy()
    out["source_csv"] = source_name
    out["source_kind"] = source_kind
    seed = extract_seed_from_source(source_name)
    if seed is not None:
        out["seed"] = int(seed)
    elif "seed" in out.columns:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    else:
        out["seed"] = np.nan
    return out


def convert_long_schema(df: pd.DataFrame, source_name: str, source_kind: str) -> pd.DataFrame:
    missing = REQUIRED_LONG_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} missing long-schema columns: {sorted(missing)}")

    keep = list(REQUIRED_LONG_COLUMNS)
    if "seed" in df.columns:
        keep.append("seed")

    out = df[keep].copy()
    out["level"] = out["level"].astype(str).str.strip().str.lower()
    out["target"] = out["target"].map(normalize_target_name)
    out["family"] = out["family"].astype(str).str.strip()
    out["model"] = out["model"].astype(str).str.strip()
    out["scenario"] = out["scenario"].astype(str).str.strip()
    out["unit_id"] = out["unit_id"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    return attach_source_metadata(out, source_name, source_kind)


def convert_wide_standard_schema(df: pd.DataFrame, source_name: str, source_kind: str) -> pd.DataFrame:
    required = {"scenario", "family", "model", "RSS_true", "RSS_pred", "RTT_true", "RTT_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} missing wide-standard columns: {sorted(missing)}")

    level = infer_level_from_file_or_df(source_name, df)
    unit_ids = make_unit_ids(df, source_name)
    rows = []

    for target, true_col, pred_col in [
        ("RSS", "RSS_true", "RSS_pred"),
        ("RTT", "RTT_true", "RTT_pred"),
    ]:
        tmp = pd.DataFrame({
            "level": level,
            "target": target,
            "family": df["family"].astype(str).str.strip(),
            "model": df["model"].astype(str).str.strip(),
            "scenario": df["scenario"].astype(str).str.strip(),
            "unit_id": unit_ids,
            "y_true": pd.to_numeric(df[true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[pred_col], errors="coerce"),
        })
        rows.append(tmp)

    return attach_source_metadata(pd.concat(rows, ignore_index=True), source_name, source_kind)


def parse_single_target_prediction_column(col_name: str) -> Optional[Tuple[str, str]]:
    text = str(col_name)
    for target in ["RSS", "RTT"]:
        prefix = f"{target}_pred_"
        if text.startswith(prefix):
            model_name = text[len(prefix):].strip()
            if model_name:
                return target, model_name
    return None


def convert_wide_single_target_multi_model_schema(df: pd.DataFrame, source_name: str, source_kind: str) -> pd.DataFrame:
    if "scenario" not in df.columns:
        raise ValueError(f"{source_name} does not contain scenario column.")

    level = infer_level_from_file_or_df(source_name, df)
    unit_ids = make_unit_ids(df, source_name)
    rows = []

    for col in df.columns:
        parsed = parse_single_target_prediction_column(col)
        if parsed is None:
            continue

        target, model_name = parsed
        true_col = f"{target}_true"

        if true_col not in df.columns:
            alternatives = [
                f"{target}_true_mean",
                f"{target}_true_mean_m",
                f"{target}_mean",
                f"{target}_mean_m",
            ]
            existing = [c for c in alternatives if c in df.columns]
            if not existing:
                raise ValueError(
                    f"{source_name}: prediction column {col} exists, but no matching "
                    f"truth column was found. Expected {true_col} or one of {alternatives}."
                )
            true_col = existing[0]

        family = infer_family_from_model_or_file(model_name, source_name)

        tmp = pd.DataFrame({
            "level": level,
            "target": target,
            "family": family,
            "model": model_name,
            "scenario": df["scenario"].astype(str).str.strip(),
            "unit_id": unit_ids,
            "y_true": pd.to_numeric(df[true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[col], errors="coerce"),
        })
        rows.append(tmp)

    if not rows:
        raise ValueError(f"{source_name} does not match single-target multi-model schema.")

    return attach_source_metadata(pd.concat(rows, ignore_index=True), source_name, source_kind)


def convert_prediction_file(path: Path, source_kind: str) -> Tuple[pd.DataFrame, dict]:
    source_name = path.name
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]

    audit = {
        "source_csv": source_name,
        "source_kind": source_kind,
        "seed": extract_seed_from_source(source_name),
        "n_input_rows": int(len(raw)),
        "status": "accepted",
        "schema": None,
        "message": "",
    }

    converters = [
        ("long", convert_long_schema),
        ("wide_standard", convert_wide_standard_schema),
        ("wide_single_target_multi_model", convert_wide_single_target_multi_model_schema),
    ]

    errors = []

    for schema_name, converter in converters:
        try:
            converted = converter(raw, source_name, source_kind)
            audit["schema"] = schema_name
            audit["n_registry_rows"] = int(len(converted))
            return converted, audit
        except Exception as exc:
            errors.append(f"{schema_name}: {exc}")

    audit["status"] = "skipped"
    audit["schema"] = "unknown"
    audit["n_registry_rows"] = 0
    audit["message"] = " | ".join(errors)

    return pd.DataFrame(), audit


# ============================================================
# Section 6 — Registry loading
# ------------------------------------------------------------
# Purpose:
# - Load either representative prediction files or all seed-specific files.
# - Keep transparent audit trails for both reporting modes.
# ============================================================

def load_prediction_registry(kind: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = list_prediction_files(kind)
    log(f"Found {len(files)} {kind} prediction CSV file(s) in: {PREDICTION_DIR}")

    tables = []
    audit_rows = []

    for idx, path in enumerate(files, start=1):
        log(f"Reading {kind} prediction file {idx}/{len(files)}: {path.name}")
        table, audit = convert_prediction_file(path, source_kind=kind)
        audit_rows.append(audit)

        if not table.empty:
            tables.append(table)
            log(f"  accepted as {audit['schema']}: {len(table):,} registry rows")
        else:
            log(f"  skipped: {audit['message']}")

    audit_df = pd.DataFrame(audit_rows)

    if not tables:
        raise RuntimeError(f"No valid {kind} prediction tables could be loaded.")

    registry = pd.concat(tables, ignore_index=True)
    registry = clean_registry(registry)

    return registry.reset_index(drop=True), audit_df


def clean_registry(registry: pd.DataFrame) -> pd.DataFrame:
    registry = registry.copy()

    registry["level"] = registry["level"].astype(str).str.lower().str.strip()
    registry["target"] = registry["target"].map(normalize_target_name)
    registry["family"] = registry["family"].astype(str).str.strip()
    registry["model"] = registry["model"].astype(str).str.strip()
    registry["scenario"] = registry["scenario"].astype(str).str.strip()
    registry["unit_id"] = registry["unit_id"].astype(str)
    registry["y_true"] = pd.to_numeric(registry["y_true"], errors="coerce")
    registry["y_pred"] = pd.to_numeric(registry["y_pred"], errors="coerce")

    if "seed" not in registry.columns:
        registry["seed"] = np.nan
    registry["seed"] = pd.to_numeric(registry["seed"], errors="coerce")

    before = len(registry)
    registry = registry.dropna(
        subset=["level", "target", "family", "model", "scenario", "unit_id", "y_true", "y_pred"]
    ).copy()
    after = len(registry)

    if after < before:
        log(f"Dropped {before - after:,} registry rows with missing/non-numeric y_true or y_pred.")

    registry["signed_error"] = registry["y_pred"] - registry["y_true"]
    registry["abs_error"] = registry["signed_error"].abs()

    dedup_cols = [
        "level", "target", "family", "model", "scenario", "unit_id",
        "y_true", "y_pred", "source_csv", "source_kind",
    ]
    duplicate_count = int(registry.duplicated(subset=dedup_cols).sum())
    if duplicate_count > 0:
        log(f"Removing {duplicate_count:,} exact duplicated prediction rows.")
        registry = registry.drop_duplicates(subset=dedup_cols).copy()

    return registry


# ============================================================
# Section 7 — Metric computation
# ------------------------------------------------------------
# Purpose:
# - Compute micro, per-scenario, and scenario-macro metrics.
#
# Trusted libraries:
# - scikit-learn is used for MAE and MSE/RMSE.
# - numpy is used for P95.
# ============================================================

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Metric length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    err = y_pred - y_true
    abs_err = np.abs(err)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "P95": float(np.percentile(abs_err, 95)),
        "n": int(len(y_true)),
    }


def compute_all_metric_tables(registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detailed_rows = []
    candidate_rows = []
    group_cols = ["level", "target", "family", "model"]

    for keys, group in registry.groupby(group_cols, sort=True):
        level, target, family, model = keys
        group = group.dropna(subset=["y_true", "y_pred"]).copy()
        if group.empty:
            continue

        micro = metric_dict(group["y_true"], group["y_pred"])
        detailed_rows.append({
            "level": level,
            "target": target,
            "family": family,
            "model": model,
            "aggregation": "micro",
            "scenario": "ALL",
            **micro,
        })

        scenario_metric_rows = []
        for scenario, gs in group.groupby("scenario", sort=True):
            m = metric_dict(gs["y_true"], gs["y_pred"])
            scenario_metric_rows.append({"scenario": scenario, **m})
            detailed_rows.append({
                "level": level,
                "target": target,
                "family": family,
                "model": model,
                "aggregation": "scenario",
                "scenario": scenario,
                **m,
            })

        scenario_metrics = pd.DataFrame(scenario_metric_rows)
        macro = {
            "MAE": float(scenario_metrics["MAE"].mean()),
            "RMSE": float(scenario_metrics["RMSE"].mean()),
            "P95": float(scenario_metrics["P95"].mean()),
            "n": int(scenario_metrics["scenario"].nunique()),
        }

        detailed_rows.append({
            "level": level,
            "target": target,
            "family": family,
            "model": model,
            "aggregation": "scenario_macro",
            "scenario": "MACRO",
            **macro,
        })

        candidate_rows.append({
            "level": level,
            "target": target,
            "family": family,
            "model": model,
            "MAE_micro": micro["MAE"],
            "RMSE_micro": micro["RMSE"],
            "P95_micro": micro["P95"],
            "MAE_macro": macro["MAE"],
            "RMSE_macro": macro["RMSE"],
            "P95_macro": macro["P95"],
            "n_total": int(len(group)),
            "n_scenarios": int(scenario_metrics["scenario"].nunique()),
            "scenarios": ",".join(sorted(group["scenario"].unique())),
            "source_csvs": ",".join(sorted(group["source_csv"].unique())),
        })

    detailed_metrics = pd.DataFrame(detailed_rows)
    candidate_metrics = pd.DataFrame(candidate_rows)

    if candidate_metrics.empty:
        raise RuntimeError("No candidate metrics were computed from the prediction registry.")

    eligibility = []
    for _, row in candidate_metrics.iterrows():
        subset = registry[
            (registry["level"] == row["level"])
            & (registry["target"] == row["target"])
        ]
        required_scenarios = set(subset["scenario"].unique())
        model_scenarios = set(str(row["scenarios"]).split(","))
        eligibility.append(required_scenarios.issubset(model_scenarios))

    candidate_metrics["eligible_all_available_scenarios"] = eligibility

    candidate_metrics = candidate_metrics.sort_values(
        ["level", "target", "family", "RMSE_macro", "P95_macro", "MAE_macro", "model"],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)

    detailed_metrics = detailed_metrics.sort_values(
        ["level", "target", "family", "model", "aggregation", "scenario"],
        ascending=True,
    ).reset_index(drop=True)

    return detailed_metrics, candidate_metrics


# ============================================================
# Section 8 — Best model per family
# ------------------------------------------------------------
# Purpose:
# - Select one representative per level-target-family for final CDF plots.
#
# Scientific protocol:
# - Selection is among already-trained candidate variants only.
# - It uses scenario-macro RMSE as the primary metric, with P95 and MAE as
#   deterministic tie-breakers.
# - It does not select seeds; seed-level reporting is handled separately.
# ============================================================

def select_best_per_family(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    eligible = candidate_metrics[candidate_metrics["eligible_all_available_scenarios"]].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible model covered all available scenarios within its level-target comparison."
        )

    selected = (
        eligible
        .sort_values(["RMSE_macro", "P95_macro", "MAE_macro", "model"], ascending=True)
        .groupby(["level", "target", "family"], as_index=False)
        .head(1)
        .copy()
        .sort_values(["level", "target", "family"])
        .reset_index(drop=True)
    )

    return selected


# ============================================================
# Section 9 — Multi-seed stability summary
# ------------------------------------------------------------
# Purpose:
# - Compute all-seed test metrics and summarize them as mean ± std.
# - Keep this separate from representative CDF reporting.
# ============================================================

def compute_seed_metric_table(seed_registry: pd.DataFrame) -> pd.DataFrame:
    if seed_registry["seed"].isna().any():
        missing = sorted(seed_registry.loc[seed_registry["seed"].isna(), "source_csv"].unique())
        raise RuntimeError(
            "Seed-specific registry contains files without parsable seed IDs. Examples: "
            + ", ".join(missing[:10])
        )

    rows = []
    group_cols = ["level", "target", "family", "model", "seed"]

    for keys, group in seed_registry.groupby(group_cols, sort=True):
        level, target, family, model, seed = keys
        scenario_rows = []
        for scenario, gs in group.groupby("scenario", sort=True):
            m = metric_dict(gs["y_true"], gs["y_pred"])
            scenario_rows.append({"scenario": scenario, **m})

        scenario_metrics = pd.DataFrame(scenario_rows)
        micro = metric_dict(group["y_true"], group["y_pred"])

        rows.append({
            "level": level,
            "target": target,
            "family": family,
            "model": model,
            "seed": int(seed),
            "MAE_micro": micro["MAE"],
            "RMSE_micro": micro["RMSE"],
            "P95_micro": micro["P95"],
            "MAE_macro": float(scenario_metrics["MAE"].mean()),
            "RMSE_macro": float(scenario_metrics["RMSE"].mean()),
            "P95_macro": float(scenario_metrics["P95"].mean()),
            "n_total": int(len(group)),
            "n_scenarios": int(scenario_metrics["scenario"].nunique()),
            "scenarios": ",".join(sorted(group["scenario"].unique())),
            "source_csvs": ",".join(sorted(group["source_csv"].unique())),
        })

    seed_metrics = pd.DataFrame(rows)
    if seed_metrics.empty:
        raise RuntimeError("No per-seed metrics were computed.")

    return seed_metrics.sort_values(
        ["level", "target", "family", "model", "seed"],
        ascending=True,
    ).reset_index(drop=True)


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["level", "target", "family", "model"]
    rows = []

    for keys, group in seed_metrics.groupby(group_cols, sort=True):
        level, target, family, model = keys
        row = {
            "level": level,
            "target": target,
            "family": family,
            "model": model,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].unique())),
        }

        for metric in [
            "RMSE_macro", "MAE_macro", "P95_macro",
            "RMSE_micro", "MAE_micro", "P95_micro",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0

        row["n_total_min"] = int(group["n_total"].min())
        row["n_total_max"] = int(group["n_total"].max())
        row["n_scenarios_min"] = int(group["n_scenarios"].min())
        row["n_scenarios_max"] = int(group["n_scenarios"].max())
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["level", "target", "family", "RMSE_macro_mean", "P95_macro_mean", "MAE_macro_mean", "model"],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)


# ============================================================
# Section 9B — Multi-seed model selection for final reporting
# ------------------------------------------------------------
# Purpose:
# - Select one model per level-target-family using mean performance
#   across seeds, not a single representative seed.
#
# Scientific protocol:
# - Primary criterion: RMSE_macro_mean.
# - Tie-breakers: P95_macro_mean, MAE_macro_mean, model name.
# - Representative prediction files are used later only for CDF plotting.
# ============================================================

def select_best_per_family_from_multiseed_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "level",
        "target",
        "family",
        "model",
        "RMSE_macro_mean",
        "MAE_macro_mean",
        "P95_macro_mean",
    }

    missing = required - set(seed_summary.columns)
    if missing:
        raise RuntimeError(
            "Multi-seed summary is missing required columns for final model selection: "
            + ", ".join(sorted(missing))
        )

    selected = (
        seed_summary
        .sort_values(
            ["level", "target", "family", "RMSE_macro_mean", "P95_macro_mean", "MAE_macro_mean", "model"],
            ascending=[True, True, True, True, True, True, True],
        )
        .groupby(["level", "target", "family"], as_index=False)
        .head(1)
        .copy()
        .sort_values(["level", "target", "family"])
        .reset_index(drop=True)
    )

    # Keep plotting compatibility with plot_error_cdf.py by exposing the
    # selected mean metrics under the standard column names.
    selected["RMSE_macro"] = selected["RMSE_macro_mean"]
    selected["MAE_macro"] = selected["MAE_macro_mean"]
    selected["P95_macro"] = selected["P95_macro_mean"]

    # Preserve available uncertainty/statistical columns for paper tables.
    keep_cols = [
        "level",
        "target",
        "family",
        "model",
        "n_seeds",
        "seeds",
        "RMSE_macro",
        "RMSE_macro_std",
        "MAE_macro",
        "MAE_macro_std",
        "P95_macro",
        "P95_macro_std",
        "RMSE_macro_mean",
        "MAE_macro_mean",
        "P95_macro_mean",
        "n_total_min",
        "n_total_max",
        "n_scenarios_min",
        "n_scenarios_max",
    ]

    selected = selected[[c for c in keep_cols if c in selected.columns]].copy()

    return selected

# ============================================================
# Section 10 — Main execution
# ============================================================

def main() -> None:
    t0 = time.time()

    log("Starting standalone final metric computation.")
    log(f"Repository root: {REPO_ROOT}")
    log(f"Prediction directory: {PREDICTION_DIR}")

    # -------------------------
    # Main final registry.
    # -------------------------
    log("Loading representative prediction files for final comparison and CDF registry.")
    registry, audit = load_prediction_registry(kind="representative")

    registry_path = PREDICTION_DIR / "_prediction_registry_long.csv"
    audit_path = RESULT_TABLE_DIR / "prediction_file_audit.csv"

    registry.to_csv(registry_path, index=False)
    audit.to_csv(audit_path, index=False)

    log(f"Saved normalized representative prediction registry: {registry_path}")
    log(f"Saved representative prediction-file audit: {audit_path}")
    log(f"Representative registry rows: {len(registry):,}")
    log("Models detected: " + ", ".join(sorted(registry["model"].astype(str).unique())))

    detailed_metrics, candidate_metrics = compute_all_metric_tables(registry)

    detailed_path = RESULT_TABLE_DIR / "detailed_micro_scenario_macro_metrics_test.csv"
    candidate_path = RESULT_TABLE_DIR / "all_candidate_metrics_test.csv"
    selected_path = RESULT_TABLE_DIR / "selected_best_model_per_family_test.csv"

    detailed_metrics.to_csv(detailed_path, index=False)
    candidate_metrics.to_csv(candidate_path, index=False)

    log(f"Saved detailed metrics: {detailed_path}")
    log(f"Saved all candidate metrics: {candidate_path}")

    # -------------------------
    # Multi-seed stability.
    # -------------------------
    log("Loading seed-specific prediction files for mean ± std stability reporting.")
    seed_registry, seed_audit = load_prediction_registry(kind="seed")

    seed_audit_path = RESULT_TABLE_DIR / "multiseed_prediction_file_audit.csv"
    seed_metrics_path = RESULT_TABLE_DIR / "multiseed_test_metrics_all_seeds.csv"
    seed_summary_path = RESULT_TABLE_DIR / "multiseed_test_summary_mean_std.csv"

    seed_audit.to_csv(seed_audit_path, index=False)
    seed_metrics = compute_seed_metric_table(seed_registry)
    seed_summary = summarize_seed_metrics(seed_metrics)

    seed_metrics.to_csv(seed_metrics_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)

    # Final paper/CDF model selection must be based on the multi-seed
    # mean summary, not on a single representative seed.
    selected_best = select_best_per_family_from_multiseed_summary(seed_summary)
    selected_best.to_csv(selected_path, index=False)

    log(f"Saved multi-seed prediction-file audit: {seed_audit_path}")
    log(f"Saved all-seed test metrics: {seed_metrics_path}")
    log(f"Saved mean ± std seed summary: {seed_summary_path}")
    log(f"Saved selected-best table from multi-seed summary: {selected_path}")


    log("Selected representative model per family for CDF/main comparison:")
    cols = [
        "level", "target", "family", "model", "n_seeds",
        "RMSE_macro", "RMSE_macro_std",
        "MAE_macro", "MAE_macro_std",
        "P95_macro", "P95_macro_std",
        ]
    
    printable = selected_best[[c for c in cols if c in selected_best.columns]].copy()
    print(printable.to_string(index=False), flush=True)

    log("Multi-seed mean ± std summary, sorted by RMSE_macro_mean:")
    summary_cols = [
        "level", "target", "family", "model", "n_seeds",
        "RMSE_macro_mean", "RMSE_macro_std",
        "MAE_macro_mean", "MAE_macro_std",
        "P95_macro_mean", "P95_macro_std",
    ]
    printable_summary = seed_summary[[c for c in summary_cols if c in seed_summary.columns]].copy()
    print(printable_summary.to_string(index=False), flush=True)

    elapsed = time.time() - t0
    log(f"performance_metrics/compute_metrics.py completed successfully in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
