# ============================================================
# plotting/plot_error_cdf.py
# ------------------------------------------------------------
# Section 100 — Final CDFs from saved representative prediction CSVs
# ------------------------------------------------------------
# Purpose:
# - Rebuild the final paper-grade absolute-error CDF plots from saved
#   prediction CSVs, not from notebook memory.
# - Run as a standalone Python script inside the repository pipeline.
#
# Scientific rule:
# - Row-level CDF: repeated scan-row absolute errors.
# - Geometry-level CDF: unique AP-RP link absolute errors.
# - Best model per family is selected by lowest scenario-macro RMSE.
# - Only representative-seed TEST predictions are used for CDF plotting.
# - The representative seed is selected upstream using validation-median
#   scenario-macro RMSE and is not a best-test seed.
# - Main quantitative claims should come from mean ± std multi-seed tables.
# - No model is trained, tuned, or selected using plots.
# - No output is fabricated if required prediction data are missing.
#
# Standalone adaptation:
# - Paths are resolved relative to the repository root.
# - The script uses the normalized representative registry and selected-best
#   table produced by performance_metrics/compute_metrics.py.
# - If those files are unavailable, it falls back to strict long-schema
#   representative prediction CSVs only.
# - Notebook-only display() calls are replaced by terminal-safe printing.
#
# Inputs, preferred:
# - output_csvs/predictions/_prediction_registry_long.csv
# - results/tables/selected_best_model_per_family_test.csv
#
# Inputs, fallback:
# - output_csvs/predictions/*representative*test_predictions.csv with long
#   prediction schema:
#       level,target,family,model,scenario,unit_id,y_true,y_pred
#
# Outputs:
# - results/figures/row_level_best_family_abs_error_cdf.png
# - results/figures/row_level_best_family_abs_error_cdf.pdf
# - results/figures/geometry_level_best_family_abs_error_cdf.png
# - results/figures/geometry_level_best_family_abs_error_cdf.pdf
# - results/tables/cdf_selected_models_used.csv
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import matplotlib

# Use a non-interactive backend so the script works reliably from CMD,
# terminals, CI jobs, and run_pipeline.py.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = REPO_ROOT / "output_csvs" / "predictions"
RESULT_FIG_DIR = REPO_ROOT / "results" / "figures"
RESULT_TABLE_DIR = REPO_ROOT / "results" / "tables"

REGISTRY_PATH = PRED_DIR / "_prediction_registry_long.csv"
SELECTED_PATH = RESULT_TABLE_DIR / "selected_best_model_per_family_test.csv"
ALL_METRICS_PATH = RESULT_TABLE_DIR / "all_candidate_metrics_test.csv"
CDF_SELECTED_PATH = RESULT_TABLE_DIR / "cdf_selected_models_used.csv"

RESULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "level", "target", "family", "model",
    "scenario", "unit_id", "y_true", "y_pred",
}

TARGET_UNITS = {
    "RSS": "dB",
    "RTT": "m",
}


# ============================================================
# Logging
# ============================================================

def log(message: str) -> None:
    print(f"[plot_error_cdf] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[plot_error_cdf][WARNING] {message}", flush=True)


# ============================================================
# Helper functions
# ============================================================

def _is_representative_prediction_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        "representative" in name
        and "test_predictions" in name
        and not name.startswith("_")
        and "audit" not in name
        and "metric" not in name
        and "summary" not in name
    )


def _standardize_prediction_table(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Standardize one saved representative prediction CSV.

    Expected schema:
        level,target,family,model,scenario,unit_id,y_true,y_pred

    This fallback function is intentionally strict. The preferred pathway is
    to use _prediction_registry_long.csv from compute_metrics.py, because that
    script already normalizes all supported prediction formats.
    """

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    keep = [
        "level", "target", "family", "model",
        "scenario", "unit_id", "y_true", "y_pred",
    ]
    df = df[keep].copy()

    df["level"] = df["level"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].astype(str).str.strip().str.upper()
    df["family"] = df["family"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["unit_id"] = df["unit_id"].astype(str).str.strip()

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    before = len(df)
    df = df.dropna(
        subset=["level", "target", "family", "model", "scenario", "unit_id", "y_true", "y_pred"]
    )
    after = len(df)

    if after < before:
        log(f"{source_name}: dropped {before - after:,} rows with invalid values.")

    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
    df["signed_error"] = df["y_pred"] - df["y_true"]
    df["source_csv"] = source_name

    return df


def _validate_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the representative prediction registry.

    The registry must contain one row per representative prediction and must
    not depend on any notebook variables.
    """

    registry = registry.copy()
    registry.columns = [str(c).strip() for c in registry.columns]

    missing = REQUIRED_COLUMNS - set(registry.columns)
    if missing:
        raise RuntimeError(
            f"Prediction registry is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(registry.columns)}"
        )

    registry["level"] = registry["level"].astype(str).str.strip().str.lower()
    registry["target"] = registry["target"].astype(str).str.strip().str.upper()
    registry["family"] = registry["family"].astype(str).str.strip()
    registry["model"] = registry["model"].astype(str).str.strip()
    registry["scenario"] = registry["scenario"].astype(str).str.strip()
    registry["unit_id"] = registry["unit_id"].astype(str).str.strip()

    registry["y_true"] = pd.to_numeric(registry["y_true"], errors="coerce")
    registry["y_pred"] = pd.to_numeric(registry["y_pred"], errors="coerce")

    before = len(registry)
    registry = registry.dropna(
        subset=["level", "target", "family", "model", "scenario", "unit_id", "y_true", "y_pred"]
    )
    after = len(registry)

    if after < before:
        log(f"Dropped {before - after:,} invalid registry rows before plotting.")

    if "abs_error" not in registry.columns:
        registry["abs_error"] = (registry["y_true"] - registry["y_pred"]).abs()

    if "signed_error" not in registry.columns:
        registry["signed_error"] = registry["y_pred"] - registry["y_true"]

    # Guard against duplicate rows caused by accidentally reading both raw
    # prediction files and an already-normalized registry.
    dedup_cols = [
        "level", "target", "family", "model",
        "scenario", "unit_id", "y_true", "y_pred",
    ]
    before = len(registry)
    registry = registry.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    after = len(registry)

    if after < before:
        log(f"Removed {before - after:,} duplicate registry rows before plotting.")

    if registry.empty:
        raise RuntimeError("Prediction registry is empty after validation.")

    return registry


def _load_prediction_registry(pred_dir: Path) -> pd.DataFrame:
    """
    Load representative predictions into one registry.

    Preferred route:
    - Use _prediction_registry_long.csv from compute_metrics.py. In the
      multi-seed protocol, this registry is built from representative files
      only.

    Fallback route:
    - Load strict long-schema representative prediction CSVs directly.
    - Skip seed-specific files to avoid mixing repeated seeds into CDF plots.
    """

    if REGISTRY_PATH.exists():
        log(f"Loading normalized representative prediction registry: {REGISTRY_PATH}")
        registry = pd.read_csv(REGISTRY_PATH)
        registry = _validate_registry(registry)
        log(f"Loaded {len(registry):,} representative prediction rows from normalized registry.")
        return registry

    log(
        "Normalized registry not found. Falling back to representative prediction "
        f"CSV files in: {pred_dir}"
    )

    files = sorted(f for f in pred_dir.glob("*.csv") if _is_representative_prediction_file(f))

    if not files:
        raise FileNotFoundError(
            f"No representative prediction CSV files found in: {pred_dir}\n"
            "Expected _prediction_registry_long.csv or *representative*test_predictions.csv.\n"
            "Run performance_metrics/compute_metrics.py first."
        )

    tables = []
    rejected = []

    for i, f in enumerate(files, start=1):
        log(f"Loading fallback representative prediction file {i}/{len(files)}: {f.name}")
        raw = pd.read_csv(f)

        try:
            tables.append(_standardize_prediction_table(raw, f.name))
        except ValueError as exc:
            rejected.append((f.name, str(exc).splitlines()[0]))
            warn(f"Skipped non-long-schema representative file: {f.name}")

    if not tables:
        details = "\n".join(f"- {name}: {reason}" for name, reason in rejected[:20])
        raise RuntimeError(
            "No representative prediction CSV could be standardized by plot_error_cdf.py.\n"
            "Run performance_metrics/compute_metrics.py first to create "
            "output_csvs/predictions/_prediction_registry_long.csv.\n"
            f"Rejected examples:\n{details}"
        )

    registry = pd.concat(tables, ignore_index=True)
    registry = _validate_registry(registry)

    log(f"Loaded {len(registry):,} representative prediction rows from {len(tables)} fallback CSV files.")
    return registry


def _compute_candidate_metrics(registry: pd.DataFrame) -> pd.DataFrame:
    """
    Compute micro and scenario-macro test metrics for each candidate model.

    Macro metrics are computed over scenario-wise metrics to avoid dominance
    by scenarios with more AP-RP samples.
    """

    rows = []
    group_cols = ["level", "target", "family", "model"]

    for keys, g in registry.groupby(group_cols, dropna=False):
        level, target, family, model = keys

        scenario_metrics = []
        for scenario, gs in g.groupby("scenario", sort=True):
            err = gs["y_pred"] - gs["y_true"]
            abs_err = err.abs()

            scenario_metrics.append({
                "scenario": scenario,
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "P95": float(np.percentile(abs_err, 95)),
                "n": int(len(gs)),
            })

        sm = pd.DataFrame(scenario_metrics)

        err_all = g["y_pred"] - g["y_true"]
        abs_all = err_all.abs()
        observed_scenarios = set(g["scenario"].unique())

        rows.append({
            "level": level,
            "target": target,
            "family": family,
            "model": model,
            "MAE_micro": float(abs_all.mean()),
            "RMSE_micro": float(np.sqrt(np.mean(err_all ** 2))),
            "P95_micro": float(np.percentile(abs_all, 95)),
            "MAE_macro": float(sm["MAE"].mean()),
            "RMSE_macro": float(sm["RMSE"].mean()),
            "P95_macro": float(sm["P95"].mean()),
            "n_total": int(len(g)),
            "n_scenarios": int(len(observed_scenarios)),
            "scenarios": ",".join(sorted(observed_scenarios)),
        })

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        raise RuntimeError("No candidate metrics could be computed from prediction registry.")

    eligible_flags = []
    for _, row in metrics.iterrows():
        subset = registry[
            (registry["level"] == row["level"]) &
            (registry["target"] == row["target"])
        ]
        required_scenarios = set(subset["scenario"].unique())
        model_scenarios = set(str(row["scenarios"]).split(","))
        eligible_flags.append(required_scenarios.issubset(model_scenarios))

    metrics["eligible_all_available_scenarios"] = eligible_flags

    return metrics.sort_values(
        ["level", "target", "family", "RMSE_macro", "P95_macro", "MAE_macro", "model"],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)


def _select_best_per_family(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best model per level-target-family using lowest RMSE_macro.

    This does not select one global winner. It preserves one candidate per
    model family for the final family-comparison CDF plots.
    """

    eligible = metrics[metrics["eligible_all_available_scenarios"]].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible models found. Check whether prediction CSVs cover "
            "the same scenarios within each level and target."
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


def _load_or_compute_selection(registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load selected-best models from compute_metrics.py if available.

    If unavailable, compute the same selection rule locally from the
    representative registry.
    """

    if ALL_METRICS_PATH.exists():
        log(f"Loading all candidate metrics: {ALL_METRICS_PATH}")
        all_metrics = pd.read_csv(ALL_METRICS_PATH)
    else:
        log("All candidate metrics table not found. Computing metrics locally for plotting.")
        all_metrics = _compute_candidate_metrics(registry)
        all_metrics.to_csv(ALL_METRICS_PATH, index=False)
        log(f"Saved all candidate metrics: {ALL_METRICS_PATH}")

    if SELECTED_PATH.exists():
        log(f"Loading selected-best table: {SELECTED_PATH}")
        selected = pd.read_csv(SELECTED_PATH)
    else:
        log("Selected-best table not found. Selecting best model per family locally.")
        selected = _select_best_per_family(all_metrics)
        selected.to_csv(SELECTED_PATH, index=False)
        log(f"Saved selected-best table: {SELECTED_PATH}")

    required_selected_cols = {"level", "target", "family", "model", "RMSE_macro"}
    missing = required_selected_cols - set(selected.columns)

    if missing:
        raise RuntimeError(
            f"Selected-best table is missing required columns: {sorted(missing)}\n"
            f"Path: {SELECTED_PATH}"
        )

    selected["level"] = selected["level"].astype(str).str.strip().str.lower()
    selected["target"] = selected["target"].astype(str).str.strip().str.upper()
    selected["family"] = selected["family"].astype(str).str.strip()
    selected["model"] = selected["model"].astype(str).str.strip()

    return all_metrics, selected


def _empirical_cdf_values(errors: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical CDF values for absolute errors.

    No model or fitting is done here. This is only the standard empirical
    distribution of observed test absolute errors.
    """

    x = np.sort(errors.dropna().to_numpy(dtype=float))

    if len(x) == 0:
        return x, x

    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def _format_label(row: pd.Series) -> str:
    """
    Compact curve label for final paper-grade CDF plots.
    """

    rmse = row.get("RMSE_macro", np.nan)

    if pd.notna(rmse):
        return f"{row['family']} | {row['model']} (RMSE_macro={float(rmse):.3f})"

    return f"{row['family']} | {row['model']}"


def _plot_level_cdf(registry: pd.DataFrame, selected: pd.DataFrame, level: str, out_stem: str) -> None:
    """
    Plot final 1x2 CDF figure for one evaluation level:
        left  = RSS
        right = RTT
    """

    log(f"Plotting {level}-level CDF figure.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for ax, target in zip(axes, ["RSS", "RTT"]):
        sel = selected[
            (selected["level"] == level) &
            (selected["target"] == target)
        ].copy()

        if sel.empty:
            ax.set_title(f"{level.capitalize()} {target}: no eligible predictions")
            ax.set_xlabel(f"Absolute error ({TARGET_UNITS.get(target, '')})")
            ax.set_ylabel("Empirical CDF")
            ax.grid(True, alpha=0.3)
            warn(f"No eligible {level}-{target} predictions. Skipping this panel.")
            continue

        curves_plotted = 0

        for _, row in sel.iterrows():
            g = registry[
                (registry["level"] == row["level"]) &
                (registry["target"] == row["target"]) &
                (registry["family"] == row["family"]) &
                (registry["model"] == row["model"])
            ]

            if g.empty:
                warn(
                    f"Selected model has no registry rows and will be skipped: "
                    f"level={row['level']}, target={row['target']}, "
                    f"family={row['family']}, model={row['model']}"
                )
                continue

            x, y = _empirical_cdf_values(g["abs_error"])

            if len(x) == 0:
                warn(f"Empty error vector for {row['family']} | {row['model']}")
                continue

            ax.plot(x, y, linewidth=2.0, label=_format_label(row))
            curves_plotted += 1

        ax.set_title(f"{level.capitalize()}-level {target}")
        ax.set_xlabel(f"Absolute error ({TARGET_UNITS.get(target, '')})")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, alpha=0.3)

        if curves_plotted > 0:
            ax.legend(fontsize=8)

    png_path = RESULT_FIG_DIR / f"{out_stem}.png"
    pdf_path = RESULT_FIG_DIR / f"{out_stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    log(f"Saved: {png_path}")
    log(f"Saved: {pdf_path}")


def _print_selected_table(selected: pd.DataFrame) -> None:
    """
    Print selected-best rows in a terminal-safe way.
    """

    display_cols = [
        "level", "target", "family", "model",
        "RMSE_macro", "MAE_macro", "P95_macro",
        "n_total", "n_scenarios",
    ]

    cols = [c for c in display_cols if c in selected.columns]

    log("Selected representative model per family:")
    print(selected[cols].to_string(index=False), flush=True)


# ============================================================
# Run final CDF reporting
# ============================================================

def main() -> None:
    start = time.time()

    log("Section 100 CSV-based final CDF plotting started.")
    log(f"Repository root: {REPO_ROOT}")
    log(f"Prediction directory: {PRED_DIR}")
    log(f"Figure directory: {RESULT_FIG_DIR}")
    log(f"Table directory: {RESULT_TABLE_DIR}")

    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {PRED_DIR}")

    registry = _load_prediction_registry(PRED_DIR)
    _, selected_best = _load_or_compute_selection(registry)

    # Keep only selected rows that are actually available in the registry.
    # This protects against stale selected_best tables after a manual cleanup.
    available_keys = registry[["level", "target", "family", "model"]].drop_duplicates()
    selected_before = len(selected_best)
    selected_best = selected_best.merge(
        available_keys,
        on=["level", "target", "family", "model"],
        how="inner",
    )

    if len(selected_best) < selected_before:
        warn(f"Removed {selected_before - len(selected_best)} stale selected model row(s).")

    if selected_best.empty:
        raise RuntimeError("No selected models remain after matching with the prediction registry.")

    selected_best.to_csv(CDF_SELECTED_PATH, index=False)
    log(f"Saved CDF selected-model table: {CDF_SELECTED_PATH}")

    _print_selected_table(selected_best)

    _plot_level_cdf(
        registry=registry,
        selected=selected_best,
        level="row",
        out_stem="row_level_best_family_abs_error_cdf",
    )

    _plot_level_cdf(
        registry=registry,
        selected=selected_best,
        level="geometry",
        out_stem="geometry_level_best_family_abs_error_cdf",
    )

    elapsed = time.time() - start
    log(f"plotting/plot_error_cdf.py completed successfully in {elapsed:.1f} s.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise
