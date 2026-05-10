# ============================================================
# plotting/plot_row_cdf_circular_zoom.py
# ------------------------------------------------------------
# Purpose:
# - Plot paper-grade row-level RSS and RTT empirical CDFs.
# - Read the finalized prediction registry and selected-model table.
# - Add clean quantitative inset zoom axes to show curve separation.
#
# Scientific protocol:
# - Uses only finalized representative prediction rows.
# - Uses selected models from selected_best_model_per_family_test.csv.
# - Does not recompute model selection.
# - Does not modify predictions or metrics.
# - Uses Matplotlib trusted inset utilities rather than decorative
#   hand-drawn zoom shapes.
#
# Outputs:
# - results/figures/row_level_best_family_abs_error_cdf_zoom_clean.png
# - results/figures/row_level_best_family_abs_error_cdf_zoom_clean.pdf
# ============================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# ============================================================
# Section 1 — Repository paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_REGISTRY_PATH = (
    REPO_ROOT / "output_csvs" / "predictions" / "_prediction_registry_long.csv"
)

SELECTED_MODELS_PATH = (
    REPO_ROOT / "results" / "tables" / "selected_best_model_per_family_test.csv"
)

FIGURE_DIR = REPO_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Section 2 — Plot configuration
# ------------------------------------------------------------
# Notes:
# - These zoom windows are intentionally exposed here so they can
#   be tuned easily without touching the plotting logic.
# - If you want a bigger zoomed graph, increase inset_bounds width
#   and height.
# ============================================================

PLOT_LEVEL = "row"

FAMILY_ORDER = ["CNN_LAMS", "GNN", "MLP", "RF", "XGBoost"]

FAMILY_PRETTY = {
    "CNN_LAMS": "CNN-LAMS",
    "GNN": "GNN",
    "MLP": "MLP",
    "RF": "RF",
    "XGBoost": "XGBoost",
}

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_MAP = {
    family: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    for i, family in enumerate(FAMILY_ORDER)
}

ZOOM_CONFIG = {
    "RSS": {
        "title": "Row-level RSS",
        "xlabel": "Absolute error (dB)",
        "xlim_main": (-0.3, 18.0),
        "ylim_main": (-0.02, 1.03),

        # Zoom source region on main panel
        "zoom_xlim": (3.0, 4.5),
        "zoom_ylim": (0.47, 0.77),

        # Inset box position [left, bottom, width, height] in axes fraction
        # Increase width/height if you want a larger zoomed panel.
        "inset_bounds": [0.57, 0.53, 0.30, 0.30],
        # "inset_bounds": [0.60, 0.56, 0.26, 0.27],

        # Use consistent connector corner logic.
        "mark_locs": (2, 3),
    },

    "RTT": {
        "title": "Row-level RTT",
        "xlabel": "Absolute error (m)",
        "xlim_main": (-0.2, 6.0),
        "ylim_main": (-0.02, 1.03),

        # Focus on the separation region
        "zoom_xlim": (0.95, 1.7),
        "zoom_ylim": (0.72, 0.85),

        # Same size and same placement as RSS
        "inset_bounds": [0.57, 0.53, 0.30, 0.30],
        # "inset_bounds": [0.60, 0.56, 0.26, 0.27],

        # Keep connector style consistent with RSS
        "mark_locs": (2, 3),
    },
}


# ============================================================
# Section 3 — Logging and validation
# ============================================================

def log(message: str) -> None:
    print(f"[plot_row_cdf_circular_zoom] {message}", flush=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_inputs(registry: pd.DataFrame, selected: pd.DataFrame) -> None:
    required_registry_cols = {
        "level",
        "target",
        "family",
        "model",
        "scenario",
        "unit_id",
        "y_true",
        "y_pred",
    }

    required_selected_cols = {
        "level",
        "target",
        "family",
        "model",
        "RMSE_macro",
    }

    missing_registry = required_registry_cols - set(registry.columns)
    missing_selected = required_selected_cols - set(selected.columns)

    if missing_registry:
        raise RuntimeError(
            "Prediction registry is missing required columns: "
            + ", ".join(sorted(missing_registry))
        )

    if missing_selected:
        raise RuntimeError(
            "Selected-best table is missing required columns: "
            + ", ".join(sorted(missing_selected))
        )


# ============================================================
# Section 4 — Data helpers
# ============================================================

def normalize_tables(
    registry: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = registry.copy()
    selected = selected.copy()

    for df in [registry, selected]:
        for col in ["level", "target", "family", "model"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    registry["level"] = registry["level"].str.lower()
    selected["level"] = selected["level"].str.lower()

    registry["target"] = registry["target"].str.upper()
    selected["target"] = selected["target"].str.upper()

    if "abs_error" not in registry.columns:
        registry["y_true"] = pd.to_numeric(registry["y_true"], errors="coerce")
        registry["y_pred"] = pd.to_numeric(registry["y_pred"], errors="coerce")
        registry["abs_error"] = (registry["y_pred"] - registry["y_true"]).abs()

    registry["abs_error"] = pd.to_numeric(registry["abs_error"], errors="coerce")
    selected["RMSE_macro"] = pd.to_numeric(selected["RMSE_macro"], errors="coerce")

    return registry, selected


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical CDF from absolute-error values.

    Scientific note:
    - This is a transparent deterministic ECDF:
      sorted values + cumulative rank / N
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = np.sort(values)

    if values.size == 0:
        return np.array([]), np.array([])

    cdf = np.arange(1, values.size + 1, dtype=float) / values.size
    return values, cdf


def selected_rows_for_target(selected: pd.DataFrame, target: str) -> pd.DataFrame:
    target = str(target).upper().strip()

    rows = selected[
        (selected["level"] == PLOT_LEVEL)
        & (selected["target"] == target)
    ].copy()

    if rows.empty:
        raise RuntimeError(
            f"No selected {PLOT_LEVEL}-level rows found for target={target}"
        )

    rows["family_order"] = rows["family"].map(
        {family: idx for idx, family in enumerate(FAMILY_ORDER)}
    ).fillna(999)

    rows = rows.sort_values(
        ["family_order", "family", "model"],
        ascending=True,
    ).reset_index(drop=True)

    return rows


def abs_error_for_model(
    registry: pd.DataFrame,
    target: str,
    family: str,
    model: str,
) -> np.ndarray:
    target = str(target).upper().strip()
    family = str(family).strip()
    model = str(model).strip()

    mask = (
        (registry["level"] == PLOT_LEVEL)
        & (registry["target"] == target)
        & (registry["family"] == family)
        & (registry["model"] == model)
    )

    values = registry.loc[mask, "abs_error"].dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise RuntimeError(
            f"No {PLOT_LEVEL}-level absolute-error values found for "
            f"target={target}, family={family}, model={model}"
        )

    return values


def legend_label(row: pd.Series) -> str:
    family = str(row["family"]).strip()
    model = str(row["model"]).strip()
    pretty_family = FAMILY_PRETTY.get(family, family)
    rmse = float(row["RMSE_macro"])

    # Keep the metric meaning explicit for the reader.
    return f"{pretty_family} | {model} (RMSE_macro={rmse:.3f})"


# ============================================================
# Section 5 — Plotting
# ============================================================

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.size": 10,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def configure_main_axis(ax: plt.Axes, target: str) -> None:
    cfg = ZOOM_CONFIG[target]

    ax.set_title(cfg["title"], pad=10)
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(*cfg["xlim_main"])
    ax.set_ylim(*cfg["ylim_main"])
    ax.grid(True, alpha=0.32, linewidth=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def configure_inset_axis(axins: plt.Axes, target: str) -> None:
    cfg = ZOOM_CONFIG[target]

    axins.set_xlim(*cfg["zoom_xlim"])
    axins.set_ylim(*cfg["zoom_ylim"])
    axins.grid(True, alpha=0.22, linewidth=0.6)

    axins.tick_params(axis="both", labelsize=7, pad=1)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axins.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for spine in axins.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.25")


def plot_target_panel(
    ax: plt.Axes,
    registry: pd.DataFrame,
    selected: pd.DataFrame,
    target: str,
) -> None:
    target = str(target).upper().strip()
    cfg = ZOOM_CONFIG[target]

    rows = selected_rows_for_target(selected, target)
    log(f"Plotting {PLOT_LEVEL}-level {target} with {len(rows)} selected model families.")

    configure_main_axis(ax, target)

    axins = ax.inset_axes(cfg["inset_bounds"])
    configure_inset_axis(axins, target)

    handles = []
    labels = []

    for _, row in rows.iterrows():
        family = str(row["family"]).strip()
        model = str(row["model"]).strip()

        values = abs_error_for_model(
            registry=registry,
            target=target,
            family=family,
            model=model,
        )

        x, y = empirical_cdf(values)
        color = COLOR_MAP.get(family, None)
        label = legend_label(row)

        line, = ax.plot(
            x,
            y,
            linewidth=2.0,
            color=color,
            label=label,
        )

        axins.plot(
            x,
            y,
            linewidth=1.8,
            color=color,
        )

        handles.append(line)
        labels.append(label)

    # Draw zoom-source rectangle and connector lines.
    # We use trusted Matplotlib utilities and keep the connector logic
    # consistent across both panels.
    pp, p1, p2 = mark_inset(
        ax,
        axins,
        loc1=cfg["mark_locs"][0],
        loc2=cfg["mark_locs"][1],
        fc="none",
        ec="0.45",
        ls="--",
        lw=1.0,
    )

    # Keep connectors behind the inset box so they do not visually cross it.
    for artist in (pp, p1, p2):
        artist.set_zorder(1)
        artist.set_clip_on(False)

    axins.set_zorder(10)
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(1.0)
    axins.patch.set_edgecolor("0.30")
    axins.patch.set_linewidth(1.0)

    # Legend back to its standard in-panel location.
    ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.70",
        handlelength=2.2,
        borderpad=0.55,
        labelspacing=0.45,
    )


# ============================================================
# Section 6 — Main execution
# ============================================================

def main() -> None:
    t0 = pd.Timestamp.now()

    configure_matplotlib()

    log("Starting clean row-level CDF plotting with inset zooms.")
    log(f"Prediction registry: {PREDICTION_REGISTRY_PATH}")
    log(f"Selected-best table: {SELECTED_MODELS_PATH}")

    require_file(PREDICTION_REGISTRY_PATH)
    require_file(SELECTED_MODELS_PATH)

    registry = pd.read_csv(PREDICTION_REGISTRY_PATH)
    selected = pd.read_csv(SELECTED_MODELS_PATH)

    validate_inputs(registry, selected)
    registry, selected = normalize_tables(registry, selected)

    log(f"Loaded registry rows: {len(registry):,}")
    log(f"Loaded selected-best rows: {len(selected):,}")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16.5, 5.6),
        dpi=180,
        constrained_layout=True,
    )

    plot_target_panel(axes[0], registry, selected, target="RSS")
    plot_target_panel(axes[1], registry, selected, target="RTT")

    png_path = FIGURE_DIR / "row_level_best_family_abs_error_cdf_zoom_clean.png"
    pdf_path = FIGURE_DIR / "row_level_best_family_abs_error_cdf_zoom_clean.pdf"

    log(f"Saving high-resolution PNG: {png_path}")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    log(f"Saving vector PDF: {pdf_path}")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    log(f"Completed successfully in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()