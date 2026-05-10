# ============================================================
# plotting/plot_compact_cdf_2x2_for_column.py
# ------------------------------------------------------------
# Purpose:
# - Plot compact paper-grade geometry-level and row-level RSS/RTT
#   empirical CDFs in one 2x2 figure.
# - Use the finalized prediction registry and selected-model table.
# - Keep the same selected-family logic used by the final pipeline.
# - Use short family-only legend labels to reduce figure size.
#
# Scientific protocol:
# - Uses only finalized representative prediction rows.
# - Uses selected models from selected_best_model_per_family_test.csv.
# - Does not recompute model selection.
# - Does not modify predictions or metrics.
# - Uses empirical CDFs of absolute prediction error.
# - Uses inset zooms only for visual readability.
#
# Outputs:
# - results/figures/compact_geometry_row_abs_error_cdf_2x2.png
# - results/figures/compact_geometry_row_abs_error_cdf_2x2.pdf
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
# - This script intentionally keeps the same scientific selection and
#   prediction registry as the existing CDF scripts.
# - Only the visual layout is made compact for column-width use.
# - Legend labels are shortened to family names only.
# ============================================================

FAMILY_ORDER = ["CNN_LAMS", "GNN", "MLP", "RF", "XGBoost"]

FAMILY_PRETTY = {
    "CNN_LAMS": "CNN-LAMS",
    "GNN": "GNN",
    "MLP": "MLP",
    "RF": "RF",
    "XGBoost": "XGB",
}

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_MAP = {
    family: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    for i, family in enumerate(FAMILY_ORDER)
}

# One zoom configuration per panel.
# Keys are (level, target).
ZOOM_CONFIG = {
    ("geometry", "RSS"): {
        "title": "Geometry-level RSS",
        "xlabel": "Absolute error (dB)",
        "xlim_main": (-0.3, 18.0),
        "ylim_main": (-0.02, 1.03),
        "zoom_xlim": (3.2, 4.4),
        "zoom_ylim": (0.62, 0.78),
        "inset_bounds": [0.55, 0.56, 0.34, 0.30],
        "mark_locs": (2, 3),
    },
    ("geometry", "RTT"): {
        "title": "Geometry-level RTT",
        "xlabel": "Absolute error (m)",
        "xlim_main": (-0.5, 38.0),
        "ylim_main": (-0.02, 1.03),
        "zoom_xlim": (1.2, 2.2),
        "zoom_ylim": (0.82, 0.96),
        "inset_bounds": [0.55, 0.56, 0.34, 0.30],
        "mark_locs": (2, 3),
    },
    ("row", "RSS"): {
        "title": "Row-level RSS",
        "xlabel": "Absolute error (dB)",
        "xlim_main": (-0.3, 18.0),
        "ylim_main": (-0.02, 1.03),
        "zoom_xlim": (3.0, 4.5),
        "zoom_ylim": (0.47, 0.77),
        "inset_bounds": [0.55, 0.56, 0.34, 0.30],
        "mark_locs": (2, 3),
    },
    ("row", "RTT"): {
        "title": "Row-level RTT",
        "xlabel": "Absolute error (m)",
        "xlim_main": (-0.2, 6.0),
        "ylim_main": (-0.02, 1.03),
        "zoom_xlim": (0.95, 1.7),
        "zoom_ylim": (0.72, 0.85),
        "inset_bounds": [0.55, 0.56, 0.34, 0.30],
        "mark_locs": (2, 3),
    },
}


# ============================================================
# Section 3 — Logging and validation
# ============================================================

def log(message: str) -> None:
    print(f"[plot_compact_cdf_2x2_for_column] {message}", flush=True)


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
    - This is deterministic and transparent:
      sort finite absolute errors, then use rank/N.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = np.sort(values)

    if values.size == 0:
        return np.array([]), np.array([])

    cdf = np.arange(1, values.size + 1, dtype=float) / values.size
    return values, cdf


def selected_rows_for_panel(
    selected: pd.DataFrame,
    level: str,
    target: str,
) -> pd.DataFrame:
    level = str(level).lower().strip()
    target = str(target).upper().strip()

    rows = selected[
        (selected["level"] == level)
        & (selected["target"] == target)
    ].copy()

    if rows.empty:
        raise RuntimeError(
            f"No selected rows found for level={level}, target={target}"
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
    level: str,
    target: str,
    family: str,
    model: str,
) -> np.ndarray:
    level = str(level).lower().strip()
    target = str(target).upper().strip()
    family = str(family).strip()
    model = str(model).strip()

    mask = (
        (registry["level"] == level)
        & (registry["target"] == target)
        & (registry["family"] == family)
        & (registry["model"] == model)
    )

    values = registry.loc[mask, "abs_error"].dropna().to_numpy(dtype=float)

    if values.size == 0:
        raise RuntimeError(
            "No absolute-error values found for "
            f"level={level}, target={target}, family={family}, model={model}"
        )

    return values


def legend_label(row: pd.Series) -> str:
    """
    Compact legend label for column-width plotting.

    Important:
    - The actual selected variant is still used internally.
    - Only the displayed label is shortened.
    """
    family = str(row["family"]).strip()
    return FAMILY_PRETTY.get(family, family)


# ============================================================
# Section 5 — Plotting
# ============================================================

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 600,
        "font.size": 7.0,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 5.7,
        "axes.linewidth": 0.65,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def configure_main_axis(ax: plt.Axes, level: str, target: str) -> None:
    cfg = ZOOM_CONFIG[(level, target)]

    ax.set_title(cfg["title"], pad=2.5)
    ax.set_xlabel(cfg["xlabel"], labelpad=1.5)
    ax.set_ylabel("CDF", labelpad=1.5)
    ax.set_xlim(*cfg["xlim_main"])
    ax.set_ylim(*cfg["ylim_main"])
    ax.grid(True, alpha=0.28, linewidth=0.45)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.tick_params(axis="both", which="major", pad=1.2, length=2.5, width=0.6)


def configure_inset_axis(axins: plt.Axes, level: str, target: str) -> None:
    cfg = ZOOM_CONFIG[(level, target)]

    axins.set_xlim(*cfg["zoom_xlim"])
    axins.set_ylim(*cfg["zoom_ylim"])
    axins.grid(True, alpha=0.20, linewidth=0.35)

    axins.tick_params(axis="both", labelsize=4.6, pad=0.5, length=1.8, width=0.45)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axins.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for spine in axins.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("0.25")


def plot_panel(
    ax: plt.Axes,
    registry: pd.DataFrame,
    selected: pd.DataFrame,
    level: str,
    target: str,
) -> None:
    level = str(level).lower().strip()
    target = str(target).upper().strip()
    cfg = ZOOM_CONFIG[(level, target)]

    rows = selected_rows_for_panel(selected, level, target)
    log(f"Plotting {level}-level {target} with {len(rows)} selected families.")

    configure_main_axis(ax, level, target)

    axins = ax.inset_axes(cfg["inset_bounds"])
    configure_inset_axis(axins, level, target)

    handles = []
    labels = []

    for _, row in rows.iterrows():
        family = str(row["family"]).strip()
        model = str(row["model"]).strip()

        values = abs_error_for_model(
            registry=registry,
            level=level,
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
            linewidth=1.15,
            color=color,
            label=label,
        )

        axins.plot(
            x,
            y,
            linewidth=1.05,
            color=color,
        )

        handles.append(line)
        labels.append(label)

    pp, p1, p2 = mark_inset(
        ax,
        axins,
        loc1=cfg["mark_locs"][0],
        loc2=cfg["mark_locs"][1],
        fc="none",
        ec="0.50",
        ls="--",
        lw=0.55,
    )

    for artist in (pp, p1, p2):
        artist.set_zorder(1)
        artist.set_clip_on(False)

    axins.set_zorder(10)
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(1.0)
    axins.patch.set_edgecolor("0.30")
    axins.patch.set_linewidth(0.55)

    ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="0.70",
        handlelength=1.35,
        borderpad=0.25,
        labelspacing=0.18,
        handletextpad=0.35,
        columnspacing=0.40,
    )


# ============================================================
# Section 6 — Main execution
# ============================================================

def main() -> None:
    t0 = pd.Timestamp.now()

    configure_matplotlib()

    log("Starting compact 2x2 CDF plotting for column-width paper use.")
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

    # Compact 2x2 layout intended for single-column scaling in Overleaf.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(7.20, 5.35),
        dpi=220,
        constrained_layout=False,
    )

    plot_panel(axes[0, 0], registry, selected, level="geometry", target="RSS")
    plot_panel(axes[0, 1], registry, selected, level="geometry", target="RTT")
    plot_panel(axes[1, 0], registry, selected, level="row", target="RSS")
    plot_panel(axes[1, 1], registry, selected, level="row", target="RTT")

    # Tighten whitespace without forcing labels to overlap.
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.080,
        top=0.945,
        wspace=0.135,
        hspace=0.300,
    )

    png_path = FIGURE_DIR / "compact_geometry_row_abs_error_cdf_2x2.png"
    pdf_path = FIGURE_DIR / "compact_geometry_row_abs_error_cdf_2x2.pdf"

    log(f"Saving high-resolution PNG: {png_path}")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.015)

    log(f"Saving vector PDF: {pdf_path}")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)

    plt.close(fig)

    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    log(f"Completed successfully in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()