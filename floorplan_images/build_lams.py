# ============================================================
# floorplan_images/build_lams.py
# ------------------------------------------------------------
# Purpose:
# - Build true 40×40 AP-to-RP aligned LAMS inputs as a standalone
#   repository script.
# - Replace notebook-runtime objects such as:
#       floorplan_registry
#       subtrain_df_cnn_hybrid
#       val_df_cnn_hybrid
#       test_df_cnn_hybrid
#       CNN_LAMS_DIR
#
# Scientific protocol:
# - This script does not train, tune, or evaluate any model.
# - It constructs deterministic image-like floorplan representations
#   from AP/RP geometry and vector wall maps.
# - RSS/RTT labels are used only later to create geometry-level target
#   tables, not to generate the LAMS images themselves.
# - The LAMS image is determined only by:
#       scenario, AP position, RP position, and floorplan geometry.
# - Repeated scans of the same AP-RP link share the same LAMS image.
# - Therefore, LAMS images are generated once per unique AP-RP geometry.
# - No missing image output is fabricated.
# - If required processed split files or vector floorplans are missing,
#   the script stops clearly.
#
# Trusted libraries used:
# - Pillow ImageDraw for wall rasterization.
# - scipy.ndimage.map_coordinates for image sampling.
# - scikit-learn StandardScaler for train-only feature scaling.
# - pandas/numpy for table construction.
#
# Inputs:
# - output_csvs/processed_features/row_train_wall_context.csv
# - output_csvs/processed_features/row_val_wall_context.csv
# - output_csvs/processed_features/row_test_wall_context.csv
# - vector_floorplans_with_internal_walls.json
#
# Outputs:
# - floorplan_images/lams/true_lams_40x40/lams_bank_40x40_float32.npy
# - floorplan_images/lams/true_lams_40x40/lams_metadata.csv
# - floorplan_images/lams/true_lams_40x40/lams_all_rows_with_ids.csv
# - floorplan_images/lams/true_lams_40x40/lams_geo_subtrain.csv
# - floorplan_images/lams/true_lams_40x40/lams_geo_val.csv
# - floorplan_images/lams/true_lams_40x40/lams_geo_test.csv
# - floorplan_images/lams/true_lams_40x40/rtt_lams_branch/*.csv
# - floorplan_images/lams/true_lams_40x40/*.joblib
# ============================================================

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from sklearn.preprocessing import StandardScaler


# ============================================================
# Section 1 — Repository paths and configuration
# ------------------------------------------------------------
# Purpose:
# - Resolve all paths relative to the repository root.
#
# Scientific protocol:
# - No absolute Windows path is hard-coded.
# - The script can be launched from the repo root or by run_pipeline.py.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = REPO_ROOT / "output_csvs" / "processed_features"
LAMS_ROOT_DIR = REPO_ROOT / "floorplan_images" / "lams"
LAMS_DIR = LAMS_ROOT_DIR / "true_lams_40x40"
RTT_LAMS_DIR = LAMS_DIR / "rtt_lams_branch"

LAMS_DIR.mkdir(parents=True, exist_ok=True)
RTT_LAMS_DIR.mkdir(parents=True, exist_ok=True)

FLOORPLAN_VECTOR_JSON_CANDIDATES = [
    REPO_ROOT
    / "outputs_floorplan_vector_builder"
    / "json"
    / "vector_floorplans_with_internal_walls.json",
    REPO_ROOT
    / "floorplan_images"
    / "corrected"
    / "vector_floorplans_with_internal_walls.json",
    REPO_ROOT
    / "floorplan_images"
    / "vector_floorplans_with_internal_walls.json",
]

LAMS_SIZE = 40
LAMS_RASTER_RES_M = 0.25
LAMS_WALL_LINE_WIDTH_PX = 2

# Paper-faithful default:
# transverse scan width equals AP-RP separation.
LAMS_SPAN_SCALE = 1.0

LAMS_MIN_DISTANCE_M = 1e-6
RTT_INVALID_PLACEHOLDER = 100000

GEOMETRY_KEY_BASE = ["scenario", "AP_x_key", "AP_y_key", "RX_x_key", "RX_y_key"]

FEATURES_GEOM_ONLY = [
    "RX_x",
    "RX_y",
    "AP_x",
    "AP_y",
    "dx",
    "dy",
    "distance",
    "LOS_flag",
    "LOS_known",
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

FEATURES_WALL_OBS_CONTEXT = (
    FEATURES_GEOM_ONLY
    + WALL_OBSTRUCTION_FEATURES
    + ENDPOINT_WALL_CONTEXT_FEATURES
)


# ============================================================
# Section 2 — Logging and file guards
# ------------------------------------------------------------
# Purpose:
# - Provide concise progress messages.
# - Stop clearly if required files are missing.
# ============================================================

def log(message: str) -> None:
    print(f"[build_lams] {message}", flush=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def find_floorplan_json() -> Path:
    for path in FLOORPLAN_VECTOR_JSON_CANDIDATES:
        if path.exists():
            return path

    checked = "\n".join(f"- {p}" for p in FLOORPLAN_VECTOR_JSON_CANDIDATES)
    raise FileNotFoundError(
        "Could not find vector_floorplans_with_internal_walls.json.\n"
        "Checked:\n"
        f"{checked}\n\n"
        "Place the finalized vector floorplan JSON in one of these locations."
    )


# ============================================================
# Section 3 — Load processed split tables
# ------------------------------------------------------------
# Purpose:
# - Load the wall-context row-level split files created by
#   build_wall_features.py.
#
# Scientific protocol:
# - These files already preserve:
#       official train/test split
#       RP-level validation split
# - This script does not reshuffle or re-split the data.
# ============================================================

def load_wall_context_splits() -> Dict[str, pd.DataFrame]:
    split_paths = {
        "subtrain": PROCESSED_DIR / "row_train_wall_context.csv",
        "val": PROCESSED_DIR / "row_val_wall_context.csv",
        "test": PROCESSED_DIR / "row_test_wall_context.csv",
    }

    split_data: Dict[str, pd.DataFrame] = {}

    for split_name, path in split_paths.items():
        require_file(path)
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)
        df["split"] = split_name
        df["row_pos_within_split"] = np.arange(len(df), dtype=int)

        split_data[split_name] = df
        log(f"Loaded {split_name}: {len(df):,} rows from {path}")

    return split_data


def validate_required_columns(split_data: Dict[str, pd.DataFrame]) -> None:
    required_columns = [
        "scenario",
        "AP_x",
        "AP_y",
        "RX_x",
        "RX_y",
        "RSS_dBm",
        "RTT_m",
    ]

    required_columns += FEATURES_WALL_OBS_CONTEXT

    for split_name, df in split_data.items():
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"{split_name} is missing required LAMS columns:\n"
                f"{missing}\n\n"
                "Run build_dataset.py and build_wall_features.py first."
            )


# ============================================================
# Section 4 — Load vector floorplan registry
# ------------------------------------------------------------
# Purpose:
# - Load manually finalized vector wall maps.
#
# Scientific protocol:
# - LAMS image construction uses only wall geometry and AP/RP coordinates.
# ============================================================

def load_floorplan_registry() -> dict:
    floorplan_json = find_floorplan_json()
    log(f"Loading vector floorplans from: {floorplan_json}")

    with floorplan_json.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    log(f"Loaded vector floorplan scenarios: {list(registry.keys())}")
    return registry


def normalize_scenario_key(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("-", "_")


def get_scenario_record(registry: dict, scenario_name: str) -> dict:
    if scenario_name in registry:
        return registry[scenario_name]

    normalized = {
        normalize_scenario_key(k): k
        for k in registry.keys()
    }

    key = normalize_scenario_key(scenario_name)

    if key in normalized:
        return registry[normalized[key]]

    raise KeyError(
        f"Scenario '{scenario_name}' was not found in vector floorplan registry.\n"
        f"Available keys: {list(registry.keys())}\n\n"
        "This is intentionally not bypassed. LAMS cannot be built without "
        "a vector floorplan for each scenario used by the LAMS branch."
    )


# ============================================================
# Section 5 — Robust wall parsing helpers
# ------------------------------------------------------------
# Purpose:
# - Convert flexible wall dictionary formats into line endpoints.
#
# Scientific protocol:
# - Parsing is defensive, but does not invent missing geometry.
# - Unsupported wall formats stop the script.
# ============================================================

def as_float_pair(obj) -> Tuple[float, float]:
    if isinstance(obj, dict):
        if {"x", "y"}.issubset(obj.keys()):
            return float(obj["x"]), float(obj["y"])
        if {"X", "Y"}.issubset(obj.keys()):
            return float(obj["X"]), float(obj["Y"])

    if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])

    raise ValueError(f"Cannot parse point: {obj}")


def extract_wall_endpoints(wall: dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # General segment.
    if {"x0", "y0", "x1", "y1"}.issubset(wall.keys()):
        return (
            (float(wall["x0"]), float(wall["y0"])),
            (float(wall["x1"]), float(wall["y1"])),
        )

    # Vertical segment.
    if {"x", "y0", "y1"}.issubset(wall.keys()):
        return (
            (float(wall["x"]), float(wall["y0"])),
            (float(wall["x"]), float(wall["y1"])),
        )

    # Horizontal segment.
    if {"y", "x0", "x1"}.issubset(wall.keys()):
        return (
            (float(wall["x0"]), float(wall["y"])),
            (float(wall["x1"]), float(wall["y"])),
        )

    # Alternative segment naming.
    if {"x1", "y1", "x2", "y2"}.issubset(wall.keys()):
        return (
            (float(wall["x1"]), float(wall["y1"])),
            (float(wall["x2"]), float(wall["y2"])),
        )

    if "start" in wall and "end" in wall:
        return as_float_pair(wall["start"]), as_float_pair(wall["end"])

    if "p1" in wall and "p2" in wall:
        return as_float_pair(wall["p1"]), as_float_pair(wall["p2"])

    if "points" in wall and len(wall["points"]) >= 2:
        return as_float_pair(wall["points"][0]), as_float_pair(wall["points"][1])

    if "coords" in wall and len(wall["coords"]) >= 2:
        return as_float_pair(wall["coords"][0]), as_float_pair(wall["coords"][1])

    raise ValueError(
        "Cannot extract wall endpoints from wall dictionary keys: "
        f"{list(wall.keys())}"
    )


def infer_wall_role(wall: dict) -> str:
    raw = str(wall.get("role", wall.get("tag", wall.get("type", "")))).lower()

    if "external" in raw or "outer" in raw or "boundary" in raw:
        return "external"

    if "internal" in raw or "inner" in raw:
        return "internal"

    # Safe default for floorplan-builder outputs:
    # explicit outer walls are normally tagged, otherwise treat as internal.
    return "internal"


def extract_walls_from_record(record: dict) -> List[dict]:
    candidate_keys = ["walls", "wall_segments", "segments", "lines"]

    walls = None
    for key in candidate_keys:
        if isinstance(record, dict) and key in record:
            walls = record[key]
            break

    if walls is None:
        raise KeyError(
            "Could not find wall list in floorplan record.\n"
            f"Available keys: {list(record.keys()) if isinstance(record, dict) else type(record)}"
        )

    parsed = []

    for wall in walls:
        p1, p2 = extract_wall_endpoints(wall)
        parsed.append(
            {
                "x1": float(p1[0]),
                "y1": float(p1[1]),
                "x2": float(p2[0]),
                "y2": float(p2[1]),
                "role": infer_wall_role(wall),
            }
        )

    return parsed


def extract_outer_box(record: dict, parsed_walls: List[dict]) -> Tuple[float, float, float, float]:
    outer = None

    if isinstance(record, dict):
        for key in ["outer_box", "bounds", "bbox"]:
            if key in record:
                outer = record[key]
                break

    if isinstance(outer, dict):
        keys = set(outer.keys())

        if {"xmin", "ymin", "xmax", "ymax"}.issubset(keys):
            return (
                float(outer["xmin"]),
                float(outer["ymin"]),
                float(outer["xmax"]),
                float(outer["ymax"]),
            )

        if {"min_x", "min_y", "max_x", "max_y"}.issubset(keys):
            return (
                float(outer["min_x"]),
                float(outer["min_y"]),
                float(outer["max_x"]),
                float(outer["max_y"]),
            )

    if isinstance(outer, (list, tuple, np.ndarray)) and len(outer) >= 4:
        return float(outer[0]), float(outer[1]), float(outer[2]), float(outer[3])

    xs = []
    ys = []

    for wall in parsed_walls:
        xs.extend([wall["x1"], wall["x2"]])
        ys.extend([wall["y1"], wall["y2"]])

    if not xs or not ys:
        raise RuntimeError("Cannot infer raster bounds because no wall coordinates were found.")

    margin = 0.5
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


# ============================================================
# Section 6 — Raster coordinate conversion
# ------------------------------------------------------------
# Purpose:
# - Convert between metric coordinates and raster pixel coordinates.
# ============================================================

def m_to_px(
    x_m: float,
    y_m: float,
    bounds: Tuple[float, float, float, float],
    res_m: float,
) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = bounds

    col = (float(x_m) - xmin) / res_m
    row = (ymax - float(y_m)) / res_m

    return float(col), float(row)


# ============================================================
# Section 7 — Build semantic wall rasters
# ------------------------------------------------------------
# Purpose:
# - Convert vector walls into semantic wall channels.
#
# Scientific protocol:
# - Pillow ImageDraw handles trusted raster drawing.
# - Channels are:
#       0 internal walls
#       1 external walls
#       2 all walls
# ============================================================

def build_lams_semantic_raster_for_scenario(
    scenario_name: str,
    floorplan_registry: dict,
) -> dict:
    record = get_scenario_record(floorplan_registry, scenario_name)
    walls = extract_walls_from_record(record)
    bounds = extract_outer_box(record, walls)

    xmin, ymin, xmax, ymax = bounds

    width_px = int(np.ceil((xmax - xmin) / LAMS_RASTER_RES_M)) + 1
    height_px = int(np.ceil((ymax - ymin) / LAMS_RASTER_RES_M)) + 1

    if width_px <= 0 or height_px <= 0:
        raise RuntimeError(
            f"Invalid raster size for scenario {scenario_name}: "
            f"{width_px} × {height_px}"
        )

    internal_img = Image.new("L", (width_px, height_px), 0)
    external_img = Image.new("L", (width_px, height_px), 0)

    draw_internal = ImageDraw.Draw(internal_img)
    draw_external = ImageDraw.Draw(external_img)

    for wall in walls:
        c1, r1 = m_to_px(wall["x1"], wall["y1"], bounds, LAMS_RASTER_RES_M)
        c2, r2 = m_to_px(wall["x2"], wall["y2"], bounds, LAMS_RASTER_RES_M)

        xy = [(c1, r1), (c2, r2)]

        if wall["role"] == "external":
            draw_external.line(xy, fill=1, width=LAMS_WALL_LINE_WIDTH_PX)
        else:
            draw_internal.line(xy, fill=1, width=LAMS_WALL_LINE_WIDTH_PX)

    internal = np.array(internal_img, dtype=np.uint8)
    external = np.array(external_img, dtype=np.uint8)
    all_walls = np.maximum(internal, external).astype(np.uint8)

    raster_chw = np.stack([internal, external, all_walls], axis=0)

    return {
        "scenario": scenario_name,
        "bounds": bounds,
        "walls": walls,
        "raster_chw": raster_chw,
        "height_px": height_px,
        "width_px": width_px,
    }


# ============================================================
# Section 8 — Generate one true LAMS image
# ------------------------------------------------------------
# Purpose:
# - Sample a 40×40 AP-to-RP aligned image from the semantic wall raster.
#
# Scientific protocol:
# - scipy.ndimage.map_coordinates handles raster sampling.
# - Nearest-neighbour sampling is used because the raster channels are
#   semantic masks, not continuous images.
# ============================================================

def generate_lams_image_from_raster(
    raster_chw: np.ndarray,
    ap_px: Tuple[float, float],
    rp_px: Tuple[float, float],
) -> np.ndarray:
    raster_chw = np.asarray(raster_chw)

    ap = np.asarray(ap_px, dtype=float)
    rp = np.asarray(rp_px, dtype=float)

    v = rp - ap
    d_px = float(np.linalg.norm(v))

    if d_px <= 0:
        raise ValueError("AP and RP pixel coordinates are identical. Cannot build LAMS image.")

    u = v / d_px
    n = np.array([-u[1], u[0]], dtype=float)

    half_span = 0.5 * LAMS_SPAN_SCALE * d_px

    # Positions along AP→RP.
    t = np.linspace(0.0, 1.0, LAMS_SIZE)

    # Perpendicular samples around the AP→RP center line.
    offsets = np.linspace(-half_span, half_span, LAMS_SIZE)

    mids = ap[None, :] + t[:, None] * v[None, :]
    pts = mids[:, None, :] + offsets[None, :, None] * n[None, None, :]

    col_coords = pts[:, :, 0]
    row_coords = pts[:, :, 1]

    sampled_channels = []

    for channel_idx in range(raster_chw.shape[0]):
        sampled = map_coordinates(
            raster_chw[channel_idx].astype(float),
            [row_coords, col_coords],
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        sampled_channels.append(sampled.astype(np.float32))

    return np.stack(sampled_channels, axis=0)


# ============================================================
# Section 9 — Prepare unique LAMS geometry table
# ------------------------------------------------------------
# Purpose:
# - Create one LAMS ID per valid unique AP-RP geometry.
#
# Scientific protocol:
# - Repeated scan rows are not treated as new LAMS images.
# - Zero-distance AP-RP geometries are excluded because propagation
#   direction is undefined.
# ============================================================

def add_geometry_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["AP_x_key"] = out["AP_x"].astype(float).round(4)
    out["AP_y_key"] = out["AP_y"].astype(float).round(4)
    out["RX_x_key"] = out["RX_x"].astype(float).round(4)
    out["RX_y_key"] = out["RX_y"].astype(float).round(4)

    return out


def build_lams_rows_and_geometries(split_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    lams_all_rows = pd.concat(
        [add_geometry_keys(df) for df in split_data.values()],
        axis=0,
        ignore_index=True,
    )

    geom_key_cols = list(GEOMETRY_KEY_BASE)

    if "AP_id_raw" in lams_all_rows.columns:
        geom_key_cols = [
            "scenario",
            "AP_id_raw",
            "AP_x_key",
            "AP_y_key",
            "RX_x_key",
            "RX_y_key",
        ]

    lams_geometries_raw = (
        lams_all_rows[
            geom_key_cols + ["AP_x", "AP_y", "RX_x", "RX_y"]
        ]
        .drop_duplicates(subset=geom_key_cols)
        .reset_index(drop=True)
    )

    lams_geometries_raw["distance"] = np.sqrt(
        (lams_geometries_raw["AP_x"] - lams_geometries_raw["RX_x"]) ** 2
        + (lams_geometries_raw["AP_y"] - lams_geometries_raw["RX_y"]) ** 2
    )

    invalid_lams_geometries = lams_geometries_raw[
        lams_geometries_raw["distance"] <= LAMS_MIN_DISTANCE_M
    ].copy()

    if len(invalid_lams_geometries) > 0:
        invalid_path = LAMS_DIR / "excluded_zero_distance_lams_geometries.csv"
        invalid_lams_geometries.to_csv(invalid_path, index=False)

        log("Excluding zero-distance AP-RP geometries.")
        log(f"  Count: {len(invalid_lams_geometries):,}")
        log(f"  Saved: {invalid_path}")

    lams_geometries = (
        lams_geometries_raw[
            lams_geometries_raw["distance"] > LAMS_MIN_DISTANCE_M
        ]
        .copy()
        .reset_index(drop=True)
    )

    lams_geometries["lams_id"] = np.arange(len(lams_geometries), dtype=int)

    n_rows_before = len(lams_all_rows)

    lams_all_rows = lams_all_rows.merge(
        lams_geometries[geom_key_cols + ["lams_id"]],
        on=geom_key_cols,
        how="inner",
        validate="many_to_one",
    )

    n_rows_after = len(lams_all_rows)
    n_rows_removed = n_rows_before - n_rows_after

    if n_rows_removed > 0:
        log(
            f"Removed {n_rows_removed:,} repeated rows linked to "
            "zero-distance AP-RP geometries."
        )

    lams_all_rows["lams_id"] = lams_all_rows["lams_id"].astype(int)

    log("LAMS geometry audit:")
    log(f"  Total repeated rows after validity filter: {len(lams_all_rows):,}")
    log(f"  Unique valid AP-RP geometries: {len(lams_geometries):,}")
    log(f"  Excluded zero-distance geometries: {len(invalid_lams_geometries):,}")
    log(f"  Scenarios: {sorted(lams_all_rows['scenario'].astype(str).unique())}")

    return lams_all_rows, lams_geometries, geom_key_cols


# ============================================================
# Section 10 — Build full LAMS image bank
# ------------------------------------------------------------
# Purpose:
# - Generate the complete true LAMS bank for all valid unique geometries.
#
# Scientific protocol:
# - Images are generated once per unique LAMS ID.
# - Progress is printed during the long-running loop.
# ============================================================

def build_lams_bank(
    lams_all_rows: pd.DataFrame,
    lams_geometries: pd.DataFrame,
    floorplan_registry: dict,
) -> Tuple[np.ndarray, pd.DataFrame]:
    scenario_names = sorted(lams_all_rows["scenario"].astype(str).unique())

    lams_raster_registry = {}

    for scenario_name in scenario_names:
        lams_raster_registry[scenario_name] = build_lams_semantic_raster_for_scenario(
            scenario_name=scenario_name,
            floorplan_registry=floorplan_registry,
        )

        rr = lams_raster_registry[scenario_name]
        pix_sum = rr["raster_chw"].sum(axis=(1, 2))

        log(
            f"Raster built for {scenario_name}: "
            f"shape={rr['raster_chw'].shape}, "
            f"internal={int(pix_sum[0])}, "
            f"external={int(pix_sum[1])}, "
            f"all={int(pix_sum[2])}"
        )

    lams_bank = np.zeros(
        (len(lams_geometries), 3, LAMS_SIZE, LAMS_SIZE),
        dtype=np.float32,
    )

    metadata_rows = []
    start_time = time.time()

    for i, row in enumerate(lams_geometries.itertuples(index=False), start=1):
        scenario_name = str(row.scenario)
        rr = lams_raster_registry[scenario_name]

        ap_px = m_to_px(row.AP_x, row.AP_y, rr["bounds"], LAMS_RASTER_RES_M)
        rp_px = m_to_px(row.RX_x, row.RX_y, rr["bounds"], LAMS_RASTER_RES_M)

        lams_img = generate_lams_image_from_raster(
            raster_chw=rr["raster_chw"],
            ap_px=ap_px,
            rp_px=rp_px,
        )

        lams_id = int(row.lams_id)
        lams_bank[lams_id] = lams_img

        distance_m = float(
            np.sqrt(
                (row.AP_x - row.RX_x) ** 2
                + (row.AP_y - row.RX_y) ** 2
            )
        )

        meta_row = {
            "lams_id": lams_id,
            "scenario": scenario_name,
            "AP_x": float(row.AP_x),
            "AP_y": float(row.AP_y),
            "RX_x": float(row.RX_x),
            "RX_y": float(row.RX_y),
            "distance": distance_m,
            "internal_pixels_lams": float(lams_img[0].sum()),
            "external_pixels_lams": float(lams_img[1].sum()),
            "all_wall_pixels_lams": float(lams_img[2].sum()),
        }

        if hasattr(row, "AP_id_raw"):
            meta_row["AP_id_raw"] = row.AP_id_raw

        metadata_rows.append(meta_row)

        if i % 500 == 0 or i == len(lams_geometries):
            elapsed = time.time() - start_time
            log(
                f"Generated {i:,}/{len(lams_geometries):,} true LAMS images "
                f"in {elapsed:.1f} s"
            )

    lams_metadata = (
        pd.DataFrame(metadata_rows)
        .sort_values("lams_id")
        .reset_index(drop=True)
    )

    return lams_bank, lams_metadata


# ============================================================
# Section 11 — Build RSS geometry-level LAMS tables
# ------------------------------------------------------------
# Purpose:
# - Build one row per AP-RP LAMS geometry for RSS modelling.
#
# Scientific protocol:
# - Repeated RSS scans of the same physical geometry are averaged.
# - Scalers are fitted on subtrain only.
# ============================================================

def build_lams_geometry_table(
    rows_df: pd.DataFrame,
    split_name: str,
    lams_metadata: pd.DataFrame,
    scalar_features: List[str],
    target_col: str,
    target_mean_col: str,
    target_std_col: str,
) -> pd.DataFrame:
    split_df = rows_df[rows_df["split"] == split_name].copy()

    if len(split_df) == 0:
        raise RuntimeError(f"No rows found for split: {split_name}")

    grouped_rows = []

    for lams_id, group in split_df.groupby("lams_id", sort=True):
        row = {
            "lams_id": int(lams_id),
            "split": split_name,
            "scenario": str(group["scenario"].iloc[0]),
            target_mean_col: float(group[target_col].mean()),
            target_std_col: float(group[target_col].std(ddof=1)) if len(group) > 1 else 0.0,
            "n_repeat": int(len(group)),
            "AP_x": float(group["AP_x"].iloc[0]),
            "AP_y": float(group["AP_y"].iloc[0]),
            "RX_x": float(group["RX_x"].iloc[0]),
            "RX_y": float(group["RX_y"].iloc[0]),
        }

        if "AP_id_raw" in group.columns:
            row["AP_id_raw"] = group["AP_id_raw"].iloc[0]

        for feature in scalar_features:
            row[feature] = float(group[feature].mean())

        grouped_rows.append(row)

    geo = pd.DataFrame(grouped_rows).sort_values("lams_id").reset_index(drop=True)

    meta_extra_cols = [
        "lams_id",
        "internal_pixels_lams",
        "external_pixels_lams",
        "all_wall_pixels_lams",
    ]

    if "distance" not in geo.columns and "distance" in lams_metadata.columns:
        meta_extra_cols.append("distance")

    available_meta_cols = [
        col for col in meta_extra_cols
        if col in lams_metadata.columns
    ]

    geo = geo.merge(
        lams_metadata[available_meta_cols],
        on="lams_id",
        how="left",
        validate="one_to_one",
    )

    return geo


def build_and_save_rss_lams_tables(
    lams_all_rows: pd.DataFrame,
    lams_metadata: pd.DataFrame,
) -> None:
    scalar_features = list(FEATURES_WALL_OBS_CONTEXT)

    if "distance" not in scalar_features and "distance" in lams_all_rows.columns:
        scalar_features.append("distance")

    missing_scalar_cols = [
        col for col in scalar_features
        if col not in lams_all_rows.columns
    ]

    if missing_scalar_cols:
        raise RuntimeError(f"Missing scalar columns needed for RSS LAMS hybrid: {missing_scalar_cols}")

    lams_geo_subtrain = build_lams_geometry_table(
        rows_df=lams_all_rows,
        split_name="subtrain",
        lams_metadata=lams_metadata,
        scalar_features=scalar_features,
        target_col="RSS_dBm",
        target_mean_col="RSS_mean_dBm",
        target_std_col="RSS_std_dBm",
    )

    lams_geo_val = build_lams_geometry_table(
        rows_df=lams_all_rows,
        split_name="val",
        lams_metadata=lams_metadata,
        scalar_features=scalar_features,
        target_col="RSS_dBm",
        target_mean_col="RSS_mean_dBm",
        target_std_col="RSS_std_dBm",
    )

    lams_geo_test = build_lams_geometry_table(
        rows_df=lams_all_rows,
        split_name="test",
        lams_metadata=lams_metadata,
        scalar_features=scalar_features,
        target_col="RSS_dBm",
        target_mean_col="RSS_mean_dBm",
        target_std_col="RSS_std_dBm",
    )

    distance_scaler = StandardScaler()
    scalar_scaler = StandardScaler()

    distance_scaler.fit(
        lams_geo_subtrain[["distance"]].to_numpy(dtype=np.float32)
    )

    scalar_scaler.fit(
        lams_geo_subtrain[scalar_features].to_numpy(dtype=np.float32)
    )

    lams_geo_audit = pd.DataFrame(
        [
            {
                "split": "subtrain",
                "n_repeated_rows": int((lams_all_rows["split"] == "subtrain").sum()),
                "n_unique_lams": len(lams_geo_subtrain),
                "rss_mean": float(lams_geo_subtrain["RSS_mean_dBm"].mean()),
                "rss_std": float(lams_geo_subtrain["RSS_mean_dBm"].std(ddof=1)),
                "mean_repeats_per_lams": float(lams_geo_subtrain["n_repeat"].mean()),
            },
            {
                "split": "val",
                "n_repeated_rows": int((lams_all_rows["split"] == "val").sum()),
                "n_unique_lams": len(lams_geo_val),
                "rss_mean": float(lams_geo_val["RSS_mean_dBm"].mean()),
                "rss_std": float(lams_geo_val["RSS_mean_dBm"].std(ddof=1)),
                "mean_repeats_per_lams": float(lams_geo_val["n_repeat"].mean()),
            },
            {
                "split": "test",
                "n_repeated_rows": int((lams_all_rows["split"] == "test").sum()),
                "n_unique_lams": len(lams_geo_test),
                "rss_mean": float(lams_geo_test["RSS_mean_dBm"].mean()),
                "rss_std": float(lams_geo_test["RSS_mean_dBm"].std(ddof=1)),
                "mean_repeats_per_lams": float(lams_geo_test["n_repeat"].mean()),
            },
        ]
    )

    lams_geo_subtrain.to_csv(LAMS_DIR / "lams_geo_subtrain.csv", index=False)
    lams_geo_val.to_csv(LAMS_DIR / "lams_geo_val.csv", index=False)
    lams_geo_test.to_csv(LAMS_DIR / "lams_geo_test.csv", index=False)
    lams_geo_audit.to_csv(LAMS_DIR / "lams_geometry_training_audit.csv", index=False)

    joblib.dump(distance_scaler, LAMS_DIR / "lams_distance_scaler.joblib")
    joblib.dump(scalar_scaler, LAMS_DIR / "lams_scalar_scaler.joblib")

    metadata = {
        "LAMS_SCALAR_FEATURES": scalar_features,
        "target": "RSS_dBm",
        "geometry_level_target": "RSS_mean_dBm",
        "lams_size": LAMS_SIZE,
        "raster_resolution_m": LAMS_RASTER_RES_M,
        "span_scale": LAMS_SPAN_SCALE,
    }

    (LAMS_DIR / "lams_rss_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    log("Saved RSS LAMS geometry tables and scalers.")
    log(f"  subtrain: {len(lams_geo_subtrain):,} unique LAMS")
    log(f"  val:      {len(lams_geo_val):,} unique LAMS")
    log(f"  test:     {len(lams_geo_test):,} unique LAMS")


# ============================================================
# Section 12 — Build RTT geometry-level LAMS tables
# ------------------------------------------------------------
# Purpose:
# - Build one row per AP-RP LAMS geometry for RTT modelling.
#
# Scientific protocol:
# - Known invalid RTT placeholders are removed before averaging.
# - Scalers are fitted on subtrain only.
# ============================================================

def build_and_save_rtt_lams_tables(
    lams_all_rows: pd.DataFrame,
    lams_metadata: pd.DataFrame,
) -> None:
    scalar_features = list(FEATURES_WALL_OBS_CONTEXT)

    if "distance" not in scalar_features and "distance" in lams_all_rows.columns:
        scalar_features.append("distance")

    missing_scalar_cols = [
        col for col in scalar_features
        if col not in lams_all_rows.columns
    ]

    if missing_scalar_cols:
        raise RuntimeError(f"Missing scalar columns needed for RTT LAMS hybrid: {missing_scalar_cols}")

    valid_rows = lams_all_rows[
        np.isfinite(lams_all_rows["RTT_m"].astype(float))
        & (lams_all_rows["RTT_m"].astype(float) != RTT_INVALID_PLACEHOLDER)
    ].copy()

    removed = len(lams_all_rows) - len(valid_rows)
    log(f"Removed {removed:,} rows with invalid RTT placeholders for RTT LAMS tables.")

    if len(valid_rows) == 0:
        raise RuntimeError("No valid RTT rows remain after placeholder filtering.")

    rtt_lams_geo_subtrain = build_lams_geometry_table(
        rows_df=valid_rows,
        split_name="subtrain",
        lams_metadata=lams_metadata,
        scalar_features=scalar_features,
        target_col="RTT_m",
        target_mean_col="RTT_mean_m",
        target_std_col="RTT_std_m",
    )

    rtt_lams_geo_val = build_lams_geometry_table(
        rows_df=valid_rows,
        split_name="val",
        lams_metadata=lams_metadata,
        scalar_features=scalar_features,
        target_col="RTT_m",
        target_mean_col="RTT_mean_m",
        target_std_col="RTT_std_m",
    )

    rtt_lams_geo_test = build_lams_geometry_table(
        rows_df=valid_rows,
        split_name="test",
        lams_metadata=lams_metadata,
        scalar_features=scalar_features,
        target_col="RTT_m",
        target_mean_col="RTT_mean_m",
        target_std_col="RTT_std_m",
    )

    rtt_target_scaler = StandardScaler()
    rtt_distance_scaler = StandardScaler()
    rtt_scalar_scaler = StandardScaler()

    rtt_target_scaler.fit(
        rtt_lams_geo_subtrain[["RTT_mean_m"]].to_numpy(dtype=np.float32)
    )

    rtt_distance_scaler.fit(
        rtt_lams_geo_subtrain[["distance"]].to_numpy(dtype=np.float32)
    )

    rtt_scalar_scaler.fit(
        rtt_lams_geo_subtrain[scalar_features].to_numpy(dtype=np.float32)
    )

    rtt_lams_geo_subtrain.to_csv(RTT_LAMS_DIR / "rtt_lams_geo_subtrain.csv", index=False)
    rtt_lams_geo_val.to_csv(RTT_LAMS_DIR / "rtt_lams_geo_val.csv", index=False)
    rtt_lams_geo_test.to_csv(RTT_LAMS_DIR / "rtt_lams_geo_test.csv", index=False)

    joblib.dump(rtt_target_scaler, RTT_LAMS_DIR / "rtt_target_scaler.joblib")
    joblib.dump(rtt_distance_scaler, RTT_LAMS_DIR / "rtt_distance_scaler.joblib")
    joblib.dump(rtt_scalar_scaler, RTT_LAMS_DIR / "rtt_scalar_scaler.joblib")

    metadata = {
        "RTT_LAMS_SCALAR_FEATURES": scalar_features,
        "target": "RTT_m",
        "geometry_level_target": "RTT_mean_m",
        "invalid_placeholder_removed": RTT_INVALID_PLACEHOLDER,
        "lams_size": LAMS_SIZE,
        "raster_resolution_m": LAMS_RASTER_RES_M,
        "span_scale": LAMS_SPAN_SCALE,
    }

    (RTT_LAMS_DIR / "rtt_lams_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    audit = pd.DataFrame(
        [
            {
                "split": "subtrain",
                "n_unique_lams": len(rtt_lams_geo_subtrain),
                "rtt_mean_m": float(rtt_lams_geo_subtrain["RTT_mean_m"].mean()),
                "rtt_std_m": float(rtt_lams_geo_subtrain["RTT_mean_m"].std(ddof=1)),
                "mean_repeats_per_lams": float(rtt_lams_geo_subtrain["n_repeat"].mean()),
            },
            {
                "split": "val",
                "n_unique_lams": len(rtt_lams_geo_val),
                "rtt_mean_m": float(rtt_lams_geo_val["RTT_mean_m"].mean()),
                "rtt_std_m": float(rtt_lams_geo_val["RTT_mean_m"].std(ddof=1)),
                "mean_repeats_per_lams": float(rtt_lams_geo_val["n_repeat"].mean()),
            },
            {
                "split": "test",
                "n_unique_lams": len(rtt_lams_geo_test),
                "rtt_mean_m": float(rtt_lams_geo_test["RTT_mean_m"].mean()),
                "rtt_std_m": float(rtt_lams_geo_test["RTT_mean_m"].std(ddof=1)),
                "mean_repeats_per_lams": float(rtt_lams_geo_test["n_repeat"].mean()),
            },
        ]
    )

    audit.to_csv(RTT_LAMS_DIR / "rtt_lams_geometry_training_audit.csv", index=False)

    log("Saved RTT LAMS geometry tables and scalers.")
    log(f"  subtrain: {len(rtt_lams_geo_subtrain):,} unique LAMS")
    log(f"  val:      {len(rtt_lams_geo_val):,} unique LAMS")
    log(f"  test:     {len(rtt_lams_geo_test):,} unique LAMS")


# ============================================================
# Section 13 — Save core LAMS image outputs
# ------------------------------------------------------------
# Purpose:
# - Save the generated image bank, metadata, and row-to-LAMS mapping.
# ============================================================

def save_core_lams_outputs(
    lams_bank: np.ndarray,
    lams_metadata: pd.DataFrame,
    lams_all_rows: pd.DataFrame,
) -> None:
    lams_bank_path = LAMS_DIR / "lams_bank_40x40_float32.npy"
    lams_meta_path = LAMS_DIR / "lams_metadata.csv"
    lams_rows_path = LAMS_DIR / "lams_all_rows_with_ids.csv"

    np.save(lams_bank_path, lams_bank)
    lams_metadata.to_csv(lams_meta_path, index=False)
    lams_all_rows.to_csv(lams_rows_path, index=False)

    log(f"Saved LAMS bank: {lams_bank_path}")
    log(f"Saved LAMS metadata: {lams_meta_path}")
    log(f"Saved row mapping: {lams_rows_path}")

    scenario_summary = (
        lams_metadata[
            [
                "scenario",
                "distance",
                "internal_pixels_lams",
                "external_pixels_lams",
                "all_wall_pixels_lams",
            ]
        ]
        .groupby("scenario")
        .agg(["count", "mean", "min", "max"])
    )

    summary_path = LAMS_DIR / "lams_metadata_summary_by_scenario.csv"
    scenario_summary.to_csv(summary_path)
    log(f"Saved LAMS metadata summary: {summary_path}")


# ============================================================
# Section 14 — Main execution
# ------------------------------------------------------------
# Purpose:
# - Execute true LAMS construction in a reproducible order.
# ============================================================

def main() -> None:
    log("Starting standalone true LAMS construction.")
    log(f"Repository root: {REPO_ROOT}")
    log("True LAMS configuration:")
    log(f"  Output image size: {LAMS_SIZE} × {LAMS_SIZE}")
    log(f"  Raster resolution: {LAMS_RASTER_RES_M} m/pixel")
    log(f"  Wall raster line width: {LAMS_WALL_LINE_WIDTH_PX} px")
    log(f"  Span scale: {LAMS_SPAN_SCALE}")
    log(f"  Output directory: {LAMS_DIR}")

    split_data = load_wall_context_splits()
    validate_required_columns(split_data)

    floorplan_registry = load_floorplan_registry()

    lams_all_rows, lams_geometries, geom_key_cols = build_lams_rows_and_geometries(
        split_data=split_data
    )

    lams_bank, lams_metadata = build_lams_bank(
        lams_all_rows=lams_all_rows,
        lams_geometries=lams_geometries,
        floorplan_registry=floorplan_registry,
    )

    save_core_lams_outputs(
        lams_bank=lams_bank,
        lams_metadata=lams_metadata,
        lams_all_rows=lams_all_rows,
    )

    build_and_save_rss_lams_tables(
        lams_all_rows=lams_all_rows,
        lams_metadata=lams_metadata,
    )

    build_and_save_rtt_lams_tables(
        lams_all_rows=lams_all_rows,
        lams_metadata=lams_metadata,
    )

    log("floorplan_images/build_lams.py completed successfully.")


if __name__ == "__main__":
    main()