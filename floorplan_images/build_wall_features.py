# ============================================================
# Section 46 — Load and audit vector floorplans
# ------------------------------------------------------------
# Purpose:
# - Load the finalized vector wall maps from the floorplan notebook.
# - Verify that the wall registry is structurally usable before feature
#   extraction.
#
# Scientific note:
# - This section does not train any model.
# - It only imports the manually finalized wall geometry.
# - The geometry-only baseline remains untouched.
# - The wall-aware branch starts here as a separate, controlled extension.
#
# Required input:
# - vector_floorplans_with_internal_walls.json
#
# Expected detailed wall scenarios:
# - building
# - corridor
# - apartment
# ============================================================



# ============================================================
# floorplan_images/build_wall_features.py
# ------------------------------------------------------------
# Purpose:
# - Build vector-floorplan wall/context features as a standalone repo script.
# - Replace notebook-runtime variables such as DATA_PARENT, OUTPUT_DIR,
#   all_links, train_df, subtrain_df, and val_df.
#
# Scientific protocol:
# - This script does not train, tune, or evaluate any model.
# - It only creates geometry-derived wall/context features.
# - RSS/RTT labels are not used to construct the wall features.
# - Shapely is used for trusted computational geometry operations.
# - The official train/test split is preserved.
# - The RP-level validation split produced by data/build_dataset.py is preserved.
# - No artificial outputs are forced. Missing required files stop the script.
#
# Inputs:
# - output_csvs/processed_features/row_train.csv
# - output_csvs/processed_features/row_val.csv
# - output_csvs/processed_features/row_test.csv
# - output_csvs/processed_features/geometry_train.csv
# - output_csvs/processed_features/geometry_val.csv
# - output_csvs/processed_features/geometry_test.csv
# - outputs_floorplan_vector_builder/json/vector_floorplans_with_internal_walls.json
#
# Outputs:
# - output_csvs/processed_features/row_train_wall_obstruction.csv
# - output_csvs/processed_features/row_val_wall_obstruction.csv
# - output_csvs/processed_features/row_test_wall_obstruction.csv
# - output_csvs/processed_features/row_train_wall_context.csv
# - output_csvs/processed_features/row_val_wall_context.csv
# - output_csvs/processed_features/row_test_wall_context.csv
# - output_csvs/processed_features/geometry_train_wall_obstruction.csv
# - output_csvs/processed_features/geometry_val_wall_obstruction.csv
# - output_csvs/processed_features/geometry_test_wall_obstruction.csv
# - output_csvs/processed_features/geometry_train_wall_context.csv
# - output_csvs/processed_features/geometry_val_wall_context.csv
# - output_csvs/processed_features/geometry_test_wall_context.csv
# - output_csvs/processed_features/wall_feature_stage/*.csv
# ============================================================

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
except Exception as exc:
    raise ImportError(
        "Shapely is required for wall feature extraction. "
        "Install it with: pip install shapely"
    ) from exc


# ============================================================
# Section 1 — Repository paths and constants
# ------------------------------------------------------------
# Purpose:
# - Resolve all required paths relative to the repository root.
#
# Scientific protocol:
# - No absolute Windows path is hard-coded.
# - The script can be run from any terminal location.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = REPO_ROOT / "output_csvs" / "processed_features"
WALL_FEATURE_DIR = PROCESSED_DIR / "wall_feature_stage"
WALL_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

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

GEOMETRY_KEYS = ["scenario", "AP_x", "AP_y", "RX_x", "RX_y"]

BASE_FEATURES = [
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

FEATURES_GEOM_ONLY = list(BASE_FEATURES)
FEATURES_WALL_OBS = list(BASE_FEATURES) + list(WALL_OBSTRUCTION_FEATURES)
FEATURES_WALL_OBS_CONTEXT = list(FEATURES_WALL_OBS) + list(ENDPOINT_WALL_CONTEXT_FEATURES)

NO_WALL_DISTANCE = 999.0
NEAR_WALL_RADIUS_M = 1.0


# ============================================================
# Section 2 — Progress logger and file guards
# ------------------------------------------------------------
# Purpose:
# - Provide concise progress notifications.
# - Fail clearly when required files are missing.
# ============================================================

def log(message: str) -> None:
    print(f"[build_wall_features] {message}", flush=True)


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
        "Place the finalized vector floorplan JSON in one of these locations, "
        "or run the floorplan vector builder first."
    )


# ============================================================
# Section 3 — Load processed split tables
# ------------------------------------------------------------
# Purpose:
# - Load row-level and geometry-level split files created by build_dataset.py.
#
# Scientific protocol:
# - These split files already preserve the official train/test split.
# - This script only adds deterministic geometry-derived columns.
# ============================================================

def load_split_tables() -> Dict[str, pd.DataFrame]:
    split_paths = {
        "row_train": PROCESSED_DIR / "row_train.csv",
        "row_val": PROCESSED_DIR / "row_val.csv",
        "row_test": PROCESSED_DIR / "row_test.csv",
        "geometry_train": PROCESSED_DIR / "geometry_train.csv",
        "geometry_val": PROCESSED_DIR / "geometry_val.csv",
        "geometry_test": PROCESSED_DIR / "geometry_test.csv",
    }

    split_data: Dict[str, pd.DataFrame] = {}

    for split_name, path in split_paths.items():
        require_file(path)
        split_data[split_name] = pd.read_csv(path)
        log(f"Loaded {split_name}: {len(split_data[split_name]):,} rows")

    return split_data


def validate_geometry_columns(df: pd.DataFrame, table_name: str) -> None:
    missing = [c for c in GEOMETRY_KEYS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required geometry columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def build_unique_link_table(split_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build one unique AP-RP geometry table across all available splits.

    Features are computed once per unique geometry and then merged back.
    This avoids recomputing Shapely intersections for repeated scans.
    """
    frames = []

    for split_name, df in split_data.items():
        validate_geometry_columns(df, split_name)
        frames.append(df[GEOMETRY_KEYS])

    unique_links = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    log(f"Unique AP-RP geometries found: {len(unique_links):,}")
    return unique_links


# ============================================================
# Section 4 — Load and audit vector floorplans
# ------------------------------------------------------------
# Purpose:
# - Load the finalized vector wall maps.
# - Verify that the wall registry is structurally usable.
#
# Scientific protocol:
# - This section only imports manually finalized wall geometry.
# - It does not alter the geometry-only baseline.
# ============================================================

def load_floorplan_registry() -> dict:
    floorplan_json = find_floorplan_json()
    log(f"Loading vector floorplans from: {floorplan_json}")

    with floorplan_json.open("r", encoding="utf-8") as f:
        floorplan_registry = json.load(f)

    log(f"Loaded vector floorplan scenarios: {list(floorplan_registry.keys())}")
    return floorplan_registry


def audit_floorplan_registry(floorplan_registry: dict) -> pd.DataFrame:
    audit_rows = []
    errors = []

    for scenario_name, fp in floorplan_registry.items():
        walls = fp.get("walls", [])
        tags = [w.get("tag", "") for w in walls]
        roles = [w.get("role", "missing") for w in walls]

        duplicate_tags = sorted(
            [tag for tag, count in Counter(tags).items() if tag and count > 1]
        )
        missing_role_count = sum(1 for r in roles if r == "missing")
        role_counts = Counter(roles)

        if duplicate_tags:
            errors.append(f"{scenario_name}: duplicate wall tags: {duplicate_tags[:10]}")

        if missing_role_count > 0:
            errors.append(f"{scenario_name}: {missing_role_count} walls missing role field")

        audit_rows.append(
            {
                "scenario": scenario_name,
                "n_walls": len(walls),
                "n_external": role_counts.get("external", 0),
                "n_internal": role_counts.get("internal", 0),
                "n_existing": role_counts.get("existing", 0),
                "n_missing_role": missing_role_count,
                "draw_outer_box": fp.get("draw_outer_box", True),
                "n_duplicate_tags": len(duplicate_tags),
            }
        )

    audit_df = pd.DataFrame(audit_rows).sort_values("scenario").reset_index(drop=True)
    audit_path = WALL_FEATURE_DIR / "floorplan_wall_registry_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    log("Floorplan wall registry audit:")
    log(audit_df.to_string(index=False))
    log(f"Saved audit table: {audit_path}")

    if errors:
        error_text = "\n".join(f"- {e}" for e in errors)
        raise RuntimeError(
            "Wall registry audit found problems:\n"
            f"{error_text}\n\n"
            "Fix the wall registry before feature extraction."
        )

    required_wall_scenarios = ["building", "corridor", "apartment"]
    missing_required = [s for s in required_wall_scenarios if s not in floorplan_registry]

    if missing_required:
        raise RuntimeError(
            f"Missing expected detailed wall scenarios: {missing_required}"
        )

    log("Vector floorplan registry is ready.")
    return audit_df


# ============================================================
# Section 5 — Convert vector walls to Shapely geometries
# ------------------------------------------------------------
# Purpose:
# - Convert JSON wall definitions into Shapely LineString objects.
#
# Scientific protocol:
# - Shapely is used for line geometry.
# - No custom computational geometry is introduced.
# ============================================================

def wall_to_linestring(wall: dict) -> LineString:
    wall_type = wall.get("type", None)

    if wall_type == "v":
        return LineString(
            [
                (float(wall["x"]), float(wall["y0"])),
                (float(wall["x"]), float(wall["y1"])),
            ]
        )

    if wall_type == "h":
        return LineString(
            [
                (float(wall["x0"]), float(wall["y"])),
                (float(wall["x1"]), float(wall["y"])),
            ]
        )

    if wall_type == "segment":
        return LineString(
            [
                (float(wall["x0"]), float(wall["y0"])),
                (float(wall["x1"]), float(wall["y1"])),
            ]
        )

    raise ValueError(f"Unsupported wall type: {wall_type}. Wall={wall}")


def convert_walls_to_shapely(floorplan_registry: dict) -> Dict[str, List[dict]]:
    scenario_wall_geoms: Dict[str, List[dict]] = {}

    for scenario_name, fp in floorplan_registry.items():
        scenario_wall_geoms[scenario_name] = []

        for wall in fp.get("walls", []):
            line = wall_to_linestring(wall)

            if line.is_empty or line.length <= 0:
                log(f"Skipping zero-length wall in {scenario_name}: {wall.get('tag')}")
                continue

            scenario_wall_geoms[scenario_name].append(
                {
                    "scenario": scenario_name,
                    "tag": wall.get("tag", ""),
                    "role": wall.get("role", "existing"),
                    "geometry": line,
                    "length_m": float(line.length),
                }
            )

    audit_rows = []

    for scenario_name, items in scenario_wall_geoms.items():
        role_counts = Counter([it["role"] for it in items])
        audit_rows.append(
            {
                "scenario": scenario_name,
                "n_lines": len(items),
                "n_external": role_counts.get("external", 0),
                "n_internal": role_counts.get("internal", 0),
                "n_existing": role_counts.get("existing", 0),
                "total_wall_length_m": sum(it["length_m"] for it in items),
            }
        )

    wall_geom_audit = pd.DataFrame(audit_rows).sort_values("scenario").reset_index(drop=True)
    wall_geom_audit_path = WALL_FEATURE_DIR / "wall_geometry_audit.csv"
    wall_geom_audit.to_csv(wall_geom_audit_path, index=False)

    log("Shapely wall geometry audit:")
    log(wall_geom_audit.to_string(index=False))
    log(f"Saved wall geometry audit: {wall_geom_audit_path}")

    return scenario_wall_geoms


# ============================================================
# Section 6 — Wall-obstruction feature extraction
# ------------------------------------------------------------
# Purpose:
# - Compute direct AP-RP wall-crossing features.
#
# Scientific protocol:
# - Features depend only on AP/RP coordinates and vector walls.
# - RSS/RTT labels are not used.
# - Features are computed once per unique AP-RP geometry.
# ============================================================

def extract_points_from_intersection(geom) -> list:
    """
    Extract point-like intersections from a Shapely intersection geometry.

    Line overlaps are not counted as wall crossings. This avoids false
    positives when an AP-RP path runs along a wall.
    """
    if geom.is_empty:
        return []

    if geom.geom_type == "Point":
        return [geom]

    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        points = []
        for g in geom.geoms:
            points.extend(extract_points_from_intersection(g))
        return points

    return []


def compute_wall_obstruction_for_link(
    row: pd.Series,
    scenario_wall_geoms: Dict[str, List[dict]],
    endpoint_eps: float = 1e-6,
) -> dict:
    scenario = row["scenario"]
    walls = scenario_wall_geoms.get(scenario, [])

    ap = (float(row["AP_x"]), float(row["AP_y"]))
    rp = (float(row["RX_x"]), float(row["RX_y"]))

    link = LineString([ap, rp])
    link_length = float(link.length)

    if link_length <= 0 or not walls:
        return {
            "wall_cross_count_total": 0,
            "wall_cross_count_internal": 0,
            "wall_cross_count_external": 0,
            "obstructed_path": 0,
        }

    seen_internal_t = set()
    seen_external_t = set()

    for wall_item in walls:
        wall_line = wall_item["geometry"]
        role = wall_item.get("role", "existing")

        intersection = link.intersection(wall_line)
        points = extract_points_from_intersection(intersection)

        for point in points:
            d_along = float(link.project(point))
            t = d_along / link_length

            if t <= endpoint_eps or t >= 1.0 - endpoint_eps:
                continue

            t_key = round(t, 6)

            if role == "external":
                seen_external_t.add(t_key)
            else:
                seen_internal_t.add(t_key)

    internal_count = len(seen_internal_t)
    external_count = len(seen_external_t)
    total_count = internal_count + external_count

    return {
        "wall_cross_count_total": int(total_count),
        "wall_cross_count_internal": int(internal_count),
        "wall_cross_count_external": int(external_count),
        "obstructed_path": int(total_count > 0),
    }


def compute_wall_obstruction_features(
    unique_links: pd.DataFrame,
    scenario_wall_geoms: Dict[str, List[dict]],
) -> pd.DataFrame:
    log("Computing wall-obstruction features.")
    start_time = time.time()

    feature_rows = []

    for i, row in unique_links.iterrows():
        if (i + 1) % 2000 == 0:
            log(f"Processed {i + 1:,}/{len(unique_links):,} unique links")

        feature_rows.append(
            compute_wall_obstruction_for_link(row, scenario_wall_geoms)
        )

    wall_features = pd.concat(
        [unique_links.reset_index(drop=True), pd.DataFrame(feature_rows)],
        axis=1,
    )

    elapsed = time.time() - start_time
    log(f"Wall-obstruction feature extraction finished in {elapsed:.2f} s")

    path = WALL_FEATURE_DIR / "unique_wall_obstruction_features.csv"
    wall_features.to_csv(path, index=False)
    log(f"Saved unique wall-obstruction features: {path}")

    return wall_features


# ============================================================
# Section 7 — Endpoint wall-context feature extraction
# ------------------------------------------------------------
# Purpose:
# - Compute local AP/RP distance-to-wall context features.
#
# Scientific protocol:
# - Uses Shapely Point.distance(LineString).
# - No custom distance algorithm is introduced.
# - Features use only geometry and floorplans.
# ============================================================

def nearest_wall_distance(
    point: Point,
    wall_items: List[dict],
    role_filter: Optional[str] = None,
    no_wall_distance: float = NO_WALL_DISTANCE,
) -> float:
    if role_filter is None:
        selected = wall_items
    else:
        selected = [w for w in wall_items if w.get("role") == role_filter]

    if len(selected) == 0:
        return float(no_wall_distance)

    return float(min(point.distance(w["geometry"]) for w in selected))


def compute_endpoint_context_for_link(
    row: pd.Series,
    scenario_wall_geoms: Dict[str, List[dict]],
) -> dict:
    scenario = row["scenario"]
    walls = scenario_wall_geoms.get(scenario, [])

    ap_point = Point(float(row["AP_x"]), float(row["AP_y"]))
    rp_point = Point(float(row["RX_x"]), float(row["RX_y"]))

    ap_dist_any = nearest_wall_distance(ap_point, walls, role_filter=None)
    ap_dist_ext = nearest_wall_distance(ap_point, walls, role_filter="external")
    ap_dist_int = nearest_wall_distance(ap_point, walls, role_filter="internal")

    rp_dist_any = nearest_wall_distance(rp_point, walls, role_filter=None)
    rp_dist_ext = nearest_wall_distance(rp_point, walls, role_filter="external")
    rp_dist_int = nearest_wall_distance(rp_point, walls, role_filter="internal")

    return {
        "AP_dist_nearest_wall": ap_dist_any,
        "AP_dist_nearest_external_wall": ap_dist_ext,
        "AP_dist_nearest_internal_wall": ap_dist_int,
        "RP_dist_nearest_wall": rp_dist_any,
        "RP_dist_nearest_external_wall": rp_dist_ext,
        "RP_dist_nearest_internal_wall": rp_dist_int,
        "AP_near_wall_1m": int(ap_dist_any <= NEAR_WALL_RADIUS_M),
        "RP_near_wall_1m": int(rp_dist_any <= NEAR_WALL_RADIUS_M),
    }


def compute_endpoint_context_features(
    unique_links: pd.DataFrame,
    scenario_wall_geoms: Dict[str, List[dict]],
) -> pd.DataFrame:
    log("Computing endpoint wall-context features.")
    start_time = time.time()

    feature_rows = []

    for i, row in unique_links.iterrows():
        if (i + 1) % 2000 == 0:
            log(f"Processed {i + 1:,}/{len(unique_links):,} unique links")

        feature_rows.append(
            compute_endpoint_context_for_link(row, scenario_wall_geoms)
        )

    endpoint_features = pd.concat(
        [unique_links.reset_index(drop=True), pd.DataFrame(feature_rows)],
        axis=1,
    )

    elapsed = time.time() - start_time
    log(f"Endpoint wall-context feature extraction finished in {elapsed:.2f} s")

    path = WALL_FEATURE_DIR / "unique_endpoint_wall_context_features.csv"
    endpoint_features.to_csv(path, index=False)
    log(f"Saved unique endpoint wall-context features: {path}")

    return endpoint_features


# ============================================================
# Section 8 — Merge and save split-level outputs
# ------------------------------------------------------------
# Purpose:
# - Merge computed wall features back into each split file.
#
# Scientific protocol:
# - Split membership is unchanged.
# - Row order is preserved.
# - The only change is adding deterministic geometry-derived columns.
# ============================================================

def merge_feature_table(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_cols: List[str],
    table_name: str,
) -> pd.DataFrame:
    validate_geometry_columns(df, table_name)

    out = df.merge(
        feature_table[GEOMETRY_KEYS + feature_cols],
        on=GEOMETRY_KEYS,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(out) != len(df):
        raise RuntimeError(
            f"Length changed after merging features for {table_name}: "
            f"{len(df)} -> {len(out)}"
        )

    for col in feature_cols:
        if col in WALL_OBSTRUCTION_FEATURES:
            out[col] = out[col].fillna(0).astype(int)
        elif col in ["AP_near_wall_1m", "RP_near_wall_1m"]:
            out[col] = out[col].fillna(0).astype(int)
        else:
            out[col] = out[col].fillna(NO_WALL_DISTANCE).astype(float)

    return out


def save_split_outputs(
    split_data: Dict[str, pd.DataFrame],
    wall_obstruction_features: pd.DataFrame,
    endpoint_context_features: pd.DataFrame,
) -> None:
    wall_context_features = wall_obstruction_features.merge(
        endpoint_context_features,
        on=GEOMETRY_KEYS,
        how="left",
        validate="one_to_one",
    )

    for split_name, df in split_data.items():
        wall_obs_df = merge_feature_table(
            df=df,
            feature_table=wall_obstruction_features,
            feature_cols=WALL_OBSTRUCTION_FEATURES,
            table_name=split_name,
        )

        wall_context_df = merge_feature_table(
            df=df,
            feature_table=wall_context_features,
            feature_cols=WALL_OBSTRUCTION_FEATURES + ENDPOINT_WALL_CONTEXT_FEATURES,
            table_name=split_name,
        )

        wall_obs_path = PROCESSED_DIR / f"{split_name}_wall_obstruction.csv"
        wall_context_path = PROCESSED_DIR / f"{split_name}_wall_context.csv"

        wall_obs_df.to_csv(wall_obs_path, index=False)
        wall_context_df.to_csv(wall_context_path, index=False)

        log(f"Saved {split_name} wall-obstruction file: {wall_obs_path}")
        log(f"Saved {split_name} wall-context file:     {wall_context_path}")

    all_wall_context_path = WALL_FEATURE_DIR / "all_unique_links_wall_context.csv"
    wall_context_features.to_csv(all_wall_context_path, index=False)
    log(f"Saved all unique-link wall-context table: {all_wall_context_path}")


# ============================================================
# Section 9 — Summaries and metadata
# ------------------------------------------------------------
# Purpose:
# - Save lightweight audit summaries for reproducibility.
# ============================================================

def save_summaries(
    split_data: Dict[str, pd.DataFrame],
    wall_obstruction_features: pd.DataFrame,
    endpoint_context_features: pd.DataFrame,
) -> None:
    wall_summary = (
        wall_obstruction_features
        .groupby("scenario")[WALL_OBSTRUCTION_FEATURES]
        .agg(["mean", "max"])
    )

    wall_summary_path = WALL_FEATURE_DIR / "wall_obstruction_feature_summary_by_scenario.csv"
    wall_summary.to_csv(wall_summary_path)
    log(f"Saved wall-obstruction summary: {wall_summary_path}")

    endpoint_summary = (
        endpoint_context_features
        .groupby("scenario")[ENDPOINT_WALL_CONTEXT_FEATURES]
        .agg(["mean", "min", "max"])
    )

    endpoint_summary_path = WALL_FEATURE_DIR / "endpoint_wall_context_summary_by_scenario.csv"
    endpoint_summary.to_csv(endpoint_summary_path)
    log(f"Saved endpoint-context summary: {endpoint_summary_path}")

    metadata = {
        "BASE_FEATURES": BASE_FEATURES,
        "WALL_OBSTRUCTION_FEATURES": WALL_OBSTRUCTION_FEATURES,
        "ENDPOINT_WALL_CONTEXT_FEATURES": ENDPOINT_WALL_CONTEXT_FEATURES,
        "FEATURES_GEOM_ONLY": FEATURES_GEOM_ONLY,
        "FEATURES_WALL_OBS": FEATURES_WALL_OBS,
        "FEATURES_WALL_OBS_CONTEXT": FEATURES_WALL_OBS_CONTEXT,
        "notes": {
            "script": "floorplan_images/build_wall_features.py",
            "uses_labels": False,
            "preserves_train_test_split": True,
            "preserves_validation_split": True,
            "geometry_engine": "shapely",
        },
    }

    metadata_path = WALL_FEATURE_DIR / "wall_feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"Saved wall feature metadata: {metadata_path}")

    row_context = pd.read_csv(PROCESSED_DIR / "row_train_wall_context.csv")
    log("Controlled feature ladder:")
    log(f"  FEATURES_GEOM_ONLY:        {len(FEATURES_GEOM_ONLY)} features")
    log(f"  FEATURES_WALL_OBS:         {len(FEATURES_WALL_OBS)} features")
    log(f"  FEATURES_WALL_OBS_CONTEXT: {len(FEATURES_WALL_OBS_CONTEXT)} features")

    missing = [c for c in FEATURES_WALL_OBS_CONTEXT if c not in row_context.columns]
    if missing:
        raise RuntimeError(f"Saved row_train_wall_context.csv is missing features: {missing}")


# ============================================================
# Section 10 — Main execution
# ------------------------------------------------------------
# Purpose:
# - Execute wall/context feature construction in a reproducible order.
# ============================================================

def main() -> None:
    log("Starting standalone wall/context feature construction.")
    log(f"Repository root: {REPO_ROOT}")

    split_data = load_split_tables()
    unique_links = build_unique_link_table(split_data)

    floorplan_registry = load_floorplan_registry()
    audit_floorplan_registry(floorplan_registry)

    scenario_wall_geoms = convert_walls_to_shapely(floorplan_registry)

    wall_obstruction_features = compute_wall_obstruction_features(
        unique_links=unique_links,
        scenario_wall_geoms=scenario_wall_geoms,
    )

    endpoint_context_features = compute_endpoint_context_features(
        unique_links=unique_links,
        scenario_wall_geoms=scenario_wall_geoms,
    )

    save_split_outputs(
        split_data=split_data,
        wall_obstruction_features=wall_obstruction_features,
        endpoint_context_features=endpoint_context_features,
    )

    save_summaries(
        split_data=split_data,
        wall_obstruction_features=wall_obstruction_features,
        endpoint_context_features=endpoint_context_features,
    )

    log("floorplan_images/build_wall_features.py completed successfully.")


if __name__ == "__main__":
    main()