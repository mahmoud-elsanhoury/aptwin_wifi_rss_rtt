# ============================================================
# data/build_dataset.py
# ------------------------------------------------------------
# Purpose:
# - Convert raw Wi-Fi RTT/RSS CSV files into standalone AP-link datasets.
# - Preserve the official train/test split.
# - Create a grouped RP-level validation split only from official training data.
# - Save row-level and geometry-level CSV files for downstream repo scripts.
#
# Scientific protocol:
# - No test data are used for validation, scaling, or model selection.
# - Repeated-scan leakage is avoided by GroupShuffleSplit on rp_key.
# - Sentinel RSS/RTT values are excluded from regression targets.
# - Dead APs are detected from TRAIN only and consistently removed.
# - Scalers are fitted on subtrain only.
#
# Outputs:
# - output_csvs/processed_features/all_ap_link_table.csv
# - output_csvs/processed_features/dead_ap_report.csv
# - output_csvs/processed_features/scenario_load_summary.csv
# - output_csvs/processed_features/dataset_audit_summary.csv
# - output_csvs/processed_features/imbalance_report.csv
# - output_csvs/processed_features/row_train.csv
# - output_csvs/processed_features/row_val.csv
# - output_csvs/processed_features/row_test.csv
# - output_csvs/processed_features/geometry_train.csv
# - output_csvs/processed_features/geometry_val.csv
# - output_csvs/processed_features/geometry_test.csv
# - models/pretraining_setup/x_scaler.joblib
# - models/pretraining_setup/y_scaler.joblib
# - models/pretraining_setup/feature_target_metadata.json
# ============================================================

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# Section 1 — Repository configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
BATCH1_DIR = DATA_DIR / "first_batch"
BATCH2_DIR = DATA_DIR / "second_batch"

AP_LOCATION_DIR = REPO_ROOT / "ap_locations"

OUTPUT_DIR = REPO_ROOT / "output_csvs" / "processed_features"
AUDIT_DIR = REPO_ROOT / "output_csvs" / "audits"
MODEL_PREP_DIR = REPO_ROOT / "models" / "pretraining_setup"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PREP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

COORD_FILE_BATCH1 = AP_LOCATION_DIR / "Floor+office+apartment_AP_coords.txt"
COORD_FILE_BATCH2 = AP_LOCATION_DIR / "lecture theatre+office+corridor_AP_position.txt"

SCENARIOS = {
    "building": {
        "batch": "batch1",
        "coord_block": "floor",
        "coord_file": COORD_FILE_BATCH1,
        "train_csv": BATCH1_DIR / "database_building_train.csv",
        "test_csv": BATCH1_DIR / "database_building_test.csv",
        "area_type": "large_full_floor",
    },
    "office": {
        "batch": "batch1",
        "coord_block": "office",
        "coord_file": COORD_FILE_BATCH1,
        "train_csv": BATCH1_DIR / "database_office_train.csv",
        "test_csv": BATCH1_DIR / "database_office_test.csv",
        "area_type": "small_los_room",
    },
    "apartment": {
        "batch": "batch1",
        "coord_block": "apartment",
        "coord_file": COORD_FILE_BATCH1,
        "train_csv": BATCH1_DIR / "database_apartment_train.csv",
        "test_csv": BATCH1_DIR / "database_apartment_test.csv",
        "area_type": "small_mixed_apartment",
    },
    "lecture_theatre": {
        "batch": "batch2",
        "coord_block": "lecture theatre",
        "coord_file": COORD_FILE_BATCH2,
        "train_csv": BATCH2_DIR / "database_lecture_theatre_train.csv",
        "test_csv": BATCH2_DIR / "database_lecture_theatre_test.csv",
        "area_type": "large_los_open",
    },
    "corridor": {
        "batch": "batch2",
        "coord_block": "corridor",
        "coord_file": COORD_FILE_BATCH2,
        "train_csv": BATCH2_DIR / "database_corridor_train.csv",
        "test_csv": BATCH2_DIR / "database_corridor_test.csv",
        "area_type": "long_narrow_nlos",
    },
    "office2": {
        "batch": "batch2",
        "coord_block": "office",
        "coord_file": COORD_FILE_BATCH2,
        "train_csv": BATCH2_DIR / "database_office2_train.csv",
        "test_csv": BATCH2_DIR / "database_office2_test.csv",
        "area_type": "medium_mixed_office",
    },
}

BASE_FEATURES = [
    "RX_x", "RX_y",
    "AP_x", "AP_y",
    "dx", "dy",
    "distance",
    "LOS_flag", "LOS_known",
]

TARGETS = ["RSS_dBm", "RTT_m"]

RSS_SENTINEL = -200.0
RTT_SENTINEL_MM = 100000.0


# ============================================================
# Section 2 — AP-coordinate parser
# ============================================================

def normalize_block_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.rstrip(":")
    name = re.sub(r"\s+", " ", name)
    return name


def _parse_float_list(line: str) -> list[float]:
    if ":" not in line:
        return []
    right = line.split(":", 1)[1]
    return [float(x.strip()) for x in right.split(",") if x.strip() != ""]


def parse_ap_coordinate_txt(path: Path) -> dict[str, list[tuple[float, float]]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {path}")

    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    blocks: dict[str, list[tuple[float, float]]] = {}
    current_name = None
    current_x = None

    for ln in lines:
        low = ln.lower().strip()

        if low.startswith("ap x"):
            current_x = _parse_float_list(ln)

        elif low.startswith("ap y"):
            y_vals = _parse_float_list(ln)

            if current_name is None or current_x is None:
                raise ValueError(f"Malformed coordinate block near line: {ln}")

            if len(current_x) != len(y_vals):
                raise ValueError(
                    f"X/Y length mismatch in block '{current_name}': "
                    f"{len(current_x)} X values vs {len(y_vals)} Y values"
                )

            block_name = normalize_block_name(current_name)
            blocks[block_name] = list(zip(current_x, y_vals))
            current_x = None

        else:
            current_name = ln.strip()

    return blocks


# ============================================================
# Section 3 — Raw CSV helpers
# ============================================================

def find_ap_measurement_columns(df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    rtt_cols: dict[int, str] = {}
    rss_cols: dict[int, str] = {}

    rtt_pattern = re.compile(r"AP\s*(\d+)\s*RTT", re.IGNORECASE)
    rss_pattern = re.compile(r"AP\s*(\d+)\s*RSS", re.IGNORECASE)

    for col in df.columns:
        col_clean = str(col).strip()

        m_rtt = rtt_pattern.search(col_clean)
        if m_rtt:
            rtt_cols[int(m_rtt.group(1))] = col

        m_rss = rss_pattern.search(col_clean)
        if m_rss:
            rss_cols[int(m_rss.group(1))] = col

    return rtt_cols, rss_cols


def find_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    x_candidates = ["x", "rx_x", "rp_x"]
    y_candidates = ["y", "rx_y", "rp_y"]

    x_col = next((cols_lower[c] for c in x_candidates if c in cols_lower), None)
    y_col = next((cols_lower[c] for c in y_candidates if c in cols_lower), None)

    if x_col is None or y_col is None:
        raise ValueError(
            "Could not identify X and Y coordinate columns. "
            f"Available columns are: {list(df.columns)}"
        )

    return x_col, y_col


def find_los_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        low = str(col).strip().lower()
        if "los" in low and "ap" in low:
            return col
    return None


def parse_los_aps(value) -> set[int]:
    if pd.isna(value):
        return set()

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return set()

    ids = re.findall(r"\d+", text)
    return set(int(x) for x in ids)


def valid_rss_array(x) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(arr) & (arr > RSS_SENTINEL + 1e-9)


def valid_rtt_mm_array(x) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(arr) & (arr < RTT_SENTINEL_MM - 1e-9)


# ============================================================
# Section 4 — Scenario loader
# ============================================================

def detect_dead_aps_from_train_wide(
    train_df: pd.DataFrame,
    rtt_cols: dict[int, str],
    rss_cols: dict[int, str],
) -> tuple[list[int], pd.DataFrame]:
    common_aps = sorted(set(rtt_cols).intersection(set(rss_cols)))
    dead_aps = []
    report_rows = []

    for ap in common_aps:
        rss_valid = valid_rss_array(train_df[rss_cols[ap]])
        rtt_valid = valid_rtt_mm_array(train_df[rtt_cols[ap]])

        is_dead = (rss_valid.sum() == 0) and (rtt_valid.sum() == 0)

        report_rows.append({
            "AP_id_raw": ap,
            "n_rows": len(train_df),
            "RSS_valid_count": int(rss_valid.sum()),
            "RTT_valid_count": int(rtt_valid.sum()),
            "RSS_valid_rate": float(rss_valid.mean()),
            "RTT_valid_rate": float(rtt_valid.mean()),
            "is_dead_ap": bool(is_dead),
        })

        if is_dead:
            dead_aps.append(ap)

    return dead_aps, pd.DataFrame(report_rows)


def get_coord_list_for_scenario(
    scenario_name: str,
    coord_blocks: dict[str, dict[str, list[tuple[float, float]]]],
) -> list[tuple[float, float]]:
    cfg = SCENARIOS[scenario_name]
    batch = cfg["batch"]
    block = normalize_block_name(cfg["coord_block"])

    blocks = coord_blocks[batch]
    if block not in blocks:
        raise KeyError(f"Coordinate block '{block}' not found for scenario '{scenario_name}'")

    return blocks[block]


def assign_ap_coordinates(
    scenario_name: str,
    usable_ap_ids: list[int],
    coord_blocks: dict[str, dict[str, list[tuple[float, float]]]],
) -> dict[int, tuple[float, float]]:
    coords = get_coord_list_for_scenario(scenario_name, coord_blocks)
    usable_ap_ids = sorted(usable_ap_ids)

    if len(usable_ap_ids) != len(coords):
        raise ValueError(
            f"Coordinate mismatch for {scenario_name}: "
            f"{len(usable_ap_ids)} usable AP columns but {len(coords)} AP coordinates."
        )

    return {ap_id: coords[i] for i, ap_id in enumerate(usable_ap_ids)}


def build_long_table_from_wide(
    df: pd.DataFrame,
    scenario_name: str,
    split_name: str,
    ap_mapping: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    cfg = SCENARIOS[scenario_name]
    rtt_cols, rss_cols = find_ap_measurement_columns(df)
    x_col, y_col = find_xy_columns(df)
    los_col = find_los_column(df)

    rx_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    rx_y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    if los_col is not None:
        los_sets = df[los_col].apply(parse_los_aps).tolist()
    else:
        los_sets = [set() for _ in range(len(df))]

    pieces = []

    for usable_index, ap_id in enumerate(sorted(ap_mapping), start=1):
        if ap_id not in rss_cols or ap_id not in rtt_cols:
            raise KeyError(
                f"{scenario_name} {split_name}: AP{ap_id} is missing RSS or RTT column."
            )

        ap_x, ap_y = ap_mapping[ap_id]

        rss_raw = pd.to_numeric(df[rss_cols[ap_id]], errors="coerce").to_numpy(dtype=float)
        rtt_raw_mm = pd.to_numeric(df[rtt_cols[ap_id]], errors="coerce").to_numpy(dtype=float)

        rss_valid = np.isfinite(rss_raw) & (rss_raw > RSS_SENTINEL + 1e-9)
        rtt_valid = np.isfinite(rtt_raw_mm) & (rtt_raw_mm < RTT_SENTINEL_MM - 1e-9)

        los_flag = np.array([1 if ap_id in s else 0 for s in los_sets], dtype=int)
        los_known = np.array([1 if los_col is not None else 0 for _ in range(len(df))], dtype=int)

        dx = rx_x - ap_x
        dy = rx_y - ap_y
        distance = np.sqrt(dx**2 + dy**2)

        tmp = pd.DataFrame({
            "scenario": scenario_name,
            "batch": cfg["batch"],
            "split": split_name,
            "area_type": cfg["area_type"],
            "raw_row_id": np.arange(len(df), dtype=int),
            "RX_x": rx_x,
            "RX_y": rx_y,
            "AP_id_raw": int(ap_id),
            "AP_index_usable": int(usable_index),
            "AP_x": float(ap_x),
            "AP_y": float(ap_y),
            "dx": dx,
            "dy": dy,
            "distance": distance,
            "LOS_flag": los_flag,
            "LOS_known": los_known,
            "RSS_dBm": rss_raw,
            "RTT_m": rtt_raw_mm / 1000.0,
            "RSS_valid": rss_valid,
            "RTT_valid": rtt_valid,
            "is_dead_ap": False,
        })

        tmp["rp_key"] = (
            tmp["scenario"].astype(str) + "_" +
            tmp["RX_x"].round(4).astype(str) + "_" +
            tmp["RX_y"].round(4).astype(str)
        )

        tmp["scan_key"] = (
            tmp["scenario"].astype(str) + "_" +
            tmp["split"].astype(str) + "_" +
            tmp["raw_row_id"].astype(str)
        )

        tmp["link_key"] = (
            tmp["scenario"].astype(str) + "_" +
            tmp["rp_key"].astype(str) + "_AP" +
            tmp["AP_id_raw"].astype(str)
        )

        pieces.append(tmp)

    return pd.concat(pieces, ignore_index=True)


def load_scenario_train_test(
    scenario_name: str,
    coord_blocks: dict[str, dict[str, list[tuple[float, float]]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = SCENARIOS[scenario_name]
    train_path = Path(cfg["train_csv"])
    test_path = Path(cfg["test_csv"])

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file for {scenario_name}: {train_path}")

    train_wide = pd.read_csv(train_path)
    rtt_cols, rss_cols = find_ap_measurement_columns(train_wide)
    common_aps = sorted(set(rtt_cols).intersection(set(rss_cols)))

    dead_aps, dead_report = detect_dead_aps_from_train_wide(train_wide, rtt_cols, rss_cols)
    usable_aps = [ap for ap in common_aps if ap not in dead_aps]

    ap_mapping = assign_ap_coordinates(scenario_name, usable_aps, coord_blocks)

    print(f"[INFO] {scenario_name}: detected AP columns = {common_aps}")
    print(f"[INFO] {scenario_name}: dead APs from TRAIN = {dead_aps}")
    print(f"[INFO] {scenario_name}: usable APs = {usable_aps}")

    train_long = build_long_table_from_wide(train_wide, scenario_name, "train", ap_mapping)

    tables = [train_long]

    if test_path.exists():
        test_wide = pd.read_csv(test_path)
        test_long = build_long_table_from_wide(test_wide, scenario_name, "test", ap_mapping)
        tables.append(test_long)
    else:
        print(f"[WARNING] {scenario_name}: test file missing, only train table was built.")

    dead_report.insert(0, "scenario", scenario_name)
    dead_report["is_removed_from_regression"] = dead_report["AP_id_raw"].isin(dead_aps)

    return pd.concat(tables, ignore_index=True), dead_report


# ============================================================
# Section 5 — Dataset building and audits
# ============================================================

def check_required_input_files() -> None:
    required_paths = [COORD_FILE_BATCH1, COORD_FILE_BATCH2]
    for cfg in SCENARIOS.values():
        required_paths.append(Path(cfg["train_csv"]))
        required_paths.append(Path(cfg["test_csv"]))

    missing = [p for p in required_paths if not p.exists()]
    for p in required_paths:
        print(f"[CHECK] {p}: {'FOUND' if p.exists() else 'MISSING'}")

    if missing:
        missing_text = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError("Missing required input files:\n" + missing_text)


def build_all_links() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coord_blocks = {
        "batch1": parse_ap_coordinate_txt(COORD_FILE_BATCH1),
        "batch2": parse_ap_coordinate_txt(COORD_FILE_BATCH2),
    }

    all_tables = []
    dead_reports = []
    load_summary_rows = []

    t0_all = time.time()

    for i, scenario_name in enumerate(SCENARIOS, start=1):
        t0 = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Loading scenario {i}/{len(SCENARIOS)}: {scenario_name}")

        tbl, dead_rep = load_scenario_train_test(scenario_name, coord_blocks)

        all_tables.append(tbl)
        dead_reports.append(dead_rep)

        n_train = int((tbl["split"] == "train").sum())
        n_test = int((tbl["split"] == "test").sum())
        n_rps = int(tbl["rp_key"].nunique())
        n_aps = int(tbl["AP_id_raw"].nunique())
        n_valid_both = int((tbl["RSS_valid"] & tbl["RTT_valid"]).sum())

        dead_list = dead_rep.loc[
            dead_rep["is_removed_from_regression"], "AP_id_raw"
        ].tolist()

        load_summary_rows.append({
            "scenario": scenario_name,
            "n_ap_links_total": len(tbl),
            "n_train_ap_links": n_train,
            "n_test_ap_links": n_test,
            "n_unique_rps": n_rps,
            "n_usable_aps": n_aps,
            "n_valid_both_links": n_valid_both,
            "dead_aps_from_train": ",".join(map(str, dead_list)) if dead_list else "",
            "load_seconds": round(time.time() - t0, 2),
        })

        print(
            f"[INFO] {scenario_name}: rows={len(tbl):,}, "
            f"train={n_train:,}, test={n_test:,}, "
            f"RPs={n_rps:,}, APs={n_aps}, valid_both={n_valid_both:,}, "
            f"dead_APs={dead_list if dead_list else 'none'}"
        )

    all_links = pd.concat(all_tables, ignore_index=True)
    dead_ap_report = pd.concat(dead_reports, ignore_index=True)
    load_summary = pd.DataFrame(load_summary_rows)

    all_links.to_csv(OUTPUT_DIR / "all_ap_link_table.csv", index=False)
    dead_ap_report.to_csv(OUTPUT_DIR / "dead_ap_report.csv", index=False)
    load_summary.to_csv(OUTPUT_DIR / "scenario_load_summary.csv", index=False)

    print(f"\n[INFO] Total AP-link rows: {len(all_links):,}")
    print(f"[INFO] Total valid RSS+RTT rows: {(all_links['RSS_valid'] & all_links['RTT_valid']).sum():,}")
    print(f"[INFO] Dataset build completed in {time.time() - t0_all:.2f} s")

    return all_links, dead_ap_report, load_summary


def write_dataset_audits(all_links: pd.DataFrame) -> None:
    audit_rows = []

    for (scenario, split), g in all_links.groupby(["scenario", "split"]):
        both_valid = g["RSS_valid"] & g["RTT_valid"]

        audit_rows.append({
            "scenario": scenario,
            "batch": g["batch"].iloc[0],
            "split": split,
            "area_type": g["area_type"].iloc[0],
            "n_ap_links": len(g),
            "n_valid_both": int(both_valid.sum()),
            "n_RPs": g["rp_key"].nunique(),
            "n_APs_usable": g["AP_id_raw"].nunique(),
            "RSS_valid_rate": float(g["RSS_valid"].mean()),
            "RTT_valid_rate": float(g["RTT_valid"].mean()),
            "LOS_rate": float(g["LOS_flag"].mean()) if g["LOS_known"].any() else np.nan,
        })

    audit_summary = pd.DataFrame(audit_rows)
    audit_summary.to_csv(OUTPUT_DIR / "dataset_audit_summary.csv", index=False)

    train_valid_for_imbalance = all_links[
        (all_links["split"] == "train") &
        (all_links["RSS_valid"]) &
        (all_links["RTT_valid"])
    ].copy()

    imbalance_report = (
        train_valid_for_imbalance
        .groupby("scenario")
        .size()
        .reset_index(name="n_train_valid_ap_links")
    )

    imbalance_report["share"] = (
        imbalance_report["n_train_valid_ap_links"] /
        imbalance_report["n_train_valid_ap_links"].sum()
    )

    imbalance_report.to_csv(OUTPUT_DIR / "imbalance_report.csv", index=False)

    print("\n[INFO] Dataset audit summary:")
    print(audit_summary)

    print("\n[INFO] Training imbalance report:")
    print(imbalance_report)

    if not imbalance_report.empty and imbalance_report["share"].max() > 0.5:
        dominant = imbalance_report.loc[imbalance_report["share"].idxmax(), "scenario"]
        share = imbalance_report["share"].max()
        print(f"\n[WARNING] Strong scenario imbalance detected: {dominant} contributes {share:.1%}.")
        print("[WARNING] Scenario-weighted training and macro-average evaluation are needed.")


# ============================================================
# Section 6 — Regression splits and scalers
# ============================================================

def make_regression_splits(all_links: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = all_links[
        (all_links["split"] == "train") &
        (all_links["RSS_valid"]) &
        (all_links["RTT_valid"])
    ].copy()

    test_df = all_links[
        (all_links["split"] == "test") &
        (all_links["RSS_valid"]) &
        (all_links["RTT_valid"])
    ].copy()

    print(f"\n[INFO] Train regression AP-links: {len(train_df):,}")
    print(f"[INFO] Test regression AP-links: {len(test_df):,}")

    missing_features = [f for f in BASE_FEATURES if f not in train_df.columns]
    missing_targets = [t for t in TARGETS if t not in train_df.columns]

    if missing_features or missing_targets:
        raise ValueError(f"Missing features={missing_features}, missing targets={missing_targets}")

    if train_df["rp_key"].nunique() < 2:
        raise ValueError("Not enough unique RPs for grouped validation split.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    sub_idx, val_idx = next(gss.split(train_df, groups=train_df["rp_key"]))

    subtrain_df = train_df.iloc[sub_idx].copy()
    val_df = train_df.iloc[val_idx].copy()

    print(f"[INFO] Subtrain AP-links: {len(subtrain_df):,}")
    print(f"[INFO] Validation AP-links: {len(val_df):,}")
    print(f"[INFO] Subtrain unique RPs: {subtrain_df['rp_key'].nunique():,}")
    print(f"[INFO] Validation unique RPs: {val_df['rp_key'].nunique():,}")

    overlap_rps = set(subtrain_df["rp_key"]).intersection(set(val_df["rp_key"]))
    print(f"[CHECK] RP overlap between subtrain and validation: {len(overlap_rps)}")

    if overlap_rps:
        raise RuntimeError("RP leakage detected between subtrain and validation.")

    return subtrain_df, val_df, test_df


def fit_and_save_scalers(subtrain_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_sub = subtrain_df[BASE_FEATURES].to_numpy(dtype=float)
    Y_sub = subtrain_df[TARGETS].to_numpy(dtype=float)

    X_val = val_df[BASE_FEATURES].to_numpy(dtype=float)
    Y_val = val_df[TARGETS].to_numpy(dtype=float)

    X_sub_s = x_scaler.fit_transform(X_sub)
    Y_sub_s = y_scaler.fit_transform(Y_sub)

    X_val_s = x_scaler.transform(X_val)
    Y_val_s = y_scaler.transform(Y_val)

    print(f"[INFO] X_sub shape: {X_sub_s.shape}")
    print(f"[INFO] Y_sub shape: {Y_sub_s.shape}")
    print(f"[INFO] X_val shape: {X_val_s.shape}")
    print(f"[INFO] Y_val shape: {Y_val_s.shape}")

    joblib.dump(x_scaler, MODEL_PREP_DIR / "x_scaler.joblib")
    joblib.dump(y_scaler, MODEL_PREP_DIR / "y_scaler.joblib")

    counts = subtrain_df["scenario"].value_counts()
    weights = subtrain_df["scenario"].map(lambda s: 1.0 / counts[s]).to_numpy(dtype=float)
    weights = weights / weights.mean()

    weight_summary = (
        pd.DataFrame({"scenario": subtrain_df["scenario"], "weight": weights})
        .groupby("scenario")
        .agg(n=("weight", "size"), mean_weight=("weight", "mean"), total_weight=("weight", "sum"))
        .reset_index()
    )
    weight_summary.to_csv(OUTPUT_DIR / "scenario_weight_summary.csv", index=False)

    metadata = {
        "base_features": BASE_FEATURES,
        "targets": TARGETS,
        "seed": SEED,
        "validation": {
            "method": "GroupShuffleSplit",
            "group_key": "rp_key",
            "test_size": 0.15,
            "random_state": SEED,
        },
        "scaler_protocol": "x_scaler and y_scaler fitted on subtrain only",
    }

    (MODEL_PREP_DIR / "feature_target_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def build_geometry_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeated scan rows into unique AP-RP geometry links.

    This supports geometry-level GNN/CNN/LAMS comparisons without changing
    the official row-level split identity.
    """
    if df.empty:
        return df.copy()

    group_cols = [
        "scenario", "batch", "split", "area_type", "rp_key",
        "AP_id_raw", "AP_index_usable",
    ]

    agg = {
        "RX_x": "first",
        "RX_y": "first",
        "AP_x": "first",
        "AP_y": "first",
        "dx": "first",
        "dy": "first",
        "distance": "first",
        "LOS_flag": "first",
        "LOS_known": "first",
        "RSS_dBm": "mean",
        "RTT_m": "mean",
    }

    out = df.groupby(group_cols, as_index=False).agg(agg)
    out["unit_id"] = (
        out["scenario"].astype(str) + "_" +
        out["rp_key"].astype(str) + "_AP" +
        out["AP_id_raw"].astype(str)
    )

    return out


def save_final_split_files(subtrain_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    subtrain_df = subtrain_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    subtrain_df["unit_id"] = (
        subtrain_df["scenario"].astype(str) + "_" +
        subtrain_df["split"].astype(str) + "_" +
        subtrain_df["raw_row_id"].astype(str) + "_AP" +
        subtrain_df["AP_id_raw"].astype(str)
    )
    val_df["unit_id"] = (
        val_df["scenario"].astype(str) + "_" +
        val_df["split"].astype(str) + "_" +
        val_df["raw_row_id"].astype(str) + "_AP" +
        val_df["AP_id_raw"].astype(str)
    )
    test_df["unit_id"] = (
        test_df["scenario"].astype(str) + "_" +
        test_df["split"].astype(str) + "_" +
        test_df["raw_row_id"].astype(str) + "_AP" +
        test_df["AP_id_raw"].astype(str)
    )

    subtrain_df.to_csv(OUTPUT_DIR / "row_train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "row_val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "row_test.csv", index=False)

    geometry_train = build_geometry_level(subtrain_df)
    geometry_val = build_geometry_level(val_df)
    geometry_test = build_geometry_level(test_df)

    geometry_train.to_csv(OUTPUT_DIR / "geometry_train.csv", index=False)
    geometry_val.to_csv(OUTPUT_DIR / "geometry_val.csv", index=False)
    geometry_test.to_csv(OUTPUT_DIR / "geometry_test.csv", index=False)

    print("\n[INFO] Saved standalone processed split files:")
    print(f"   row_train:      {OUTPUT_DIR / 'row_train.csv'}")
    print(f"   row_val:        {OUTPUT_DIR / 'row_val.csv'}")
    print(f"   row_test:       {OUTPUT_DIR / 'row_test.csv'}")
    print(f"   geometry_train: {OUTPUT_DIR / 'geometry_train.csv'}")
    print(f"   geometry_val:   {OUTPUT_DIR / 'geometry_val.csv'}")
    print(f"   geometry_test:  {OUTPUT_DIR / 'geometry_test.csv'}")


# ============================================================
# Section 7 — Main execution
# ============================================================

def main() -> None:
    print("[INFO] Starting standalone AP-link dataset build.")
    print(f"[INFO] Repository root: {REPO_ROOT}")

    check_required_input_files()

    all_links, dead_ap_report, load_summary = build_all_links()
    write_dataset_audits(all_links)

    subtrain_df, val_df, test_df = make_regression_splits(all_links)
    fit_and_save_scalers(subtrain_df, val_df, test_df)
    save_final_split_files(subtrain_df, val_df, test_df)

    print("\n[INFO] data/build_dataset.py completed successfully.")


if __name__ == "__main__":
    main()
