# ============================================================
# Section 2 — Paths, dataset registry, and AP-coordinate files
# ------------------------------------------------------------
# Purpose:
# - Centralize all dataset paths and output paths.
# - Register the six scenarios and their native train/test files.
# - Register the two AP-coordinate TXT files.
# - Avoid scattered hardcoded paths throughout the notebook.
#
# Scientific note:
# - The six scenarios come from two batches.
# - The official train/test split is the primary evaluation protocol.
# - AP IDs are scenario-specific and should not be assumed to be
#   AP1...APN in every scenario.
# - The AP-coordinate TXT files provide the physical AP layout needed
#   for geometry-aware prediction.
#
# Expected scenarios:
# - Batch 1: building/floor, office, apartment
# - Batch 2: lecture theatre, corridor, office2
#
# Progress/logging:
# - Print whether all expected files exist.
# ============================================================


# ============================================================
# Section 2 — Paths, dataset registry, and AP-coordinate files
# ------------------------------------------------------------
# Purpose:
# - Centralize all dataset paths and output paths.
# - Match the actual folder structure:
#       DATA_PARENT/
#       ├── first batch/
#       └── second batch/
# - Avoid scattered hardcoded paths throughout the notebook.
#
# Scientific note:
# - The official train/test split is preserved.
# - Each batch has its own CSV files and AP-coordinate TXT file.
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd


# CHANGE ONLY THIS LINE to your real parent dataset folder.
DATA_PARENT = Path(r"C:\Users\melsanho\OneDrive - University of Vaasa\ownCloud\Sultan Qabos University\IPIN 2026 paper APTwin\Github repo for APTwin")

AP_LOCATIONS = DATA_PARENT / "ap_locations"

BATCH1_DIR = DATA_PARENT / "data" / "first_batch"
BATCH2_DIR = DATA_PARENT / "data" / "second_batch"

OUTPUT_DIR = Path("outputs_clean_six_scenario")
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

COORD_FILE_BATCH1 = AP_LOCATIONS / "Floor+office+apartment_AP_coords.txt"
COORD_FILE_BATCH2 = AP_LOCATIONS / "lecture theatre+office+corridor_AP_position.txt"

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

expected_files = [COORD_FILE_BATCH1, COORD_FILE_BATCH2]
for cfg in SCENARIOS.values():
    expected_files.append(cfg["train_csv"])
    expected_files.append(cfg["test_csv"])

for f in expected_files:
    print(f"[CHECK] {f}: {'FOUND' if f.exists() else 'MISSING'}")




# ============================================================
# Section 3 — AP-coordinate TXT parser
# ------------------------------------------------------------
# Purpose:
# - Parse the AP-coordinate TXT files into a structured dictionary.
# - Normalize block names safely, e.g., "Apartment:" -> "apartment".
# - Prevent coordinate block lookup errors caused by capitalization,
#   trailing colons, or extra spaces.
#
# Scientific note:
# - AP coordinates are physical inputs to the geometry-aware digital twin.
# - Coordinate parsing should be deterministic and transparent.
# - We do not infer missing AP coordinates blindly.
# - If a CSV exposes more AP columns than the coordinate file provides,
#   this is handled later through TRAIN-based dead AP detection.
#
# Expected coordinate-file format:
#     <block name>
#     AP X: x1, x2, ...
#     AP Y: y1, y2, ...
#
# Progress/logging:
# - Print parsed coordinate blocks and number of AP coordinates per block.
# - Print the exact normalized block names so mismatch errors are visible.
# ============================================================

def normalize_block_name(name):
    """
    Normalize coordinate block names for robust lookup.

    Examples:
        "Apartment:"       -> "apartment"
        " Office "         -> "office"
        "lecture theatre:" -> "lecture theatre"
    """
    name = str(name).strip().lower()
    name = name.rstrip(":")
    name = re.sub(r"\s+", " ", name)
    return name


def _parse_float_list(line):
    """
    Extract comma-separated numbers from a line such as:
        AP X: 1, 2, 3
    """
    if ":" not in line:
        return []

    right = line.split(":", 1)[1]
    return [float(x.strip()) for x in right.split(",") if x.strip() != ""]


def parse_ap_coordinate_txt(path):
    """
    Parse the project AP-coordinate TXT format.

    Returns
    -------
    blocks : dict
        normalized block name -> list of (x, y) coordinates in metadata order.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {path}")

    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    blocks = {}
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
            # This is the block/scenario name line.
            current_name = ln.strip()

    return blocks


coord_blocks = {
    "batch1": parse_ap_coordinate_txt(COORD_FILE_BATCH1),
    "batch2": parse_ap_coordinate_txt(COORD_FILE_BATCH2),
}

for batch, blocks in coord_blocks.items():
    print(f"\n[INFO] Parsed coordinate blocks in {batch}:")
    for name, coords in blocks.items():
        print(f"   '{name}': {len(coords)} AP coordinates")

# Quick required-block check before moving forward.
required_blocks = {
    "batch1": ["floor", "office", "apartment"],
    "batch2": ["lecture theatre", "office", "corridor"],
}

for batch, names in required_blocks.items():
    available = set(coord_blocks[batch].keys())
    for name in names:
        if normalize_block_name(name) not in available:
            raise KeyError(
                f"Required coordinate block '{name}' not found in {batch}. "
                f"Available blocks are: {sorted(available)}"
            )

print("\n[INFO] Coordinate parser check passed.")