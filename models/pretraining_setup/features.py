# ============================================================
# Section 9 — Feature set, targets, and leakage control
# ------------------------------------------------------------
# Purpose:
# - Define the base geometry/context features.
# - Define RSS/RTT regression targets.
# - Keep feature design explicit and leakage-safe.
#
# Scientific note:
# - The primary feature set uses AP/RP geometry and LOS/NLOS context.
# - RSS and RTT are never used as inputs to predict each other in the
#   main model.
# - AP identity and scenario identity are not included in the primary
#   model to reduce memorization.
# - Optional identity features may be tested only as ablations later.
#
# Base features:
# - RX_x, RX_y
# - AP_x, AP_y
# - dx, dy
# - distance
# - LOS_flag, LOS_known
#
# Targets:
# - RSS_dBm
# - RTT_m
# ============================================================

BASE_FEATURES = [
    "RX_x", "RX_y",
    "AP_x", "AP_y",
    "dx", "dy",
    "distance",
    "LOS_flag", "LOS_known",
]

TARGETS = ["RSS_dBm", "RTT_m"]

print("[INFO] Base features:")
for f in BASE_FEATURES:
    print("   ", f)

print("[INFO] Targets:")
for t in TARGETS:
    print("   ", t)

missing_features = [f for f in BASE_FEATURES if f not in train_df.columns]
missing_targets = [t for t in TARGETS if t not in train_df.columns]

if missing_features or missing_targets:
    raise ValueError(f"Missing features={missing_features}, missing targets={missing_targets}")