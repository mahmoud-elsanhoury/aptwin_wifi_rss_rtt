# ============================================================
# training/train_cnn_lams.py
# ------------------------------------------------------------
# Purpose:
# - Train standalone true LAMS CNN branches for RSS and RTT.
# - Remove notebook-memory dependencies such as lams_bank,
#   lams_id_subtrain, X_scalar_lams_subtrain, and RTT_LAMS_DIR.
# - Add the agreed fixed multi-seed reporting protocol without changing
#   the CNN architectures, LAMS inputs, feature handling, or data splits.
#
# Scientific protocol:
# - Uses PyTorch for CNN training.
# - Uses the true 40×40 AP-to-RP LAMS bank already built by
#   floorplan_images/build_lams.py.
# - Uses fixed architecture and fixed training settings.
# - No architecture search or hyperparameter sweep is performed.
# - Early stopping uses validation RMSE.
# - RSS and RTT are trained at geometry level because the LAMS image is
#   identical for repeated scans of the same AP-RP link.
# - Repeats the same fixed protocol over fixed seeds.
# - Main multi-seed reporting uses mean ± std over seeds.
# - The representative seed is used only for CDF plotting and is selected
#   from validation scenario-macro RMSE using the validation-median rule.
# - No test metric is used for seed selection.
# ============================================================

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    raise ImportError("PyTorch is required for training/train_cnn_lams.py. Install torch first.") from exc

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from common_training import (
    FEATURES_WALL_OBS_CONTEXT,
    MODEL_DIR,
    PREDICTION_DIR,
    RESULTS_TABLE_DIR,
    REPO_ROOT,
    SEEDS,
    log,
    set_all_seeds,
    summarize_seed_metrics,
)

PREFIX = "train_cnn_lams"
FAMILY = "CNN_LAMS"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMS_DIR = REPO_ROOT / "floorplan_images" / "lams" / "true_lams_40x40"
RTT_LAMS_DIR = LAMS_DIR / "rtt_lams_branch"
MODEL_OUT_DIR = MODEL_DIR / "cnn_lams"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "batch_size": 128,
    "max_epochs_rss": 80,
    "max_epochs_rtt": 100,
    "patience_rss": 8,
    "patience_rtt": 12,
    "lr": 1e-3,
    "weight_decay_rss": 0.0,
    "weight_decay_rtt": 1e-4,
    "dropout_rss": 0.0,
    "dropout_rtt": 0.15,
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


class LAMSGeometryDataset(Dataset):
    def __init__(self, bank: np.ndarray, lams_ids, scalar, y):
        self.bank = np.asarray(bank, dtype=np.float32)
        self.lams_ids = np.asarray(lams_ids, dtype=np.int64)
        self.scalar = np.asarray(scalar, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        if not (len(self.lams_ids) == len(self.scalar) == len(self.y)):
            raise ValueError("lams_ids, scalar, and y lengths do not match.")

    def __len__(self):
        return len(self.lams_ids)

    def __getitem__(self, idx):
        img = self.bank[self.lams_ids[idx]].astype(np.float32)
        scalar = self.scalar[idx].astype(np.float32)
        target = self.y[idx].astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(scalar), torch.tensor(target)


class LAMSEncoderNoEarlyPooling(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 8, kernel_size=4, stride=3, padding=0), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=0), nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            self.out_dim = int(self.conv(torch.zeros(1, in_channels, 40, 40)).shape[1])

    def forward(self, x):
        return self.conv(x)


class LAMSOnlyCNN(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        self.encoder = LAMSEncoderNoEarlyPooling(in_channels=in_channels, dropout=dropout)
        layers = [nn.Linear(self.encoder.out_dim + 1, 32), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, img, distance_scalar):
        z = self.encoder(img)
        z = torch.cat([z, distance_scalar[:, :1]], dim=1)
        return self.head(z).squeeze(-1)


class LAMSHybridCNN(nn.Module):
    def __init__(self, in_channels: int = 3, scalar_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = LAMSEncoderNoEarlyPooling(in_channels=in_channels, dropout=dropout)
        scalar_layers = [nn.Linear(scalar_dim, 64), nn.ReLU()]
        if dropout > 0:
            scalar_layers.append(nn.Dropout(dropout))
        scalar_layers += [nn.Linear(64, 32), nn.ReLU()]
        self.scalar_branch = nn.Sequential(*scalar_layers)
        head_layers = [nn.Linear(self.encoder.out_dim + 32, 64), nn.ReLU()]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers += [nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, img, scalar):
        z_img = self.encoder(img)
        z_scalar = self.scalar_branch(scalar)
        return self.head(torch.cat([z_img, z_scalar], dim=1)).squeeze(-1)


def make_loader(ds: Dataset, shuffle: bool, seed: int | None = None) -> DataLoader:
    """Build a DataLoader; seed the shuffle order only for training loaders."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def rmse_np(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def train_regressor(model, train_loader, val_loader, model_name: str, max_epochs: int, patience: int, lr: float, weight_decay: float, seed: int, inverse_fn=None):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        ep0 = time.time()
        model.train()
        sse = 0.0
        n = 0
        for batch_idx, (img, scalar, y) in enumerate(train_loader, start=1):
            img, scalar, y = img.to(DEVICE), scalar.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(img, scalar)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            sse += torch.sum((pred.detach() - y) ** 2).item()
            n += len(y)
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                log(PREFIX, f"{model_name} | seed {seed} | epoch {epoch:03d}/{max_epochs} | batch {batch_idx:,}/{len(train_loader):,}")

        train_rmse = float(np.sqrt(sse / max(n, 1)))
        val_true, val_pred = predict_with_targets(model, val_loader)
        if inverse_fn is not None:
            val_true = inverse_fn(val_true)
            val_pred = inverse_fn(val_pred)
        val_rmse = rmse_np(val_true, val_pred)
        improved = val_rmse < best_val - 1e-8
        if improved:
            best_val = val_rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
            status = "IMPROVED"
        else:
            wait += 1
            status = "NO_IMPROVE"
        history.append({"seed": int(seed), "epoch": epoch, "train_rmse_scaled": train_rmse, "val_rmse": val_rmse, "best_val_rmse": best_val, "best_epoch": best_epoch, "status": status})
        log(PREFIX, f"{model_name} | seed {seed} | epoch {epoch:03d} | trainRMSE={train_rmse:.4f} | valRMSE={val_rmse:.4f} | best={best_val:.4f}@{best_epoch:03d} | patience={wait}/{patience} | {status} | epoch={time.time()-ep0:.1f}s")
        if wait >= patience:
            log(PREFIX, f"{model_name} | seed {seed}: early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} | seed {seed}: no valid checkpoint was created.")
    model.load_state_dict(best_state)
    log(PREFIX, f"{model_name} | seed {seed}: restored best checkpoint. Total time={(time.time()-t0)/60.0:.2f} min.")
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_with_targets(model, loader):
    model.eval()
    preds, ys = [], []
    for img, scalar, y in loader:
        img, scalar = img.to(DEVICE), scalar.to(DEVICE)
        pred = model(img, scalar)
        preds.append(pred.detach().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def build_eval_df(geo_df: pd.DataFrame, true_col: str, pred_cols: dict[str, np.ndarray], target_name: str) -> pd.DataFrame:
    out = geo_df[[c for c in ["lams_id", "scenario", "AP_x", "AP_y", "RX_x", "RX_y", "n_repeat"] if c in geo_df.columns]].copy()
    out[f"{target_name}_true"] = geo_df[true_col].to_numpy(dtype=float)
    for name, values in pred_cols.items():
        out[f"{target_name}_pred_{name}"] = np.asarray(values, dtype=float)
    return out


def metrics_for_single_target(eval_df: pd.DataFrame, target: str, pred_col: str, model: str, level: str, seed: int | None = None) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "scenario": eval_df["scenario"],
        f"{target}_true": eval_df[f"{target}_true"],
        f"{target}_pred": eval_df[pred_col],
    })
    rows = []
    valid = tmp.dropna()
    yt, yp = valid[f"{target}_true"], valid[f"{target}_pred"]
    err = yp - yt
    base = {"family": FAMILY, "model": model, "target": target, "level": level}
    if seed is not None:
        base["seed"] = int(seed)
    rows.append({**base, "aggregation": "micro", "scenario": "ALL", "MAE": float(mean_absolute_error(yt, yp)), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "P95": float(np.percentile(np.abs(err), 95)), "n": int(len(valid))})
    scen = []
    for s, g in valid.groupby("scenario", sort=True):
        yts, yps = g[f"{target}_true"], g[f"{target}_pred"]
        e = yps - yts
        m = {"MAE": float(mean_absolute_error(yts, yps)), "RMSE": float(np.sqrt(mean_squared_error(yts, yps))), "P95": float(np.percentile(np.abs(e), 95)), "n": int(len(g))}
        scen.append(m)
        rows.append({**base, "aggregation": "scenario", "scenario": s, **m})
    sdf = pd.DataFrame(scen)
    rows.append({**base, "aggregation": "scenario_macro", "scenario": "MACRO", "MAE": float(sdf["MAE"].mean()), "RMSE": float(sdf["RMSE"].mean()), "P95": float(sdf["P95"].mean()), "n": int(valid["scenario"].nunique())})
    return pd.DataFrame(rows)


def macro_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    macro = metrics_df[metrics_df["aggregation"] == "scenario_macro"].copy()
    out = macro[["family", "model", "seed", "level", "target", "RMSE", "MAE", "P95", "n"]].copy()
    return out.rename(columns={"RMSE": "RMSE_macro", "MAE": "MAE_macro", "P95": "P95_macro", "n": "n_scenarios"})


def prepare_rss_data(bank):
    geo_train = pd.read_csv(require_file(LAMS_DIR / "lams_geo_subtrain.csv"))
    geo_val = pd.read_csv(require_file(LAMS_DIR / "lams_geo_val.csv"))
    geo_test = pd.read_csv(require_file(LAMS_DIR / "lams_geo_test.csv"))
    scalar_features = [f for f in FEATURES_WALL_OBS_CONTEXT if f in geo_train.columns]
    if "distance" not in scalar_features and "distance" in geo_train.columns:
        scalar_features.append("distance")
    dist_scaler = StandardScaler().fit(geo_train[["distance"]].to_numpy(dtype=np.float32))
    scalar_scaler = StandardScaler().fit(geo_train[scalar_features].to_numpy(dtype=np.float32))
    return geo_train, geo_val, geo_test, scalar_features, dist_scaler, scalar_scaler


def prepare_rtt_data():
    geo_train = pd.read_csv(require_file(RTT_LAMS_DIR / "rtt_lams_geo_subtrain.csv"))
    geo_val = pd.read_csv(require_file(RTT_LAMS_DIR / "rtt_lams_geo_val.csv"))
    geo_test = pd.read_csv(require_file(RTT_LAMS_DIR / "rtt_lams_geo_test.csv"))
    target_scaler = joblib.load(require_file(RTT_LAMS_DIR / "rtt_target_scaler.joblib"))
    scalar_features = [f for f in FEATURES_WALL_OBS_CONTEXT if f in geo_train.columns]
    if "distance" not in scalar_features and "distance" in geo_train.columns:
        scalar_features.append("distance")
    dist_scaler = StandardScaler().fit(geo_train[["distance"]].to_numpy(dtype=np.float32))
    scalar_scaler = StandardScaler().fit(geo_train[scalar_features].to_numpy(dtype=np.float32))
    return geo_train, geo_val, geo_test, scalar_features, dist_scaler, scalar_scaler, target_scaler


def select_representative_seeds(validation_macro: pd.DataFrame) -> pd.DataFrame:
    summary = validation_macro.groupby(["family", "model", "seed"], as_index=False).agg(validation_RMSE_macro_mean=("RMSE_macro", "mean"))
    rows = []
    for (family, model), g in summary.groupby(["family", "model"], sort=True):
        med = float(g["validation_RMSE_macro_mean"].median())
        gg = g.copy()
        gg["distance_to_validation_median_RMSE"] = (gg["validation_RMSE_macro_mean"] - med).abs()
        chosen = gg.sort_values(["distance_to_validation_median_RMSE", "validation_RMSE_macro_mean", "seed"]).iloc[0]
        rows.append({"family": family, "model": model, "representative_seed": int(chosen["seed"]), "selection_rule": "validation_median_RMSE_macro_mean_over_available_targets", "validation_RMSE_macro_mean": float(chosen["validation_RMSE_macro_mean"]), "validation_RMSE_macro_median": med})
    return pd.DataFrame(rows)


def make_representative_eval(target: str, split_name: str, representative: pd.DataFrame, seed_eval_tables: dict[tuple[str, int, str], pd.DataFrame], base_df: pd.DataFrame) -> pd.DataFrame:
    """Create one representative prediction CSV, allowing each CNN model to use its own representative seed."""
    id_cols = [c for c in ["lams_id", "scenario", "AP_x", "AP_y", "RX_x", "RX_y", "n_repeat"] if c in base_df.columns]
    out = base_df[id_cols].copy()
    true_col = "RSS_mean_dBm" if target == "RSS" else "RTT_mean_m"
    out[f"{target}_true"] = base_df[true_col].to_numpy(dtype=float)
    for _, row in representative.iterrows():
        model = row["model"]
        if row.get("target", target) != target and "target" in row.index:
            continue
        seed = int(row["representative_seed"])
        eval_df = seed_eval_tables[(target, seed, split_name)]
        pred_col = f"{target}_pred_{model}"
        if pred_col in eval_df.columns:
            out[pred_col] = eval_df[pred_col].to_numpy(dtype=float)
    return out


def main() -> None:
    log(PREFIX, f"Starting standalone CNN/LAMS multi-seed training on {DEVICE}.")
    log(PREFIX, f"Fixed seed set: {SEEDS}")
    lams_bank = np.load(require_file(LAMS_DIR / "lams_bank_40x40_float32.npy"))

    rss_train, rss_val, rss_test, rss_scalar_features, rss_dist_scaler, rss_scalar_scaler = prepare_rss_data(lams_bank)
    rtt_train, rtt_val, rtt_test, rtt_scalar_features, rtt_dist_scaler, rtt_scalar_scaler, rtt_target_scaler = prepare_rtt_data()

    y_rss_train = rss_train["RSS_mean_dBm"].to_numpy(dtype=np.float32)
    y_rss_val = rss_val["RSS_mean_dBm"].to_numpy(dtype=np.float32)
    y_rss_test = rss_test["RSS_mean_dBm"].to_numpy(dtype=np.float32)

    y_rtt_train_raw = rtt_train[["RTT_mean_m"]].to_numpy(dtype=np.float32)
    y_rtt_val_raw = rtt_val[["RTT_mean_m"]].to_numpy(dtype=np.float32)
    y_rtt_test_raw = rtt_test[["RTT_mean_m"]].to_numpy(dtype=np.float32)
    y_rtt_train = rtt_target_scaler.transform(y_rtt_train_raw).ravel().astype(np.float32)
    y_rtt_val = rtt_target_scaler.transform(y_rtt_val_raw).ravel().astype(np.float32)
    y_rtt_test = rtt_target_scaler.transform(y_rtt_test_raw).ravel().astype(np.float32)
    inv_rtt = lambda arr: rtt_target_scaler.inverse_transform(np.asarray(arr, dtype=np.float32).reshape(-1, 1)).ravel()

    validation_macro_all = []
    test_macro_all = []
    seed_eval_tables = {}

    for seed in SEEDS:
        set_all_seeds(seed)
        log(PREFIX, "=" * 78)
        log(PREFIX, f"Starting CNN/LAMS seed {seed}.")

        # ---------------- RSS branch ----------------
        ds_rss_only_train = LAMSGeometryDataset(lams_bank, rss_train["lams_id"], rss_dist_scaler.transform(rss_train[["distance"]]), y_rss_train)
        ds_rss_only_val = LAMSGeometryDataset(lams_bank, rss_val["lams_id"], rss_dist_scaler.transform(rss_val[["distance"]]), y_rss_val)
        ds_rss_only_test = LAMSGeometryDataset(lams_bank, rss_test["lams_id"], rss_dist_scaler.transform(rss_test[["distance"]]), y_rss_test)
        rss_only = LAMSOnlyCNN(in_channels=lams_bank.shape[1], dropout=CONFIG["dropout_rss"])
        rss_only, hist_rss_only = train_regressor(rss_only, make_loader(ds_rss_only_train, True, seed), make_loader(ds_rss_only_val, False), "TRUE_LAMS_ONLY_CNN", CONFIG["max_epochs_rss"], CONFIG["patience_rss"], CONFIG["lr"], CONFIG["weight_decay_rss"], seed=seed)
        _, pred_rss_only_val = predict_with_targets(rss_only, make_loader(ds_rss_only_val, False))
        _, pred_rss_only_test = predict_with_targets(rss_only, make_loader(ds_rss_only_test, False))

        ds_rss_h_train = LAMSGeometryDataset(lams_bank, rss_train["lams_id"], rss_scalar_scaler.transform(rss_train[rss_scalar_features]), y_rss_train)
        ds_rss_h_val = LAMSGeometryDataset(lams_bank, rss_val["lams_id"], rss_scalar_scaler.transform(rss_val[rss_scalar_features]), y_rss_val)
        ds_rss_h_test = LAMSGeometryDataset(lams_bank, rss_test["lams_id"], rss_scalar_scaler.transform(rss_test[rss_scalar_features]), y_rss_test)
        rss_hybrid = LAMSHybridCNN(in_channels=lams_bank.shape[1], scalar_dim=len(rss_scalar_features), dropout=CONFIG["dropout_rss"])
        rss_hybrid, hist_rss_hybrid = train_regressor(rss_hybrid, make_loader(ds_rss_h_train, True, seed), make_loader(ds_rss_h_val, False), "TRUE_LAMS_HYBRID_CNN", CONFIG["max_epochs_rss"], CONFIG["patience_rss"], CONFIG["lr"], CONFIG["weight_decay_rss"], seed=seed)
        _, pred_rss_hybrid_val = predict_with_targets(rss_hybrid, make_loader(ds_rss_h_val, False))
        _, pred_rss_hybrid_test = predict_with_targets(rss_hybrid, make_loader(ds_rss_h_test, False))

        rss_val_eval = build_eval_df(rss_val, "RSS_mean_dBm", {"TRUE_LAMS_ONLY_CNN": pred_rss_only_val, "TRUE_LAMS_HYBRID_CNN": pred_rss_hybrid_val}, "RSS")
        rss_test_eval = build_eval_df(rss_test, "RSS_mean_dBm", {"TRUE_LAMS_ONLY_CNN": pred_rss_only_test, "TRUE_LAMS_HYBRID_CNN": pred_rss_hybrid_test}, "RSS")
        rss_val_eval["seed"] = seed
        rss_test_eval["seed"] = seed
        rss_val_eval.to_csv(PREDICTION_DIR / f"cnn_lams_rss_seed{seed}_geometry_val_predictions.csv", index=False)
        rss_test_eval.to_csv(PREDICTION_DIR / f"cnn_lams_rss_seed{seed}_geometry_test_predictions.csv", index=False)
        seed_eval_tables[("RSS", seed, "val")] = rss_val_eval
        seed_eval_tables[("RSS", seed, "test")] = rss_test_eval

        hist_rss_only.to_csv(LAMS_DIR / f"true_lams_only_seed{seed}_history.csv", index=False)
        hist_rss_hybrid.to_csv(LAMS_DIR / f"true_lams_hybrid_seed{seed}_history.csv", index=False)
        torch.save(rss_only.state_dict(), MODEL_OUT_DIR / f"true_lams_only_rss_seed{seed}.pt")
        torch.save(rss_hybrid.state_dict(), MODEL_OUT_DIR / f"true_lams_hybrid_rss_seed{seed}.pt")

        # ---------------- RTT branch ----------------
        ds_rtt_only_train = LAMSGeometryDataset(lams_bank, rtt_train["lams_id"], rtt_dist_scaler.transform(rtt_train[["distance"]]), y_rtt_train)
        ds_rtt_only_val = LAMSGeometryDataset(lams_bank, rtt_val["lams_id"], rtt_dist_scaler.transform(rtt_val[["distance"]]), y_rtt_val)
        ds_rtt_only_test = LAMSGeometryDataset(lams_bank, rtt_test["lams_id"], rtt_dist_scaler.transform(rtt_test[["distance"]]), y_rtt_test)
        rtt_only = LAMSOnlyCNN(in_channels=lams_bank.shape[1], dropout=CONFIG["dropout_rtt"])
        rtt_only, hist_rtt_only = train_regressor(rtt_only, make_loader(ds_rtt_only_train, True, seed), make_loader(ds_rtt_only_val, False), "CNN_LAMS_ONLY_RTT_GEO", CONFIG["max_epochs_rtt"], CONFIG["patience_rtt"], CONFIG["lr"], CONFIG["weight_decay_rtt"], seed=seed, inverse_fn=inv_rtt)
        _, pred_rtt_only_val_s = predict_with_targets(rtt_only, make_loader(ds_rtt_only_val, False))
        _, pred_rtt_only_test_s = predict_with_targets(rtt_only, make_loader(ds_rtt_only_test, False))
        pred_rtt_only_val = inv_rtt(pred_rtt_only_val_s)
        pred_rtt_only_test = inv_rtt(pred_rtt_only_test_s)

        ds_rtt_h_train = LAMSGeometryDataset(lams_bank, rtt_train["lams_id"], rtt_scalar_scaler.transform(rtt_train[rtt_scalar_features]), y_rtt_train)
        ds_rtt_h_val = LAMSGeometryDataset(lams_bank, rtt_val["lams_id"], rtt_scalar_scaler.transform(rtt_val[rtt_scalar_features]), y_rtt_val)
        ds_rtt_h_test = LAMSGeometryDataset(lams_bank, rtt_test["lams_id"], rtt_scalar_scaler.transform(rtt_test[rtt_scalar_features]), y_rtt_test)
        rtt_hybrid = LAMSHybridCNN(in_channels=lams_bank.shape[1], scalar_dim=len(rtt_scalar_features), dropout=CONFIG["dropout_rtt"])
        rtt_hybrid, hist_rtt_hybrid = train_regressor(rtt_hybrid, make_loader(ds_rtt_h_train, True, seed), make_loader(ds_rtt_h_val, False), "CNN_LAMS_HYBRID_RTT_GEO", CONFIG["max_epochs_rtt"], CONFIG["patience_rtt"], CONFIG["lr"], CONFIG["weight_decay_rtt"], seed=seed, inverse_fn=inv_rtt)
        _, pred_rtt_hybrid_val_s = predict_with_targets(rtt_hybrid, make_loader(ds_rtt_h_val, False))
        _, pred_rtt_hybrid_test_s = predict_with_targets(rtt_hybrid, make_loader(ds_rtt_h_test, False))
        pred_rtt_hybrid_val = inv_rtt(pred_rtt_hybrid_val_s)
        pred_rtt_hybrid_test = inv_rtt(pred_rtt_hybrid_test_s)

        rtt_val_eval = build_eval_df(rtt_val, "RTT_mean_m", {"CNN_LAMS_ONLY_RTT_GEO": pred_rtt_only_val, "CNN_LAMS_HYBRID_RTT_GEO": pred_rtt_hybrid_val}, "RTT")
        rtt_test_eval = build_eval_df(rtt_test, "RTT_mean_m", {"CNN_LAMS_ONLY_RTT_GEO": pred_rtt_only_test, "CNN_LAMS_HYBRID_RTT_GEO": pred_rtt_hybrid_test}, "RTT")
        rtt_val_eval["seed"] = seed
        rtt_test_eval["seed"] = seed
        rtt_val_eval.to_csv(PREDICTION_DIR / f"cnn_lams_rtt_seed{seed}_geometry_val_predictions.csv", index=False)
        rtt_test_eval.to_csv(PREDICTION_DIR / f"cnn_lams_rtt_seed{seed}_geometry_test_predictions.csv", index=False)
        seed_eval_tables[("RTT", seed, "val")] = rtt_val_eval
        seed_eval_tables[("RTT", seed, "test")] = rtt_test_eval

        hist_rtt_only.to_csv(RTT_LAMS_DIR / f"cnn_lams_only_rtt_seed{seed}_history.csv", index=False)
        hist_rtt_hybrid.to_csv(RTT_LAMS_DIR / f"cnn_lams_hybrid_rtt_seed{seed}_history.csv", index=False)
        torch.save(rtt_only.state_dict(), MODEL_OUT_DIR / f"cnn_lams_only_rtt_seed{seed}.pt")
        torch.save(rtt_hybrid.state_dict(), MODEL_OUT_DIR / f"cnn_lams_hybrid_rtt_seed{seed}.pt")

        metrics = []
        for eval_df, split_name in [(rss_val_eval, "val"), (rss_test_eval, "test")]:
            metrics.append(metrics_for_single_target(eval_df, "RSS", "RSS_pred_TRUE_LAMS_ONLY_CNN", "TRUE_LAMS_ONLY_CNN", "geometry", seed=seed))
            metrics.append(metrics_for_single_target(eval_df, "RSS", "RSS_pred_TRUE_LAMS_HYBRID_CNN", "TRUE_LAMS_HYBRID_CNN", "geometry", seed=seed))
            out_metrics = pd.concat(metrics[-2:], ignore_index=True)
            out_metrics.to_csv(RESULTS_TABLE_DIR / f"cnn_lams_rss_seed{seed}_{split_name}_metrics.csv", index=False)
        for eval_df, split_name in [(rtt_val_eval, "val"), (rtt_test_eval, "test")]:
            metrics.append(metrics_for_single_target(eval_df, "RTT", "RTT_pred_CNN_LAMS_ONLY_RTT_GEO", "CNN_LAMS_ONLY_RTT_GEO", "geometry", seed=seed))
            metrics.append(metrics_for_single_target(eval_df, "RTT", "RTT_pred_CNN_LAMS_HYBRID_RTT_GEO", "CNN_LAMS_HYBRID_RTT_GEO", "geometry", seed=seed))
            out_metrics = pd.concat(metrics[-2:], ignore_index=True)
            out_metrics.to_csv(RESULTS_TABLE_DIR / f"cnn_lams_rtt_seed{seed}_{split_name}_metrics.csv", index=False)

        val_metrics = pd.concat([
            metrics_for_single_target(rss_val_eval, "RSS", "RSS_pred_TRUE_LAMS_ONLY_CNN", "TRUE_LAMS_ONLY_CNN", "geometry", seed=seed),
            metrics_for_single_target(rss_val_eval, "RSS", "RSS_pred_TRUE_LAMS_HYBRID_CNN", "TRUE_LAMS_HYBRID_CNN", "geometry", seed=seed),
            metrics_for_single_target(rtt_val_eval, "RTT", "RTT_pred_CNN_LAMS_ONLY_RTT_GEO", "CNN_LAMS_ONLY_RTT_GEO", "geometry", seed=seed),
            metrics_for_single_target(rtt_val_eval, "RTT", "RTT_pred_CNN_LAMS_HYBRID_RTT_GEO", "CNN_LAMS_HYBRID_RTT_GEO", "geometry", seed=seed),
        ], ignore_index=True)
        test_metrics = pd.concat([
            metrics_for_single_target(rss_test_eval, "RSS", "RSS_pred_TRUE_LAMS_ONLY_CNN", "TRUE_LAMS_ONLY_CNN", "geometry", seed=seed),
            metrics_for_single_target(rss_test_eval, "RSS", "RSS_pred_TRUE_LAMS_HYBRID_CNN", "TRUE_LAMS_HYBRID_CNN", "geometry", seed=seed),
            metrics_for_single_target(rtt_test_eval, "RTT", "RTT_pred_CNN_LAMS_ONLY_RTT_GEO", "CNN_LAMS_ONLY_RTT_GEO", "geometry", seed=seed),
            metrics_for_single_target(rtt_test_eval, "RTT", "RTT_pred_CNN_LAMS_HYBRID_RTT_GEO", "CNN_LAMS_HYBRID_RTT_GEO", "geometry", seed=seed),
        ], ignore_index=True)
        validation_macro_all.append(macro_rows(val_metrics))
        test_macro_all.append(macro_rows(test_metrics))

    validation_macro_all = pd.concat(validation_macro_all, ignore_index=True)
    test_macro_all = pd.concat(test_macro_all, ignore_index=True)
    validation_macro_all.to_csv(RESULTS_TABLE_DIR / "cnn_lams_multiseed_validation_metrics.csv", index=False)
    test_macro_all.to_csv(RESULTS_TABLE_DIR / "cnn_lams_multiseed_test_metrics.csv", index=False)
    summarize_seed_metrics(test_macro_all).to_csv(RESULTS_TABLE_DIR / "cnn_lams_multiseed_test_summary_mean_std.csv", index=False)

    representative = select_representative_seeds(validation_macro_all)
    representative.to_csv(RESULTS_TABLE_DIR / "cnn_lams_representative_seed_for_cdf.csv", index=False)

    # Build representative prediction CSVs. A different CNN candidate may have
    # a different representative seed, so we construct the representative table
    # column-by-column instead of blindly copying a whole seed file.
    rss_rep = make_representative_eval("RSS", "test", representative, seed_eval_tables, rss_test)
    rtt_rep = make_representative_eval("RTT", "test", representative, seed_eval_tables, rtt_test)
    rss_rep.to_csv(PREDICTION_DIR / "cnn_lams_rss_representative_geometry_test_predictions.csv", index=False)
    rtt_rep.to_csv(PREDICTION_DIR / "cnn_lams_rtt_representative_geometry_test_predictions.csv", index=False)

    metadata = {
        "family": FAMILY,
        "config": CONFIG,
        "seeds": SEEDS,
        "primary_metric": "scenario-macro RMSE",
        "secondary_metrics": ["scenario-macro MAE", "scenario-macro P95"],
        "representative_seed_rule": "validation-median RMSE_macro mean over available targets",
        "rss_scalar_features": rss_scalar_features,
        "rtt_scalar_features": rtt_scalar_features,
    }
    (MODEL_OUT_DIR / "cnn_lams_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(PREFIX, "training/train_cnn_lams.py completed successfully.")


if __name__ == "__main__":
    main()
