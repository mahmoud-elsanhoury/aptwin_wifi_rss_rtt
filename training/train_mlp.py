# ============================================================
# training/train_mlp.py
# ------------------------------------------------------------
# Purpose:
# - Train standalone geometry-aware Gaussian MLP models for RSS/RTT.
# - Remove notebook-memory dependencies such as HAS_TORCH, X_sub_s,
#   Y_sub_s, w_sub_scenario, y_scaler, and test_df.
# - Add the agreed fixed multi-seed reporting protocol without changing
#   the model variants, architecture, calibration logic, or data splits.
#
# Scientific protocol:
# - Uses PyTorch because scikit-learn does not provide a native multi-output
#   Gaussian neural regressor with learned sigma.
# - Uses a small fixed MLP with mean and sigma heads.
# - Trains two variants only:
#     1) RawPooledGaussianMLP
#     2) ScenarioWeightedGaussianMLP
# - Early stopping uses validation NLL.
# - Sigma calibration uses validation data only.
# - Repeats the same fixed protocol over fixed seeds.
# - Main multi-seed reporting uses mean ± std over seeds.
# - The representative seed is used only for CDF plotting and is selected
#   from validation scenario-macro RMSE using the validation-median rule.
# - No test metric is used for seed selection.
# - Official test data are evaluated only after training/calibration.
# Random Forest and XGBoost are treated as deterministic tabular regressors optimized for point prediction. 
# In contrast, the Gaussian MLP is formulated as a probabilistic regressor that predicts both mean and uncertainty; 
# therefore, it is trained using Gaussian negative log-likelihood and calibrated on the validation split.
# ============================================================

from __future__ import annotations

import shutil
import time
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    raise ImportError("PyTorch is required for training/train_mlp.py. Install torch first.") from exc

try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from common_training import (
    FEATURES_GEOM_ONLY,
    MODEL_DIR,
    PREDICTION_DIR,
    RESULTS_TABLE_DIR,
    SEEDS,
    TARGETS,
    build_standard_scalers,
    extract_scenario_macro_metrics,
    load_row_splits,
    log,
    save_json,
    save_prediction_bundle,
    scenario_balanced_weights,
    set_all_seeds,
    summarize_seed_metrics,
)

PREFIX = "train_mlp"
FAMILY = "MLP"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLP_CONFIG = {
    "hidden": [128, 128, 64],
    "lr": 1e-3,
    "batch_size": 4096,
    "max_epochs": 60,
    "patience": 10,
    "weight_decay": 1e-5,
}


def gaussian_nll_np(y_true, mu, sigma) -> float:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((y_true - mu) / sigma) ** 2
    return float(np.mean(nll))


class TabularGaussianDataset(Dataset):
    def __init__(self, X, Y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.weights[idx]


class GaussianMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2, hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, output_dim)
        self.log_sigma_head = nn.Linear(prev, output_dim)

    def forward(self, x):
        z = self.backbone(x)
        mu = self.mu_head(z)
        log_sigma = torch.clamp(self.log_sigma_head(z), min=-5.0, max=3.0)
        sigma = torch.exp(log_sigma)
        return mu, sigma


def weighted_gaussian_nll(y, mu, sigma, weights):
    sigma = torch.clamp(sigma, min=1e-6)
    nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2
    nll = nll.mean(dim=1)
    return (nll * weights).mean()


@torch.no_grad()
def eval_scaled_nll(model, X, Y) -> float:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    w_t = torch.ones(len(X_t), dtype=torch.float32, device=DEVICE)
    mu, sigma = model(X_t)
    return float(weighted_gaussian_nll(Y_t, mu, sigma, w_t).cpu().item())


def train_gaussian_mlp(X_train, Y_train, w_train, X_val, Y_val, model_name: str, seed: int) -> GaussianMLP:
    """
    Train one Gaussian MLP instance for one fixed seed.

    Scientific note:
    - Architecture and optimizer settings are unchanged.
    - The DataLoader generator is seeded so shuffle order is controlled.
    """
    train_ds = TabularGaussianDataset(X_train, Y_train, w_train)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        train_ds,
        batch_size=MLP_CONFIG["batch_size"],
        shuffle=True,
        generator=generator,
    )

    model = GaussianMLP(
        X_train.shape[1],
        Y_train.shape[1],
        hidden=tuple(MLP_CONFIG["hidden"]),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_CONFIG["lr"],
        weight_decay=MLP_CONFIG["weight_decay"],
    )

    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    history = []
    t0 = time.time()

    log(PREFIX, f"{model_name} | seed {seed}: training started on {DEVICE}.")
    for epoch in range(1, MLP_CONFIG["max_epochs"] + 1):
        ep0 = time.time()
        model.train()
        losses = []
        for xb, yb, wb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            wb = wb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            mu, sigma = model(xb)
            loss = weighted_gaussian_nll(yb, mu, sigma, wb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses))
        val_loss = eval_scaled_nll(model, X_val, Y_val)
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
            status = "IMPROVED"
        else:
            wait += 1
            status = "NO_IMPROVE"

        history.append({
            "seed": int(seed),
            "epoch": epoch,
            "train_nll": train_loss,
            "val_nll": val_loss,
            "best_val_nll": best_val,
            "best_epoch": best_epoch,
            "status": status,
        })
        if epoch <= 5 or epoch % 5 == 0 or improved or wait >= max(1, MLP_CONFIG["patience"] - 3):
            log(
                PREFIX,
                f"{model_name} | seed {seed} | epoch {epoch:03d} | train NLL={train_loss:.4f} | "
                f"val NLL={val_loss:.4f} | best={best_val:.4f}@{best_epoch:03d} | "
                f"patience={wait}/{MLP_CONFIG['patience']} | {status} | epoch={time.time()-ep0:.1f}s",
            )
        if wait >= MLP_CONFIG["patience"]:
            log(PREFIX, f"{model_name} | seed {seed}: early stopping at epoch {epoch}. Best epoch={best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} | seed {seed}: no valid checkpoint was stored.")
    model.load_state_dict(best_state)
    model.history_ = pd.DataFrame(history)
    model.best_epoch_ = best_epoch
    model.best_val_nll_ = best_val
    log(PREFIX, f"{model_name} | seed {seed}: restored best checkpoint. Total time={(time.time()-t0)/60.0:.2f} min.")
    return model


@torch.no_grad()
def predict_gaussian(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mu_s, sigma_s = model(X_t)
    return mu_s.cpu().numpy(), sigma_s.cpu().numpy()


def inverse_mu_sigma(mu_s, sigma_s, y_scaler):
    mu = y_scaler.inverse_transform(mu_s)
    sigma = np.maximum(sigma_s * y_scaler.scale_, 1e-6)
    return mu, sigma


def fit_sigma_scale(y_true, mu, sigma):
    def objective(alpha):
        return gaussian_nll_np(y_true, mu, sigma * alpha)

    if HAS_SCIPY:
        res = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
        return float(res.x)

    alphas = np.logspace(np.log10(0.05), np.log10(10.0), 200)
    vals = np.array([objective(a) for a in alphas])
    return float(alphas[np.argmin(vals)])


def select_representative_seeds_for_models(validation_macro: pd.DataFrame) -> pd.DataFrame:
    """Select one validation-median RMSE seed per MLP variant."""
    summary = (
        validation_macro
        .groupby(["family", "model", "seed"], as_index=False)
        .agg(validation_RMSE_macro_mean=("RMSE_macro", "mean"))
    )

    rows = []
    for (family, model), g in summary.groupby(["family", "model"], sort=True):
        median_rmse = float(g["validation_RMSE_macro_mean"].median())
        gg = g.copy()
        gg["distance_to_validation_median_RMSE"] = (
            gg["validation_RMSE_macro_mean"] - median_rmse
        ).abs()
        chosen = gg.sort_values(
            ["distance_to_validation_median_RMSE", "validation_RMSE_macro_mean", "seed"],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append({
            "family": family,
            "model": model,
            "representative_seed": int(chosen["seed"]),
            "selection_rule": "validation_median_RMSE_macro_mean_over_level_target_pairs",
            "validation_RMSE_macro_mean": float(chosen["validation_RMSE_macro_mean"]),
            "validation_RMSE_macro_median": median_rmse,
            "distance_to_validation_median_RMSE": float(chosen["distance_to_validation_median_RMSE"]),
        })
    return pd.DataFrame(rows)


def copy_representative_predictions(model_prefix: str, representative_seed: int) -> None:
    """Copy seed-specific prediction files into representative filenames."""
    for level in ["row", "geometry"]:
        src = PREDICTION_DIR / f"{model_prefix}_seed{representative_seed}_{level}_test_predictions.csv"
        dst = PREDICTION_DIR / f"{model_prefix}_representative_{level}_test_predictions.csv"
        if not src.exists():
            raise FileNotFoundError(f"Representative source prediction file not found: {src}")
        shutil.copy2(src, dst)
        log(PREFIX, f"Saved representative {level} predictions: {dst}")


def main() -> None:
    log(PREFIX, "Starting standalone Gaussian MLP multi-seed training.")
    log(PREFIX, f"Fixed seed set: {SEEDS}")

    splits = load_row_splits(kind="base", prefix=PREFIX)
    train_df, val_df, test_df = splits["subtrain"], splits["val"], splits["test"]

    x_scaler, y_scaler = build_standard_scalers(train_df, FEATURES_GEOM_ONLY)
    X_train = x_scaler.transform(train_df[FEATURES_GEOM_ONLY].to_numpy(dtype=np.float32)).astype(np.float32)
    X_val = x_scaler.transform(val_df[FEATURES_GEOM_ONLY].to_numpy(dtype=np.float32)).astype(np.float32)
    X_test = x_scaler.transform(test_df[FEATURES_GEOM_ONLY].to_numpy(dtype=np.float32)).astype(np.float32)
    Y_train = y_scaler.transform(train_df[TARGETS].to_numpy(dtype=np.float32)).astype(np.float32)
    Y_val_scaled = y_scaler.transform(val_df[TARGETS].to_numpy(dtype=np.float32)).astype(np.float32)
    Y_val = val_df[TARGETS].to_numpy(dtype=float)

    weights = {
        "RawPooledGaussianMLP": np.ones(len(train_df), dtype=np.float32),
        "ScenarioWeightedGaussianMLP": scenario_balanced_weights(train_df),
    }

    prefix_lookup = {
        "RawPooledGaussianMLP": "mlp_raw_pooled",
        "ScenarioWeightedGaussianMLP": "mlp_scenario_weighted",
    }

    model_dir = MODEL_DIR / "mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "y_scaler.joblib")

    validation_macro_all = []
    test_macro_all = []
    all_calibration = {}

    for seed in SEEDS:
        set_all_seeds(seed)
        log(PREFIX, "=" * 78)
        log(PREFIX, f"Starting MLP seed {seed}.")

        for model_name, w_train in weights.items():
            model = train_gaussian_mlp(
                X_train,
                Y_train,
                w_train,
                X_val,
                Y_val_scaled,
                model_name=model_name,
                seed=seed,
            )
            model.history_.to_csv(model_dir / f"{model_name}_seed{seed}_history.csv", index=False)

            mu_val_s, sigma_val_s = predict_gaussian(model, X_val)
            mu_val, sigma_val = inverse_mu_sigma(mu_val_s, sigma_val_s, y_scaler)
            alpha_rss = fit_sigma_scale(Y_val[:, 0], mu_val[:, 0], sigma_val[:, 0])
            alpha_rtt = fit_sigma_scale(Y_val[:, 1], mu_val[:, 1], sigma_val[:, 1])
            alpha = np.array([alpha_rss, alpha_rtt], dtype=float)
            all_calibration[f"{model_name}_seed{seed}"] = alpha.tolist()
            log(PREFIX, f"{model_name} | seed {seed}: sigma calibration alpha RSS={alpha_rss:.3f}, RTT={alpha_rtt:.3f}")

            mu_val_cal = mu_val
            _, _, val_metrics = save_prediction_bundle(
                row_df=val_df,
                y_pred=mu_val_cal,
                model=model_name,
                family=FAMILY,
                prefix=f"{prefix_lookup[model_name]}_seed{seed}",
                split_name="val",
                seed=seed,
            )
            validation_macro_all.append(extract_scenario_macro_metrics(val_metrics, seed=seed))

            mu_test_s, sigma_test_s = predict_gaussian(model, X_test)
            mu_test, sigma_test = inverse_mu_sigma(mu_test_s, sigma_test_s, y_scaler)
            sigma_test = sigma_test * alpha

            _, _, test_metrics = save_prediction_bundle(
                row_df=test_df,
                y_pred=mu_test,
                model=model_name,
                family=FAMILY,
                prefix=f"{prefix_lookup[model_name]}_seed{seed}",
                split_name="test",
                seed=seed,
            )
            test_macro_all.append(extract_scenario_macro_metrics(test_metrics, seed=seed))

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "features": FEATURES_GEOM_ONLY,
                    "config": MLP_CONFIG,
                    "seed": int(seed),
                    "best_epoch": int(model.best_epoch_),
                    "best_val_nll": float(model.best_val_nll_),
                    "sigma_calibration": alpha.tolist(),
                },
                model_dir / f"{model_name}_seed{seed}.pt",
            )

    validation_macro_all = pd.concat(validation_macro_all, ignore_index=True)
    test_macro_all = pd.concat(test_macro_all, ignore_index=True)

    validation_macro_all.to_csv(RESULTS_TABLE_DIR / "mlp_multiseed_validation_metrics.csv", index=False)
    test_macro_all.to_csv(RESULTS_TABLE_DIR / "mlp_multiseed_test_metrics.csv", index=False)

    test_summary = summarize_seed_metrics(test_macro_all)
    test_summary.to_csv(RESULTS_TABLE_DIR / "mlp_multiseed_test_summary_mean_std.csv", index=False)

    representative = select_representative_seeds_for_models(validation_macro_all)
    representative.to_csv(RESULTS_TABLE_DIR / "mlp_representative_seed_for_cdf.csv", index=False)

    for _, row in representative.iterrows():
        copy_representative_predictions(
            model_prefix=prefix_lookup[row["model"]],
            representative_seed=int(row["representative_seed"]),
        )

    save_json(
        {
            "family": FAMILY,
            "features": FEATURES_GEOM_ONLY,
            "config": MLP_CONFIG,
            "seeds": SEEDS,
            "primary_metric": "scenario-macro RMSE",
            "secondary_metrics": ["scenario-macro MAE", "scenario-macro P95"],
            "representative_seed_rule": "validation-median RMSE_macro mean across available level-target pairs",
            "calibration": all_calibration,
        },
        model_dir / "mlp_metadata.json",
    )
    log(PREFIX, "training/train_mlp.py completed successfully.")


if __name__ == "__main__":
    main()
