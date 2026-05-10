# ============================================================
# training/train_gnn.py
# ------------------------------------------------------------
# Purpose:
# - Train and evaluate a GraphSAGE edge-regression GNN as a standalone
#   repository script.
# - Remove notebook-memory dependencies such as GNN_GRAPHS, test_graphs,
#   gnn_target_scaler, and wall_xgb_predictions.
# - Save final RSS and RTT predictions in the unified long CSV format
#   expected by performance_metrics/compute_metrics.py.
# - Add the agreed fixed multi-seed reporting protocol without changing
#   the GraphSAGE architecture, graph construction, or split logic.
#
# Scientific protocol:
# - This script uses PyTorch Geometric, a trusted graph-learning library.
# - If PyTorch Geometric is unavailable, the script stops clearly.
# - No fake GNN output is created.
# - No hyperparameter search is performed.
# - One fixed GraphSAGE architecture is trained.
# - The same fixed protocol is repeated over fixed seeds.
# - Main multi-seed reporting uses mean ± std over seeds.
# - The representative seed is used only for CDF plotting and is selected
#   from validation scenario-macro RMSE using the validation-median rule.
# - No test metric is used for seed selection.
# - The same official train/validation/test split produced earlier in
#   the repository pipeline is preserved.
# - Training uses scenario-balanced loss:
#       loss = average(loss per scenario graph)
#   so the largest scenario does not dominate the GNN objective.
# - Validation loss is used only for early stopping and checkpoint
#   restoration.
# - Test data is used only once for final reporting.
#
# Inputs:
# - output_csvs/processed_features/geometry_train_wall_context.csv
# - output_csvs/processed_features/geometry_val_wall_context.csv
# - output_csvs/processed_features/geometry_test_wall_context.csv
#
# Outputs:
# - models/saved_models/gnn/graphsage_edge_regressor_best.pt
# - models/saved_models/gnn/gnn_training_history.csv
# - models/saved_models/gnn/gnn_graph_summary.csv
# - output_csvs/predictions/gnn_test_predictions.csv
# - results/tables/gnn_test_metrics.csv
# ============================================================

from __future__ import annotations

import importlib.util
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise ImportError(
        "PyTorch is required for the GNN branch. Install torch first."
    ) from exc

if importlib.util.find_spec("torch_geometric") is None:
    raise RuntimeError(
        "PyTorch Geometric is missing. No GNN experiment was run.\n"
        "Install torch-geometric first if you want the optional GNN branch.\n"
        "The pipeline may continue if train_gnn is configured as optional."
    )

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from common_training import (
    SEEDS,
    set_all_seeds,
    summarize_seed_metrics,
)


# ============================================================
# Section 1 — Paths and fixed configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = REPO_ROOT / "output_csvs" / "processed_features"
PREDICTION_DIR = REPO_ROOT / "output_csvs" / "predictions"
MODEL_DIR = REPO_ROOT / "models" / "saved_models" / "gnn"
RESULTS_TABLE_DIR = REPO_ROOT / "results" / "tables"

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# The legacy single seed is replaced by the shared fixed seed set in common_training.SEEDS.
# Each seed is used as repeated experimental randomness, not as a hyperparameter.

MAX_EPOCHS = 300
PATIENCE = 35
PRINT_EVERY = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
DROPOUT = 0.15

GEOMETRY_SPLIT_FILES = {
    "subtrain": PROCESSED_DIR / "geometry_train_wall_context.csv",
    "val": PROCESSED_DIR / "geometry_val_wall_context.csv",
    "test": PROCESSED_DIR / "geometry_test_wall_context.csv",
}

EDGE_FEATURES_REQUESTED = [
    "dx",
    "dy",
    "distance",
    "LOS_flag",
    "LOS_known",
    "wall_cross_count_total",
    "wall_cross_count_internal",
    "wall_cross_count_external",
    "obstructed_path",
    "AP_dist_nearest_wall",
    "AP_dist_nearest_external_wall",
    "AP_dist_nearest_internal_wall",
    "RP_dist_nearest_wall",
    "RP_dist_nearest_external_wall",
    "RP_dist_nearest_internal_wall",
    "AP_near_wall_1m",
    "RP_near_wall_1m",
]


# ============================================================
# Section 2 — Logging and deterministic setup
# ============================================================

def log(message: str) -> None:
    print(f"[train_gnn] {message}", flush=True)


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms are requested when possible. Some PyG/CUDA
    # kernels may still be hardware-dependent, so this is a best-effort guard.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


# ============================================================
# Section 3 — Metrics
# ------------------------------------------------------------
# Purpose:
# - Compute deterministic regression metrics separately for RSS and RTT.
#
# Scientific protocol:
# - Do not compare one prediction column against a two-column target array.
# - RSS and RTT are evaluated separately, then reported in the same table.
# ============================================================

def metric_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Metric shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    err = y_pred - y_true
    abs_err = np.abs(err)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "P95": float(np.percentile(abs_err, 95)),
        "Bias": float(np.mean(err)),
        "n": int(len(y_true)),
    }


def compute_gnn_metrics(eval_df: pd.DataFrame, split_name: str, seed: int) -> pd.DataFrame:
    rows = []

    target_specs = [
        ("RSS", "RSS_true", "RSS_pred_GNN"),
        ("RTT", "RTT_true", "RTT_pred_GNN"),
    ]

    for target, true_col, pred_col in target_specs:
        micro = metric_dict(eval_df[true_col], eval_df[pred_col])

        scenario_rows = []
        for scenario, g in eval_df.groupby("scenario", sort=True):
            sm = metric_dict(g[true_col], g[pred_col])
            sm["scenario"] = scenario
            scenario_rows.append(sm)

        scenario_df = pd.DataFrame(scenario_rows)

        rows.append(
            {
                "model": "GraphSAGE_GNN",
                "family": "GNN",
                "target": target,
                "level": "geometry",
                "MAE_micro": micro["MAE"],
                "RMSE_micro": micro["RMSE"],
                "P95_micro": micro["P95"],
                "seed": int(seed),
                "split": split_name,
                "MAE_macro": float(scenario_df["MAE"].mean()),
                "RMSE_macro": float(scenario_df["RMSE"].mean()),
                "P95_macro": float(scenario_df["P95"].mean()),
                "n_total": int(len(eval_df)),
                "n_scenarios": int(eval_df["scenario"].nunique()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Section 4 — Load geometry-level split tables
# ============================================================

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_geometry_splits() -> Dict[str, pd.DataFrame]:
    split_tables: Dict[str, pd.DataFrame] = {}

    for split_name, path in GEOMETRY_SPLIT_FILES.items():
        require_file(path)
        df = pd.read_csv(path).reset_index(drop=True)

        if "split" not in df.columns:
            df["split"] = split_name

        required = ["scenario", "AP_x", "AP_y", "RX_x", "RX_y", "RSS_dBm", "RTT_m"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise RuntimeError(f"{split_name} is missing required GNN columns: {missing}")

        if "dx" not in df.columns:
            df["dx"] = df["RX_x"].astype(float) - df["AP_x"].astype(float)

        if "dy" not in df.columns:
            df["dy"] = df["RX_y"].astype(float) - df["AP_y"].astype(float)

        if "distance" not in df.columns:
            df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

        split_tables[split_name] = df
        log(f"Loaded {split_name}: {len(df):,} geometry links from {path}")

    return split_tables


def select_edge_features(split_tables: Dict[str, pd.DataFrame]) -> List[str]:
    features = [
        c for c in EDGE_FEATURES_REQUESTED
        if all(c in df.columns for df in split_tables.values())
    ]

    missing = [c for c in EDGE_FEATURES_REQUESTED if c not in features]

    log("GNN edge features used:")
    for feature in features:
        log(f"  - {feature}")

    if missing:
        log("Requested edge features unavailable in all splits and excluded:")
        for feature in missing:
            log(f"  - {feature}")

    if len(features) < 3:
        raise RuntimeError(
            "Too few edge features are available. At minimum dx, dy, and distance are expected."
        )

    return features


# ============================================================
# Section 5 — Fit scalers on subtrain only
# ============================================================

@dataclass
class GNNScalers:
    node_coord_scaler: StandardScaler
    edge_scaler: StandardScaler
    target_scaler: StandardScaler


def fit_scalers(split_tables: Dict[str, pd.DataFrame], edge_features: List[str]) -> GNNScalers:
    train_df = split_tables["subtrain"]

    node_coord_train = np.vstack(
        [
            train_df[["AP_x", "AP_y"]].drop_duplicates().to_numpy(dtype=float),
            train_df[["RX_x", "RX_y"]].drop_duplicates().to_numpy(dtype=float),
        ]
    )

    edge_train = train_df[edge_features].to_numpy(dtype=float)
    target_train = train_df[["RSS_dBm", "RTT_m"]].to_numpy(dtype=float)

    scalers = GNNScalers(
        node_coord_scaler=StandardScaler(),
        edge_scaler=StandardScaler(),
        target_scaler=StandardScaler(),
    )

    scalers.node_coord_scaler.fit(node_coord_train)
    scalers.edge_scaler.fit(edge_train)
    scalers.target_scaler.fit(target_train)

    joblib.dump(scalers.node_coord_scaler, MODEL_DIR / "gnn_node_coord_scaler.joblib")
    joblib.dump(scalers.edge_scaler, MODEL_DIR / "gnn_edge_scaler.joblib")
    joblib.dump(scalers.target_scaler, MODEL_DIR / "gnn_target_scaler.joblib")

    log("Fitted node, edge, and target scalers on subtrain only.")
    return scalers


# ============================================================
# Section 6 — Build PyTorch Geometric graphs
# ============================================================

def make_node_key(kind: str, x: float, y: float, ap_id=None) -> tuple:
    if kind == "AP" and ap_id is not None:
        return ("AP", str(ap_id), round(float(x), 4), round(float(y), 4))

    return (kind, round(float(x), 4), round(float(y), 4))


def add_geometry_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AP_x_key"] = out["AP_x"].astype(float).round(4)
    out["AP_y_key"] = out["AP_y"].astype(float).round(4)
    out["RX_x_key"] = out["RX_x"].astype(float).round(4)
    out["RX_y_key"] = out["RX_y"].astype(float).round(4)
    return out


def build_graph_for_scenario(
    df: pd.DataFrame,
    split_name: str,
    scenario_name: str,
    edge_features: List[str],
    scalers: GNNScalers,
) -> Tuple[Data, pd.DataFrame]:
    g = df[df["scenario"].astype(str) == str(scenario_name)].copy().reset_index(drop=True)

    if g.empty:
        raise RuntimeError(f"{split_name}/{scenario_name}: empty scenario table.")

    node_keys = []
    node_xy = []
    node_type = []
    node_to_idx = {}

    def add_node(key, x, y, is_ap: bool) -> int:
        if key in node_to_idx:
            return node_to_idx[key]

        idx = len(node_keys)
        node_to_idx[key] = idx
        node_keys.append(key)
        node_xy.append([float(x), float(y)])
        node_type.append([1.0, 0.0] if is_ap else [0.0, 1.0])
        return idx

    src_list = []
    dst_list = []
    src_rev_list = []
    dst_rev_list = []
    edge_feature_rows = []
    target_rows = []
    edge_table_rows = []

    for _, row in g.iterrows():
        ap_id = row["AP_id_raw"] if "AP_id_raw" in g.columns else None

        ap_key = make_node_key("AP", row["AP_x"], row["AP_y"], ap_id=ap_id)
        rp_key = make_node_key("RP", row["RX_x"], row["RX_y"], ap_id=None)

        ap_idx = add_node(ap_key, row["AP_x"], row["AP_y"], is_ap=True)
        rp_idx = add_node(rp_key, row["RX_x"], row["RX_y"], is_ap=False)

        src_list.append(ap_idx)
        dst_list.append(rp_idx)
        src_rev_list.append(rp_idx)
        dst_rev_list.append(ap_idx)

        edge_feature_rows.append(row[edge_features].to_numpy(dtype=float))
        target_rows.append([float(row["RSS_dBm"]), float(row["RTT_m"])])

        edge_table_rows.append(row.to_dict())

    node_xy = np.asarray(node_xy, dtype=float)
    node_type = np.asarray(node_type, dtype=float)

    node_xy_scaled = scalers.node_coord_scaler.transform(node_xy)
    x_node = np.hstack([node_xy_scaled, node_type]).astype(np.float32)

    edge_features_scaled = scalers.edge_scaler.transform(
        np.asarray(edge_feature_rows, dtype=float)
    ).astype(np.float32)

    targets_raw = np.asarray(target_rows, dtype=float)
    targets_scaled = scalers.target_scaler.transform(targets_raw).astype(np.float32)

    edge_index_forward = np.vstack([src_list, dst_list])
    edge_index_reverse = np.vstack([src_rev_list, dst_rev_list])
    edge_index = np.hstack([edge_index_forward, edge_index_reverse]).astype(np.int64)

    data = Data(
        x=torch.tensor(x_node, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_label_index=torch.tensor(edge_index_forward, dtype=torch.long),
        edge_label_attr=torch.tensor(edge_features_scaled, dtype=torch.float32),
        y=torch.tensor(targets_scaled, dtype=torch.float32),
        y_raw=torch.tensor(targets_raw, dtype=torch.float32),
    )

    data.scenario = str(scenario_name)
    data.split = split_name
    data.n_nodes = int(len(node_keys))
    data.n_prediction_edges = int(len(src_list))

    edge_table = pd.DataFrame(edge_table_rows)
    edge_table["gnn_edge_pos"] = np.arange(len(edge_table), dtype=int)

    return data, edge_table


def build_graphs(
    split_tables: Dict[str, pd.DataFrame],
    edge_features: List[str],
    scalers: GNNScalers,
) -> Tuple[Dict[str, List[Data]], Dict[str, pd.DataFrame]]:
    graphs: Dict[str, List[Data]] = {"subtrain": [], "val": [], "test": []}
    edge_tables: Dict[str, List[pd.DataFrame]] = {"subtrain": [], "val": [], "test": []}

    for split_name, raw_df in split_tables.items():
        df = add_geometry_keys(raw_df)
        log(f"Building PyG graphs for split: {split_name}")

        for scenario_name in sorted(df["scenario"].astype(str).unique()):
            data, edge_table = build_graph_for_scenario(
                df=df,
                split_name=split_name,
                scenario_name=scenario_name,
                edge_features=edge_features,
                scalers=scalers,
            )

            graphs[split_name].append(data)
            edge_tables[split_name].append(edge_table)

            log(
                f"  {scenario_name}: nodes={data.n_nodes:,}, "
                f"AP-RP edges={data.n_prediction_edges:,}"
            )

    edge_tables_final = {
        split: pd.concat(tables, ignore_index=True)
        for split, tables in edge_tables.items()
    }

    summary_rows = []
    for split_name, graph_list in graphs.items():
        for data in graph_list:
            summary_rows.append(
                {
                    "split": split_name,
                    "scenario": data.scenario,
                    "n_nodes": data.n_nodes,
                    "n_prediction_edges": data.n_prediction_edges,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MODEL_DIR / "gnn_graph_summary.csv", index=False)

    for split_name, table in edge_tables_final.items():
        table.to_csv(MODEL_DIR / f"gnn_{split_name}_edge_table.csv", index=False)

    return graphs, edge_tables_final


# ============================================================
# Section 7 — GraphSAGE model
# ============================================================

class GraphSAGEEdgeRegressor(nn.Module):
    """
    Simple edge-regression GNN.
    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_encoder(data.x)

        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, data.edge_index)
        x = F.relu(x)

        src = data.edge_label_index[0]
        dst = data.edge_label_index[1]

        edge_input = torch.cat(
            [x[src], x[dst], data.edge_label_attr],
            dim=1,
        )

        return self.edge_decoder(edge_input)


# ============================================================
# Section 8 — Training
# ============================================================

def scenario_balanced_loss(model: nn.Module, graphs: List[Data]) -> torch.Tensor:
    losses = []

    for data in graphs:
        pred = model(data)
        loss = F.mse_loss(pred, data.y)
        losses.append(loss)

    return torch.stack(losses).mean()


@torch.no_grad()
def scenario_balanced_eval_loss(model: nn.Module, graphs: List[Data]) -> float:
    model.eval()
    return float(scenario_balanced_loss(model, graphs).detach().cpu().item())


def train_gnn_model(graphs: Dict[str, List[Data]], device: torch.device, seed: int) -> Tuple[nn.Module, pd.DataFrame]:
    train_graphs = [g.to(device) for g in graphs["subtrain"]]
    val_graphs = [g.to(device) for g in graphs["val"]]

    if not train_graphs or not val_graphs:
        raise RuntimeError("GNN train and validation graphs must be non-empty.")

    node_in_dim = train_graphs[0].x.shape[1]
    edge_in_dim = train_graphs[0].edge_label_attr.shape[1]

    model = GraphSAGEEdgeRegressor(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    log(f"GNN training configuration for seed {seed}:")
    log(f"  device       = {device}")
    log(f"  node_in_dim  = {node_in_dim}")
    log(f"  edge_in_dim  = {edge_in_dim}")
    log(f"  hidden_dim   = {HIDDEN_DIM}")
    log(f"  dropout      = {DROPOUT}")
    log(f"  max_epochs   = {MAX_EPOCHS}")
    log(f"  patience     = {PATIENCE}")
    log(f"  lr           = {LEARNING_RATE}")
    log(f"  weight_decay = {WEIGHT_DECAY}")

    history = []
    best_val_loss = np.inf
    best_epoch = -1
    best_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        train_loss = scenario_balanced_loss(model, train_graphs)
        train_loss.backward()
        optimizer.step()

        val_loss = scenario_balanced_eval_loss(model, val_graphs)

        improved = val_loss < best_val_loss - 1e-8

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss_scaled": float(train_loss.detach().cpu().item()),
                "val_loss_scaled": float(val_loss),
                "best_val_loss_scaled": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "patience_counter": int(patience_counter),
                "elapsed_min": float((time.time() - start_time) / 60.0),
            }
        )

        should_print = (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or improved
            or patience_counter >= max(1, PATIENCE - 5)
        )

        if should_print:
            log(
                f"epoch {epoch:03d} | "
                f"train_loss={float(train_loss.detach().cpu().item()):.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"best={best_val_loss:.6f}@{best_epoch} | "
                f"patience={patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            log(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best GNN checkpoint was recorded.")

    model.load_state_dict(best_state)
    model.eval()

    history_df = pd.DataFrame(history)
    history_df.to_csv(MODEL_DIR / f"gnn_training_history_seed{seed}.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, MODEL_DIR / f"graphsage_edge_regressor_best_seed{seed}.pt")

    log(f"Restored best GNN checkpoint from epoch {best_epoch}.")
    log(f"Saved model: {MODEL_DIR / f'graphsage_edge_regressor_best_seed{seed}.pt'}")

    return model, history_df


# ============================================================
# Section 9 — Evaluation and prediction export
# ============================================================

@torch.no_grad()
def evaluate_test_graphs(
    model: nn.Module,
    test_graphs: List[Data],
    test_edge_table: pd.DataFrame,
    target_scaler: StandardScaler,
    device: torch.device,
) -> pd.DataFrame:
    frames = []

    model.eval()

    for data in test_graphs:
        data = data.to(device)

        pred_scaled = model(data).detach().cpu().numpy()
        true_scaled = data.y.detach().cpu().numpy()

        pred_raw = target_scaler.inverse_transform(pred_scaled)
        true_raw = target_scaler.inverse_transform(true_scaled)

        frame = pd.DataFrame(
            {
                "scenario": data.scenario,
                "RSS_true": true_raw[:, 0],
                "RTT_true": true_raw[:, 1],
                "RSS_pred_GNN": pred_raw[:, 0],
                "RTT_pred_GNN": pred_raw[:, 1],
                "gnn_edge_pos": np.arange(len(true_raw), dtype=int),
            }
        )

        frames.append(frame)

    pred_df = pd.concat(frames, ignore_index=True)

    if len(pred_df) != len(test_edge_table):
        raise RuntimeError(
            "GNN prediction table and test edge table have different lengths:\n"
            f"predictions={len(pred_df)}, edge_table={len(test_edge_table)}"
        )

    eval_df = pd.concat(
        [
            test_edge_table.reset_index(drop=True),
            pred_df.drop(columns=["scenario", "gnn_edge_pos"]).reset_index(drop=True),
        ],
        axis=1,
    )

    return eval_df


def export_long_predictions(eval_df: pd.DataFrame, pred_path: Path, seed: int | None = None) -> pd.DataFrame:
    rows = []

    for idx, row in eval_df.reset_index(drop=True).iterrows():
        unit_id_base = row.get("lams_id", idx)

        rows.append(
            {
                "level": "geometry",
                "target": "RSS",
                "family": "GNN",
                "model": "GraphSAGE_GNN",
                "scenario": row["scenario"],
                "unit_id": f"{row['scenario']}__{unit_id_base}",
                "y_true": float(row["RSS_true"]),
                "y_pred": float(row["RSS_pred_GNN"]),
                **({"seed": int(seed)} if seed is not None else {}),
            }
        )

        rows.append(
            {
                "level": "geometry",
                "target": "RTT",
                "family": "GNN",
                "model": "GraphSAGE_GNN",
                "scenario": row["scenario"],
                "unit_id": f"{row['scenario']}__{unit_id_base}",
                "y_true": float(row["RTT_true"]),
                "y_pred": float(row["RTT_pred_GNN"]),
                **({"seed": int(seed)} if seed is not None else {}),
            }
        )

    out = pd.DataFrame(rows)

    out.to_csv(pred_path, index=False)

    log(f"Saved unified GNN prediction CSV: {pred_path}")
    return out


# ============================================================
# Section 10 — Main
# ============================================================

def macro_metric_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert GNN metric rows to the shared multi-seed macro schema.
    """
    out = metrics_df[[
        "family", "model", "seed", "level", "target",
        "RMSE_macro", "MAE_macro", "P95_macro", "n_scenarios",
    ]].copy()
    return out.reset_index(drop=True)


def select_representative_seed(validation_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Select one representative GNN seed using validation-median RMSE.

    Because one GNN checkpoint predicts both RSS and RTT, the representative
    seed is shared across both targets.
    """
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
            "selection_rule": "validation_median_RMSE_macro_mean_over_targets",
            "validation_RMSE_macro_mean": float(chosen["validation_RMSE_macro_mean"]),
            "validation_RMSE_macro_median": median_rmse,
            "distance_to_validation_median_RMSE": float(chosen["distance_to_validation_median_RMSE"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# Section 10 — Main
# ============================================================

def main() -> None:
    log("Starting standalone GraphSAGE GNN multi-seed training.")
    log(f"Repository root: {REPO_ROOT}")
    log(f"Fixed seed set: {SEEDS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"PyTorch version: {torch.__version__}")
    log(f"Device: {device}")

    split_tables = load_geometry_splits()
    edge_features = select_edge_features(split_tables)

    # Scalers and graph construction remain outside the seed loop because they
    # are deterministic transformations fitted on subtrain only.
    scalers = fit_scalers(split_tables, edge_features)

    graphs, edge_tables = build_graphs(
        split_tables=split_tables,
        edge_features=edge_features,
        scalers=scalers,
    )

    validation_macro_all = []
    test_macro_all = []
    all_test_eval_paths = []

    for seed in SEEDS:
        set_all_seeds(seed)
        set_reproducibility(seed)
        log("=" * 78)
        log(f"Starting GraphSAGE seed {seed}.")

        model, history_df = train_gnn_model(
            graphs=graphs,
            device=device,
            seed=seed,
        )

        val_eval_df = evaluate_test_graphs(
            model=model,
            test_graphs=graphs["val"],
            test_edge_table=edge_tables["val"],
            target_scaler=scalers.target_scaler,
            device=device,
        )
        val_eval_df["seed"] = int(seed)
        val_eval_path = MODEL_DIR / f"gnn_val_predictions_geometry_level_seed{seed}.csv"
        val_eval_df.to_csv(val_eval_path, index=False)
        log(f"Saved validation geometry-level GNN evaluation table: {val_eval_path}")

        test_eval_df = evaluate_test_graphs(
            model=model,
            test_graphs=graphs["test"],
            test_edge_table=edge_tables["test"],
            target_scaler=scalers.target_scaler,
            device=device,
        )
        test_eval_df["seed"] = int(seed)
        test_eval_path = MODEL_DIR / f"gnn_test_predictions_geometry_level_seed{seed}.csv"
        test_eval_df.to_csv(test_eval_path, index=False)
        all_test_eval_paths.append(str(test_eval_path))
        log(f"Saved test geometry-level GNN evaluation table: {test_eval_path}")

        val_metrics = compute_gnn_metrics(val_eval_df, split_name="val", seed=seed)
        test_metrics = compute_gnn_metrics(test_eval_df, split_name="test", seed=seed)

        val_metrics.to_csv(RESULTS_TABLE_DIR / f"gnn_seed{seed}_validation_metrics.csv", index=False)
        test_metrics.to_csv(RESULTS_TABLE_DIR / f"gnn_seed{seed}_test_metrics.csv", index=False)

        validation_macro_all.append(macro_metric_rows(val_metrics))
        test_macro_all.append(macro_metric_rows(test_metrics))

        export_long_predictions(
            test_eval_df,
            pred_path=PREDICTION_DIR / f"gnn_seed{seed}_test_predictions.csv",
            seed=seed,
        )

    validation_macro_all = pd.concat(validation_macro_all, ignore_index=True)
    test_macro_all = pd.concat(test_macro_all, ignore_index=True)

    validation_macro_all.to_csv(RESULTS_TABLE_DIR / "gnn_multiseed_validation_metrics.csv", index=False)
    test_macro_all.to_csv(RESULTS_TABLE_DIR / "gnn_multiseed_test_metrics.csv", index=False)

    test_summary = summarize_seed_metrics(test_macro_all)
    test_summary.to_csv(RESULTS_TABLE_DIR / "gnn_multiseed_test_summary_mean_std.csv", index=False)

    representative = select_representative_seed(validation_macro_all)
    representative.to_csv(RESULTS_TABLE_DIR / "gnn_representative_seed_for_cdf.csv", index=False)
    representative_seed = int(representative.iloc[0]["representative_seed"])

    representative_eval_path = MODEL_DIR / f"gnn_test_predictions_geometry_level_seed{representative_seed}.csv"
    representative_eval_df = pd.read_csv(representative_eval_path)
    export_long_predictions(
        representative_eval_df,
        pred_path=PREDICTION_DIR / "gnn_representative_test_predictions.csv",
        seed=representative_seed,
    )

    # Backward-compatible aliases for downstream scripts that still look for
    # the previous single-seed filenames. These aliases contain the agreed
    # representative seed, not a best test seed.
    representative_eval_df.to_csv(MODEL_DIR / "gnn_test_predictions_geometry_level.csv", index=False)
    representative_metrics = compute_gnn_metrics(representative_eval_df, split_name="test", seed=representative_seed)
    representative_metrics.to_csv(RESULTS_TABLE_DIR / "gnn_test_metrics.csv", index=False)
    export_long_predictions(
        representative_eval_df,
        pred_path=PREDICTION_DIR / "gnn_test_predictions.csv",
        seed=representative_seed,
    )

    metadata = {
        "model": "GraphSAGE_GNN",
        "family": "GNN",
        "level": "geometry",
        "targets": ["RSS", "RTT"],
        "edge_features": edge_features,
        "seeds": SEEDS,
        "representative_seed": representative_seed,
        "representative_seed_rule": "validation-median RMSE_macro mean over RSS and RTT",
        "primary_metric": "scenario-macro RMSE",
        "secondary_metrics": ["scenario-macro MAE", "scenario-macro P95"],
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "test_eval_paths": all_test_eval_paths,
    }

    (MODEL_DIR / "gnn_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    log("training/train_gnn.py completed successfully.")


if __name__ == "__main__":
    main()
