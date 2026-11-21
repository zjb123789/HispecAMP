#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import json
import itertools

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau

from film_align import AlignmentModel

# =========================
# Config
# =========================
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 500
PATIENCE = 30
PATH_EMB_PATH = "../pathogen_embeddings.pt"

TAXONOMY_JSON_PATH = "../pathogen_taxonomy_by_taxid.json"

OUT_DIR = "./results_gridsearch"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "grid_search.log")

INDEPENDENT_TEST_PATH = "../warmstart/split_data/test.csv"

GS_MAX_EPOCHS = 300
GS_PATIENCE = 20

# =========================
# Logger
# =========================
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_name2taxid(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data.get("pathogen_to_taxonomy", data)
    name2taxid = {}
    for taxid, info in data.items():
        if not isinstance(info, dict):
            continue
        name = (info.get("scientific_name") or "").strip()
        if not name:
            continue
        name2taxid.setdefault(name, str(taxid).strip())
    return name2taxid

def evaluate(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "PCC": pearsonr(y_true, y_pred)[0],
        "Kendall": kendalltau(y_true, y_pred)[0],
    }

# =========================
# Dataset
# =========================
class AlignmentDataset(Dataset):
    def __init__(self, X, y, pid):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.pid = torch.tensor(pid, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pid[idx]

# =========================
# Feature builder
# =========================
def build_X(df, feature_cols, pathogen_emb, name2taxid, warn_tag=""):
    seq_feat = df[feature_cols].values.astype(np.float32)
    path_dim = next(iter(pathogen_emb.values())).numel()

    path_feat = []
    missing_map = set()
    missing_emb = set()

    for p in df["pathogen"].astype(str).str.strip():
        taxid = name2taxid.get(p)
        emb = None
        if taxid is not None:
            emb = pathogen_emb.get(taxid)
            if emb is None:
                missing_emb.add((p, taxid))
        else:
            missing_map.add(p)

        if emb is None:
            emb = torch.zeros(path_dim, dtype=torch.float32)

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().view(-1).numpy()
        path_feat.append(np.asarray(emb, dtype=np.float32).reshape(-1))

    if missing_map:
        log(f"  [Warn{(' ' + warn_tag) if warn_tag else ''}] "
            f"{len(missing_map)} 个 pathogen 名字未在 taxonomy JSON 中找到（零向量兜底）: "
            f"{sorted(missing_map)[:10]}{' ...' if len(missing_map) > 10 else ''}")
    if missing_emb:
        log(f"  [Warn{(' ' + warn_tag) if warn_tag else ''}] "
            f"{len(missing_emb)} 个 taxid 未在 {os.path.basename(PATH_EMB_PATH)} 中找到（零向量兜底）: "
            f"{sorted(list(missing_emb))[:10]}{' ...' if len(missing_emb) > 10 else ''}")

    return np.concatenate([seq_feat, np.asarray(path_feat)], axis=1)

# =========================
# Early stopping training
# =========================
def train_es(model, optimizer, train_loader, val_loader, max_epochs, patience):
    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y, pid in train_loader:
            x, y, pid = x.to(DEVICE), y.to(DEVICE), pid.to(DEVICE)
            optimizer.zero_grad()
            _, loss, _ = model(x, mic=y, path_id=pid)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, pid in val_loader:
                x, y, pid = x.to(DEVICE), y.to(DEVICE), pid.to(DEVICE)
                _, loss, _ = model(x, mic=y, path_id=pid)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            log(f"  Epoch {epoch}/{max_epochs}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, model

# =========================
# Main: Grid Search only
# =========================
def grid_search():
    set_seed(SEED)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    data_5fold = pd.read_csv('../warmstart/split_data/data_5folds.csv')
    test_df = pd.read_csv(INDEPENDENT_TEST_PATH)

    feature_cols = [c for c in data_5fold.columns if c.startswith("ESM_")]

    pathogen_emb = torch.load(PATH_EMB_PATH, map_location="cpu")

    name2taxid = load_name2taxid(TAXONOMY_JSON_PATH)
    log(f"[Taxonomy] 载入 name->taxid: {len(name2taxid)} 条")

    all_pathogens = pd.concat([
        data_5fold["pathogen"].astype(str).str.strip(),
        test_df["pathogen"].astype(str).str.strip(),
    ]).unique()
    pathogen2id = {p: i for i, p in enumerate(all_pathogens)}

    seq_dim = len(feature_cols)
    path_dim = next(iter(pathogen_emb.values())).numel()

    # =========================
    # Grid Search: fold0 作验证，fold1-4 作训练
    # =========================
    log(f"{'='*70}")
    log("GRID SEARCH (Validation: fold0, Training: fold1-4)")
    log(f"{'='*70}")

    train_idx_gs = data_5fold['fold'] != 0
    val_idx_gs = data_5fold['fold'] == 0

    train_df_gs = data_5fold[train_idx_gs].copy()
    val_df_gs = data_5fold[val_idx_gs].copy()

    y_train_raw = train_df_gs["mic"].values
    y_val_raw = val_df_gs["mic"].values

    y_train_log = np.log1p(y_train_raw)
    y_mean_gs, y_std_gs = y_train_log.mean(), y_train_log.std()

    y_train = (y_train_log - y_mean_gs) / y_std_gs
    y_val = (np.log1p(y_val_raw) - y_mean_gs) / y_std_gs

    train_df_gs_copy = train_df_gs.copy()
    val_df_gs_copy = val_df_gs.copy()
    for df in [train_df_gs_copy, val_df_gs_copy]:
        df["pathogen"] = df["pathogen"].astype(str).str.strip()
        df["pid"] = df["pathogen"].map(pathogen2id)

    X_train_gs = build_X(train_df_gs_copy, feature_cols, pathogen_emb, name2taxid, warn_tag="gs-train")
    X_val_gs   = build_X(val_df_gs_copy,   feature_cols, pathogen_emb, name2taxid, warn_tag="gs-val")

    param_grid = {
        "hidden_dim": [1024, 512, 256],
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [128, 64, 32],
    }
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    n_total = int(np.prod([len(v) for v in values]))
    n_combinations = n_total
    
    log(f"\nGrid Search Space: {param_grid}")
    log(f"Total combinations: {n_total}")
    log(f"Combinations to run: {n_combinations}")
    log(f"GS_MAX_EPOCHS={GS_MAX_EPOCHS}, GS_PATIENCE={GS_PATIENCE}")

    best_val_loss = float("inf")
    best_params = None

    all_results = []   

    run_idx = 0
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        run_idx += 1
        set_seed(SEED)   
        log(f"\n[{run_idx}/{n_combinations}] Testing: {params}")

        train_loader = DataLoader(
            AlignmentDataset(X_train_gs, y_train, train_df_gs_copy["pid"].values),
            batch_size=params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            AlignmentDataset(X_val_gs, y_val, val_df_gs_copy["pid"].values),
            batch_size=params["batch_size"], shuffle=False
        )

        model = AlignmentModel(
            seq_dim=seq_dim, path_dim=path_dim,
            hidden_dim=params["hidden_dim"],
            use_align=True, use_cross=True,
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])

        val_loss, _ = train_es(model, optimizer, train_loader, val_loader,
                               GS_MAX_EPOCHS, GS_PATIENCE)
        log(f"  Best Val Loss: {val_loss:.6f}")

        all_results.append({**params, "val_loss": float(val_loss)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()
            log(f"  >>> NEW BEST PARAMS! Val Loss: {val_loss:.6f}")

    log(f"\n{'='*70}")
    log("GRID SEARCH COMPLETE")
    log(f"{'='*70}")
    log(f"Best Params: {best_params}")
    log(f"Best Val Loss (fold0): {best_val_loss:.6f}")

    log("\n--- All Combinations Sorted by Val Loss ---")
    for r in sorted(all_results, key=lambda x: x["val_loss"]):
        log(f"hidden_dim={r['hidden_dim']:>4d}, lr={r['lr']:.0e}, "
            f"batch_size={r['batch_size']:>3d}  ->  Val Loss: {r['val_loss']:.6f}")

    summary = {
        "search_setting": {
            "validation_fold": 0,
            "train_folds": [1, 2, 3, 4],
            "gs_max_epochs": GS_MAX_EPOCHS,
            "gs_patience": GS_PATIENCE,
            "seed": SEED,
        },
        "best_params": best_params,
        "best_val_loss": float(best_val_loss),
        "all_results": all_results,
    }
    out_json = os.path.join(OUT_DIR, "grid_search_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to: {out_json}")

    best_params_path = os.path.join(OUT_DIR, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    log(f"Best params saved to: {best_params_path}")

    log("=" * 70)
    log("GRID SEARCH DONE!")
    log("=" * 70)

    return best_params


if __name__ == "__main__":
    grid_search()
