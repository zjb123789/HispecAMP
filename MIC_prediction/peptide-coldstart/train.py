# -*- coding: utf-8 -*-
"""
MIC Regression for peptide-coldstart
1. Grid search on com1 to find best hyperparameters
2. Train on combinations 1-5 using best parameters
"""

import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import json
import itertools
import datetime

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
MAX_EPOCHS = 1
PATIENCE = 30
PATH_EMB_PATH = "../pathogen_embeddings.pt"

TAXONOMY_JSON_PATH = "../pathogen_taxonomy_by_taxid.json"

BASE_DATA_PATH = "./data/com_{}"
COMS = [1, 2, 3, 4, 5]

OUT_DIR = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_OUT_DIR = os.path.join(OUT_DIR, "grid_search")
os.makedirs(GRID_OUT_DIR, exist_ok=True)
GRID_LOG_FILE = os.path.join(GRID_OUT_DIR, "grid_search.log")

BEST_MODEL_DIR = os.path.join(GRID_OUT_DIR, "best_model")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.pt")
BEST_CONFIG_PATH = os.path.join(BEST_MODEL_DIR, "model_config.json")
BEST_NORM_PATH = os.path.join(BEST_MODEL_DIR, "normalization_params.json")
BEST_MAPPING_PATH = os.path.join(BEST_MODEL_DIR, "pathogen_mapping.json")

TRAIN_OUT_DIR = os.path.join(OUT_DIR, "train_all_coms")
os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
TRAIN_LOG_FILE = os.path.join(TRAIN_OUT_DIR, "training_all_s.log")

# =========================
# Logger
# =========================
def get_logger(log_file):
    def log(msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {msg}"
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return log

grid_log = get_logger(GRID_LOG_FILE)
train_log = get_logger(TRAIN_LOG_FILE)

# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "PCC": pearsonr(y_true, y_pred)[0],
        "Kendall": kendalltau(y_true, y_pred)[0],
    }

def load_name2taxid(json_path):
    """从 taxonomy JSON 构建 scientific_name -> taxid(str) 映射。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容 {"pathogen_to_taxonomy": {...}} 或直接 {...} 两种结构
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
# Feature builder (Modified)
# =========================
def build_X(df, feature_cols, pathogen_emb, name2taxid, warn_tag="", log_func=None):
    if log_func is None:
        log_func = lambda x: None

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
        log_func(f"  [Warn{(' ' + warn_tag) if warn_tag else ''}] "
                 f"{len(missing_map)} 个 pathogen 名字未在 taxonomy JSON 中找到（零向量兜底）: "
                 f"{sorted(missing_map)[:10]}{' ...' if len(missing_map) > 10 else ''}")
    if missing_emb:
        log_func(f"  [Warn{(' ' + warn_tag) if warn_tag else ''}] "
                 f"{len(missing_emb)} 个 taxid 未在 {os.path.basename(PATH_EMB_PATH)} 中找到（零向量兜底）: "
                 f"{sorted(list(missing_emb))[:10]}{' ...' if len(missing_emb) > 10 else ''}")

    return np.concatenate([seq_feat, np.asarray(path_feat)], axis=1)

# =========================
# Early stopping training
# =========================
def train_es(model, optimizer, train_loader, val_loader, max_epochs, patience, verbose=True, log_func=None):
    if log_func is None:
        log_func = lambda x: None
    
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
        
        if verbose and (epoch % 20 == 0 or epoch == 1):
            log_func(f"  Epoch {epoch}/{max_epochs}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    log_func(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss

# =========================
# Model save/load
# =========================
def save_model(model, pathogen2id, y_mean, y_std, feature_cols,
               seq_dim, path_dim, hidden_dim, save_dir, prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f"{prefix}best_model.pt" if prefix else "best_model.pt")
    config_path = os.path.join(save_dir, f"{prefix}model_config.json" if prefix else "model_config.json")
    norm_params_path = os.path.join(save_dir, f"{prefix}normalization_params.json" if prefix else "normalization_params.json")
    mapping_path = os.path.join(save_dir, f"{prefix}pathogen_mapping.json" if prefix else "pathogen_mapping.json")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_dim': seq_dim,
        'path_dim': path_dim,
        'hidden_dim': hidden_dim,
    }, model_path)

    model_config = {
        'seq_dim': seq_dim,
        'path_dim': path_dim,
        'hidden_dim': hidden_dim,
        'use_align': True,
        'use_cross': True,
        'num_pathogens': len(pathogen2id),
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)

    norm_params = {
        'y_mean': float(y_mean),
        'y_std': float(y_std),
    }
    with open(norm_params_path, 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, indent=2)

    pathogen_mapping = {k: int(v) for k, v in pathogen2id.items()}
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(pathogen_mapping, f, indent=2)
    
    return model_path

def load_model(model_path, config_path, device=DEVICE):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model = AlignmentModel(
        seq_dim=config['seq_dim'],
        path_dim=config['path_dim'],
        hidden_dim=config['hidden_dim'],
        use_align=config.get('use_align', True),
        use_cross=config.get('use_cross', True),
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

# =========================
# Step 1: Grid Search on com1
# =========================
def grid_search_com1():
    """Perform grid search on com1 to find best hyperparameters"""

    if os.path.exists(GRID_LOG_FILE):
        os.remove(GRID_LOG_FILE)
    
    grid_log("=" * 70)
    grid_log("STEP 1: GRID SEARCH ON COM1")
    grid_log("=" * 70)
    grid_log(f"Device: {DEVICE}")
    grid_log(f"Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")

    train_df = pd.read_csv(BASE_DATA_PATH.format(1) + "/train.csv")
    val_df = pd.read_csv(BASE_DATA_PATH.format(1) + "/val.csv")
    test_df = pd.read_csv(BASE_DATA_PATH.format(1) + "/test.csv")
    
    grid_log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    y_train_raw = train_df["mic"].values
    y_val_raw = val_df["mic"].values
    y_test_raw = test_df["mic"].values
    
    y_train_log = np.log1p(y_train_raw)
    y_mean, y_std = y_train_log.mean(), y_train_log.std()
    
    y_train = (y_train_log - y_mean) / y_std
    y_val = (np.log1p(y_val_raw) - y_mean) / y_std
    y_test = (np.log1p(y_test_raw) - y_mean) / y_std

    feature_cols = [c for c in train_df.columns if c.startswith("ESM_")]
    
    pathogen_emb = torch.load(PATH_EMB_PATH, map_location="cpu")
    name2taxid = load_name2taxid(TAXONOMY_JSON_PATH)
    grid_log(f"[Taxonomy] 载入 name->taxid: {len(name2taxid)} 条")
    
    all_pathogens = pd.concat([
        train_df["pathogen"], val_df["pathogen"], test_df["pathogen"]
    ]).unique()
    pathogen2id = {p: i for i, p in enumerate(all_pathogens)}
    
    seq_dim = len(feature_cols)
    path_dim = next(iter(pathogen_emb.values())).numel()
    
    for df in [train_df, val_df, test_df]:
        df["pathogen"] = df["pathogen"].astype(str).str.strip()
        df["pid"] = df["pathogen"].map(pathogen2id)
    
    X_train = build_X(train_df, feature_cols, pathogen_emb, name2taxid, warn_tag="com1-train", log_func=grid_log)
    X_val = build_X(val_df, feature_cols, pathogen_emb, name2taxid, warn_tag="com1-val", log_func=grid_log)
    X_test = build_X(test_df, feature_cols, pathogen_emb, name2taxid, warn_tag="com1-test", log_func=grid_log)

    param_grid = {
        "hidden_dim": [1024, 512, 256],
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [128, 64, 32],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    n_combinations = np.prod([len(v) for v in values])
    
    grid_log(f"\nGrid Search Space:")
    for k, v in param_grid.items():
        grid_log(f"  {k}: {v}")
    grid_log(f"Total combinations: {n_combinations}")
    grid_log("=" * 70)

    best_val_loss = float("inf")
    best_params = None
    
    for idx, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        
        grid_log(f"\n[{idx+1}/{n_combinations}] Testing: {params}")
        
        train_loader = DataLoader(
            AlignmentDataset(X_train, y_train, train_df["pid"].values),
            batch_size=params["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            AlignmentDataset(X_val, y_val, val_df["pid"].values),
            batch_size=params["batch_size"],
            shuffle=False
        )
        
        model = AlignmentModel(
            seq_dim=seq_dim,
            path_dim=path_dim,
            hidden_dim=params["hidden_dim"],
            use_align=True,
            use_cross=True,
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        
        val_loss = train_es(model, optimizer, train_loader, val_loader, 
                           MAX_EPOCHS, PATIENCE, verbose=True, log_func=grid_log)
        
        grid_log(f"  Best Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()

            save_model(
                model=model,
                pathogen2id=pathogen2id,
                y_mean=y_mean,
                y_std=y_std,
                feature_cols=feature_cols,
                seq_dim=seq_dim,
                path_dim=path_dim,
                hidden_dim=params["hidden_dim"],
                save_dir=BEST_MODEL_DIR,
                prefix=""
            )
            
            grid_log(f"  >>> NEW BEST MODEL SAVED! Val Loss: {val_loss:.6f}")

    grid_summary = {
        "best_params": best_params,
        "best_val_loss": float(best_val_loss),
        "param_grid": {k: [str(v) for v in vals] for k, vals in param_grid.items()},
    }
    with open(os.path.join(GRID_OUT_DIR, "grid_search_summary.json"), 'w') as f:
        json.dump(grid_summary, f, indent=2)
    
    grid_log(f"\nGrid Search Complete!")
    grid_log(f"Best Params: {best_params}")
    grid_log(f"Best Val Loss: {best_val_loss:.6f}")
    grid_log("=" * 70)
    
    return best_params

# =========================
# Step 2: Train on all combinations
# =========================
def train_evaluate_com(com_num, best_params, log_func):
    """Train and evaluate on a single combination using best params"""
    
    log_func(f"\n{'='*60}")
    log_func(f"STEP 2: Training Combination {com_num}")
    log_func(f"{'='*60}")
    
    set_seed(SEED)

    data_path = BASE_DATA_PATH.format(com_num)
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    
    log_func(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

    y_train_raw = train_df["mic"].values
    y_val_raw = val_df["mic"].values
    y_test_raw = test_df["mic"].values
    
    y_train_log = np.log1p(y_train_raw)
    y_mean, y_std = y_train_log.mean(), y_train_log.std()
    
    y_train = (y_train_log - y_mean) / y_std
    y_val = (np.log1p(y_val_raw) - y_mean) / y_std
    y_test = (np.log1p(y_test_raw) - y_mean) / y_std
    
    log_func(f"Normalization: mean={y_mean:.6f}, std={y_std:.6f}")

    feature_cols = [c for c in train_df.columns if c.startswith("ESM_")]
    
    pathogen_emb = torch.load(PATH_EMB_PATH, map_location="cpu")
    name2taxid = load_name2taxid(TAXONOMY_JSON_PATH)
    log_func(f"[Taxonomy] 载入 name->taxid: {len(name2taxid)} 条")
    
    all_pathogens = pd.concat([
        train_df["pathogen"], val_df["pathogen"], test_df["pathogen"]
    ]).unique()
    pathogen2id = {p: i for i, p in enumerate(all_pathogens)}
    
    seq_dim = len(feature_cols)
    path_dim = next(iter(pathogen_emb.values())).numel()
    
    for df in [train_df, val_df, test_df]:
        df["pathogen"] = df["pathogen"].astype(str).str.strip()
        df["pid"] = df["pathogen"].map(pathogen2id)
    
    X_train = build_X(train_df, feature_cols, pathogen_emb, name2taxid, warn_tag=f"com{com_num}-train", log_func=log_func)
    X_val = build_X(val_df, feature_cols, pathogen_emb, name2taxid, warn_tag=f"com{com_num}-val", log_func=log_func)
    X_test = build_X(test_df, feature_cols, pathogen_emb, name2taxid, warn_tag=f"com{com_num}-test", log_func=log_func)
    
    log_func(f"X_train shape: {X_train.shape}")
    log_func(f"X_val shape: {X_val.shape}")
    log_func(f"X_test shape: {X_test.shape}")

    train_loader = DataLoader(
        AlignmentDataset(X_train, y_train, train_df["pid"].values),
        batch_size=best_params["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        AlignmentDataset(X_val, y_val, val_df["pid"].values),
        batch_size=best_params["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        AlignmentDataset(X_test, y_test, test_df["pid"].values),
        batch_size=best_params["batch_size"],
        shuffle=False
    )

    model = AlignmentModel(
        seq_dim=seq_dim,
        path_dim=path_dim,
        hidden_dim=best_params["hidden_dim"],
        use_align=True,
        use_cross=True,
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

    log_func("Training Alignment Model...")
    val_loss = train_es(model, optimizer, train_loader, val_loader, 
                       MAX_EPOCHS, PATIENCE, verbose=True, log_func=log_func)
    log_func(f"Best validation loss: {val_loss:.6f}")

    com_out_dir = os.path.join(TRAIN_OUT_DIR, f"com{com_num}")
    os.makedirs(com_out_dir, exist_ok=True)

    model_path = save_model(
        model=model,
        pathogen2id=pathogen2id,
        y_mean=y_mean,
        y_std=y_std,
        feature_cols=feature_cols,
        seq_dim=seq_dim,
        path_dim=path_dim,
        hidden_dim=best_params["hidden_dim"],
        save_dir=com_out_dir,
        prefix="alignment_"
    )
    log_func(f"Model saved to: {model_path}")

    model.eval()
    preds_norm = []
    with torch.no_grad():
        for x, y, pid in test_loader:
            x, pid = x.to(DEVICE), pid.to(DEVICE)
            pred = model(x, path_id=pid)
            preds_norm.append(pred.cpu().numpy())
    
    preds_norm = np.concatenate(preds_norm)
    assert len(preds_norm) == len(test_df), f"Predictions count mismatch"

    sequence_col = None
    for col in ['sequence']:
        if col in test_df.columns:
            sequence_col = col
            break

    strain_col = None
    for col in ['pathogen']:
        if col in test_df.columns:
            strain_col = col
            break
    
    if strain_col is None:
        if 'id' in test_df.columns:
            log_func("Warning: No strain column found, using 'id' column as strain identifier")
            strain_col = 'id'
        else:
            log_func("Warning: No strain or id column found, using index")
            test_df = test_df.copy()
            test_df['strain'] = [f'sample_{i}' for i in range(len(test_df))]
            strain_col = 'strain'

    predictions_df = pd.DataFrame({
        'sequence': test_df[sequence_col].values if sequence_col else [''] * len(test_df),
        'strain': test_df[strain_col].values,
        'mic_true': y_test,
        'mic_predict': preds_norm
    })
    
    predictions_path = os.path.join(com_out_dir, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    log_func(f"Test predictions saved to: {predictions_path}")
    log_func(f"Predictions are in normalized space (mean={y_mean:.4f}, std={y_std:.4f})")

    metrics_norm = evaluate(y_test, preds_norm)
    
    log_func("\nTest Performance (Normalized Space):")
    for k, v in metrics_norm.items():
        log_func(f"  {k}: {v:.4f}")

    summary = {
        "com": com_num,
        "val_loss": float(val_loss),
        "test_metrics_normalized": {k: float(v) for k, v in metrics_norm.items()},
        "best_params": best_params,
        "random_seed": SEED,
        "test_samples": len(test_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "normalization": {
            "mean": float(y_mean),
            "std": float(y_std)
        }
    }
    
    with open(os.path.join(com_out_dir, "summary_metrics.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "com": com_num,
        "val_loss": val_loss,
        "metrics_normalized": metrics_norm,
        "predictions_file": predictions_path,
        "summary": summary
    }

# =========================
# Step 3: Aggregate results
# =========================
def aggregate_results(all_results, log_func):
    """Aggregate results from all combinations"""
    
    log_func(f"\n{'='*60}")
    log_func("AGGREGATED RESULTS ACROSS COMBINATIONS 1-5 (Normalized Space)")
    log_func(f"{'='*60}")

    summary_df = pd.DataFrame([
        {
            "com": r["com"],
            "val_loss": r["val_loss"],
            "MSE": r["metrics_normalized"]["MSE"],
            "RMSE": r["metrics_normalized"]["RMSE"],
            "R2": r["metrics_normalized"]["R2"],
            "PCC": r["metrics_normalized"]["PCC"],
            "Kendall": r["metrics_normalized"]["Kendall"]
        }
        for r in all_results
    ])
    
    log_func("\nPer-combination test performance (Normalized Space):")
    log_func(summary_df.to_string(index=False))
    
    log_func("\nMean ± Std across combinations (Normalized Space):")
    for col in ["MSE", "RMSE", "R2", "PCC", "Kendall"]:
        mean_val = summary_df[col].mean()
        std_val = summary_df[col].std()
        log_func(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")

    summary_df.to_csv(os.path.join(TRAIN_OUT_DIR, "aggregated_results_all_coms.csv"), index=False)
    
    log_func(f"\n{'='*60}")
    log_func(f"All results saved to: {TRAIN_OUT_DIR}")
    log_func("Files saved:")
    log_func(f"  - aggregated_results_all_coms.csv: Summary across all combinations")
    log_func(f"  - com1/, com2/, ...: Individual combination results")
    log_func(f"    - test_predictions.csv: Predictions with sequence, strain, mic_true (normalized), mic_predict (normalized)")
    log_func(f"    - summary_metrics.json: Performance metrics and normalization parameters")
    log_func(f"    - alignment_best_model.pt: Best model checkpoint")
    log_func(f"{'='*60}")

# =========================
# Main Pipeline
# =========================
def main():
    """Complete training pipeline"""

    if os.path.exists(GRID_LOG_FILE):
        os.remove(GRID_LOG_FILE)
    if os.path.exists(TRAIN_LOG_FILE):
        os.remove(TRAIN_LOG_FILE)
    
    grid_log("=" * 70)
    grid_log("COMPLETE TRAINING PIPELINE FOR ALIGNMENT MODEL (Peptide-Coldstart)")
    grid_log(f"Fixed random seed: {SEED}")
    grid_log(f"Device: {DEVICE}")
    grid_log("=" * 70)

    best_params = grid_search_com1()

    all_results = []
    for com in COMS:
        try:
            result = train_evaluate_com(com, best_params, train_log)
            all_results.append(result)
        except Exception as e:
            train_log(f"Error processing combination {com}: {str(e)}")
            import traceback
            train_log(traceback.format_exc())

    aggregate_results(all_results, train_log)

    grid_log("\n" + "=" * 70)
    grid_log("PIPELINE COMPLETE!")
    grid_log(f"Best Parameters from Grid Search: {best_params}")
    grid_log("=" * 70)

if __name__ == "__main__":
    main()
