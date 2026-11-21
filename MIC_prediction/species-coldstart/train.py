# -*- coding: utf-8 -*-
"""
MIC Regression for species-coldstart
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
MAX_EPOCHS = 500
PATIENCE = 30
PATH_EMB_PATH = "../pathogen_embeddings.pt"

COM1_PATH = "./data/com1/"
BASE_DATA_PATH = "./data/com{}"
COMS = [1, 2, 3, 4, 5]

OUT_DIR = "./results"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "complete_pipeline.log")
BEST_MODEL_DIR = os.path.join(OUT_DIR, "best_model_from_grid")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.pt")
BEST_CONFIG_PATH = os.path.join(BEST_MODEL_DIR, "model_config.json")
BEST_NORM_PATH = os.path.join(BEST_MODEL_DIR, "normalization_params.json")
BEST_MAPPING_PATH = os.path.join(BEST_MODEL_DIR, "pathogen_mapping.json")

# =========================
# Logger
# =========================
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {msg}"
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
def build_X(df, feature_cols, pathogen_emb):
    seq_feat = df[feature_cols].values.astype(np.float32)
    path_feat = []
    for p in df["pathogen"]:
        emb = pathogen_emb[p]
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().view(-1).numpy()
        path_feat.append(emb.astype(np.float32))
    return np.concatenate([seq_feat, np.asarray(path_feat)], axis=1)

# =========================
# Early stopping training
# =========================
def train_es(model, optimizer, train_loader, val_loader, max_epochs, patience, verbose=True):
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
            log(f"  Epoch {epoch}/{max_epochs}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    log(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss

# =========================
# Model save/load
# =========================
def save_model_generic(model, pathogen2id, y_mean, y_std, feature_cols,
                       seq_dim, path_dim, hidden_dim, save_dir, prefix=""):
    """Generic model saver with prefix for different runs"""
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
    
    log("=" * 70)
    log("STEP 1: GRID SEARCH ON COM1")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")

    train_df = pd.read_csv(os.path.join(COM1_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(COM1_PATH, "val.csv"))
    test_df = pd.read_csv(os.path.join(COM1_PATH, "test.csv"))
    
    log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    y_train_raw = train_df["mic"].values
    y_val_raw = val_df["mic"].values
    y_test_raw = test_df["mic"].values
    
    y_train_log = np.log1p(y_train_raw)
    y_mean, y_std = y_train_log.mean(), y_train_log.std()
    
    y_train = (y_train_log - y_mean) / y_std
    y_val = (np.log1p(y_val_raw) - y_mean) / y_std
    y_test = (np.log1p(y_test_raw) - y_mean) / y_std

    feature_cols = [c for c in train_df.columns if c.startswith("ESM_")]
    pathogen_emb = torch.load(PATH_EMB_PATH)
    
    all_pathogens = pd.concat([
        train_df["pathogen"], val_df["pathogen"], test_df["pathogen"]
    ]).unique()
    pathogen2id = {p: i for i, p in enumerate(all_pathogens)}
    
    seq_dim = len(feature_cols)
    path_dim = next(iter(pathogen_emb.values())).numel()
    
    for df in [train_df, val_df, test_df]:
        df["pid"] = df["pathogen"].map(pathogen2id)
    
    X_train = build_X(train_df, feature_cols, pathogen_emb)
    X_val = build_X(val_df, feature_cols, pathogen_emb)
    X_test = build_X(test_df, feature_cols, pathogen_emb)

    param_grid = {
        "hidden_dim": [1024, 512, 256],
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [128, 64, 32],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    n_combinations = np.prod([len(v) for v in values])
    
    log(f"\nGrid Search Space:")
    for k, v in param_grid.items():
        log(f"  {k}: {v}")
    log(f"Total combinations: {n_combinations}")
    log("=" * 70)
    
    best_val_loss = float("inf")
    best_params = None
    
    for idx, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        
        log(f"\n[{idx+1}/{n_combinations}] Testing: {params}")
        
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
                           MAX_EPOCHS, PATIENCE, verbose=True)
        
        log(f"  Best Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()

            save_model_generic(
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
            
            log(f"  >>> NEW BEST MODEL SAVED! Val Loss: {val_loss:.6f}")

    grid_summary = {
        "best_params": best_params,
        "best_val_loss": float(best_val_loss),
        "param_grid": {k: [str(v) for v in vals] for k, vals in param_grid.items()},
    }
    with open(os.path.join(OUT_DIR, "grid_search_summary.json"), 'w') as f:
        json.dump(grid_summary, f, indent=2)
    
    log(f"\nGrid Search Complete!")
    log(f"Best Params: {best_params}")
    log(f"Best Val Loss: {best_val_loss:.6f}")
    log("=" * 70)
    
    return best_params

# =========================
# Step 2: Train on all combinations
# =========================
def train_evaluate_com(com_num, best_params):
    """Train and evaluate on a single combination using best params"""
    
    log(f"\n{'='*60}")
    log(f"STEP 2: Training Combination {com_num}")
    log(f"{'='*60}")
    
    set_seed(SEED)

    data_path = BASE_DATA_PATH.format(com_num)
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    
    log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    y_train_raw = train_df["mic"].values
    y_val_raw = val_df["mic"].values
    y_test_raw = test_df["mic"].values
    
    y_train_log = np.log1p(y_train_raw)
    y_mean, y_std = y_train_log.mean(), y_train_log.std()
    
    y_train = (y_train_log - y_mean) / y_std
    y_val = (np.log1p(y_val_raw) - y_mean) / y_std
    y_test = (np.log1p(y_test_raw) - y_mean) / y_std
    
    log(f"Normalization: mean={y_mean:.6f}, std={y_std:.6f}")

    feature_cols = [c for c in train_df.columns if c.startswith("ESM_")]
    pathogen_emb = torch.load(PATH_EMB_PATH)
    
    all_pathogens = pd.concat([
        train_df["pathogen"], val_df["pathogen"], test_df["pathogen"]
    ]).unique()
    pathogen2id = {p: i for i, p in enumerate(all_pathogens)}
    
    seq_dim = len(feature_cols)
    path_dim = next(iter(pathogen_emb.values())).numel()
    
    for df in [train_df, val_df, test_df]:
        df["pid"] = df["pathogen"].map(pathogen2id)
    
    X_train = build_X(train_df, feature_cols, pathogen_emb)
    X_val = build_X(val_df, feature_cols, pathogen_emb)
    X_test = build_X(test_df, feature_cols, pathogen_emb)

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

    log("Training model...")
    val_loss = train_es(model, optimizer, train_loader, val_loader, 
                       MAX_EPOCHS, PATIENCE, verbose=True)
    log(f"Best validation loss: {val_loss:.6f}")

    com_out_dir = os.path.join(OUT_DIR, f"com{com_num}")
    os.makedirs(com_out_dir, exist_ok=True)

    save_model_generic(
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
    log(f"Model saved to: {com_out_dir}")

    model.eval()
    preds_norm = []
    with torch.no_grad():
        for x, y, pid in test_loader:
            x, pid = x.to(DEVICE), pid.to(DEVICE)
            pred = model(x, path_id=pid)
            preds_norm.append(pred.cpu().numpy())
    
    preds_norm = np.concatenate(preds_norm)

    metrics_norm = evaluate(y_test, preds_norm)
    
    log("\nTest Performance (Normalized Space):")
    for k, v in metrics_norm.items():
        log(f"  {k}: {v:.4f}")

    y_test_orig = y_test * y_std + y_mean
    preds_orig = preds_norm * y_std + y_mean
    y_test_mic = np.expm1(y_test_orig)
    preds_mic = np.expm1(preds_orig)
    
    metrics_mic = evaluate(y_test_mic, preds_mic)
    
    log("\nTest Performance (Original MIC Space):")
    for k, v in metrics_mic.items():
        log(f"  {k}: {v:.4f}")

    pathogens = test_df["pathogen"].values
    per_pathogen_records = []
    
    log("\nPer-pathogen Performance (Original MIC Space):")
    for pathogen in np.unique(pathogens):
        mask = pathogens == pathogen
        if mask.sum() < 2:
            log(f"{pathogen:25s} | n={mask.sum():4d} | Skipped (n<2)")
            continue
        
        yt = y_test_mic[mask]
        yp = preds_mic[mask]
        metrics = evaluate(yt, yp)
        per_pathogen_records.append({
            "pathogen": pathogen,
            "n_samples": mask.sum(),
            **metrics
        })
        
        log(f"{pathogen:25s} | n={mask.sum():4d} | "
            f"R2={metrics['R2']:.4f} | PCC={metrics['PCC']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f}")

    macro_avg = None
    if len(per_pathogen_records) > 0:
        macro_avg = {
            "MSE": np.mean([m["MSE"] for m in per_pathogen_records]),
            "RMSE": np.mean([m["RMSE"] for m in per_pathogen_records]),
            "R2": np.mean([m["R2"] for m in per_pathogen_records]),
            "PCC": np.mean([m["PCC"] for m in per_pathogen_records]),
            "Kendall": np.mean([m["Kendall"] for m in per_pathogen_records])
        }
        log("\nPer-pathogen Macro Average (Original MIC):")
        for k, v in macro_avg.items():
            log(f"  {k}: {v:.4f}")

    pd.DataFrame(per_pathogen_records).to_csv(
        os.path.join(com_out_dir, "per_pathogen_metrics_mic.csv"), index=False
    )
    
    pred_df = test_df.copy()
    pred_df["predicted_mic_norm"] = preds_norm
    pred_df["true_mic_norm"] = y_test
    pred_df["predicted_mic"] = preds_mic
    pred_df["true_mic"] = y_test_mic
    pred_df.to_csv(os.path.join(com_out_dir, "test_predictions.csv"), index=False)

    summary = {
        "com": com_num,
        "val_loss": float(val_loss),
        "test_metrics_normalized": {k: float(v) for k, v in metrics_norm.items()},
        "test_metrics_original_mic": {k: float(v) for k, v in metrics_mic.items()},
        "macro_average_metrics": {k: float(v) for k, v in macro_avg.items()} if macro_avg else None,
        "best_params": best_params,
        "random_seed": SEED
    }
    with open(os.path.join(com_out_dir, "summary_metrics.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "com": com_num,
        "val_loss": val_loss,
        "metrics_normalized": metrics_norm,
        "metrics_original": metrics_mic,
        "macro_avg": macro_avg,
        "per_pathogen": pd.DataFrame(per_pathogen_records)
    }

# =========================
# Step 3: Aggregate results
# =========================
def aggregate_results(all_results):
    """Aggregate results from all combinations"""
    
    log(f"\n{'='*60}")
    log("AGGREGATED RESULTS ACROSS COMBINATIONS 1-5")
    log(f"{'='*60}")

    summary_df = pd.DataFrame([
        {
            "com": r["com"],
            "val_loss": r["val_loss"],
            "test_MSE": r["metrics_original"]["MSE"],
            "test_RMSE": r["metrics_original"]["RMSE"],
            "test_R2": r["metrics_original"]["R2"],
            "test_PCC": r["metrics_original"]["PCC"],
            "test_Kendall": r["metrics_original"]["Kendall"]
        }
        for r in all_results
    ])
    
    log("\nPer-combination test performance (original MIC space):")
    log(summary_df.to_string(index=False))
    
    log("\nMean ± Std across combinations (Original MIC Space):")
    for col in ["test_MSE", "test_RMSE", "test_R2", "test_PCC", "test_Kendall"]:
        mean_val = summary_df[col].mean()
        std_val = summary_df[col].std()
        log(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    
    summary_df.to_csv(os.path.join(OUT_DIR, "aggregated_results_all_coms.csv"), index=False)

    all_per_pathogen = [r["per_pathogen"] for r in all_results if not r["per_pathogen"].empty]
    if all_per_pathogen:
        combined = pd.concat(all_per_pathogen, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, "combined_per_pathogen_all_coms.csv"), index=False)

        pathogen_summary = combined.groupby("pathogen").agg({
            "n_samples": "first",
            "MSE": ["mean", "std"],
            "RMSE": ["mean", "std"],
            "R2": ["mean", "std"],
            "PCC": ["mean", "std"],
            "Kendall": ["mean", "std"]
        }).round(4)
        pathogen_summary.to_csv(os.path.join(OUT_DIR, "per_pathogen_summary_across_coms.csv"))

        pathogen_avg_r2 = combined.groupby("pathogen")["R2"].mean().sort_values()
        log(f"\nTop 5 pathogens by average R2:")
        for p, r2 in pathogen_avg_r2.tail(5).items():
            log(f"  {p}: R2={r2:.4f}")
        
        log(f"\nBottom 5 pathogens by average R2:")
        for p, r2 in pathogen_avg_r2.head(5).items():
            log(f"  {p}: R2={r2:.4f}")

    macro_avg_df = pd.DataFrame([
        {
            "com": r["com"],
            "macro_MSE": r["macro_avg"]["MSE"] if r["macro_avg"] else None,
            "macro_RMSE": r["macro_avg"]["RMSE"] if r["macro_avg"] else None,
            "macro_R2": r["macro_avg"]["R2"] if r["macro_avg"] else None,
            "macro_PCC": r["macro_avg"]["PCC"] if r["macro_avg"] else None,
            "macro_Kendall": r["macro_avg"]["Kendall"] if r["macro_avg"] else None
        }
        for r in all_results if r["macro_avg"]
    ])
    
    if not macro_avg_df.empty:
        macro_avg_df.to_csv(os.path.join(OUT_DIR, "macro_averages_all_coms.csv"), index=False)
        log(f"\nMacro-average across combinations:")
        for col in ["macro_MSE", "macro_RMSE", "macro_R2", "macro_PCC", "macro_Kendall"]:
            mean_val = macro_avg_df[col].mean()
            std_val = macro_avg_df[col].std()
            log(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    
    log(f"\n{'='*60}")
    log("All results saved to: {}".format(OUT_DIR))
    log("Files saved:")
    log("  - aggregated_results_all_coms.csv")
    log("  - combined_per_pathogen_all_coms.csv")
    log("  - per_pathogen_summary_across_coms.csv")
    log("  - macro_averages_all_coms.csv")
    log("  - com1/, com2/, ...: Individual combination results")
    log("  - best_model_from_grid/: Best model from grid search")
    log(f"{'='*60}")

# =========================
# Main Pipeline
# =========================
def main():
    """Complete training pipeline"""

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    log("=" * 70)
    log("COMPLETE TRAINING PIPELINE FOR ALIGNMENT MODEL")
    log(f"Fixed random seed: {SEED}")
    log(f"Device: {DEVICE}")
    log("=" * 70)

    best_params = grid_search_com1()

    all_results = []
    for com in COMS:
        try:
            result = train_evaluate_com(com, best_params)
            all_results.append(result)
        except Exception as e:
            log(f"Error processing combination {com}: {str(e)}")
            import traceback
            log(traceback.format_exc())

    aggregate_results(all_results)
    
    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE!")
    log("=" * 70)

if __name__ == "__main__":
    main()
