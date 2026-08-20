# -*- coding: utf-8 -*-
"""
MIC Regression for warmstart
1. Grid search on fold0 to find best hyperparameters
2. Train on fold 0-4 using best parameters
"""
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

from HispecAMP.film_align import AlignmentModel

# =========================
# Config
# =========================
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 500
PATIENCE = 30
PATH_EMB_PATH = "../54_species_embeddings.pt"

OUT_DIR = "./results_5fold_cv"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "5fold_cv.log")
FOLD_MODELS_DIR = os.path.join(OUT_DIR, "fold_models")
os.makedirs(FOLD_MODELS_DIR, exist_ok=True)

INDEPENDENT_TEST_PATH = "./data/test.csv"

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
# Model save/load
# =========================
def save_model(model, pathogen2id, y_mean, y_std, feature_cols,
               seq_dim, path_dim, hidden_dim, model_path, config_path,
               norm_params_path, mapping_path):
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
# Train on specific fold and evaluate on test.csv
# =========================
def train_on_fold_and_evaluate(train_df, val_df, test_df, fold_id, best_params, 
                                feature_cols, pathogen_emb, pathogen2id, seq_dim, path_dim):
    
    log(f"\n{'='*70}")
    log(f"Training on Fold {fold_id}")
    log(f"{'='*70}")
    log(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 标准化
    y_train_raw = train_df["mic"].values
    y_val_raw = val_df["mic"].values
    
    y_train_log = np.log1p(y_train_raw)
    y_mean, y_std = y_train_log.mean(), y_train_log.std()
    
    y_train = (y_train_log - y_mean) / y_std
    y_val = (np.log1p(y_val_raw) - y_mean) / y_std

    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    test_df_copy = test_df.copy()
    
    for df in [train_df_copy, val_df_copy, test_df_copy]:
        df["pid"] = df["pathogen"].map(pathogen2id)
    
    X_train = build_X(train_df_copy, feature_cols, pathogen_emb)
    X_val = build_X(val_df_copy, feature_cols, pathogen_emb)
    X_test = build_X(test_df_copy, feature_cols, pathogen_emb)
    
    y_test_raw = test_df_copy["mic"].values
    y_test = (np.log1p(y_test_raw) - y_mean) / y_std
    
    train_loader = DataLoader(
        AlignmentDataset(X_train, y_train, train_df_copy["pid"].values),
        batch_size=best_params["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        AlignmentDataset(X_val, y_val, val_df_copy["pid"].values),
        batch_size=best_params["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        AlignmentDataset(X_test, y_test, test_df_copy["pid"].values),
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
    
    val_loss, trained_model = train_es(model, optimizer, train_loader, val_loader, 
                                        MAX_EPOCHS, PATIENCE)
    
    log(f"Fold {fold_id} best val loss: {val_loss:.6f}")

    fold_model_dir = os.path.join(FOLD_MODELS_DIR, f"fold_{fold_id}")
    os.makedirs(fold_model_dir, exist_ok=True)
    
    save_model(
        model=trained_model,
        pathogen2id=pathogen2id,
        y_mean=y_mean,
        y_std=y_std,
        feature_cols=feature_cols,
        seq_dim=seq_dim,
        path_dim=path_dim,
        hidden_dim=best_params["hidden_dim"],
        model_path=os.path.join(fold_model_dir, "model.pt"),
        config_path=os.path.join(fold_model_dir, "config.json"),
        norm_params_path=os.path.join(fold_model_dir, "norm_params.json"),
        mapping_path=os.path.join(fold_model_dir, "mapping.json")
    )

    trained_model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y, pid in test_loader:
            pred = trained_model(x.to(DEVICE), path_id=pid.to(DEVICE))
            preds.append(pred.cpu().numpy())
            gts.append(y.numpy())
    
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    metrics_norm = evaluate(gts, preds)

    gts_log = gts * y_std + y_mean
    preds_log = preds * y_std + y_mean
    gts_mic = np.expm1(gts_log)
    preds_mic = np.expm1(preds_log)

    results_df = test_df_copy.copy()
    results_df['predicted_mic'] = preds_mic
    results_df['true_mic'] = gts_mic
    results_df['predicted_mic_norm'] = preds
    results_df['true_mic_norm'] = gts
    results_df.to_csv(os.path.join(fold_model_dir, "test_predictions.csv"), index=False)
    
    log(f"Fold {fold_id} - MSE={metrics_norm['MSE']:.4f}, RMSE={metrics_norm['RMSE']:.4f}, R2={metrics_norm['R2']:.4f}, PCC={metrics_norm['PCC']:.4f}, Kendall={metrics_norm['Kendall']:.4f}")
    
    return {
        "fold": fold_id,
        "val_loss": val_loss,
        "test_metrics_norm": metrics_norm,
        "predictions_norm": preds,
        "targets_norm": gts,
        "predictions": preds_mic,
        "targets": gts_mic,
        "pathogens": test_df["pathogen"].values,
        "y_mean": y_mean,
        "y_std": y_std
    }

# =========================
# Main 5-fold CV function
# =========================
def five_fold_cv():
    set_seed(SEED)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE) 
    
    # =========================
    # Load data
    # =========================
    data_5fold = pd.read_csv('data/data_5fold.csv')  
    test_df = pd.read_csv(INDEPENDENT_TEST_PATH)    
    feature_cols = [c for c in data_5fold.columns if c.startswith("ESM_")]

    pathogen_emb = torch.load(PATH_EMB_PATH)
    all_pathogens = pd.concat([
        data_5fold["pathogen"], test_df["pathogen"]
    ]).unique()
    pathogen2id = {p: i for i, p in enumerate(all_pathogens)}
    
    seq_dim = len(feature_cols)
    path_dim = next(iter(pathogen_emb.values())).numel()
    
    # =========================
    # Grid Search
    # =========================
    log(f"{'='*70}")
    log("STEP 1: GRID SEARCH (Validation: fold0, Training: fold1-4)")
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
        df["pid"] = df["pathogen"].map(pathogen2id)
    
    X_train_gs = build_X(train_df_gs_copy, feature_cols, pathogen_emb)
    X_val_gs = build_X(val_df_gs_copy, feature_cols, pathogen_emb)

    param_grid = {
        "hidden_dim": [1024, 512, 256],
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [128, 64, 32],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    n_combinations = np.prod([len(v) for v in values])
    
    log(f"\nGrid Search Space: {param_grid}")
    log(f"Total combinations: {n_combinations}")

    best_val_loss = float("inf")
    best_params = None
    
    for idx, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        
        log(f"\n[{idx+1}/{n_combinations}] Testing: {params}")
        
        train_loader = DataLoader(
            AlignmentDataset(X_train_gs, y_train, train_df_gs_copy["pid"].values),
            batch_size=params["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            AlignmentDataset(X_val_gs, y_val, val_df_gs_copy["pid"].values),
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
        
        val_loss, _ = train_es(model, optimizer, train_loader, val_loader, 
                                MAX_EPOCHS, PATIENCE)
        
        log(f"  Best Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params.copy()
            log(f"  >>> NEW BEST PARAMS! Val Loss: {val_loss:.6f}")
    
    log(f"\n{'='*70}")
    log(f"GRID SEARCH COMPLETE")
    log(f"Best Params: {best_params}")
    log(f"Best Val Loss: {best_val_loss:.6f}")
    
    # =========================
    # 5-fold cross-validation
    # =========================
    log(f"\n{'='*70}")
    log("STEP 2: 5-FOLD CV WITH BEST PARAMETERS")
    log(f"{'='*70}")

    all_fold_results = []
    
    for fold_id in range(5):
        val_idx = data_5fold['fold'] == fold_id
        train_idx = data_5fold['fold'] != fold_id
        
        train_df_fold = data_5fold[train_idx].copy()
        val_df_fold = data_5fold[val_idx].copy()
        result = train_on_fold_and_evaluate(
            train_df_fold, val_df_fold, test_df, fold_id, 
            best_params, feature_cols, pathogen_emb, pathogen2id, seq_dim, path_dim
        )
        all_fold_results.append(result)
    
    # =========================
    # Results
    # =========================
    log(f"\n{'='*70}")
    log("FINAL RESULTS - 5-FOLD CROSS VALIDATION ON test.csv (Normalized Space)")
    log(f"{'='*70}")

    norm_metrics = {metric: [] for metric in ['MSE', 'RMSE', 'R2', 'PCC', 'Kendall']}
    
    for result in all_fold_results:
        for metric in norm_metrics:
            norm_metrics[metric].append(result['test_metrics_norm'][metric])
    
    log("\n--- Normalized Space Metrics on test.csv ---")
    for metric in norm_metrics:
        mean_val = np.mean(norm_metrics[metric])
        std_val = np.std(norm_metrics[metric])
        log(f"{metric:10s}: {mean_val:.4f} ± {std_val:.4f}")

    log("\n--- Per-fold Results on test.csv ---")
    for result in all_fold_results:
        fold = result['fold']
        metrics = result['test_metrics_norm']
        log(f"Fold {fold}: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}, PCC={metrics['PCC']:.4f}, Kendall={metrics['Kendall']:.4f}")

    log("\n--- Per-pathogen Statistics Across 5 Folds ---")

    pathogen_fold_metrics = {}
    
    for result in all_fold_results:
        pathogens = result['pathogens']
        predictions_norm = result['predictions_norm']
        targets_norm = result['targets_norm']

        fold_pathogen_metrics = {}
        for i, pathogen in enumerate(pathogens):
            if pathogen not in fold_pathogen_metrics:
                fold_pathogen_metrics[pathogen] = {'targets': [], 'predictions': []}
            fold_pathogen_metrics[pathogen]['targets'].append(targets_norm[i])
            fold_pathogen_metrics[pathogen]['predictions'].append(predictions_norm[i])

        for pathogen, data in fold_pathogen_metrics.items():
            if len(data['targets']) >= 2:
                metrics = evaluate(np.array(data['targets']), np.array(data['predictions']))
                if pathogen not in pathogen_fold_metrics:
                    pathogen_fold_metrics[pathogen] = []
                pathogen_fold_metrics[pathogen].append(metrics)

    for pathogen, metrics_list in pathogen_fold_metrics.items():
        if len(metrics_list) == 5:
            r2_list = [m['R2'] for m in metrics_list]
            pcc_list = [m['PCC'] for m in metrics_list]
            rmse_list = [m['RMSE'] for m in metrics_list]
            
            log(f"{pathogen:25s} | MSE: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f} | "
                f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}"
                f"R2: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f} | "
                f"PCC: {np.mean(pcc_list):.4f} ± {np.std(pcc_list):.4f} | "
                f"Kendall: {np.mean([m['Kendall'] for m in metrics_list]):.4f} ± {np.std([m['Kendall'] for m in metrics_list]):.4f}")
            
    log("\n--- Ensemble Prediction Statistics ---")
    all_predictions_norm = np.array([r['predictions_norm'] for r in all_fold_results])
    pred_mean_norm = np.mean(all_predictions_norm, axis=0)

    targets_norm = all_fold_results[0]['targets_norm']
    ensemble_metrics_norm = evaluate(targets_norm, pred_mean_norm)
    log(f"Ensemble (mean): R2={ensemble_metrics_norm['R2']:.4f}, PCC={ensemble_metrics_norm['PCC']:.4f}, RMSE={ensemble_metrics_norm['RMSE']:.4f}")

    ensemble_results = test_df.copy()
    all_predictions_mic = np.array([r['predictions'] for r in all_fold_results])
    ensemble_results['predicted_mic_mean'] = np.mean(all_predictions_mic, axis=0)
    ensemble_results['predicted_mic_std'] = np.std(all_predictions_mic, axis=0)
    for i, result in enumerate(all_fold_results):
        ensemble_results[f'fold_{i}_predicted_mic'] = result['predictions']
    ensemble_results.to_csv(os.path.join(OUT_DIR, "ensemble_predictions.csv"), index=False)
    
    # =========================
    # Save results
    # =========================
    summary = {
        "best_params": best_params,
        "best_val_loss_in_grid_search": float(best_val_loss),
        "cv_results_on_test_csv": {
            "normalized_space": {metric: {"mean": float(np.mean(norm_metrics[metric])), 
                                          "std": float(np.std(norm_metrics[metric]))} 
                                for metric in norm_metrics},
            "per_fold": [
                {
                    "fold": r["fold"],
                    "val_loss": r["val_loss"],
                    "test_metrics_norm": r["test_metrics_norm"]
                }
                for r in all_fold_results
            ]
        },
        "ensemble_metrics_norm": ensemble_metrics_norm
    }
    
    with open(os.path.join(OUT_DIR, "5fold_cv_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"\nResults saved to: {os.path.join(OUT_DIR, '5fold_cv_summary.json')}")
    log("=" * 70)
    log("5-FOLD CROSS VALIDATION COMPLETE!")
    log("=" * 70)

if __name__ == "__main__":
    five_fold_cv()