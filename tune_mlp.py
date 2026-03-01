"""
Optuna hyperparameter tuning for MLP baseline on DBP_dataset_DWTP_B.csv

Key design choices:
  - Search space includes 0 hidden layers (linear baseline)
  - lr and weight_decay use log-uniform sampling
  - StandardScaler is fit independently inside each CV fold
  - 5-fold CV mean MSE as objective
  - Final retrain on all 141 train samples with best params
  
FIX: Runs a separate Optuna study for each target to avoid MSE scaling
     issues and underfitting on smaller targets.
"""

import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
N_TRIALS = 200
N_FOLDS = 5
MAX_EPOCHS = 2000
PATIENCE = 80

FEATURE_COLS = [
    "pH", "COD_mg_L", "NH4_N_mg_L", "NO2_N_mg_L", "NO3_N_mg_L",
    "Br_mg_L", "TOC_mg_L", "UV254_A_cm", "temp_C",
]
TARGET_COLS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("DBP_dataset_DWTP_B.csv")
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)

X_train_all = train_df[FEATURE_COLS].values.astype(np.float32)
X_test_raw  = test_df[FEATURE_COLS].values.astype(np.float32)

print(f"Train: {len(train_df)},  Test: {len(test_df)}")
print(f"Features: {len(FEATURE_COLS)},  Targets: {len(TARGET_COLS)}")
print(f"Optuna trials: {N_TRIALS},  CV folds: {N_FOLDS}\n")


# ── Model builder ────────────────────────────────────────────────────────────
def build_model(in_dim, out_dim, n_layers, hidden_dim, dropout, activation_name):
    if n_layers == 0:
        return nn.Linear(in_dim, out_dim)

    act_fn = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU,
              "SiLU": nn.SiLU, "Tanh": nn.Tanh}[activation_name]
    layers = []
    prev = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden_dim), act_fn(), nn.Dropout(dropout)]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── Single fold training ─────────────────────────────────────────────────────
def train_one_fold(model, X_tr, Y_tr, X_va, Y_va, lr, weight_decay, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_tr, Y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), Y_va).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    return best_val, best_epoch


# We will store models, scalers, and predictions for each target
models = {}
scalers_x = {}
scalers_y = {}
best_params_all = {}
best_cv_mse_all = {}
best_epochs_all = {}

for target_idx, target_name in enumerate(TARGET_COLS):
    print("\n" + "=" * 60)
    print(f"Tuning model for: {target_name}")
    print("=" * 60)
    
    # Target-specific data
    Y_train_target = train_df[[target_name]].values.astype(np.float32)

    # ── Objective ────────────────────────────────────────────────────────────
    def objective(trial):
        n_layers   = trial.suggest_int("n_layers", 0, 3)
        hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]) if n_layers > 0 else 0
        dropout    = trial.suggest_float("dropout", 0.0, 0.5, step=0.1) if n_layers > 0 else 0.0
        activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "SiLU", "Tanh"]) if n_layers > 0 else "ReLU"
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_losses = []

        for train_idx, val_idx in kf.split(X_train_all):
            scaler_x = StandardScaler().fit(X_train_all[train_idx])
            scaler_y = StandardScaler().fit(Y_train_target[train_idx])

            X_tr = torch.tensor(scaler_x.transform(X_train_all[train_idx]), dtype=torch.float32)
            Y_tr = torch.tensor(scaler_y.transform(Y_train_target[train_idx]), dtype=torch.float32)
            X_va = torch.tensor(scaler_x.transform(X_train_all[val_idx]),   dtype=torch.float32)
            Y_va = torch.tensor(scaler_y.transform(Y_train_target[val_idx]),   dtype=torch.float32)

            torch.manual_seed(SEED + target_idx)
            model = build_model(len(FEATURE_COLS), 1,
                                n_layers, hidden_dim, dropout, activation)
            val_mse, _ = train_one_fold(model, X_tr, Y_tr, X_va, Y_va, lr, wd, batch_size)
            fold_losses.append(val_mse)

        return np.mean(fold_losses)

    # ── Run Optuna ───────────────────────────────────────────────────────────
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED + target_idx))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\nBest trial")
    best = study.best_trial
    print(f"  CV MSE: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    
    best_cv_mse_all[target_name] = best.value
    best_params_all[target_name] = best.params

    # ── Final retrain with best params on all train data ─────────────────────
    print("\nFinal retrain on full training set ...")
    bp = best.params
    n_layers   = bp["n_layers"]
    hidden_dim = bp.get("hidden_dim", 0)
    dropout    = bp.get("dropout", 0.0)
    activation = bp.get("activation", "ReLU")

    # Step 1: Re-run CV with best params to collect best epoch per fold
    print("  Determining optimal epoch count from CV ...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_best_epochs = []
    for train_idx, val_idx in kf.split(X_train_all):
        scaler_x_f = StandardScaler().fit(X_train_all[train_idx])
        scaler_y_f = StandardScaler().fit(Y_train_target[train_idx])
        X_tr_f = torch.tensor(scaler_x_f.transform(X_train_all[train_idx]), dtype=torch.float32)
        Y_tr_f = torch.tensor(scaler_y_f.transform(Y_train_target[train_idx]), dtype=torch.float32)
        X_va_f = torch.tensor(scaler_x_f.transform(X_train_all[val_idx]),   dtype=torch.float32)
        Y_va_f = torch.tensor(scaler_y_f.transform(Y_train_target[val_idx]),   dtype=torch.float32)
        torch.manual_seed(SEED + target_idx)
        m = build_model(len(FEATURE_COLS), 1, n_layers, hidden_dim, dropout, activation)
        _, best_ep = train_one_fold(m, X_tr_f, Y_tr_f, X_va_f, Y_va_f, bp["lr"], bp["weight_decay"], bp["batch_size"])
        fold_best_epochs.append(best_ep)

    fixed_epochs = int(np.median(fold_best_epochs))
    print(f"  Fold best epochs: {fold_best_epochs}  →  median = {fixed_epochs}")
    best_epochs_all[target_name] = fixed_epochs

    # Step 2: Train on ALL training data for fixed_epochs
    scaler_x = StandardScaler().fit(X_train_all)
    scaler_y = StandardScaler().fit(Y_train_target)

    X_tr = torch.tensor(scaler_x.transform(X_train_all), dtype=torch.float32)
    Y_tr = torch.tensor(scaler_y.transform(Y_train_target), dtype=torch.float32)

    torch.manual_seed(SEED + target_idx)
    final_model = build_model(len(FEATURE_COLS), 1,
                              n_layers, hidden_dim, dropout, activation)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_tr, Y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=bp["batch_size"], shuffle=True)

    for epoch in range(1, fixed_epochs + 1):
        final_model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(final_model(xb), yb).backward()
            optimizer.step()

    print(f"  Trained for {fixed_epochs} epochs on all {len(X_tr)} training samples")
    
    final_model.eval()
    models[target_name] = final_model
    scalers_x[target_name] = scaler_x
    scalers_y[target_name] = scaler_y


# ── Global Final Evaluation ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final evaluation on test set (original scale)")
print("=" * 60)

for target_name in TARGET_COLS:
    model = models[target_name]
    scaler_x = scalers_x[target_name]
    scaler_y = scalers_y[target_name]

    model.eval()
    X_te = torch.tensor(scaler_x.transform(X_test_raw), dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(X_te).numpy()
    
    pred_raw = scaler_y.inverse_transform(pred_scaled)
    y_pred = pred_raw[:, 0]
    
    y_true = test_df[target_name].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {target_name:15s}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
torch.save({
    "model_states": {name: m.state_dict() for name, m in models.items()},
    "scalers_x": scalers_x,
    "scalers_y": scalers_y,
    "best_params": best_params_all,
    "cv_mse": best_cv_mse_all,
    "fixed_epochs": best_epochs_all,
}, "mlp_tuned_checkpoint_separate.pt")
print("\nModels saved to mlp_tuned_checkpoint_separate.pt")
