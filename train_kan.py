"""
KAN baseline for DBP prediction on DBP_dataset_DWTP_B.csv.
Predicts: T_THMs, DBCM, BDCM from 9 water quality features.

The data split / scaling / validation / test evaluation flow is kept
identical to train_mlp.py for fair comparison.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Keep KAN/matplotlib cache in writable paths.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from kan import KAN

# -- Reproducibility ----------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -- Hyperparameters ----------------------------------------------------------
KAN_HIDDEN_DIMS = [32, 16]
KAN_GRID = 8
KAN_K = 3
KAN_BASE_FUN = "silu"
KAN_SYMBOLIC_ENABLED = False

LR = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularization
BATCH_SIZE = 16
MAX_EPOCHS = 2000
PATIENCE = 100               # early-stopping patience
VAL_FRACTION = 0.15          # hold-out validation fraction from training data only

# -- Data loading -------------------------------------------------------------
FEATURE_COLS = [
    "pH", "COD_mg_L", "NH4_N_mg_L", "NO2_N_mg_L", "NO3_N_mg_L",
    "Br_mg_L", "TOC_mg_L", "UV254_A_cm", "temp_C",
]
TARGET_COLS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]

df = pd.read_csv("DBP_dataset_DWTP_B.csv")
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

print(f"Train samples: {len(train_df)},  Test samples: {len(test_df)}")
print(f"Features: {len(FEATURE_COLS)},   Targets: {len(TARGET_COLS)}")

# -- Train/validation split from training set only (no test leakage) ---------
train_sub_df, val_df = train_test_split(
    train_df,
    test_size=VAL_FRACTION,
    random_state=SEED,
)
print(f"Train subset: {len(train_sub_df)}, Validation: {len(val_df)}")

# -- Feature scaling (fit on train subset only) -------------------------------
scaler_x = StandardScaler().fit(train_sub_df[FEATURE_COLS])
scaler_y = StandardScaler().fit(train_sub_df[TARGET_COLS])

X_train = torch.tensor(scaler_x.transform(train_sub_df[FEATURE_COLS]), dtype=torch.float32)
Y_train = torch.tensor(scaler_y.transform(train_sub_df[TARGET_COLS]), dtype=torch.float32)
X_val = torch.tensor(scaler_x.transform(val_df[FEATURE_COLS]), dtype=torch.float32)
Y_val = torch.tensor(scaler_y.transform(val_df[TARGET_COLS]), dtype=torch.float32)
X_test = torch.tensor(scaler_x.transform(test_df[FEATURE_COLS]), dtype=torch.float32)

# Keep raw targets for evaluation in original scale
Y_test_raw = test_df[TARGET_COLS].values

# -- DataLoader ---------------------------------------------------------------
train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


def build_kan(in_dim: int, hidden_dims: list, out_dim: int) -> nn.Module:
    width = [in_dim] + hidden_dims + [out_dim]
    return KAN(
        width=width,
        grid=KAN_GRID,
        k=KAN_K,
        base_fun=KAN_BASE_FUN,
        symbolic_enabled=KAN_SYMBOLIC_ENABLED,
        save_act=False,
        auto_save=False,
        seed=SEED,
        device="cpu",
    )


# -- Model --------------------------------------------------------------------
model = build_kan(len(FEATURE_COLS), KAN_HIDDEN_DIMS, len(TARGET_COLS))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# -- Training with early stopping ---------------------------------------------
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    # --- train ---
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(train_ds)

    # --- validate on hold-out validation set (from training data) ---
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, Y_val).item()

    # --- early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if epoch % 100 == 0 or patience_counter == PATIENCE:
        print(
            f"Epoch {epoch:4d} | train MSE: {epoch_loss:.4f} | "
            f"val MSE: {val_loss:.4f} | patience: {patience_counter}/{PATIENCE}"
        )

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# -- Load best model & evaluate -----------------------------------------------
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    pred_scaled = model(X_test).numpy()

# Inverse-transform to original scale
pred_raw = scaler_y.inverse_transform(pred_scaled)

print("\n" + "=" * 60)
print("Final evaluation on test set (original scale)")
print("=" * 60)

for i, name in enumerate(TARGET_COLS):
    y_true = Y_test_raw[:, i]
    y_pred = pred_raw[:, i]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:15s}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")

# -- Save model ---------------------------------------------------------------
torch.save({
    "model_state": best_state,
    "scaler_x": scaler_x,
    "scaler_y": scaler_y,
    "feature_cols": FEATURE_COLS,
    "target_cols": TARGET_COLS,
    "hyperparams": {
        "kan_hidden_dims": KAN_HIDDEN_DIMS,
        "kan_grid": KAN_GRID,
        "kan_k": KAN_K,
        "kan_base_fun": KAN_BASE_FUN,
        "kan_symbolic_enabled": KAN_SYMBOLIC_ENABLED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "val_fraction": VAL_FRACTION,
    },
}, "kan_checkpoint.pt")
print("\nModel saved to kan_checkpoint.pt")
