"""
MLP baseline for DBP prediction on DBP_dataset_DWTP_B.csv
Predicts: T_THMs, DBCM, BDCM from 9 water quality features.

FIX 1: early stopping uses a held-out validation split from TRAINING data,
       not the test set. Test set is only touched for final evaluation.
FIX 2: trains a SEPARATE model for each target to avoid MSE scaling issues
       and underfitting on smaller targets.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters ──────────────────────────────────────────────────────────
HIDDEN_DIMS = [32, 16]
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularisation
BATCH_SIZE = 16
MAX_EPOCHS = 2000
PATIENCE = 100                # early-stopping patience
VAL_FRACTION = 0.15           # fraction of training data held out for validation

# ── Data loading ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "pH", "COD_mg_L", "NH4_N_mg_L", "NO2_N_mg_L", "NO3_N_mg_L",
    "Br_mg_L", "TOC_mg_L", "UV254_A_cm", "temp_C",
]
TARGET_COLS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]

df = pd.read_csv("DBP_dataset_DWTP_B.csv")
train_df = df[df["split"] == "train"]
test_df  = df[df["split"] == "test"]

print(f"Train samples: {len(train_df)},  Test samples: {len(test_df)}")
print(f"Features: {len(FEATURE_COLS)},   Targets: {len(TARGET_COLS)}")

# ── Train / Validation split (from training data only) ───────────────────────
train_sub_df, val_df = train_test_split(
    train_df, test_size=VAL_FRACTION, random_state=SEED
)
print(f"  → Train subset: {len(train_sub_df)},  Validation: {len(val_df)}")

# ── Model Definition ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# We will store models, scalers, and predictions for each target
models = {}
scalers_x = {}
scalers_y = {}

print("\n" + "=" * 60)
print("Training independent models for each target")
print("=" * 60)

for target_idx, target_name in enumerate(TARGET_COLS):
    print(f"\n▶ Training model for: {target_name}")

    # ── Feature scaling for THIS target ──────────────────────────────────────
    scaler_x = StandardScaler().fit(train_sub_df[FEATURE_COLS])
    scaler_y = StandardScaler().fit(train_sub_df[[target_name]])

    X_train = torch.tensor(scaler_x.transform(train_sub_df[FEATURE_COLS]), dtype=torch.float32)
    Y_train = torch.tensor(scaler_y.transform(train_sub_df[[target_name]]), dtype=torch.float32)
    X_val   = torch.tensor(scaler_x.transform(val_df[FEATURE_COLS]),      dtype=torch.float32)
    Y_val   = torch.tensor(scaler_y.transform(val_df[[target_name]]),        dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # ── Init model (output dim = 1) ──────────────────────────────────────────
    torch.manual_seed(SEED + target_idx) # different seed per target initialization
    model = MLP(len(FEATURE_COLS), HIDDEN_DIMS, 1, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # ── Training with early stopping ─────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch:4d} (best epoch: {best_epoch:4d}) | final val MSE: {best_val_loss:.4f}")
            break

    model.load_state_dict(best_state)
    
    models[target_name] = model
    scalers_x[target_name] = scaler_x
    scalers_y[target_name] = scaler_y


print("\n" + "=" * 60)
print("Final evaluation on test set (original scale)")
print("=" * 60)

for target_name in TARGET_COLS:
    model = models[target_name]
    scaler_x = scalers_x[target_name]
    scaler_y = scalers_y[target_name]

    model.eval()
    X_te = torch.tensor(scaler_x.transform(test_df[FEATURE_COLS]), dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(X_te).numpy()
    
    # Inverse-transform to original scale
    pred_raw = scaler_y.inverse_transform(pred_scaled)
    y_pred = pred_raw[:, 0]
    y_true = test_df[target_name].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {target_name:15s}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")

# ── Save models ──────────────────────────────────────────────────────────────
torch.save({
    "model_states": {name: m.state_dict() for name, m in models.items()},
    "scalers_x": scalers_x,
    "scalers_y": scalers_y,
    "feature_cols": FEATURE_COLS,
    "target_cols": TARGET_COLS,
    "hyperparams": {
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    },
}, "mlp_checkpoint_separate.pt")
print("\nModels saved to mlp_checkpoint_separate.pt")
