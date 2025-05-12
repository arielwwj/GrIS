# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:22:37 2025

@author: wjw23vcn
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import optuna
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# ===== Configuration =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = r'C:\Users\yourname\Documents\lake_data'  # <<< Change to your actual folder path
Lake_i = 0  # Only process lake index 0
start_date, end_date = '1958-01-01', '2022-12-31'
date_range0 = pd.date_range(start=start_date, end=end_date, freq='D')

# ===== Load data =====
final_data = pd.DataFrame(index=date_range0)
files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and f != 'Landsat_merged_data.csv']
for file in files:
    df = pd.read_csv(os.path.join(data_folder, file), parse_dates=[0], dayfirst=True)
    if df.shape[1] < 2:
        continue
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df.set_index(df.columns[0], inplace=True)
    df = df.iloc[:, Lake_i] - 273.15 if file == 't2m_merged_output.csv' else df.iloc[:, Lake_i]
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    col_name = file.replace('merged', '').replace('_', '').replace('.csv', '').replace('output', '')
    final_data = final_data.join(df.rename(col_name), how='left')

obs = pd.read_csv(os.path.join(data_folder, 'Landsat_merged_data.csv'))
obs['time'] = pd.to_datetime(obs['time'], dayfirst=True, errors='coerce')
obs.set_index('time', inplace=True)
lake_observed = obs.iloc[:, Lake_i]
lake_observed_aligned = lake_observed.loc[final_data.index]

# ===== Feature selection =====
correlation_with_target = final_data.corrwith(lake_observed_aligned)
low_corr = correlation_with_target[abs(correlation_with_target) < 0.1].index.tolist()
nan_corr = correlation_with_target[correlation_with_target.isna()].index.tolist()
final_data_filtered = final_data.drop(columns=set(low_corr + nan_corr))
cor_matrix = final_data_filtered.corr()
to_drop = set()
for i in range(len(cor_matrix.columns)):
    for j in range(i):
        if abs(cor_matrix.iloc[i, j]) > 0.8:
            f1, f2 = cor_matrix.columns[i], cor_matrix.columns[j]
            if abs(correlation_with_target[f1]) < abs(correlation_with_target[f2]):
                to_drop.add(f1)
            else:
                to_drop.add(f2)
final_data_filtered = final_data_filtered.drop(columns=list(to_drop))
final_data_filtered['sin_day'] = np.sin(2 * np.pi * final_data_filtered.index.dayofyear / 365)
final_data_filtered['cos_day'] = np.cos(2 * np.pi * final_data_filtered.index.dayofyear / 365)

# ===== Standardize features =====
scaler = StandardScaler()
final_data_scaled = pd.DataFrame(
    scaler.fit_transform(final_data_filtered),
    columns=final_data_filtered.columns,
    index=final_data_filtered.index
)

# ===== Sliding window function =====
def create_windowed_dataset(window_size):
    X_windows, y_targets = [], []
    for date in lake_observed_aligned.dropna().index:
        start = date - pd.Timedelta(days=window_size)
        if start < final_data_scaled.index[0]:
            continue
        X_window = final_data_scaled.loc[start:date - pd.Timedelta(days=1)].values
        if X_window.shape[0] == window_size:
            X_windows.append(X_window)
            y_targets.append(lake_observed_aligned.loc[date])
    return np.array(X_windows), np.array(y_targets).reshape(-1, 1)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== LSTM model =====
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def weighted_mse_loss(pred, target):
    weights = (target > 0).float() * 2.0 + 1.0
    return ((pred - target) ** 2 * weights).mean()

# ===== Hyperparameter tuning =====
def objective(trial):
    window_size = trial.suggest_int("window_size", 15, 60)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    X, y = create_windowed_dataset(window_size)
    tscv = TimeSeriesSplit(n_splits=5)
    val_losses = []

    for train_idx, val_idx in tscv.split(X):
        model = LSTMRegressor(X.shape[2], hidden_size, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size)

        best_val_loss, patience, wait = float("inf"), 5, 0
        for epoch in range(50):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = weighted_mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += weighted_mse_loss(model(xb), yb).item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        val_losses.append(best_val_loss)
    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Best hyperparameters:", study.best_params)

# ===== Train final model =====
params = study.best_params
window_size = params['window_size']
X, y = create_windowed_dataset(window_size)
model = LSTMRegressor(X.shape[2], params['hidden_size'], params['dropout']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
train_loader = DataLoader(TimeSeriesDataset(X, y), batch_size=params['batch_size'], shuffle=True)

for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = weighted_mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()

# ===== Save model and parameters =====
output_dir = os.path.join(data_folder, "Lake_model_output")
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, f"lake_{Lake_i}_lstm_model.pth"))
with open(os.path.join(output_dir, f"lake_{Lake_i}_best_params.json"), 'w') as f:
    json.dump(study.best_params, f, indent=4)

# ===== Model evaluation =====
def evaluate_model(X_eval, y_eval, title, fig_name):
    model.eval()
    X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().squeeze()
    r2 = r2_score(y_eval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    mae = mean_absolute_error(y_eval, y_pred)
    plt.figure(figsize=(6,6))
    plt.scatter(y_eval, y_pred, alpha=0.5)
    plt.plot([min(y_eval), max(y_eval)], [min(y_eval), max(y_eval)], 'r--')
    plt.title(f"{title}\nR2={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fig_name))
    plt.close()
    return r2, rmse, mae

split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

evaluate_model(X_train, y_train, "Training Set", f"lake_{Lake_i}_training_eval.png")
evaluate_model(X_val, y_val, "Validation Set", f"lake_{Lake_i}_validation_eval.png")
evaluate_model(X, y, "All Data", f"lake_{Lake_i}_alldata_eval.png")

# ===== Save predictions =====
train_preds = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
pd.DataFrame({'Observed': y_train.flatten(), 'Predicted': train_preds}).to_csv(
    os.path.join(output_dir, f"lake_{Lake_i}_train_results.csv"), index=False)
pd.DataFrame({'Observed': y_val.flatten(), 'Predicted': val_preds}).to_csv(
    os.path.join(output_dir, f"lake_{Lake_i}_val_results.csv"), index=False)

# ===== Simulated time series plot =====
X_all, dates_all = [], []
for i in range(window_size, len(final_data_scaled)):
    window = final_data_scaled.iloc[i - window_size:i].values
    if not np.isnan(window).any():
        X_all.append(window)
        dates_all.append(final_data_scaled.index[i])

X_all_tensor = torch.tensor(np.array(X_all), dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred = model(X_all_tensor).cpu().numpy().squeeze()

simulated = pd.Series(y_pred, index=dates_all, name="simulated_temp")
simulated.to_csv(os.path.join(output_dir, f"lake_{Lake_i}_simulated_timeseries.csv"))

plt.figure(figsize=(14, 5))
plt.plot(simulated, label="Simulated", alpha=0.8)
plt.scatter(lake_observed_aligned.index, lake_observed_aligned.values, color='r', s=10, label="Observed")
plt.title("Lake Surface Water Temperature: Simulated vs Observed")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"lake_{Lake_i}_simulated_vs_observed.png"))
plt.close()
