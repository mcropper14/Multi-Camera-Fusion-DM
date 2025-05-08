import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data and scalers
with open("extracted_future_gnss_prediction.pkl", "rb") as f:
    data = pickle.load(f)

with open("trained_model_future/feature_scaler.pkl", "rb") as f:
    feature_scaler = pickle.load(f)
with open("trained_model_future/label_scaler.pkl", "rb") as f:
    label_scaler = pickle.load(f)

X = np.array([entry["features"] for entry in data])
y = np.array([entry["gnss"] for entry in data])

X_scaled = feature_scaler.transform(X)
y_scaled = label_scaler.transform(y)

# Dataset class
class FutureGNSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

dataset = FutureGNSSDataset(X_scaled, y_scaled)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

class GNSSLSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_dim, 6)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNSSLSTMRegressor(input_dim=X.shape[1]).to(device)
model.load_state_dict(torch.load("trained_model_future/lstm_future_gnss.pth"))
model.eval()


all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

preds = np.vstack(all_preds)
targets = np.vstack(all_targets)

# Inverse transform
preds_real = label_scaler.inverse_transform(preds)
targets_real = label_scaler.inverse_transform(targets)

# Metrics
mse = mean_squared_error(targets_real, preds_real)
mae = mean_absolute_error(targets_real, preds_real)
r2 = r2_score(targets_real, preds_real)

print("\n Future GNSS Prediction Evaluation:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²:  {r2:.4f}")

labels = ["Latitude", "Longitude", "Altitude", "Roll", "Pitch", "Azimuth"]
mse_each = mean_squared_error(targets_real, preds_real, multioutput="raw_values")
r2_each = r2_score(targets_real, preds_real, multioutput="raw_values")

print("\n Per-GNSS Component Performance:")
for i, label in enumerate(labels):
    print(f"  {label}: MSE = {mse_each[i]:.4f}, R² = {r2_each[i]:.4f}")
