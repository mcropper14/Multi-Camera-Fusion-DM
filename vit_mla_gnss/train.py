import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Load extracted future prediction data
with open("extracted_future_gnss_prediction.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([entry["features"] for entry in data])        # ViT+MLA+GNSS_t
y = np.array([entry["gnss"] for entry in data])            # GNSS_t+1

# Normalize features and labels
feature_scaler = StandardScaler()
label_scaler = StandardScaler()

X_scaled = feature_scaler.fit_transform(X)
y_scaled = label_scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

class FutureGNSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]  # [batch, seq, feature]

train_dataset = FutureGNSSDataset(X_train, y_train)
test_dataset = FutureGNSSDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class GNSSLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: [batch, seq, input_dim]
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNSSLSTMRegressor(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(500):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    print(f"Epoch {epoch+1}/50 | Train Loss: {total_loss / len(train_dataset):.6f}")

# Save model + scalers
os.makedirs("trained_model_future", exist_ok=True)
torch.save(model.state_dict(), "trained_model_future/lstm_future_gnss.pth")
with open("trained_model_future/feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open("trained_model_future/label_scaler.pkl", "wb") as f:
    pickle.dump(label_scaler, f)

print("\n Training complete and model saved.")
