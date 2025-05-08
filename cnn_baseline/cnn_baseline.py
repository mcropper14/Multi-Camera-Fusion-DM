import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
import pandas as pd
from datetime import datetime
import glob

from load_data import load_matched_data
# Dataset Directory and Camera setup
DATASET_DIR = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002"
CAMERAS = ["image_00", "image_01", "image_02", "image_03", "image_04", "image_05", "image_06"]



def load_camera_timestamps():

    camera_timestamps = {}
    camera_image_paths = {}

    for cam in CAMERAS:
        timestamp_file = os.path.join(DATASET_DIR, "labeled", cam, "timestamps.txt")
        timestamps = pd.read_csv(timestamp_file,  header=None, names=["timestamp"])
        
        timestamps["timestamp"] = timestamps["timestamp"].apply(
            lambda x: datetime.strptime(x[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        )
        
        camera_timestamps[cam] = timestamps["timestamp"].tolist()

    # Find common timestamps across all cameras
    common_timestamps = set(camera_timestamps[CAMERAS[0]])  
    for cam in CAMERAS[1:]:
        common_timestamps.intersection_update(set(camera_timestamps[cam]))  
    
    common_timestamps = sorted(common_timestamps)

    #assign image to timestamp
    for i, timestamp in enumerate(common_timestamps):
        image_paths = {
            cam: os.path.join(DATASET_DIR, "labeled", cam, "data", f"{str(i).zfill(10)}.png")
            for cam in CAMERAS
        }
        camera_image_paths[timestamp] = image_paths  # Store paths in dictionary

    return common_timestamps, camera_image_paths

#mtach gns with time
def load_gnss_ins_data():
    """
    Reads and matches GNSS/INS data files with camera timestamps.
    Returns:
        - gnss_data_dict: Dictionary where keys are timestamps and values are GNSS/INS data.
    """
    ah_fuck = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002/labeled/novatel"
    gnss_dir = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002/labeled/novatel/data" 
    gnss_files = sorted(glob.glob(os.path.join(gnss_dir, "*.txt")))  # Get all GNSS text files
    

    #/home/cropthecoder/Documents/cadc_devkit/data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002/labeled/novatel/timestamps.txt

    gnss_timestamps_file = os.path.join(ah_fuck, "timestamps.txt")
    gnss_timestamps = pd.read_csv(gnss_timestamps_file, header=None, names=["timestamp"])
    
    gnss_timestamps["timestamp"] = gnss_timestamps["timestamp"].apply(
        lambda x: datetime.strptime(x[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    )
    gnss_timestamps_list = gnss_timestamps["timestamp"].tolist()
    
    
    column_names = [
        "latitude", "longitude", "altitude", "undulation",
        "latitude_std", "longitude_std", "altitude_std",
        "roll", "pitch", "azimuth",
        "roll_std", "pitch_std", "azimuth_std",
        "ins_status", "position_type"
    ]
    
    gnss_data_dict = {}

    for i, gnss_file in enumerate(gnss_files):
        gnss_data = pd.read_csv(gnss_file, sep=r"\s+",  names=column_names)
        gnss_timestamp = gnss_timestamps_list[i]
        gnss_data_dict[gnss_timestamp] = gnss_data.iloc[0].to_dict()  # Store row as dict
    
    return gnss_data_dict

def match_gnss_to_camera(matched_timestamps, gnss_data_dict):

    matched_gnss_data = {}
    gnss_timestamps_sorted = sorted(gnss_data_dict.keys())

    for cam_timestamp in matched_timestamps:
        closest_gnss_timestamp = min(gnss_timestamps_sorted, key=lambda x: abs(x - cam_timestamp))
        matched_gnss_data[cam_timestamp] = gnss_data_dict[closest_gnss_timestamp]

    return matched_gnss_data






# Loading matched data
def load_matched_data():
    matched_timestamps, camera_image_paths = load_camera_timestamps()
    gnss_data_dict = load_gnss_ins_data()
    matched_gnss_data = match_gnss_to_camera(matched_timestamps, gnss_data_dict)

    matched_data = []
    for timestamp in matched_timestamps:
        if timestamp in matched_gnss_data:
            matched_data.append({
                "timestamp": timestamp,
                "images": camera_image_paths[timestamp],
                "gnss": matched_gnss_data[timestamp]
            })

    return matched_data

# Process and load data
matched_data = load_matched_data()

# Prepare data for the CNN
class CNNImageDataset(Dataset):
    def __init__(self, matched_data, transform=None):
        self.matched_data = matched_data
        self.transform = transform

    def __len__(self):
        return len(self.matched_data)

    def __getitem__(self, idx):
        entry = self.matched_data[idx]
        image_paths = entry["images"]
        image = cv2.imread(image_paths["image_00"])  # Read image from one of the cameras (adjust if needed)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        gnss_data = entry["gnss"]
        gnss_features = [
            gnss_data["latitude"],
            gnss_data["longitude"],
            gnss_data["altitude"],
            gnss_data["roll"],
            gnss_data["pitch"],
            gnss_data["azimuth"]
        ]
        
        return image, np.array(gnss_features, dtype=np.float32)

# Data Preprocessing and Scaling
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset and create DataLoader
dataset = CNNImageDataset(matched_data, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

# Dataset Directory and Camera setup
DATASET_DIR = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002"
CAMERAS = ["image_00", "image_01", "image_02", "image_03", "image_04", "image_05", "image_06"]

# Loading matched data
def load_matched_data():
    matched_timestamps, camera_image_paths = load_camera_timestamps()
    gnss_data_dict = load_gnss_ins_data()
    matched_gnss_data = match_gnss_to_camera(matched_timestamps, gnss_data_dict)

    matched_data = []
    for timestamp in matched_timestamps:
        if timestamp in matched_gnss_data:
            matched_data.append({
                "timestamp": timestamp,
                "images": camera_image_paths[timestamp],
                "gnss": matched_gnss_data[timestamp]
            })

    return matched_data

# Process and load data
matched_data = load_matched_data()

# Prepare data for the CNN
class CNNImageDataset(Dataset):
    def __init__(self, matched_data, transform=None):
        self.matched_data = matched_data
        self.transform = transform

    def __len__(self):
        return len(self.matched_data)

    def __getitem__(self, idx):
        entry = self.matched_data[idx]
        image_paths = entry["images"]
        image = cv2.imread(image_paths["image_00"])  # Read image from one of the cameras (adjust if needed)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        gnss_data = entry["gnss"]
        gnss_features = [
            gnss_data["latitude"],
            gnss_data["longitude"],
            gnss_data["altitude"],
            gnss_data["roll"],
            gnss_data["pitch"],
            gnss_data["azimuth"]
        ]
        
        return image, np.array(gnss_features, dtype=np.float32)

# Data Preprocessing and Scaling
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset and create DataLoader
dataset = CNNImageDataset(matched_data, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)  # After pooling, adjust for input size
        self.fc2 = nn.Linear(1024, 6)  # Output GNSS values: latitude, longitude, altitude, roll, pitch, azimuth
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 128 * 28 * 28)  # Adjust based on your image size after pooling
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output final prediction (6 GNSS values)
        
        return x

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 80
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (batch_features, batch_targets) in enumerate(dataloader):
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_features)

        # Compute the loss
        loss = criterion(predictions, batch_targets)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "cnn_baseline_model.pth")
print("Model saved as cnn_baseline_model.pth")

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)  # After pooling, adjust for input size
        self.fc2 = nn.Linear(1024, 6)  # Output GNSS values: latitude, longitude, altitude, roll, pitch, azimuth
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 128 * 28 * 28)  # Adjust based on your image size after pooling
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output final prediction (6 GNSS values)
        
        return x

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 80
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (batch_features, batch_targets) in enumerate(dataloader):
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_features)

        # Compute the loss
        loss = criterion(predictions, batch_targets)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "cnn_baseline_model.pth")
print("Model saved as cnn_baseline_model.pth")

# Evaluation Metrics
all_targets = []
all_predictions = []

# Run inference and collect metrics
with torch.no_grad():
    for batch_features, batch_targets in dataloader:
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

        predictions = model(batch_features)
        
        all_targets.append(batch_targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

# Convert targets and predictions to numpy arrays
all_targets = np.vstack(all_targets)
all_predictions = np.vstack(all_predictions)

# Calculate MSE, MAE, and R^2
mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

gnss_labels = ["Latitude", "Longitude", "Altitude", "Roll", "Pitch", "Azimuth"]
mse_per_feature = mean_squared_error(all_targets, all_predictions, multioutput="raw_values")
r2_per_feature = r2_score(all_targets, all_predictions, multioutput="raw_values")

# Print the evaluation results
print("\nEvaluation Complete!")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Overall R^2 Score: {r2:.4f}")

print("Per-GNSS Feature Metrics:")
for i, label in enumerate(gnss_labels):
    print(f"  {label}:")
    print(f"    - MSE: {mse_per_feature[i]:.6f}")
    print(f"    - R² Score: {r2_per_feature[i]:.4f}")
