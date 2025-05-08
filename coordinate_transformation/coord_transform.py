import torch
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch.nn as nn

# WGS84 ellipsoid parameters
a = 6378137.0  # semi-major axis in meters
f = 1 / 298.257223563  # flattening factor
e2 = 2 * f - f**2  # square of the eccentricity

def latlon_to_xyz(lat, lon, alt):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # Calculate X, Y, Z
    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = ((1 - e2) * N + alt) * np.sin(lat_rad)
    
    return X, Y, Z

def rotation_matrix(roll, pitch, azimuth):
    """
    Create a rotation matrix from roll, pitch, and azimuth (direction).
    This rotates the local coordinate system (ENU) to global coordinates.
    """
    # Roll, pitch, and azimuth are in radians
    R = np.array([
        [np.cos(azimuth) * np.cos(pitch), 
         np.cos(azimuth) * np.sin(pitch) * np.sin(roll) - np.sin(azimuth) * np.cos(roll),
         np.cos(azimuth) * np.sin(pitch) * np.cos(roll) + np.sin(azimuth) * np.sin(roll)],

        [np.sin(azimuth) * np.cos(pitch),
         np.sin(azimuth) * np.sin(pitch) * np.sin(roll) + np.cos(azimuth) * np.cos(roll),
         np.sin(azimuth) * np.sin(pitch) * np.cos(roll) - np.cos(azimuth) * np.sin(roll)],

        [-np.sin(pitch), 
         np.cos(pitch) * np.sin(roll), 
         np.cos(pitch) * np.cos(roll)]
    ])
    return R

# Load the extracted features for evaluation
print("Loading Extracted Features for Evaluation...")
with open("extracted_features_test_2.pkl", "rb") as f:
    test_features = pickle.load(f)

valid_entries = []
feature_values = []
gnss_values = []
max_feature_dim = 0

# Prepare the features and GNSS values for evaluation
for entry in test_features:
    if "features" in entry and isinstance(entry["features"], np.ndarray):
        feature_vector = entry["features"].squeeze(0)
        valid_entries.append(entry)
        feature_values.append(feature_vector)
        max_feature_dim = max(max_feature_dim, feature_vector.shape[0])  

        gnss = entry["gnss_data"]
        gnss_values.append([
            gnss["latitude"],
            gnss["longitude"],
            gnss["altitude"],
            gnss["roll"],
            gnss["pitch"],
            gnss["azimuth"]
        ])

xyz_coordinates = []

for gnss in gnss_values:
    lat, lon, alt, roll, pitch, azimuth = gnss
    
    X, Y, Z = latlon_to_xyz(lat, lon, alt)
    
    # Apply rotation based on roll, pitch, azimuth
    R = rotation_matrix(roll, pitch, azimuth)
    movement_vector = np.array([1, 0, 0])  #Movement in ENU frame (1 meter forward in the local X direction)
    
    # Transform the movement vector to the global XYZ frame
    global_movement = np.dot(R, movement_vector)
    xyz_coordinates.append((X, Y, Z, global_movement))

print("\nConverted XYZ Coordinates and Adjusted Movement:")
for i, coords in enumerate(xyz_coordinates):
    X, Y, Z, movement = coords
    print(f"Entry {i+1}:")
    print(f"  Latitude, Longitude, Altitude -> X, Y, Z: {X:.4f}, {Y:.4f}, {Z:.4f}")
    print(f"  Adjusted Movement Vector: {movement}")
    print("-" * 50)


# Save the XYZ coordinates and the evaluation metrics
evaluation_results = {
    "xyz_coordinates": xyz_coordinates,
    "evaluation_metrics": {
        "mse": mean_squared_error(all_targets, all_predictions),
        "mae": mean_absolute_error(all_targets, all_predictions),
        "r2": r2_score(all_targets, all_predictions)
    }
}

# Save the results to a pickle file
with open("evaluation_results_with_xyz.pkl", "wb") as f:
    pickle.dump(evaluation_results, f)
    print(evaluation_results)

print("Evaluation results and XYZ coordinates saved.")