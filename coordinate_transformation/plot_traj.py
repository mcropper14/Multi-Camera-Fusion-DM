import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Left and right image directories
left_img_dir = "/Documents/cadc_devkit/.../image_02/data"
right_img_dir = "/Documents/cadc_devkit/.../image_03/data"

# Load trajectory
with open("evaluation_results_with_xyz.pkl", "rb") as f:
    data = pickle.load(f)
    xyz_coordinates = data["xyz_coordinates"]

# Sort image files
left_files = sorted([f for f in os.listdir(left_img_dir) if f.endswith(".png") or f.endswith(".jpg")])
right_files = sorted([f for f in os.listdir(right_img_dir) if f.endswith(".png") or f.endswith(".jpg")])

# Frame indices to plot
selected_indices = [0, min(16, len(xyz_coordinates) - 1)]

# Extract and normalize trajectory
X = np.array([entry[0] for entry in xyz_coordinates])
Y = np.array([entry[1] for entry in xyz_coordinates])

# Image size
sample_img = cv2.imread(os.path.join(left_img_dir, left_files[0]))
img_h, img_w, _ = sample_img.shape
img_center_x = img_w // 2

# Normalize to fit image
X_norm = (X - np.min(X)) / (np.max(X) - np.min(X)) * img_w * 0.95
Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) * img_h * 0.95

# Plotting
fig, axes = plt.subplots(len(selected_indices), 2, figsize=(14, 6))
for row, idx in enumerate(selected_indices):
    # Load images
    left_img = cv2.imread(os.path.join(left_img_dir, left_files[idx]))
    right_img = cv2.imread(os.path.join(right_img_dir, right_files[idx]))
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # Prepare split trajectory
    x_traj = X_norm[:idx+1]
    y_traj = Y_norm[:idx+1]

    left_mask = x_traj <= img_center_x
    right_mask = x_traj > img_center_x

    # Plot LEFT camera view with left-half trajectory
    ax_left = axes[row, 0]
    ax_left.imshow(left_rgb)
    ax_left.plot(x_traj[left_mask], y_traj[left_mask], color='red', linewidth=2)
    ax_left.set_title(f"Left Camera - Frame {idx}")
    ax_left.axis("off")

    # Plot RIGHT camera view with right-half trajectory
    ax_right = axes[row, 1]
    ax_right.imshow(right_rgb)
    ax_right.plot(x_traj[right_mask], y_traj[right_mask], color='red', linewidth=2)
    ax_right.set_title(f"Right Camera - Frame {idx}")
    ax_right.axis("off")

plt.tight_layout()
plt.show()
