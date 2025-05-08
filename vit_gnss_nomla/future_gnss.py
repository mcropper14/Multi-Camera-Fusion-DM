import os
import cv2
import torch
import pickle
import numpy as np
from transformers import ViTModel, ViTImageProcessor
from load_data import load_matched_data

# Create output directory
OUTPUT_PATH = "extracted_future_gnss_prediction_no_mla.pkl"
os.makedirs("outputs", exist_ok=True)

# Load ViT model and processor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).eval()

# Extract features using ViT
def extract_vit_features(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state  # Shape: [1, 197, 768]

# Process dataset to predict GNSS(t+1) from image(t) + GNSS(t)
def process_and_save_future_prediction_features(n=100):
    data = load_matched_data()[:n+1]  # Need n+1 entries for future label
    all_entries = []

    for i in range(n):
        entry = data[i]
        future_entry = data[i+1]

        timestamp = entry["timestamp"]
        image_paths = entry["images"]
        gnss_now = entry["gnss"]
        gnss_future = future_entry["gnss"]

        cam5_path = image_paths.get("image_05")
        cam6_path = image_paths.get("image_06")

        if not cam5_path or not cam6_path:
            print(f"[WARNING] Missing images for {timestamp}")
            continue

        img5 = cv2.imread(cam5_path)
        img6 = cv2.imread(cam6_path)

        # ViT feature extraction
        feat5 = extract_vit_features(img5, vit_model, feature_extractor)  # [1, 197, 768]
        feat6 = extract_vit_features(img6, vit_model, feature_extractor)  # [1, 197, 768]

        # Concatenate feature maps along sequence dimension and average
        combined = torch.cat([feat5, feat6], dim=1)  # [1, 394, 768]
        fused = combined.mean(dim=1).squeeze(0)      # [768]

        gnss_vec_now = np.array([
            gnss_now["latitude"],
            gnss_now["longitude"],
            gnss_now["altitude"],
            gnss_now["roll"],
            gnss_now["pitch"],
            gnss_now["azimuth"]
        ], dtype=np.float32)

        input_features = np.concatenate([fused.detach().numpy(), gnss_vec_now], axis=0)

        gnss_vec_future = np.array([
            gnss_future["latitude"],
            gnss_future["longitude"],
            gnss_future["altitude"],
            gnss_future["roll"],
            gnss_future["pitch"],
            gnss_future["azimuth"]
        ], dtype=np.float32)

        all_entries.append({
            "timestamp": timestamp,
            "features": input_features,
            "gnss": gnss_vec_future
        })

        print(f"[INFO] Processed future GNSS target for: {timestamp}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_entries, f)

    print(f"\nSaved {len(all_entries)} future prediction entries to {OUTPUT_PATH}")


process_and_save_future_prediction_features(n=100)
