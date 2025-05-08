import os
import cv2
import torch
import pickle
import numpy as np
from transformers import ViTModel, ViTImageProcessor
import torch.nn as nn
from load_data import load_matched_data 

# Create output directory
OUTPUT_PATH = "extracted_future_gnss_prediction.pkl"
os.makedirs("outputs", exist_ok=True)

# Load ViT model and processor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).eval()

class MLAFusionLayer(nn.Module):
    def __init__(self, input_dim=768, num_heads=8):
        super(MLAFusionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, features_1, features_2):
        combined = torch.cat([features_1, features_2], dim=1).squeeze(0).unsqueeze(1)
        attn_output, _ = self.attention(combined, combined, combined)
        fused = self.fc(attn_output)
        return fused

# Extract features using ViT
def extract_vit_features(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state  # Shape: [1, 197, 768]

# Process dataset to predict GNSS(t+1) from image(t) + GNSS(t)
def process_and_save_future_prediction_features(n=100):
    data = load_matched_data()[:n+1]  # Need n+1 entries for future label
    fusion_layer = MLAFusionLayer()
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
        feat5 = extract_vit_features(img5, vit_model, feature_extractor)
        feat6 = extract_vit_features(img6, vit_model, feature_extractor)

        # MLA fusion
        fused = fusion_layer(feat5, feat6)  # [seq_len * 2, 1, 768]
        fused = fused.mean(dim=0).squeeze(0)  # mean over sequence [768]

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
    print(f"\n Saved {len(all_entries)} future prediction entries to {OUTPUT_PATH}")


process_and_save_future_prediction_features(n=100)
