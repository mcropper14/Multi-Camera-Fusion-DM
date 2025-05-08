import os
import cv2
import torch
import pickle
import numpy as np
from transformers import ViTModel, ViTImageProcessor
import torch.nn as nn
from load_data import load_matched_data 

# Set output path
OUTPUT_PATH = "outputs/extracted_future_gnss_prediction.pkl"
os.makedirs("outputs", exist_ok=True)

# Load ViT model and processor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).eval()

# MLA Block from DeepSeek
class MLA_Block(nn.Module):
    def __init__(self, latent_dim, num_heads=8, dropout=0.1):
        super(MLA_Block, self).__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, latents, context):
        latents2 = self.norm1(latents)
        context2 = self.norm1(context)
        attn_output, _ = self.attn(latents2, context2, context2)
        latents = latents + attn_output  # Residual

        latents2 = self.norm2(latents)
        ffn_output = self.ffn(latents2)
        latents = latents + ffn_output  # Residual

        return latents

#From DeepSeek paper
class DeepSeekMLAFusion(nn.Module):
    def __init__(self, input_dim=768, num_latents=8, latent_dim=768, num_heads=8, num_layers=2, dropout=0.1):
        super(DeepSeekMLAFusion, self).__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))  # [1, N, D]
        self.layers = nn.ModuleList([
            MLA_Block(latent_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, features_1, features_2):
        x = torch.cat([features_1, features_2], dim=1)  # [1, seq_len*2, dim]
        x = x.squeeze(0).unsqueeze(0)  # Ensure shape [1, L, D]

        B = x.size(0)
        latents = self.latents.expand(B, -1, -1).contiguous()  # [B, num_latents, D]

        for layer in self.layers:
            latents = layer(latents, x)

        return latents  # [B, num_latents, D]


def extract_vit_features(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state  # [1, 197, 768]


def process_and_save_future_prediction_features(n=100):
    data = load_matched_data()[:n+1]
    fusion_layer = DeepSeekMLAFusion()
    all_entries = []

    for i in range(n):
        entry = data[i]
        future_entry = data[i + 1]

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
        fused = fusion_layer(feat5, feat6)  # [1, num_latents, 768]
        fused = fused.mean(dim=1).squeeze(0)  # Mean over latent tokens -> [768]

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
