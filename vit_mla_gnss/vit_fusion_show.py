import os
import cv2
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from transformers import ViTModel, ViTImageProcessor
from load_data import load_matched_data


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


model_name = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name, output_attentions=True).eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MLAFusionLayer(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MLAFusionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, features_cam_1, features_cam_2):
        combined_features = torch.cat([features_cam_1, features_cam_2], dim=0)
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        fused_features = self.fc(attn_output)
        return fused_features

def extract_features(image, model, feature_extractor):
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs.last_hidden_state

def process_and_fuse_features(image_cam_05, image_cam_06, model, feature_extractor):
    features_cam_05 = extract_features(image_cam_05, model, feature_extractor)
    features_cam_06 = extract_features(image_cam_06, model, feature_extractor)
    fusion_layer = MLAFusionLayer(input_dim=features_cam_05.size(-1))
    fused_features = fusion_layer(features_cam_05, features_cam_06)
    return features_cam_05, features_cam_06, fused_features


def process_first_n_images_with_gnss(n):
    matched_data = load_matched_data()[:n]
    fused_features_list = []
    cam05_features_list = []
    cam06_features_list = []

    for entry in matched_data:
        timestamp = entry["timestamp"]
        image_paths = entry["images"]
        gnss_data = entry["gnss"]

        image_cam_05 = cv2.imread(image_paths["image_05"])
        image_cam_06 = cv2.imread(image_paths["image_06"])

        features_cam_05, features_cam_06, fused_features = process_and_fuse_features(image_cam_05, image_cam_06, model, feature_extractor)

        gnss_tensor = torch.tensor([
            gnss_data["latitude"], gnss_data["longitude"], gnss_data["altitude"],
            gnss_data["roll"], gnss_data["pitch"], gnss_data["azimuth"]
        ], dtype=torch.float32).unsqueeze(0)

        features_cam_05 = features_cam_05.flatten(start_dim=1)[0]
        features_cam_06 = features_cam_06.flatten(start_dim=1)[0]
        fused_features = fused_features.flatten(start_dim=1)[0]
        gnss_tensor = gnss_tensor.squeeze(0)

        try:
            final_fused = torch.cat((fused_features, gnss_tensor), dim=0)
        except Exception as e:
            print(f"[ERROR] Failed to concatenate features: {e}")
            continue

        print(f"Processed {timestamp} | Final Feature Shape: {final_fused.shape}")

        fused_features_list.append({
            "timestamp": timestamp,
            "features": final_fused.detach().numpy(),
        })

        cam05_features_list.append({
            "timestamp": timestamp,
            "features": features_cam_05.detach().numpy(),
        })

        cam06_features_list.append({
            "timestamp": timestamp,
            "features": features_cam_06.detach().numpy(),
        })

    return fused_features_list, cam05_features_list, cam06_features_list

# === T-SNE VISUALIZATION ===
def visualize_tsne(features_list, title, filename):
    feature_matrix = np.stack([item["features"] for item in features_list], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_result = tsne.fit_transform(feature_matrix)

    plt.figure(figsize=(7, 7))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', s=10)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()

if __name__ == "__main__":
    print("Extracting features from matched data...")
    fused_feats, cam05_feats, cam06_feats = process_first_n_images_with_gnss(10)

    fused_train, _ = train_test_split(fused_feats, test_size=0.2, random_state=42)
    cam05_train, _ = train_test_split(cam05_feats, test_size=0.2, random_state=42)
    cam06_train, _ = train_test_split(cam06_feats, test_size=0.2, random_state=42)

    # Save
    with open("extracted_features_fused.pkl", "wb") as f:
        pickle.dump(fused_train, f)
    with open("extracted_features_cam05.pkl", "wb") as f:
        pickle.dump(cam05_train, f)
    with open("extracted_features_cam06.pkl", "wb") as f:
        pickle.dump(cam06_train, f)

    # Visualizations
    print("Generating t-SNE visualizations...")
    visualize_tsne(fused_train, "t-SNE of Fused ViT + GNSS Features", "tsne_fused_features.png")
    visualize_tsne(cam05_train, "t-SNE of ViT Features from Camera 5", "tsne_cam05_features.png")
    visualize_tsne(cam06_train, "t-SNE of ViT Features from Camera 6", "tsne_cam06_features.png")
