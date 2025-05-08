
# Adaptive Multi-Camera Sensor Fusion for Autonomous Vehicles

This project implements a multi-modal sensor fusion pipeline to predict the future position and orientation (GNSS pose) of an autonomous vehicle using:

- Vision Transformers (ViT) for multi-camera feature extraction
- Multi-Head Latent Attention (MLA) for cross-view fusion
- GNSS (latitude, longitude, altitude, roll, pitch, azimuth) for spatial grounding
- A BiLSTM-based regression model for future GNSS trajectory prediction

Tested on the Canadian Adverse Driving Conditions (CADC) dataset.

---

##  Project Structure

```bash
├── camera_info/                 # Calibration utilities
├── cnn_baseline/               # CNN-based baseline for GNSS prediction
├── coordinate_transformation/  # GPS -> XYZ coordinate transforms
├── data_loading/               # Matched data loaders
├── vit_gnss_nomla/             # ViT + GNSS (no fusion)
├── vit_mla_gnss/               # ViT + MLA + GNSS + LSTM (main pipeline)
│   ├── eval.py                 # Evaluates trained LSTM model
│   ├── eval_no_lstm.py         # Variant for direct regression
│   ├── train.py                # Trains LSTM using fused ViT + GNSS(t) -> GNSS(t+1)
│   ├── future_gnss_lstm.py     # GNSS + LSTM  + ViT pipeline
│   ├── future_gnss_mla.py      # GNSS + LSTM + ViT + MLA pipeline
│   ├── future_gnss.py          # Feature extractor to generate GNSS prediction data
│   ├── vit_fusion_show.py      # Visualizations: attention, t-SNE, saliency maps
├── README.md
├── requirements.txt
```
##  Method Overview

###  Vision Transformer (ViT)
- Extracts spatial features from two ego cameras (Camera 5 & 6)
- Fine-tuned on CADC for improved winter-road generalization

###  Multi-Head Latent Attention (MLA)
- Aligns and fuses features across camera views
- Learns view-aware attention using latent space alignment

### GNSS Integration
- GNSS at time *t* is concatenated to fused visual features
- Predicts GNSS at time *t+1* (6-DoF pose):  
  `latitude, longitude, altitude, roll, pitch, azimuth`

###  BiLSTM Model
- Captures temporal continuity in vehicle motion
- Predicts future GNSS using a fused feature sequence

---

##  Setup

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt

2. Download CADC Dataset [View the CADC dataset](https://github.com/mpitropov/cadc_devkit/tree/master)


3. Ensure the Directory is setup as:
```
/path/to/cadc/
  └── sequence_01/
      ├── image_05/
      ├── image_06/
      └── gnss.json
```
4. Run load_data.py for data data synchronization 

## Pipeline 

1. Extract Features + gnss: ```python vit_mla_gnss/future_gnss_mla.py```

2. Train for GNSS Prediction: ```python vit_mla_gnss/train.py```

3. Evaluate Model Accuracy: ```python vit_mla_gnss/eval.py```

