# DepthWizard 🧙‍♂️
### Single-View Height Estimation & 3D Flythrough

> **SIH 2026 Prototype**  
> *Demonstrating single-view relative height estimation, physical shadow cue fusion, spatial confidence mapping, and interactive 3D surface reconstruction from aerial and satellite imagery.*

---

## 📌 Problem & Solution

### Problem
Ordinary 2D satellite and aerial images (JPG, PNG) lack vertical elevation data, making 3D spatial visualization and height assessment challenging without expensive LiDAR or multi-view stereo imagery. Furthermore, standard RGB images do not inherently contain absolute real-world metric scales.

### Solution
**DepthWizard** addresses this challenge by combining state-of-the-art AI monocular depth estimation (**Depth Anything V2**) with physical shadow detection cues, confidence-weighted data fusion, spatial evidence mapping, and real-time interactive 3D surface rendering.

---

## 🔄 System Pipeline Architecture

```
                                  USER UPLOADS IMAGE (JPG / PNG / GeoTIFF)
                                                     │
                                                     ▼
                                        IMAGE PREPROCESSING & VALIDATION
                                                     │
                                                     ▼
                                    DEPTH ANYTHING V2 (MONOCULAR DEPTH)
                                                     │
                                    ┌────────────────┴────────────────┐
                                    ▼                                 ▼
                              AI DEPTH MAP                    SHADOW DETECTION
                                    │                                 │
                                    └────────────────┬────────────────┘
                                                     ▼
                                         CONFIDENCE-WEIGHTED FUSION
                                                     │
                                                     ▼
                                            RELATIVE HEIGHT MAP
                                                     │
                                    ┌────────────────┴────────────────┐
                                    ▼                                 ▼
                          CONFIDENCE EVIDENCE MAP           INTERACTIVE 3D TERRAIN
                                                                      │
                                                                      ▼
                                                      (OPTIONAL) DEM / GCP CALIBRATION
                                                                      │
                                                                      ▼
                                                         ESTIMATED METRIC ELEVATION
```

---

## 💡 Key Operating Modes

1. **Relative Height Mode (Normal JPG / PNG)**
   - Estimates relative elevation and depth variations.
   - Generates interactive 3D surface models.
   - *Note:* Does not pretend to output exact metric elevation without ground-truth reference data.

2. **Calibrated Metric Elevation Mode (GeoTIFF + Reference DEM)**
   - Uses reference elevation control points or DEM data.
   - Applies linear scale & offset calibration (`metric_val = scale * relative_val + offset`).
   - Produces estimated absolute Digital Surface Models (DSM).

---

## 🛠️ Technology Stack

- **Language:** Python 3.12
- **Deep Learning Framework:** PyTorch (CUDA GPU Acceleration)
- **AI Model:** Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf` via Hugging Face Transformers)
- **Computer Vision:** OpenCV, Pillow, NumPy
- **3D Surface Rendering:** Plotly Graph Objects
- **Web UI:** Gradio
- **Geospatial Processing (Optional):** Rasterio

---

## 📁 Project Structure

```
DepthWizard/
│
├── app/                      # Web UI & Application Core
│   ├── __init__.py
│   ├── main.py               # Main launcher & pre-loading logic
│   └── ui.py                 # Gradio Blocks layout & workflow
│
├── depth/                    # Monocular Depth Estimation
│   ├── __init__.py
│   ├── depth_model.py        # Depth Anything V2 singleton wrapper
│   └── preprocessing.py      # Image validation & resizing
│
├── shadow/                   # Physical Shadow Cues
│   ├── __init__.py
│   ├── shadow_detector.py   # LAB/HSV color-space shadow detection
│   └── shadow_geometry.py   # Shadow direction & area geometry
│
├── fusion/                   # Multi-Modal Data Fusion
│   ├── __init__.py
│   └── fusion.py             # Confidence-weighted AI + Shadow fusion
│
├── calibration/              # Elevation Calibration
│   ├── __init__.py
│   ├── calibrator.py         # Scale & offset linear calibration
│   └── dem_loader.py         # Reference DEM & GeoTIFF loader
│
├── visualization/            # 2D & 3D Visualizations
│   ├── __init__.py
│   ├── depth_visualizer.py   # Color-mapped depth visualizer
│   ├── terrain_3d.py         # Plotly 3D interactive mesh surface
│   └── confidence_map.py     # Spatial evidence / confidence heatmap
│
├── evaluation/               # Metrics & Validation
│   ├── __init__.py
│   ├── metrics.py            # MAE, RMSE, Pearson correlation
│   └── evaluate.py           # Evaluation runner
│
├── data/                     # Input & Reference Data Directories
│   ├── input/
│   ├── reference/
│   └── samples/
│
├── outputs/                  # Rendered Output Directories
│   ├── depth/
│   ├── shadows/
│   ├── height/
│   ├── confidence/
│   └── 3d/
│
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── run.py                    # Root execution entrypoint
└── tests/                    # Integration & Pipeline Tests
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ or Python 3.12
- NVIDIA GPU with CUDA support (e.g., RTX 3050 4GB VRAM) or CPU fallback

### 2. Installation

Clone the repository and install dependencies:

```bash
cd DepthWizard
py -3.12 -m pip install -r requirements.txt
```

### 3. Run E2E Test Suite
Verify that model download, CUDA acceleration, and pipeline execution work:

```bash
py -3.12 tests/test_pipeline.py
```

### 4. Launch Web Application
Start the interactive local web dashboard:

```bash
py -3.12 run.py
```

Open your browser and navigate to:  
`http://127.0.0.1:7860`

---

## 📷 Usage Workflow

1. Open `http://127.0.0.1:7860` in your web browser.
2. Upload any satellite or aerial image (JPG/PNG).
3. Click **🚀 ANALYZE IMAGE**.
4. View the generated outputs across tabs:
   - **🌋 3D Interactive Terrain:** Rotate, zoom, and pan around the 3D surface model.
   - **🗺️ Height & Depth Maps:** Inspect the Depth Anything V2 map, detected shadow mask, and fused relative height map.
   - **🛡️ Confidence Map:** View the spatial evidence reliability map.
