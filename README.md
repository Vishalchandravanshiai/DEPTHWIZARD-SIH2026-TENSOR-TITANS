<div align="center">

# 🧙‍♂️ DepthWizard

### Single-View Height Estimation & 3D Flythrough

**Smart India Hackathon 2026 · Problem Statement 26175 · Team Tensor Titans**

*AI-powered single-view height estimation and 3D terrain reconstruction — turning one satellite or aerial photo into an interactive, navigable 3D terrain.*

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-V1%20Prototype-yellow)
![License](https://img.shields.io/badge/License-Active%20Development-lightgrey)

</div>

---

## 📖 Table of Contents

- [Project Status](#-project-status)
- [Problem](#-problem)
- [Proposed Solution](#-proposed-solution)
- [Current V1 Architecture](#-current-v1-architecture)
- [V1 Capabilities](#-v1-capabilities)
- [V1 Outputs](#-v1-outputs)
- [Final Project Architecture](#️-final-project-architecture)
- [Two Operating Modes (Final System)](#-two-operating-modes-in-the-final-system)
- [Planned Final Enhancements](#-planned-final-enhancements)
- [Technology Stack](#️-technology-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage Workflow](#-usage-workflow)
- [Evaluation](#-evaluation)
- [Development Roadmap](#️-development-roadmap)
- [SIH 2026 Alignment](#-sih-2026-alignment)
- [Team](#-team)

---

## 🚧 Project Status

### ✅ V1 Prototype — Working

The current prototype implements a complete end-to-end pipeline for:

| ✔ | Capability |
|---|---|
| ✅ | Single-view monocular depth estimation |
| ✅ | Shadow detection and shadow geometry analysis |
| ✅ | AI depth + shadow-based fusion |
| ✅ | Relative surface height / rDSM generation |
| ✅ | Spatial confidence / evidence mapping |
| ✅ | Terrain slope analysis |
| ✅ | Interactive 3D terrain visualization |
| ✅ | 3D and elevation asset export |

> **Current V1 output:** Relative Digital Surface Model (**rDSM**).
> Absolute metric elevation is part of the planned final architecture and requires geospatial reference information such as SRTM/DEM data or Ground Control Points (GCPs).

---

## 📌 Problem

Traditional 3D terrain and elevation reconstruction relies on:

- 🛰️ LiDAR
- 📷 Stereo imagery
- 📡 InSAR
- 🔺 Multi-view reconstruction

These approaches require multiple images, specialized sensors, expensive data, or significant computational resources.

A single RGB satellite or aerial image contains rich visual information but does **not** directly provide dense per-pixel elevation or absolute height.

**The challenge:** estimate useful terrain height from a *single optical image*, while addressing the scale ambiguity and domain gap inherent to monocular depth estimation.

---

## 💡 Proposed Solution

DepthWizard combines AI-based monocular depth estimation with visual and physical terrain cues:

- 🧠 **Depth Anything V2** for initial monocular depth estimation
- 🌓 **Shadow detection** for additional structural cues
- 📐 **Shadow geometry analysis**
- ⚖️ **Confidence-weighted fusion**
- 🗺️ **Relative height / rDSM generation**
- 📊 **Terrain slope analysis**
- 🛡️ **Spatial confidence / evidence mapping**
- 🌍 **Interactive 3D terrain reconstruction**

The system is designed to evolve from relative height estimation toward **metric, georeferenced DSM generation** using reference elevation data.

---

## 🔄 Current V1 Architecture

```text
                    USER UPLOADS IMAGE
                 (JPG / PNG / TIFF / HDF5)
                            │
                            ▼
               IMAGE PREPROCESSING
                  & VALIDATION
                            │
                            ▼
                  DEPTH ANYTHING V2
                MONOCULAR DEPTH MODEL
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          AI DEPTH MAP           SHADOW DETECTION
                                        │
                                        ▼
                                SHADOW GEOMETRY
                │                       │
                └───────────┬───────────┘
                            ▼
                 DEPTH + SHADOW FUSION
                            │
                            ▼
                 RELATIVE HEIGHT / rDSM
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
       CONFIDENCE        SLOPE          3D TERRAIN
       / EVIDENCE       ANALYSIS       VISUALIZATION
          MAP
                            │
                            ▼
                     OUTPUT EXPORTS
                  OBJ / NPY / PNG
```

---

## 🚀 V1 Capabilities

### 1️⃣ Image Preprocessing

Accepts optical imagery and prepares it for inference.

**Supported formats:** `.jpg` `.jpeg` `.png` `.tif` `.tiff` `.h5` `.hdf5` `.he5`

Large images are resized during preprocessing for practical local inference.

### 2️⃣ Monocular Depth Estimation

Uses **Depth Anything V2 — Small** to estimate relative depth from a single optical image.

```
Optical Image → Depth Anything V2 → Relative Depth Map
```

Automatically uses CUDA when available, with CPU fallback.

### 3️⃣ Shadow Detection

Analyzes image intensity and color information to detect potential shadow regions using:

- LAB color space
- HSV information
- Luminance analysis
- Morphological processing

Shadow geometry extraction provides:
- Shadow presence · Shadow area · Shadow centroid · Approximate shadow direction

> ⚠️ The V1 shadow system is currently a computer-vision heuristic cue. More physically grounded shadow-height estimation is planned for the final system.

### 4️⃣ Confidence-Weighted Fusion

```
        AI Depth ──┐
                    ├──► Fusion ──► Relative Height
Shadow Evidence ────┘
```

The fusion stage also generates spatial evidence/confidence information.

### 5️⃣ Relative Digital Surface Model (rDSM)

The fused output represents relative surface-height variation. It should **not** be interpreted as real-world elevation in meters without reference calibration.

### 6️⃣ Terrain Analysis

- Mean slope angle
- Maximum slope angle
- Steep terrain percentage
- Slope heatmap

### 7️⃣ Interactive 3D Terrain

The relative height map is converted into an interactive 3D terrain surface using Plotly, with camera modes:

- 45° Perspective · Drone Flythrough · Top-Down Overhead · Structural Profile

The original optical image textures the reconstructed surface.

---

## 📊 V1 Outputs

| Output | Description |
|---|---|
| Depth Map | Monocular depth estimated using Depth Anything V2 |
| Shadow Map | Detected potential shadow regions |
| Shadow Geometry | Spatial shadow characteristics |
| Relative Height Map | Fused relative surface-height representation |
| Confidence / Evidence Map | Spatial evidence visualization |
| Slope Map | Terrain slope visualization |
| 3D Terrain | Interactive reconstructed terrain |
| OBJ | 3D terrain mesh |
| NPY | Raw relative elevation matrix |
| 16-bit PNG | Relative elevation representation |

---

## 🏗️ Final Project Architecture

The V1 prototype is the foundation of the final SIH solution. The planned final architecture extends the relative-height pipeline with geospatial reference information and metric calibration:

```text
                         USER INPUT
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
       JPG / PNG                     GEOREFERENCED
       Normal Image                    GeoTIFF
             │                             │
             │                             ▼
             │                    GEOSPATIAL METADATA
             │                    CRS / TRANSFORM / BOUNDS
             │                             │
             └──────────────┬──────────────┘
                            ▼
                 IMAGE PREPROCESSING
                            │
                            ▼
                  DEPTH ANYTHING V2
                            │
                            ▼
                    RELATIVE DEPTH
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
       SHADOW ANALYSIS              REFERENCE ELEVATION
             │                       SRTM / DEM / GCP
             │                             │
             ▼                             ▼
       PHYSICAL SHADOW              SCALE CALIBRATION
       HEIGHT CUES                   SCALE + OFFSET
             │                             │
             └──────────────┬──────────────┘
                            ▼
                   CONFIDENCE-WEIGHTED
                        FUSION
                            │
                            ▼
                  METRIC / RELATIVE DSM
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        HEIGHT MAP      SLOPE MAP     CONFIDENCE MAP
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                TEXTURED 3D TERRAIN
                            │
                            ▼
                INTERACTIVE FLYTHROUGH
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Height Inspection            Terrain Analysis
       Structure Analysis           Slope Analysis
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 GEOREFERENCED DSM EXPORT
                         GeoTIFF
```

---

## 🌍 Two Operating Modes in the Final System

<table>
<tr>
<th>🖼️ Mode 1 — Relative Height</th>
<th>🛰️ Mode 2 — Metric Elevation</th>
</tr>
<tr>
<td>

For normal JPG/PNG imagery:
```
JPG / PNG
   ↓
Depth Anything V2
   ↓
Shadow + Depth Fusion
   ↓
Relative Height / rDSM
   ↓
3D Visualization
```
Provides relative terrain structure and visualization.

</td>
<td>

For georeferenced imagery with reference elevation data:
```
GeoTIFF
   ↓
Geospatial Metadata
   ↓
Depth Anything V2 → Relative Depth
   ↓
SRTM / DEM / GCP
   ↓
Scale & Offset Calibration
   ↓
Metric Elevation → Georeferenced DSM
```

</td>
</tr>
</table>

**Calibration model:**

```
Metric Elevation = Scale × Relative Height + Offset
```

> This mode is part of the **planned final implementation** and is not yet fully represented in the current V1 prototype.

---

## 🧭 Planned Final Enhancements

<details>
<summary><b>🌐 Geospatial Reconstruction</b></summary>

- Robust GeoTIFF metadata handling
- CRS and affine-transform preservation
- SRTM / DEM integration
- Reference DEM alignment
- GCP-based calibration
- Metric elevation estimation
- Georeferenced DSM GeoTIFF export
</details>

<details>
<summary><b>📐 Improved Height Estimation</b></summary>

- More robust shadow segmentation
- Sun elevation and azimuth integration
- Shadow-length based physical height estimation
- Improved depth + shadow fusion
- Confidence-aware multi-source fusion
</details>

<details>
<summary><b>🛰️ Remote-Sensing Adaptation</b></summary>

- Evaluation on remote-sensing datasets
- Domain adaptation / fine-tuning
- Testing across urban environments
- Testing across sparse landscapes
- Testing across hilly terrain
- Testing across forested environments
</details>

<details>
<summary><b>🌋 Visualization</b></summary>

- Improved first-person navigation
- More realistic terrain texturing
- Interactive height inspection
- Structural height measurement
- Advanced terrain and slope analysis
- Standalone deployment
</details>

---

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| **Core** | Python 3.12 · PyTorch · Hugging Face Transformers |
| **AI Model** | Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) |
| **Computer Vision** | OpenCV · Pillow · NumPy |
| **Visualization** | Plotly (interactive 3D surface) |
| **Interface** | Gradio |
| **Geospatial** | Rasterio (DEM / SRTM support planned for metric calibration) |
| **Evaluation** | MAE · RMSE · Pearson Correlation |

---

## 📁 Project Structure

```
DepthWizard/
│
├── app/
│   ├── main.py
│   └── ui.py
│
├── calibration/
│   ├── calibrator.py
│   ├── dem_loader.py
│   └── exporter.py
│
├── data/
│   └── samples/
│
├── depth/
│   ├── depth_model.py
│   └── preprocessing.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluate.py
│
├── fusion/
│   └── fusion.py
│
├── models/
│
├── shadow/
│   ├── shadow_detector.py
│   └── shadow_geometry.py
│
├── tests/
│   └── test_pipeline.py
│
├── visualization/
│   ├── depth_visualizer.py
│   ├── terrain_3d.py
│   ├── confidence_map.py
│   └── slope_analysis.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA support (recommended, 4 GB+ VRAM) — CPU fallback available

### 2. Clone the Repository

```bash
git clone https://github.com/Vishalchandravanshiai/DEPTHWIZARD-SIH2026-TENSOR-TITANS.git
cd DEPTHWIZARD-SIH2026-TENSOR-TITANS
```

### 3. Create a Virtual Environment (Windows)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Prototype

```bash
python run.py
```

Then open: **http://127.0.0.1:7860**

---

## 📷 Usage Workflow

| Step | Action |
|---|---|
| 1 | Launch the application |
| 2 | Upload a supported aerial or satellite image |
| 3 | Click **🚀 RUN ELEVATION & 3D PIPELINE** |
| 4 | Explore: 🌋 3D Terrain · 🗺️ Depth Map · 🌑 Shadow Map · 📐 rDSM · 🛡️ Confidence Map · 📊 Slope Analysis |
| 5 | Download the generated terrain assets |

---

## 🧪 Evaluation

Implemented metrics for comparing estimated surfaces against reference depth/elevation data:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Square Error
- **Pearson Correlation** — correlation between predicted and reference surface values

The final evaluation plan will compare performance across different terrain and land-cover conditions.

---

## 🗺️ Development Roadmap

```text
                 DEPTHWIZARD DEVELOPMENT
                          │
                          ▼
                 ┌─────────────────┐
                 │   V1 PROTOTYPE  │
                 └────────┬────────┘
                          │
                          ▼
             Single-view relative height
                          │
                          ▼
              Shadow + depth fusion
                          │
                          ▼
                  3D visualization
                          │
                          ▼
                 ┌─────────────────┐
                 │   V2 GEO MODE   │
                 └────────┬────────┘
                          │
                          ▼
                 GeoTIFF processing
                          │
                          ▼
                   DEM / SRTM / GCP
                          │
                          ▼
                 Metric calibration
                          │
                          ▼
                    Metric DSM
                          │
                          ▼
                 ┌─────────────────┐
                 │ FINAL PLATFORM  │
                 └────────┬────────┘
                          │
                          ▼
             Georeferenced DSM + 3D
                          │
                          ▼
              Interactive flythrough
                          │
                          ▼
             Height & slope inspection
                          │
                          ▼
                  Evaluation & Demo
```

---

## 🎯 SIH 2026 Alignment

DepthWizard is being developed toward the Smart India Hackathon 2026 problem statement:

> **DepthWizard — Single-View Height Estimation & 3D Flythrough**

<table>
<tr>
<th>🔹 Current Prototype</th>
<th>🎯 Final Target</th>
</tr>
<tr>
<td>

```
Single Optical Image
        ↓
Monocular Depth
        ↓
Shadow Cues
        ↓
Fusion
        ↓
Relative Height
        ↓
3D Terrain
```

</td>
<td>

```
Georeferenced Optical Image
        ↓
Monocular Depth
        ↓
Shadow / Physical Cues
        ↓
Reference DEM / SRTM / GCP
        ↓
Scale Calibration
        ↓
Metric DSM
        ↓
Georeferenced 3D Terrain
        ↓
Interactive Flythrough
        ↓
Height / Slope / Structural Analysis
```

</td>
</tr>
</table>

---

## 👥 Team

<div align="center">

### 🏆 Team Tensor Titans
**AKS University, Satna**

Developed for **Smart India Hackathon 2026**

</div>

---

## 📌 Project Philosophy

> **Prototype first → Validate → Calibrate → Georeference → Visualize → Evaluate**

The V1 prototype establishes the core AI and visualization pipeline, while the planned geospatial calibration and metric DSM stages extend it toward the complete SIH solution.

---

## 📜 License

This project is currently under active development as a Smart India Hackathon 2026 prototype.

<div align="center">

*Made with 🧠 and ☕ by Team Tensor Titans*

</div>
