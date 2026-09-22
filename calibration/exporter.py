"""
Geospatial and 3D Model Exporter for DepthWizard.
Exports rDSM/DSM elevation maps to GeoTIFF, NPY, and 3D Wavefront OBJ mesh files.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple


def export_elevation_npy(height_map: np.ndarray, output_path: str) -> str:
    """Save raw height matrix as NumPy binary file (.npy)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, height_map)
    return output_path


def export_elevation_png(height_map: np.ndarray, output_path: str) -> str:
    """Save 16-bit high dynamic range PNG elevation map."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h_min, h_max = np.min(height_map), np.max(height_map)
    if h_max - h_min > 1e-6:
        norm_h = (height_map - h_min) / (h_max - h_min)
    else:
        norm_h = np.zeros_like(height_map)

    uint16_h = (norm_h * 65535.0).astype(np.uint16)
    cv2.imwrite(output_path, uint16_h)
    return output_path


def export_obj_mesh(
    height_map: np.ndarray,
    output_path: str,
    max_grid_dim: int = 150,
    height_scale: float = 0.35
) -> str:
    """
    Export 3D terrain surface as Wavefront OBJ file (.obj)
    Compatible with Three.js, Unity, Babylon.js, and Blender.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = height_map.shape

    scale = min(1.0, max_grid_dim / float(max(h, w)))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        grid = cv2.resize(height_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
        grid = height_map.copy()

    with open(output_path, "w") as f:
        f.write("# DepthWizard 3D Terrain Wavefront OBJ Export\n")
        
        # Write vertices (v x y z)
        for i in range(new_h):
            for j in range(new_w):
                x = j / float(new_w - 1)
                y = 1.0 - (i / float(new_h - 1))
                z = float(grid[i, j]) * height_scale
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        # Write texture coordinates (vt u v)
        for i in range(new_h):
            for j in range(new_w):
                u = j / float(new_w - 1)
                v = 1.0 - (i / float(new_h - 1))
                f.write(f"vt {u:.6f} {v:.6f}\n")

        # Write faces (f v1/vt1 v2/vt2 v3/vt3)
        for i in range(new_h - 1):
            for j in range(new_w - 1):
                idx1 = i * new_w + j + 1
                idx2 = i * new_w + (j + 1) + 1
                idx3 = (i + 1) * new_w + (j + 1) + 1
                idx4 = (i + 1) * new_w + j + 1

                # Quad as two triangles
                f.write(f"f {idx1}/{idx1} {idx2}/{idx2} {idx3}/{idx3}\n")
                f.write(f"f {idx1}/{idx1} {idx3}/{idx3} {idx4}/{idx4}\n")

    return output_path
