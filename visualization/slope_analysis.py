"""
Slope Analysis and Terrain Surface Gradient Module for DepthWizard.
Computes terrain slope angles (0° - 90°) and renders slope safety heatmaps.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any


def calculate_slope_map(
    height_map: np.ndarray,
    pixel_spacing: float = 1.0
) -> Tuple[np.ndarray, Image.Image, Dict[str, float]]:
    """
    Compute terrain slope in degrees using Sobel spatial gradients.

    Args:
        height_map: 2D numpy array [0.0, 1.0] or metric elevation.
        pixel_spacing: Distance scale per pixel.

    Returns:
        Tuple[np.ndarray, Image.Image, Dict]:
            - slope_degrees: 2D float32 numpy array of slope values in degrees [0, 90].
            - slope_vis: PIL Image color-coded slope heatmap (Green=Flat, Yellow=Moderate, Red=Steep).
            - stats: Dictionary with mean_slope, max_slope, and steep_ratio.
    """
    # Compute spatial gradients using Sobel operators
    grad_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3) / (8.0 * pixel_spacing)
    grad_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3) / (8.0 * pixel_spacing)

    # Slope magnitude in radians and degrees
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Generate Colorized Slope Heatmap
    # Normalize 0-45+ degrees to 0-255 uint8 for colormap
    norm_slope = np.clip(slope_deg / 45.0, 0.0, 1.0)
    uint8_slope = (norm_slope * 255.0).astype(np.uint8)

    # Apply COLORMAP_TURBO (Green -> Yellow -> Red)
    color_mapped = cv2.applyColorMap(uint8_slope, cv2.COLORMAP_TURBO)
    rgb_slope = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)
    slope_vis = Image.fromarray(rgb_slope)

    # Summary Stats
    mean_slope = float(np.mean(slope_deg))
    max_slope = float(np.max(slope_deg))
    steep_pixels = np.count_nonzero(slope_deg > 30.0)
    steep_ratio = float(steep_pixels / float(slope_deg.size))

    stats = {
        "mean_slope_deg": mean_slope,
        "max_slope_deg": max_slope,
        "steep_terrain_ratio": steep_ratio
    }

    return slope_deg, slope_vis, stats
