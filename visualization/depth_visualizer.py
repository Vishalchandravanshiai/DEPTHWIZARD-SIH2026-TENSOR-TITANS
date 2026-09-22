"""
Depth Visualization utilities.
Generates normalized, color-mapped depth and height visualizations.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional


def visualize_depth(
    depth_map: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Convert a 2D depth array into a colorized PIL Image for display.

    Args:
        depth_map: 2D numpy array (float32).
        colormap: OpenCV colormap (default cv2.COLORMAP_INFERNO).
        save_path: Optional path to save image file.

    Returns:
        PIL.Image: Colorized depth map.
    """
    d_min, d_max = np.min(depth_map), np.max(depth_map)
    if d_max - d_min > 1e-6:
        norm_depth = (depth_map - d_min) / (d_max - d_min)
    else:
        norm_depth = np.zeros_like(depth_map, dtype=np.float32)

    uint8_depth = (norm_depth * 255.0).astype(np.uint8)
    color_mapped = cv2.applyColorMap(uint8_depth, colormap)
    
    # Convert BGR to RGB
    rgb_depth = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)
    img_out = Image.fromarray(rgb_depth)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_out.save(save_path)

    return img_out
