"""
Confidence / Evidence Map Visualizer.
Renders confidence array as a clear color map (Red=Low Confidence, Green=High Confidence).
"""

import cv2
import numpy as np
from PIL import Image


def visualize_confidence(confidence_map: np.ndarray) -> Image.Image:
    """
    Colorize spatial confidence map using TURBO / JET colormap.

    Args:
        confidence_map: 2D numpy array [0.0, 1.0].

    Returns:
        PIL.Image: Colored confidence heatmap.
    """
    conf_clipped = np.clip(confidence_map, 0.0, 1.0)
    uint8_conf = (conf_clipped * 255.0).astype(np.uint8)

    # Use COLORMAP_TURBO or COLORMAP_JET for confidence
    color_mapped = cv2.applyColorMap(uint8_conf, cv2.COLORMAP_TURBO)
    rgb_conf = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_conf)
