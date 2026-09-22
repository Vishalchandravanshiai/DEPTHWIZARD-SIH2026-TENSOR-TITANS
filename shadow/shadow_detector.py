"""
OpenCV-based Shadow Detector for Aerial/Satellite imagery.
Detects dark regions, calculates shadow masks and confidence scores.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Union


def detect_shadows(
    image: Union[Image.Image, np.ndarray],
    threshold_factor: float = 0.6
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Detect shadows in an RGB image using LAB and HSV color space thresholding.

    Args:
        image: PIL Image or RGB Numpy array.
        threshold_factor: Sensitivity for shadow thresholding.

    Returns:
        Tuple[np.ndarray, float, Dict]:
            - shadow_mask: uint8 array (0 for non-shadow, 255 for shadow)
            - shadow_confidence: float [0.0, 1.0] representing detection reliability
            - metadata: Dict of debug stats (shadow_area_ratio, mean_intensity, etc.)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l_channel = lab[:, :, 0].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32)

    # Calculate shadow score index: C_s = (H + 1) / (V + 1) or low L with high S
    # Aerial shadow index: low L & (S+1)/(V+1) ratio higher than background average
    mean_l = np.mean(l_channel)
    std_l = np.std(l_channel)

    # Threshold for low luminance
    shadow_thresh = max(10.0, mean_l - threshold_factor * std_l)
    raw_mask = (l_channel < shadow_thresh).astype(np.uint8) * 255

    # Morphological cleaning to remove isolated noise pixels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_small)
    shadow_mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)

    # Compute Shadow Area Ratio
    total_pixels = shadow_mask.size
    shadow_pixels = np.count_nonzero(shadow_mask)
    shadow_ratio = shadow_pixels / float(total_pixels)

    # Calculate Confidence Score based on contrast difference between shadow and non-shadow regions
    if shadow_pixels > 50 and shadow_ratio < 0.8:
        shadow_l_mean = np.mean(l_channel[shadow_mask > 0])
        non_shadow_l_mean = np.mean(l_channel[shadow_mask == 0])
        contrast_diff = (non_shadow_l_mean - shadow_l_mean) / 255.0

        # High confidence if distinct shadow regions exist (5% to 40% of image) with clear contrast
        if 0.02 <= shadow_ratio <= 0.5 and contrast_diff > 0.15:
            confidence = min(1.0, contrast_diff * 2.5)
        else:
            confidence = max(0.1, min(0.5, contrast_diff))
    else:
        confidence = 0.05  # Low confidence if negligible or total darkness

    metadata = {
        "shadow_ratio": float(shadow_ratio),
        "mean_luminance": float(mean_l),
        "shadow_pixels": int(shadow_pixels),
        "confidence": float(confidence)
    }

    return shadow_mask, confidence, metadata
