"""
Physical and Geometric Shadow Analysis for DepthWizard.
Extracts shadow area, direction vectors, and geometric cues.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


def estimate_shadow_geometry(shadow_mask: np.ndarray) -> Dict[str, Any]:
    """
    Extract geometric cues from a binary shadow mask.

    Returns:
        Dict containing:
            - shadow_present: bool
            - total_area: int
            - shadow_direction_deg: float (approximate orientation in degrees)
            - centroid: Tuple[float, float]
    """
    shadow_pixels = np.count_nonzero(shadow_mask)
    if shadow_pixels == 0:
        return {
            "shadow_present": False,
            "total_area": 0,
            "shadow_direction_deg": 0.0,
            "centroid": (0.0, 0.0)
        }

    # Find connected component contours
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate moments for total centroid
    M = cv2.moments(shadow_mask)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx, cy = 0.0, 0.0

    # Calculate dominant orientation using image moments / principal axes
    if M["m00"] != 0:
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        angle_deg = float(np.degrees(angle_rad))
    else:
        angle_deg = 0.0

    return {
        "shadow_present": True,
        "total_area": int(shadow_pixels),
        "shadow_direction_deg": angle_deg,
        "centroid": (cx, cy)
    }
