"""
Confidence-Weighted Fusion of AI Monocular Depth and Physical Shadow Cues.
Produces normalized relative height map and 2D spatial confidence map.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any


def fuse_depth_and_shadows(
    depth_map: np.ndarray,
    shadow_mask: np.ndarray,
    shadow_confidence: float,
    weight_ai: float = 0.85
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fuse AI monocular depth map with shadow geometric cues in a confidence-weighted manner.

    Args:
        depth_map: 2D numpy array of raw AI depth predictions.
        shadow_mask: 2D uint8 numpy array (0 = non-shadow, 255 = shadow).
        shadow_confidence: Global confidence score of shadow detection [0.0, 1.0].
        weight_ai: Base importance weight given to AI monocular depth model.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]:
            - fused_height: 2D numpy array [0.0, 1.0] of relative height.
            - confidence_map: 2D numpy array [0.0, 1.0] of spatial confidence.
            - stats: Dict containing fusion metrics.
    """
    # 1. Normalize depth map to range [0.0, 1.0]
    d_min, d_max = np.min(depth_map), np.max(depth_map)
    if d_max - d_min > 1e-6:
        norm_depth = (depth_map - d_min) / (d_max - d_min)
    else:
        norm_depth = np.zeros_like(depth_map, dtype=np.float32)

    # Note: Depth Anything output usually represents inverse depth or relative proximity (closer/higher objects have higher values)
    
    # 2. Extract spatial shadow gradient / boundary cues
    shadow_norm = (shadow_mask > 0).astype(np.float32)
    shadow_dist = cv2.distanceTransform((shadow_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    shadow_dist_norm = np.clip(shadow_dist / 30.0, 0.0, 1.0)

    # Effective shadow contribution weight dynamically adjusted by shadow_confidence
    effective_shadow_weight = (1.0 - weight_ai) * shadow_confidence

    # 3. Perform confidence-weighted fusion
    if shadow_confidence > 0.15:
        # Structure height cue: regions immediately adjacent to shadows get subtle height enhancement
        shadow_edge_boost = (1.0 - shadow_dist_norm) * shadow_norm * 0.1
        fused_height = norm_depth * (1.0 - effective_shadow_weight) + shadow_edge_boost * effective_shadow_weight
    else:
        # AI depth dominates completely when shadow confidence is low
        fused_height = norm_depth

    # Re-normalize fused height to [0.0, 1.0]
    f_min, f_max = np.min(fused_height), np.max(fused_height)
    if f_max - f_min > 1e-6:
        fused_height = (fused_height - f_min) / (f_max - f_min)

    # 4. Generate Spatial Confidence Map
    # Base confidence comes from AI depth consistency + shadow presence consistency
    ai_confidence = 0.75 + 0.20 * norm_depth  # higher confidence on distinct elevated structures
    shadow_evidence = shadow_confidence * (1.0 - shadow_dist_norm)
    spatial_confidence = np.clip(ai_confidence + 0.15 * shadow_evidence, 0.0, 1.0)

    stats = {
        "shadow_confidence_used": float(shadow_confidence),
        "effective_shadow_weight": float(effective_shadow_weight),
        "fused_min": float(np.min(fused_height)),
        "fused_max": float(np.max(fused_height))
    }

    return fused_height.astype(np.float32), spatial_confidence.astype(np.float32), stats
