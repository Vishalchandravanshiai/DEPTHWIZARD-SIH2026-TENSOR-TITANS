"""
Quantitative evaluation metrics (MAE, RMSE, Pearson Correlation) against reference DEM ground truth.
"""

import numpy as np
from typing import Dict, Any, Optional


def calculate_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~np.isnan(prediction) & ~np.isnan(target)
    if np.count_nonzero(mask) == 0:
        return 0.0
    return float(np.mean(np.abs(prediction[mask] - target[mask])))


def calculate_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~np.isnan(prediction) & ~np.isnan(target)
    if np.count_nonzero(mask) == 0:
        return 0.0
    return float(np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2)))


def calculate_metrics(prediction: np.ndarray, target: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Compute MAE, RMSE, and Correlation between prediction and target elevation.
    If target is None, returns status indicating no reference data available.
    """
    if target is None:
        return {
            "reference_available": False,
            "message": "No reference DEM provided. Metrics are unavailable for uncalibrated relative height."
        }

    # Ensure shape alignment
    if prediction.shape != target.shape:
        import cv2
        target_resized = cv2.resize(target, (prediction.shape[1], prediction.shape[0]))
    else:
        target_resized = target

    mask = ~np.isnan(prediction) & ~np.isnan(target_resized)
    if np.count_nonzero(mask) == 0:
        return {
            "reference_available": False,
            "message": "Zero valid overlapping pixels between prediction and reference DEM."
        }

    p_flat = prediction[mask].flatten()
    t_flat = target_resized[mask].flatten()

    mae = calculate_mae(p_flat, t_flat)
    rmse = calculate_rmse(p_flat, t_flat)
    
    # Calculate Pearson Correlation Coefficient
    corr = float(np.corrcoef(p_flat, t_flat)[0, 1]) if len(p_flat) > 1 else 0.0

    return {
        "reference_available": True,
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "num_valid_pixels": int(len(p_flat))
    }
