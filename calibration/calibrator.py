"""
Scale and Offset Height Calibrator.
Converts relative height values to estimated metric elevation (meters) when reference GCP/DEM data is present.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class HeightCalibrator:
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale = scale
        self.offset = offset
        self.is_calibrated = False

    def fit(self, relative_height: np.ndarray, reference_elevation: np.ndarray) -> Dict[str, float]:
        """
        Fit linear calibration parameters using reference elevation data:
        metric_elevation = scale * relative_height + offset
        """
        valid_mask = ~np.isnan(reference_elevation) & ~np.isnan(relative_height)
        if np.count_nonzero(valid_mask) < 2:
            raise ValueError("Insufficient valid reference points for elevation calibration.")

        rel_pts = relative_height[valid_mask].flatten()
        ref_pts = reference_elevation[valid_mask].flatten()

        # Least squares line fit
        slope, intercept = np.polyfit(rel_pts, ref_pts, 1)
        self.scale = float(slope)
        self.offset = float(intercept)
        self.is_calibrated = True

        return {"scale": self.scale, "offset": self.offset}

    def calibrate(self, relative_height: np.ndarray) -> np.ndarray:
        """
        Transform relative height array [0.0, 1.0] to estimated metric elevation (in meters).
        """
        if not self.is_calibrated:
            return relative_height  # Return unchanged if not calibrated
        return self.scale * relative_height + self.offset
