"""
Evaluation script wrapper for DepthWizard.
Runs metrics reporting on depth predictions against reference DEM datasets.
"""

from typing import Dict, Any, Optional
import numpy as np
from .metrics import calculate_metrics


def run_evaluation(
    predicted_height: np.ndarray,
    reference_dem: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run complete quantitative evaluation.
    """
    return calculate_metrics(predicted_height, reference_dem)
