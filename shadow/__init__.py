"""
Shadow Detection and Analysis Module for DepthWizard.
"""
from .shadow_detector import detect_shadows
from .shadow_geometry import estimate_shadow_geometry

__all__ = ["detect_shadows", "estimate_shadow_geometry"]
