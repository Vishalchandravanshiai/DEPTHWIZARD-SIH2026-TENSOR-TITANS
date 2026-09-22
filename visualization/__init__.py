"""
Visualization Package for DepthWizard.
Provides 2D colormaps, interactive Plotly 3D terrain surfaces, slope maps, and confidence maps.
"""
from .depth_visualizer import visualize_depth
from .terrain_3d import create_3d_terrain_figure
from .confidence_map import visualize_confidence
from .slope_analysis import calculate_slope_map

__all__ = ["visualize_depth", "create_3d_terrain_figure", "visualize_confidence", "calculate_slope_map"]
