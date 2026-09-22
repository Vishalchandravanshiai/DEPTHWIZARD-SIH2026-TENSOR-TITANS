"""
Interactive 3D Terrain Visualization using Plotly.
Generates 3D surface mesh from height maps with optional RGB texture overlay.
"""

import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
from typing import Optional, Union


def create_3d_terrain_figure(
    height_map: np.ndarray,
    rgb_image: Optional[Union[Image.Image, np.ndarray]] = None,
    max_resolution: int = 180,
    height_scale: float = 0.35,
    title: str = "3D Relative Surface Reconstruction",
    z_label: str = "Relative Height",
    camera_mode: str = "45_perspective"
) -> go.Figure:
    """
    Generate an interactive Plotly 3D Surface figure from a height map.

    Args:
        height_map: 2D numpy array (height/elevation values).
        rgb_image: Optional PIL Image or RGB Numpy array to texture/color the mesh.
        max_resolution: Maximum grid dimension (downsampling for fast rendering).
        height_scale: Vertical exaggeration multiplier.
        title: Plot title.
        z_label: Label for Z axis.
        camera_mode: Camera perspective preset ('45_perspective', 'drone_flythrough', 'top_down', 'profile').

    Returns:
        plotly.graph_objects.Figure: Interactive 3D Plotly figure.
    """
    h, w = height_map.shape

    # Downsample matrix for performance if larger than max_resolution
    scale = min(1.0, max_resolution / float(max(h, w)))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        z_grid = cv2.resize(height_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
        z_grid = height_map.copy()

    x_coords = np.linspace(0, 1, new_w)
    y_coords = np.linspace(0, 1, new_h)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Invert Y so top of image corresponds to top of plot
    Z = z_grid * height_scale

    # Process RGB texture colors if provided
    surfacecolor = None
    colorscale = "Viridis"
    cmin, cmax = None, None
    showscale = True

    if rgb_image is not None:
        if isinstance(rgb_image, Image.Image):
            rgb_pil = rgb_image
        else:
            rgb_pil = Image.fromarray(rgb_image)

        # Resize PIL image to match grid dimensions
        rgb_resized_pil = rgb_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Quantize to 256 color palette for true RGB satellite texture mapping in Plotly
        quantized = rgb_resized_pil.quantize(colors=256)
        palette = quantized.getpalette()  # [r0, g0, b0, r1, g1, b1, ...]
        indices = np.array(quantized)     # 2D array of palette indices
        
        num_colors = max(int(indices.max()) + 1, 2)
        custom_colorscale = []
        for i in range(num_colors):
            r, g, b = palette[i * 3 : i * 3 + 3]
            val = i / float(num_colors - 1)
            custom_colorscale.append([val, f"rgb({r},{g},{b})"])

        surfacecolor = indices
        colorscale = custom_colorscale
        cmin = 0
        cmax = num_colors - 1
        showscale = False

    # Build Plotly Surface trace with true satellite RGB texture
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=surfacecolor,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        showscale=showscale,
        colorbar=dict(title=z_label, len=0.7) if showscale else None,
        lighting=dict(ambient=0.75, diffuse=0.85, roughness=0.3, specular=0.15)
    )

    fig = go.Figure(data=[surface])

    # Select Camera Presets
    camera_presets = {
        "drone_flythrough": dict(eye=dict(x=-1.6, y=-1.6, z=0.45)),  # Low drone angle
        "top_down": dict(eye=dict(x=0.0, y=0.0, z=2.4)),             # Direct overhead
        "profile": dict(eye=dict(x=-2.2, y=0.0, z=0.3)),             # Structural profile
        "45_perspective": dict(eye=dict(x=-1.35, y=-1.35, z=1.2))   # Standard isometric
    }
    camera_eye = camera_presets.get(camera_mode, camera_presets["45_perspective"])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="X (East-West)", showgrid=False, zeroline=False),
            yaxis=dict(title="Y (North-South)", showgrid=False, zeroline=False, autorange="reversed"),
            zaxis=dict(title=z_label),
            aspectratio=dict(x=1.0, y=float(new_h)/float(new_w), z=height_scale),
            camera=camera_eye
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        autosize=True
    )

    return fig
