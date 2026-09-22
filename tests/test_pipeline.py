"""
End-to-end integration test script for DepthWizard pipeline.
Generates a sample satellite/aerial-like image with a building and shadows,
and runs depth estimation, shadow detection, fusion, confidence map, and 3D rendering.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from depth.depth_model import get_depth_estimator
from depth.preprocessing import preprocess_image
from shadow.shadow_detector import detect_shadows
from shadow.shadow_geometry import estimate_shadow_geometry
from fusion.fusion import fuse_depth_and_shadows
from visualization.depth_visualizer import visualize_depth
from visualization.terrain_3d import create_3d_terrain_figure
from visualization.confidence_map import visualize_confidence


def create_sample_aerial_image(width=512, height=512) -> Image.Image:
    """Create a synthetic satellite/aerial image of a building casting a shadow on ground."""
    # Base ground (light terrain green/beige)
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    img_arr[:, :] = [120, 140, 100]  # Ground terrain

    # Draw a building structure (bright grey roof)
    # Building box: x: 150..300, y: 150..300
    img_arr[150:300, 150:300] = [210, 215, 220]

    # Draw building shadow (dark region cast to the right/down)
    # Shadow box: x: 300..360, y: 170..320
    img_arr[170:320, 300:360] = [35, 40, 35]

    # Add minor noise for texture
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr)


def test_full_pipeline():
    print("[Test Pipeline] Starting DepthWizard E2E test...")
    
    # 1. Create sample image
    sample_path = os.path.join("data", "samples", "sample_aerial.png")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    sample_img = create_sample_aerial_image()
    sample_img.save(sample_path)
    print(f"[Test Pipeline] Created sample aerial image at {sample_path}")

    # 2. Preprocess
    pil_img, orig_size, gt_depth = preprocess_image(sample_path)
    print(f"[Test Pipeline] Image loaded and validated: {orig_size}")

    # 2b. Test HDF5 dataset loader (.h5)
    h5_sample_path = os.path.join("data", "samples", "test_sample.h5")
    if os.path.exists(h5_sample_path):
        h5_img, h5_size, h5_gt = preprocess_image(h5_sample_path)
        print(f"[Test Pipeline] HDF5 file successfully loaded: size={h5_size}, has_gt={h5_gt is not None}")

    # 3. Depth Anything V2 Inference
    print("[Test Pipeline] Running Depth Anything V2 model...")
    estimator = get_depth_estimator()
    depth_map = estimator.predict_depth(pil_img)
    print(f"[Test Pipeline] Depth map predicted: shape={depth_map.shape}, min={depth_map.min():.4f}, max={depth_map.max():.4f}")

    # 4. Shadow Detection
    shadow_mask, shadow_conf, shadow_meta = detect_shadows(pil_img)
    shadow_geom = estimate_shadow_geometry(shadow_mask)
    print(f"[Test Pipeline] Shadow detection: ratio={shadow_meta['shadow_ratio']:.3f}, confidence={shadow_conf:.3f}")

    # 5. Fusion
    fused_height, spatial_conf, stats = fuse_depth_and_shadows(depth_map, shadow_mask, shadow_conf)
    print(f"[Test Pipeline] Fused height shape: {fused_height.shape}")

    # 6. Visualization & 3D Surface
    for sub in ["depth", "shadows", "height", "confidence", "3d"]:
        os.makedirs(os.path.join("outputs", sub), exist_ok=True)

    depth_vis = visualize_depth(depth_map, save_path=os.path.join("outputs", "depth", "test_depth.png"))
    shadow_vis = Image.fromarray(shadow_mask)
    shadow_vis.save(os.path.join("outputs", "shadows", "test_shadow.png"))
    fused_vis = visualize_depth(fused_height, save_path=os.path.join("outputs", "height", "test_height.png"))
    conf_vis = visualize_confidence(spatial_conf)
    conf_vis.save(os.path.join("outputs", "confidence", "test_confidence.png"))

    fig = create_3d_terrain_figure(fused_height, rgb_image=pil_img)
    print(f"[Test Pipeline] 3D Figure generated successfully.")
    
    print("\n[SUCCESS] [Test Pipeline] ALL STAGES (Phase 1 to Phase 5) COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    test_full_pipeline()
