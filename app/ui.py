"""
Gradio Interface for DepthWizard.
Provides an intuitive UI for single-view height estimation, slope analysis, 
and 3D flythrough terrain visualization aligned with SIH 2026 problem statement.
"""

import os
import numpy as np
import gradio as gr
from PIL import Image

from depth.depth_model import get_depth_estimator
from depth.preprocessing import preprocess_image
from shadow.shadow_detector import detect_shadows
from shadow.shadow_geometry import estimate_shadow_geometry
from fusion.fusion import fuse_depth_and_shadows
from visualization.depth_visualizer import visualize_depth
from visualization.terrain_3d import create_3d_terrain_figure
from visualization.confidence_map import visualize_confidence
from visualization.slope_analysis import calculate_slope_map
from calibration.exporter import export_obj_mesh, export_elevation_npy, export_elevation_png
from evaluation.metrics import calculate_metrics


# Global storage for export generation
_last_state = {
    "fused_height": None,
    "pil_img": None
}


def process_image(
    input_file,
    camera_preset="45_perspective",
    height_scale=0.35,
    progress=gr.Progress(track_tqdm=False)
):
    """
    Main processing pipeline for DepthWizard.
    """
    if input_file is None:
        return (
            None, None, None, None, None, None, None,
            "⚠️ Please upload a satellite image or HDF5 (.h5) dataset file first.",
            "N/A", "N/A", "N/A", None, None, None
        )

    try:
        # Extract file path string from Gradio file object
        file_path = input_file.name if hasattr(input_file, "name") else input_file

        # Step 1: Preprocess and Validate Image (JPG, PNG, TIFF, HDF5 .h5)
        progress(0.1, desc="Loading and validating optical image / HDF5 dataset...")
        pil_img, orig_size, gt_depth = preprocess_image(file_path, max_dim=1280)
        
        # Step 2: Run Depth Anything V2 Monocular Depth Estimation
        progress(0.3, desc="Running Depth Anything V2 monocular depth backbone...")
        estimator = get_depth_estimator()
        depth_map = estimator.predict_depth(pil_img)

        # Step 3: Detect Shadows and Analyze Geometry
        progress(0.5, desc="Extracting physical shadow cues & LAB/HSV geometry...")
        shadow_mask, shadow_conf, shadow_meta = detect_shadows(pil_img)
        shadow_geom = estimate_shadow_geometry(shadow_mask)

        # Step 4: Confidence-Weighted Fusion
        progress(0.65, desc="Fusing AI depth predictions with shadow geometry...")
        fused_height, spatial_confidence, fusion_stats = fuse_depth_and_shadows(
            depth_map=depth_map,
            shadow_mask=shadow_mask,
            shadow_confidence=shadow_conf
        )

        # Step 5: Evaluation against GT if HDF5/DEM GT Depth was loaded
        gt_metrics_md = ""
        if gt_depth is not None:
            eval_res = calculate_metrics(fused_height, gt_depth)
            if eval_res.get("reference_available"):
                gt_metrics_md = f"""
- **Ground Truth Evaluation:** `MAE: {eval_res['mae']:.4f}`, `RMSE: {eval_res['rmse']:.4f}`, `Correlation r: {eval_res['correlation']:.4f}`
"""

        # Step 6: Slope & Terrain Surface Analysis
        progress(0.75, desc="Calculating terrain slope angles and surface gradients...")
        slope_deg, slope_vis, slope_stats = calculate_slope_map(fused_height)

        # Save to global cache for export triggers
        _last_state["fused_height"] = fused_height
        _last_state["pil_img"] = pil_img

        # Step 7: Visualizations & 2D Maps
        progress(0.85, desc="Generating colorized 2D elevation, shadow, and confidence maps...")
        depth_vis = visualize_depth(depth_map)
        shadow_vis = Image.fromarray(shadow_mask)
        fused_vis = visualize_depth(fused_height)
        confidence_vis = visualize_confidence(spatial_confidence)

        # Step 8: 3D Interactive Flythrough Surface Mesh
        progress(0.95, desc="Generating satellite-textured 3D terrain surface mesh...")
        fig_3d = create_3d_terrain_figure(
            height_map=fused_height,
            rgb_image=pil_img,
            height_scale=height_scale,
            title="3D Satellite Terrain Reconstruction & Flythrough",
            z_label="Relative Height (rDSM)",
            camera_mode=camera_preset
        )

        # Generate default exported files for downloads tab
        os.makedirs("outputs", exist_ok=True)
        obj_file = os.path.join("outputs", "depthwizard_3d_mesh.obj")
        npy_file = os.path.join("outputs", "depthwizard_rdsm.npy")
        png_file = os.path.join("outputs", "depthwizard_rdsm_16bit.png")

        export_obj_mesh(fused_height, obj_file, height_scale=height_scale)
        export_elevation_npy(fused_height, npy_file)
        export_elevation_png(fused_height, png_file)

        progress(1.0, desc="Done.")

        # Summary Status Output
        status_md = f"""
### 📊 Pipeline Summary & Execution Diagnostics

- **Operating Mode:** `Relative Digital Surface Model (rDSM)`
- **AI Backbone:** `Depth Anything V2 (Small)`
- **Inference Hardware:** `{estimator.device.type.upper()}` ({torch_device_info()})
- **Resolution:** `{orig_size[0]} x {orig_size[1]} px`
- **Shadow Cue Confidence:** `{shadow_conf:.2f}` ({'High' if shadow_conf > 0.4 else 'Low/Moderate'})
- **Shadow Area Coverage:** `{shadow_meta['shadow_ratio']*100:.1f}%`{gt_metrics_md}

> ℹ️ **Geospatial Mode Note:** Normal JPG/PNG/H5 optical images produce relative height representations (**rDSM**). Georeferenced GeoTIFF files with SRTM/GCP reference data enable absolute metric elevation calibration (**DSM**).
"""

        return (
            pil_img,
            depth_vis,
            shadow_vis,
            fused_vis,
            confidence_vis,
            slope_vis,
            fig_3d,
            status_md,
            f"{slope_stats['mean_slope_deg']:.1f}°",
            f"{slope_stats['max_slope_deg']:.1f}°",
            f"{slope_stats['steep_terrain_ratio']*100:.1f}%",
            obj_file,
            npy_file,
            png_file
        )

    except Exception as e:
        import traceback
        err_msg = f"❌ **Error processing image:** {str(e)}\n```\n{traceback.format_exc()}\n```"
        print(f"[DepthWizard Error] {e}")
        return (
            None, None, None, None, None, None, None,
            err_msg, "N/A", "N/A", "N/A", None, None, None
        )


def torch_device_info() -> str:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def update_camera_view(camera_preset, height_scale):
    """Dynamically update camera view on existing height map."""
    if _last_state["fused_height"] is None:
        return None
    fig = create_3d_terrain_figure(
        height_map=_last_state["fused_height"],
        rgb_image=_last_state["pil_img"],
        height_scale=height_scale,
        title="3D Satellite Terrain Reconstruction & Flythrough",
        z_label="Relative Height (rDSM)",
        camera_mode=camera_preset
    )
    return fig


def create_ui() -> gr.Blocks:
    """
    Build SIH 2026 Compliant Gradio UI interface for DepthWizard.
    """
    custom_css = """
    .gradio-container { font-family: 'Segoe UI', system-ui, sans-serif; }
    .header-box { text-align: center; margin-bottom: 15px; }
    """

    with gr.Blocks(title="DepthWizard - Single-View Height Estimation & 3D Flythrough") as demo:
        with gr.Row(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🧙‍♂️ DEPTHWIZARD
                ### Single-View Height Estimation & 3D Flythrough Pipeline
                *SIH 2026 Software Prototype | Earthflow GAMUS Dataset & Depth Anything V2 Backbone*
                """
            )

        with gr.Row():
            # Left Control Column
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Upload Satellite File (.jpg, .png, .tif, .h5, .hdf5)",
                    file_count="single",
                    file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".h5", ".hdf5", ".he5"]
                )
                
                with gr.Accordion("⚙️ 3D Render & Camera Settings", open=True):
                    camera_dropdown = gr.Dropdown(
                        choices=[
                            ("45° Perspective View", "45_perspective"),
                            ("🛸 Low-Altitude Drone Flythrough", "drone_flythrough"),
                            ("🛰️ Top-Down Overhead Aerial", "top_down"),
                            ("📐 Structural Side Profile", "profile")
                        ],
                        value="45_perspective",
                        label="3D Camera Angle Preset"
                    )
                    
                    height_scale_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.35,
                        step=0.05,
                        label="Vertical Elevation Exaggeration"
                    )

                analyze_btn = gr.Button("🚀 RUN ELEVATION & 3D PIPELINE", variant="primary", size="lg")
                
                status_output = gr.Markdown(
                    """
                    *Upload an optical satellite image and click **RUN ELEVATION & 3D PIPELINE** to extract heights, shadows, slopes, and interactive 3D terrain.*
                    """
                )

            # Right Output Display Column
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🌋 3D Interactive Flythrough"):
                        plot_3d = gr.Plot(label="Interactive 3D Satellite Terrain (WebGL)")
                        
                    with gr.TabItem("🗺️ Elevation & Shadow Maps"):
                        with gr.Row():
                            orig_view = gr.Image(label="Original Optical RGB Image", type="pil")
                            depth_view = gr.Image(label="Monocular Depth Map (Depth Anything V2)", type="pil")
                        with gr.Row():
                            shadow_view = gr.Image(label="Shadow Detection Mask", type="pil")
                            fused_view = gr.Image(label="Fused Relative Surface Model (rDSM)", type="pil")

                    with gr.TabItem("📐 Terrain Slope & Structural Analysis"):
                        with gr.Row():
                            slope_view = gr.Image(label="Terrain Slope Heatmap (0°-90°)", type="pil")
                            with gr.Column():
                                gr.Markdown("### 📊 Structural Terrain Metrics")
                                mean_slope_txt = gr.Textbox(label="Mean Slope Angle", value="N/A")
                                max_slope_txt = gr.Textbox(label="Max Slope Angle", value="N/A")
                                steep_ratio_txt = gr.Textbox(label="Steep Terrain Area (>30°)", value="N/A")
                                gr.Markdown(
                                    """
                                    - **Green:** Flat / Gentle Terrain (0° - 15°)
                                    - **Yellow:** Moderate Slope (15° - 30°)
                                    - **Red:** Steep Structures / Cliff Edges (>30°)
                                    """
                                )

                    with gr.TabItem("🛡️ Confidence Evidence Map"):
                        conf_view = gr.Image(label="Spatial Evidence / Confidence Heatmap", type="pil")

                    with gr.TabItem("💾 3D & Geospatial Asset Exports"):
                        gr.Markdown("### 📦 Download Reconstructed Assets")
                        with gr.Row():
                            obj_download = gr.File(label="Wavefront 3D OBJ Mesh (.obj)")
                            npy_download = gr.File(label="Raw Elevation Matrix (.npy)")
                            png_download = gr.File(label="16-Bit HDR Elevation Image (.png)")
                        gr.Markdown("*The exported Wavefront .obj file can be imported into Three.js, Unity, Blender, or WebGL engines.*")

                    with gr.TabItem("ℹ️ Dataset & Reference Calibration"):
                        gr.Markdown(
                            """
                            ### 🗄️ Earthflow GAMUS Dataset & Scale Calibration Strategy

                            - **Monocular Backbone:** Pretrained *Depth Anything V2* adapted for top-down remote sensing imagery.
                            - **Recommended Fine-Tuning Dataset:** **GAMUS Dataset** (Hugging Face: `earthflow/GAMUS`).
                            - **Scale Calibration Framework:**
                              $$\\text{Elevation}_{\\text{metric}} = \\text{scale} \\times \\text{rDSM} + \\text{offset}$$
                              When georeferenced GeoTIFF and reference DEM data (e.g. SRTM 30m) or Ground Control Points (GCPs) are supplied, the calibrator fits scale and offset parameters via least-squares linear regression.
                            """
                        )

        # Wire Event Handlers
        analyze_btn.click(
            fn=process_image,
            inputs=[input_file, camera_dropdown, height_scale_slider],
            outputs=[
                orig_view,
                depth_view,
                shadow_view,
                fused_view,
                conf_view,
                slope_view,
                plot_3d,
                status_output,
                mean_slope_txt,
                max_slope_txt,
                steep_ratio_txt,
                obj_download,
                npy_download,
                png_download
            ]
        )

        camera_dropdown.change(
            fn=update_camera_view,
            inputs=[camera_dropdown, height_scale_slider],
            outputs=[plot_3d]
        )

        height_scale_slider.change(
            fn=update_camera_view,
            inputs=[camera_dropdown, height_scale_slider],
            outputs=[plot_3d]
        )

    return demo
