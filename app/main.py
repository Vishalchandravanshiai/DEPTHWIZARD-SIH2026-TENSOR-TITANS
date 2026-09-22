"""
Main entry point for DepthWizard application.
"""

import sys
import os

# Ensure DepthWizard root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from depth.depth_model import get_depth_estimator
from app.ui import create_ui


def main():
    print("=" * 60)
    print("      DEPTHWIZARD: Height Estimation & 3D Flythrough")
    print("=" * 60)
    
    # Pre-initialize depth model to avoid latency on first user click
    print("\n[DepthWizard] Pre-loading Depth Anything V2 model...")
    estimator = get_depth_estimator()

    print("\n[DepthWizard] Building Gradio UI...")
    demo = create_ui()
    
    print("\n[DepthWizard] Launching local server...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
