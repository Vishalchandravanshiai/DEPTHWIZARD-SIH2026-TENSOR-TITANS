"""
Depth Anything V2 Model Wrapper for DepthWizard.
Provides singleton loading and VRAM-aware inference.
"""

import sys
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    _instance: Optional["DepthEstimator"] = None

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Initialize Depth Anything V2 model.
        Automatically selects CUDA if available, with CPU fallback.
        """
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._print_hardware_info()

        print(f"[DepthWizard] Loading model '{self.model_id}' on {self.device.type.upper()}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            print("[DepthWizard] Model loaded successfully.")
        except Exception as e:
            print(f"[DepthWizard] Error loading model on {self.device}: {e}")
            if self.device.type == "cuda":
                print("[DepthWizard] Falling back to CPU...")
                self.device = torch.device("cpu")
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()
                print("[DepthWizard] Model loaded on CPU successfully.")
            else:
                raise e

    def _print_hardware_info(self) -> None:
        """Print startup system and GPU information."""
        print("=" * 50)
        print("DEPTHWIZARD - HARDWARE INITIALIZATION")
        print("=" * 50)
        print(f"Device: {self.device.type.upper()}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU Name: {gpu_name}")
            print(f"Total VRAM: {vram_total:.2f} GB")
        else:
            print("GPU Name: N/A (Running on CPU)")
        print(f"Model ID: {self.model_id}")
        print("=" * 50)

    @torch.no_grad()
    def predict_depth(self, image: Image.Image) -> np.ndarray:
        """
        Predict relative depth map from PIL Image.

        Returns:
            np.ndarray: 2D numpy array (float32) of predicted relative depth values.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
        
        orig_w, orig_h = image.size

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            # Resize predicted depth back to original image dimensions
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_np = prediction.cpu().numpy().astype(np.float32)
            return depth_np

        except torch.cuda.OutOfMemoryError:
            print("[DepthWizard WARNING] CUDA Out of Memory during inference! Falling back to CPU for this request...")
            torch.cuda.empty_cache()
            
            # Temporary CPU prediction fallback
            self.model.to("cpu")
            inputs = self.processor(images=image, return_tensors="pt").to("cpu")
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_np = prediction.cpu().numpy().astype(np.float32)
            # Move back to GPU if possible
            try:
                self.model.to(self.device)
            except Exception:
                pass
            return depth_np


_global_estimator: Optional[DepthEstimator] = None


def get_depth_estimator(model_id: str = "depth-anything/Depth-Anything-V2-Small-hf") -> DepthEstimator:
    """
    Get or create the global singleton DepthEstimator instance.
    """
    global _global_estimator
    if _global_estimator is None:
        _global_estimator = DepthEstimator(model_id=model_id)
    return _global_estimator
