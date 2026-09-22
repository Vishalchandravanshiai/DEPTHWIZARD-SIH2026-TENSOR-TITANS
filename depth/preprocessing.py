"""
Preprocessing utilities for DepthWizard.
Handles image validation, channel conversion, and aspect-ratio preserving resizing.
"""

import os
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image


def load_h5_data(file_path: str) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Extract RGB optical image array and optional ground-truth depth from an HDF5 (.h5 / .hdf5) file.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py library is required to read HDF5 (.h5) files. Run 'pip install h5py'.")

    with h5py.File(file_path, "r") as hf:
        rgb_arr = None
        gt_depth = None

        # Search for RGB or optical dataset keys
        for key in ["rgb", "image", "optical", "img", "data", "colors"]:
            if key in hf:
                rgb_arr = np.array(hf[key])
                break

        if rgb_arr is None:
            # Fallback: find first 2D or 3D dataset array
            for key in hf.keys():
                item = hf[key]
                if isinstance(item, h5py.Dataset):
                    arr = np.array(item)
                    if arr.ndim in (2, 3):
                        rgb_arr = arr
                        break

        if rgb_arr is None:
            raise ValueError(f"No valid image dataset found inside HDF5 file '{file_path}'. Keys: {list(hf.keys())}")

        # Search for optional ground truth depth/elevation dataset
        for key in ["depth", "elevation", "gt", "dem", "dsm", "target", "height"]:
            if key in hf:
                gt_depth = np.array(hf[key]).astype(np.float32)
                break

    # Process RGB numpy array to uint8
    if rgb_arr.ndim == 2:
        # Grayscale to RGB
        norm = ((rgb_arr - rgb_arr.min()) / max(1e-6, rgb_arr.max() - rgb_arr.min()) * 255.0).astype(np.uint8)
        img_out = Image.fromarray(norm).convert("RGB")
    elif rgb_arr.ndim == 3:
        # Channel transpose if shape is (C, H, W)
        if rgb_arr.shape[0] in (1, 3, 4) and rgb_arr.shape[2] not in (1, 3, 4):
            rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
        
        if rgb_arr.dtype != np.uint8:
            norm = ((rgb_arr - rgb_arr.min()) / max(1e-6, rgb_arr.max() - rgb_arr.min()) * 255.0).astype(np.uint8)
            img_out = Image.fromarray(norm[:, :, :3]).convert("RGB")
        else:
            img_out = Image.fromarray(rgb_arr[:, :, :3]).convert("RGB")
    else:
        raise ValueError(f"Invalid HDF5 image dataset shape: {rgb_arr.shape}")

    return img_out, gt_depth


def load_and_validate_image(
    image_input: Union[str, np.ndarray, Image.Image]
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Load image from path, HDF5 file, numpy array, or PIL Image, validate format,
    and convert to RGB PIL Image.

    Returns:
        Tuple[Image.Image, Optional[np.ndarray]]: (RGB Image, GT Depth if present)
    """
    gt_depth = None

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        
        ext = os.path.splitext(image_input)[1].lower()
        if ext in [".h5", ".hdf5", ".he5"]:
            img, gt_depth = load_h5_data(image_input)
        else:
            img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img = Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Invalid image array shape: {image_input.shape}")
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img, gt_depth


def preprocess_image(
    image_input: Union[str, np.ndarray, Image.Image],
    max_dim: int = 1280
) -> Tuple[Image.Image, Tuple[int, int], Optional[np.ndarray]]:
    """
    Preprocess image for Depth Estimation.
    - Validates and converts to RGB (supports JPG, PNG, GeoTIFF, HDF5 .h5).
    - Resizes while preserving aspect ratio if larger than max_dim.
    
    Returns:
        Tuple[Image.Image, Tuple[int, int], Optional[np.ndarray]]: (preprocessed PIL image, original (width, height), gt_depth if present)
    """
    img, gt_depth = load_and_validate_image(image_input)
    orig_size = img.size  # (width, height)

    w, h = orig_size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    return img, orig_size, gt_depth
