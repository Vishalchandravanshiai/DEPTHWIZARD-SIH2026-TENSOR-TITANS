"""
Reference DEM / GeoTIFF Loader for DepthWizard.
Loads reference digital elevation models for metric calibration if available.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any


class DEMLoader:
    def __init__(self, dem_path: Optional[str] = None):
        self.dem_path = dem_path
        self.dem_data: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}

    def load_dem(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load Digital Elevation Model file (GeoTIFF, NPY, or TIFF).
        Falls back gracefully if rasterio is not installed.
        """
        if not os.path.exists(file_path):
            print(f"[DEMLoader] File not found: {file_path}")
            return None

        self.dem_path = file_path

        # Try loading via rasterio if available
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                self.dem_data = src.read(1).astype(np.float32)
                self.metadata = {
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "nodata": src.nodata
                }
                print(f"[DEMLoader] Loaded GeoTIFF DEM via rasterio: shape={self.dem_data.shape}")
                return self.dem_data
        except ImportError:
            print("[DEMLoader] rasterio not installed. Trying fallback loaders (PIL / OpenCV / NumPy)...")
        except Exception as e:
            print(f"[DEMLoader] rasterio load failed: {e}. Trying fallback loaders...")

        # Fallback for .npy or image files
        try:
            if file_path.endswith(".npy"):
                self.dem_data = np.load(file_path).astype(np.float32)
            else:
                from PIL import Image
                img = Image.open(file_path)
                self.dem_data = np.array(img).astype(np.float32)
            print(f"[DEMLoader] Loaded DEM via fallback loader: shape={self.dem_data.shape}")
            return self.dem_data
        except Exception as err:
            print(f"[DEMLoader Error] Failed to load DEM file '{file_path}': {err}")
            return None
