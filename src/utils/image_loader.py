"""
image_loader.py

Load image bytes into a NumPy RGB array.

Supported:
- JPG
- JPEG
- PNG
- WEBP
- TIFF
- BMP
- HEIC
- HEIF
- RAW
"""

from __future__ import annotations

import io

import numpy as np
import rawpy

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

Image.MAX_IMAGE_PIXELS = None


def load_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into RGB NumPy array.
    """

    buffer = io.BytesIO(image_bytes)

    try:

        with Image.open(buffer) as img:

            img = img.convert("RGB")

            return np.array(img)

    except UnidentifiedImageError:

        buffer.seek(0)

        try:

            with rawpy.imread(buffer) as raw:

                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                )

                return rgb

        except Exception as exc:

            raise ValueError(
                "Unsupported or corrupted image."
            ) from exc