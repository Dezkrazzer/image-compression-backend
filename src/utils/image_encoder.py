"""
image_encoder.py

Encode NumPy images into
JPEG Base64.
"""

from __future__ import annotations

import io
import base64

import numpy as np

from PIL import Image


def encode_base64_jpeg(
    image: np.ndarray,
    *,
    quality: int = 90,
) -> tuple[str, bytes]:
    """
    Encode image into JPEG Base64.

    Returns
    -------
    base64_string

    jpeg_bytes
    """

    pil = Image.fromarray(image)

    output = io.BytesIO()

    pil.save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=0,
    )

    jpeg_bytes = output.getvalue()

    encoded = base64.b64encode(
        jpeg_bytes
    ).decode("utf-8")

    return encoded, jpeg_bytes