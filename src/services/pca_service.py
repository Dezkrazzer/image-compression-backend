"""
pca_service.py

Adaptive Tile-Based PCA Image Compression Service.
"""

from __future__ import annotations

import time

from src.core.device import DEVICE, device_name
from src.core.tile_engine import (
    batch_tiles,
    merge_tiles,
    select_batch_size,
    select_tile_size,
    split_tiles,
)

from src.core.pca_math import (
    cleanup,
    numpy_batch_to_tensor,
    run_batch_pca,
    tensor_batch_to_numpy,
)

from src.utils.image_encoder import encode_base64_jpeg
from src.utils.image_loader import load_image


def compress_image_pca(
    image_bytes: bytes,
    components: int,
) -> dict:

    if components < 1:
        raise ValueError(
            "Components must be greater than zero."
        )

    start = time.perf_counter()

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------

    image = load_image(image_bytes)

    height, width, _ = image.shape

    components = min(
        components,
        min(height, width),
    )

    # --------------------------------------------------
    # Tile Configuration
    # --------------------------------------------------

    tile_size = select_tile_size(
        width,
        height,
    )

    batch_size = select_batch_size()

    tiles = split_tiles(
        image,
        tile_size,
    )

    processed_tiles = []

    pixel_difference = 0.0
    batch_count = 0

    # --------------------------------------------------
    # PCA Processing
    # --------------------------------------------------

    for tile_batch in batch_tiles(
        tiles,
        batch_size,
    ):

        batch_count += 1

        images = [
            tile.image
            for tile in tile_batch
        ]

        tensor = numpy_batch_to_tensor(
            images
        )

        reconstructed, diff = run_batch_pca(
            tensor,
            components,
        )

        pixel_difference += diff

        reconstructed_images = tensor_batch_to_numpy(
            reconstructed
        )

        for tile, img in zip(
            tile_batch,
            reconstructed_images,
        ):
            tile.image = img
            processed_tiles.append(tile)

    cleanup()

    pixel_difference /= max(batch_count, 1)

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    merged = merge_tiles(
        processed_tiles,
        width,
        height,
    )

    # --------------------------------------------------
    # JPEG
    # --------------------------------------------------

    encoded, jpeg = encode_base64_jpeg(
        merged,
        quality=90,
    )

    elapsed = time.perf_counter() - start

    return {

        "compressed_image": encoded,

        "compressed_size": len(jpeg),

        "original_size": len(image_bytes),

        "components_used": components,

        "tile_size": tile_size,

        "batch_size": batch_size,

        "processing_mode": "tile",

        "device": device_name(),

        "backend": DEVICE.type,

        "pixel_difference": round(
            pixel_difference,
            4,
        ),

        "process_time": round(
            elapsed,
            3,
        ),

    }
