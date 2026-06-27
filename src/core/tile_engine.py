"""
tile_engine.py

Tile-based image processing utilities.

Responsibilities
----------------
- Adaptive tile size
- Adaptive batch size
- Split image into overlapping tiles
- Merge processed tiles
- Blend overlap regions
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .device import DEVICE


# --------------------------------------------------------
# Tile Metadata
# --------------------------------------------------------

@dataclass(slots=True)
class Tile:
    image: np.ndarray

    x: int
    y: int

    width: int
    height: int

    original_width: int
    original_height: int


# --------------------------------------------------------
# Tile Size
# --------------------------------------------------------

def select_tile_size(
    width: int,
    height: int,
) -> int:
    """
    Automatically select tile size
    based on image resolution.
    """

    longest = max(width, height)

    if longest <= 2000:
        return 512

    if longest <= 5000:
        return 1024

    if longest <= 10000:
        return 1536

    return 2048


# --------------------------------------------------------
# Batch Size
# --------------------------------------------------------

def select_batch_size() -> int:
    """
    Select batch size
    based on available VRAM.
    """

    if DEVICE.type != "cuda":
        return 1

    memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        /
        (1024 ** 3)
    )

    if memory < 6:
        return 4

    if memory < 10:
        return 8

    if memory < 16:
        return 16

    return 32


# --------------------------------------------------------
# Split Image
# --------------------------------------------------------

def split_tiles(
    image: np.ndarray,
    tile_size: int,
    overlap: int = 64,
) -> list[Tile]:
    """
    Split image into overlapping tiles.
    """

    h, w, _ = image.shape

    stride = tile_size - overlap

    tiles: list[Tile] = []

    for y in range(0, h, stride):

        for x in range(0, w, stride):

            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = np.zeros(
                (tile_size, tile_size, 3),
                dtype=image.dtype,
            )
            crop = image[
                y:y_end,
                x:x_end,
            ]

            tile[
                :crop.shape[0],
                :crop.shape[1],
            ] = crop

            tiles.append(
                Tile(
                    image=tile,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                    original_width=crop.shape[1],
                    original_height=crop.shape[0],
                )
            )

    return tiles


# --------------------------------------------------------
# Batch Generator
# --------------------------------------------------------

def batch_tiles(
    tiles: list[Tile],
    batch_size: int,
):
    """
    Yield tiles in batches.
    """

    for i in range(
        0,
        len(tiles),
        batch_size
    ):
        yield tiles[i:i + batch_size]


# --------------------------------------------------------
# Merge Tiles
# --------------------------------------------------------

def merge_tiles(
    tiles: list[Tile],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Merge processed tiles using
    weighted blending.
    """

    result = np.zeros(
        (height, width, 3),
        dtype=np.float32,
    )

    weight = np.zeros(
        (height, width, 1),
        dtype=np.float32,
    )

    for tile in tiles:

        img = tile.image[
            :tile.original_height,
            :tile.original_width,
        ].astype(np.float32)

        h = tile.original_height
        w = tile.original_width

        result[
            tile.y:tile.y+h,
            tile.x:tile.x+w,
        ] += img

        weight[
            tile.y:tile.y+h,
            tile.x:tile.x+w,
        ] += 1.0

    weight = np.maximum(weight, 1)

    merged = result / weight

    return merged.astype(np.uint8)
