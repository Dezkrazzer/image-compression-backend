from __future__ import annotations

import numpy as np
import torch

from .device import DEVICE


# --------------------------------------------------------
# NumPy -> Tensor
# --------------------------------------------------------

def numpy_batch_to_tensor(
    images: list[np.ndarray],
) -> torch.Tensor:
    """
    Convert list of NumPy images into
    batched Torch tensor.

    Shape

    (N,H,W,C)

    →

    (N,C,H,W)
    """

    tensor = torch.stack(

        [

            torch.from_numpy(img)

            for img in images

        ]

    )

    return (

        tensor

        .float()

        .permute(0,3,1,2)

        .to(DEVICE)

    )


# --------------------------------------------------------
# Tensor -> NumPy
# --------------------------------------------------------

def tensor_batch_to_numpy(
    tensor: torch.Tensor,
) -> list[np.ndarray]:

    tensor = (

        tensor

        .permute(0,2,3,1)

        .cpu()

        .numpy()

        .clip(0,255)

        .astype(np.uint8)

    )

    return [

        img

        for img in tensor

    ]


# --------------------------------------------------------
# Batched PCA
# --------------------------------------------------------

def run_batch_pca(
    batch: torch.Tensor,
    components: int,
) -> tuple[torch.Tensor,float]:

    with torch.inference_mode():

        U,S,Vh = torch.linalg.svd(

            batch,

            full_matrices=False

        )

        U = U[...,:components]

        S = S[...,:components]

        Vh = Vh[...,:components,:]

        reconstructed = (

            U

            @ torch.diag_embed(S)

            @ Vh

        )

        reconstructed = reconstructed.clamp(

            0,

            255

        )

        diff = torch.abs(

            reconstructed

            -

            batch

        )

        pixel_diff = float(

            diff.mean()

            /

            255

            *100

        )

        return reconstructed,pixel_diff


# --------------------------------------------------------
# Cleanup
# --------------------------------------------------------

def cleanup():

    if DEVICE.type=="cuda":

        torch.cuda.empty_cache()