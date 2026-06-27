import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import time
import base64
import torch

# Deteksi hardware secara dinamis: CUDA (NVIDIA), MPS (Apple Silicon), atau CPU (Multicore x86/AMD)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def compress_image_pca(image_bytes: bytes, k: int) -> dict:
    if k < 1:
        raise ValueError("Components must be greater than or equal to 1.")

    start_time = time.time()
    
    try:
        # Buka gambar menggunakan Pillow dan konversi ke RGB
        with Image.open(io.BytesIO(image_bytes)) as source_img:
            img = source_img.convert('RGB')
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    img_array = np.array(img)
    h, w, c = img_array.shape
    
    # Batasi nilai k agar tidak melebihi dimensi gambar terkecil
    max_k = min(h, w)
    k = min(k, max_k)
    
    # 1. Pindahkan data ke PyTorch Tensor dan arahkan ke device yang tersedia
    # 2. Ubah urutan dimensi dari (H, W, C) menjadi (C, H, W) agar siap untuk Batched SVD
    tensor_img = torch.tensor(img_array, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    # Eksekusi SVD secara paralel untuk ketiga channel (R, G, B) sekaligus tanpa perulangan
    U, S, Vh = torch.linalg.svd(tensor_img, full_matrices=False)
    
    # Slicing untuk mengambil sejumlah komponen k yang diminta
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    # Rekonstruksi gambar
    # torch.diag_embed mengubah vektor 1D S menjadi matriks diagonal untuk perkalian matriks
    reconstructed_tensor = U_k @ torch.diag_embed(S_k) @ Vh_k
    
    # Kembalikan ke format awal (H, W, C), tarik memori kembali ke CPU, dan konversi ke NumPy
    compressed_array = reconstructed_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Pastikan nilai pixel tetap dalam rentang valid 0-255 dan ubah tipe data
    compressed_array = np.clip(compressed_array, 0, 255).astype(np.uint8)
    
    # Hitung perbedaan pixel untuk evaluasi loss
    diff = np.abs(img_array.astype(np.float32) - compressed_array.astype(np.float32))
    pixel_diff_percentage = float((np.sum(diff) / (img_array.size * 255.0)) * 100)
    
    # Konversi array numpy kembali menjadi gambar JPEG
    compressed_img = Image.fromarray(compressed_array)
    img_byte_arr = io.BytesIO()
    compressed_img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
    compressed_bytes = img_byte_arr.getvalue()
    
    process_time = time.time() - start_time
    
    # Ubah gambar ke format Base64 untuk dikirim via JSON response
    compressed_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
    
    return {
        "compressed_image": compressed_b64,
        "compressed_size": len(compressed_bytes),
        "original_size": len(image_bytes),
        "components_used": k,
        "process_time": process_time,
        "pixel_difference": float(pixel_diff_percentage),
        "compute_device": str(device) # Menambahkan info device agar kelihatan di respon API
    }