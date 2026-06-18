import numpy as np
from PIL import Image
import io
import time
import base64

def compress_image_pca(image_bytes: bytes, k: int) -> dict:
    start_time = time.time()
    
    # Buka gambar menggunakan Pillow dan konversi ke RGB (berupa matriks 3D)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(img)
    
    # Pisahkan channel warna Red, Green, Blue
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Batasi nilai k (komponen PCA) agar tidak melebihi dimensi gambar
    max_k = min(R.shape)
    k = min(k, max_k)
    
    def compress_channel(channel_matrix, components):
        # Dekomposisi matriks menggunakan SVD (Singular Value Decomposition)
        U, S, Vt = np.linalg.svd(channel_matrix, full_matrices=False)
        
        # Rekonstruksi matriks hanya menggunakan sejumlah komponen k
        reconstructed = np.dot(U[:, :components], np.dot(np.diag(S[:components]), Vt[:components, :]))
        return reconstructed

    # Terapkan PCA pada masing-masing matriks warna
    R_compressed = compress_channel(R, k)
    G_compressed = compress_channel(G, k)
    B_compressed = compress_channel(B, k)
    
    # Gabungkan kembali ketiga channel warna tersebut
    compressed_array = np.dstack((R_compressed, G_compressed, B_compressed))
    
    # Pastikan nilai pixel tetap dalam rentang valid 0-255
    compressed_array = np.clip(compressed_array, 0, 255).astype(np.uint8)
    
    # Hitung perbedaan pixel
    diff = np.abs(img_array.astype(np.float32) - compressed_array.astype(np.float32))
    pixel_diff_percentage = float((np.sum(diff) / (img_array.size * 255.0)) * 100)
    
    # Konversi array numpy kembali menjadi gambar JPEG
    compressed_img = Image.fromarray(compressed_array)
    img_byte_arr = io.BytesIO()
    compressed_img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
    compressed_bytes = img_byte_arr.getvalue()
    
    process_time = time.time() - start_time
    
    # Ubah gambar ke format Base64
    compressed_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
    
    return {
        "compressed_image": compressed_b64,
        "compressed_size": len(compressed_bytes),
        "process_time": process_time,
        "pixel_difference": float(pixel_diff_percentage)
    }