# Image Compression Backend

A FastAPI-based backend for image compression using **Principal Component Analysis (PCA)** with an adaptive tile-based processing pipeline. The application supports CPU execution and optional GPU acceleration through PyTorch.

---

## Features

* PCA-based image compression
* Adaptive tile-based processing for large images
* Modular architecture
* FastAPI REST API
* Automatic CPU/GPU device detection
* Optional CUDA acceleration (NVIDIA)
* Apple Silicon (MPS) support
* CPU fallback for unsupported hardware
* Support for JPEG, PNG, BMP, TIFF, HEIC, and RAW images
* Automatic tile size selection
* Adaptive batch processing
* Interactive Swagger API documentation

---

## Project Structure

```text
src/
│
├── api/
├── core/
├── services/
├── utils/
├── schemas/
├── config/
│
├── main.py
└── requirements.txt
```

---

# Requirements

* Python **3.10 or newer**
* Windows / Linux / macOS
* 8 GB RAM recommended
* NVIDIA GPU (optional)

---

# Clone Repository

```bash
git clone https://github.com/<username>/image-compression-backend.git

cd image-compression-backend
```

---

# Create Virtual Environment

### Windows

```powershell
py -3 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

# Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install PyTorch

PyTorch is installed separately depending on your hardware.

## CPU Only

```bash
pip install torch torchvision torchaudio
```

---

## NVIDIA CUDA

Install the version matching your CUDA runtime.

Example (CUDA 12.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For the latest installation command, visit:

https://pytorch.org/get-started/locally/

---

## Apple Silicon (M1 / M2 / M3)

Install the standard PyTorch package.

```bash
pip install torch torchvision torchaudio
```

PyTorch will automatically use Apple's Metal Performance Shaders (MPS) when available.

---

# Run the Server

```bash
python -m uvicorn src.main:app --reload
```

or

```bash
python src/main.py
```

---

# API Documentation

After starting the server:

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

# Supported Image Formats

* JPEG
* PNG
* BMP
* TIFF
* HEIC
* RAW (Camera RAW)

---

# Processing Pipeline

```text
Image

↓

Load Image

↓

Adaptive Tile Engine

↓

Batch Processing

↓

PCA Compression

↓

Tile Merge

↓

JPEG Encoding

↓

JSON Response
```

---

# Hardware Support

| Hardware      | Supported    |
| ------------- | ------------ |
| Intel CPU     | ✅            |
| AMD CPU       | ✅            |
| Apple Silicon | ✅            |
| NVIDIA CUDA   | ✅            |
| AMD Radeon    | CPU Fallback |
| Intel Arc     | CPU Fallback |

---

# Verify PyTorch Installation

```bash
python -c "import torch; print(torch.__version__)"
```

Verify GPU support:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

CPU

```
False
```

CUDA

```
True
```

---

# Install Missing Packages

If a dependency is missing:

```bash
pip install fastapi uvicorn[standard] python-multipart pillow rawpy pillow-heif numpy opencv-python
```

---

# Development

Run with auto reload:

```bash
uvicorn src.main:app --reload
```

---

# License

This project is intended for educational and research purposes.
