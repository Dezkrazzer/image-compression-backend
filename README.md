# image-compression-backend
Backend for Image Compression App

## Setup

Use Python 3.10 or newer.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m uvicorn src.main:app --reload
```

Or run the entrypoint directly:

```powershell
python src/main.py
```

The API is available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.
