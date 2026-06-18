from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import api_router

app = FastAPI(title="Image Compression PCA API", version="1.0")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Izin akses dari frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan rute API
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Image Compression PCA API! Use the /compress endpoint to compress your images."}