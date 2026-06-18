from fastapi import HTTPException

def validate_image_file(filename: str) -> None:
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Cek apakah ekstensi file valid
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Format not supported. Allowed formats: {', '.join(allowed_extensions)}"
        )