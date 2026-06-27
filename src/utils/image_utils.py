from fastapi import HTTPException
from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image_file(filename: str | None) -> None:
    if not filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")
    
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Format not supported. Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}."
        )
