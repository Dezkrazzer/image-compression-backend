from fastapi import HTTPException
from pathlib import Path

def validate_image_file(filename: str | None) -> None:
    if not filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")
    
    extension = Path(filename).suffix.lower()
    if not extension:
        raise HTTPException(
            status_code=400, 
            detail="File must have an extension."
        )