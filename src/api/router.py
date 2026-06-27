from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.services.pca_service import compress_image_pca
from src.utils.image_utils import validate_image_file

api_router = APIRouter()

@api_router.post("/compress")
async def compress_image(
    image: UploadFile = File(...),
    components: int = Form(...)
):
    try:
        # Validasi file
        validate_image_file(image.filename)
        
        # Baca file
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        
        # Proses kompresi
        result = compress_image_pca(image_bytes, k=components)
        
        return result
        
    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to compress image.") from e
    finally:
        await image.close()
