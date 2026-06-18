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
        
        # Proses kompresi
        result = compress_image_pca(image_bytes, k=components)
        
        return result
        
    except HTTPException as he:
        raise he # 404 Bad Request
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))