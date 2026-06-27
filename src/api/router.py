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

        validate_image_file(image.filename)

        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        result = compress_image_pca(
            image_bytes,
            components
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        await image.close()