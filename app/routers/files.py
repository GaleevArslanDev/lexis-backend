from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
import aiofiles
import os
from uuid import uuid4
from ..dependencies.auth import get_current_user
from ..models import User, AssessmentImage
from ..db import get_session
from sqlmodel import Session

router = APIRouter(prefix="/files", tags=["files"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    new_name = f"{uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, new_name)
    try:
        async with aiofiles.open(path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save file")
    return {"file_url": path}


@router.get("/work-image/{work_id}")
async def get_work_image(
    work_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Отдаёт изображение работы по ID.
    Проверяет права: учитель может смотреть работы своих классов,
    ученик — только свои.
    """
    image = session.get(AssessmentImage, work_id)
    if not image:
        raise HTTPException(status_code=404, detail="Work not found")

    # Проверка прав
    if current_user.role == "student" and image.student_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = image.original_image_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found on server")

    # Определяем media type
    ext = os.path.splitext(file_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    return FileResponse(file_path, media_type=media_type)