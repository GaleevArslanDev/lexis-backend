from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles
import os
from uuid import uuid4

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
    # TODO: upload to Firebase Storage and return public URL
    return {"file_url": path}
