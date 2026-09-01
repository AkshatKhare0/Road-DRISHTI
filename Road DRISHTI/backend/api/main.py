from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import tempfile
import os

from inference import detect_image, detect_video

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "Road Drishti inference service is running"}


@app.post("/detect/image")
async def image_detection(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = detect_image(image, filename=file.filename)

    return {
        "filename": file.filename,
        **result
    }


@app.post("/detect/video")
async def video_detection(file: UploadFile = File(...)):

    video_bytes = await file.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp_file:

        temp_file.write(video_bytes)
        temp_path = temp_file.name

    try:
        result = detect_video(temp_path)

        return {
            "filename": file.filename,
            **result
        }

    finally:
        os.remove(temp_path)