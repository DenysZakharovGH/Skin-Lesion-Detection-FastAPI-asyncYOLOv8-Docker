import base64

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

from starlette.responses import StreamingResponse, HTMLResponse

from app.app_config import FRONTEND_STORAGE
from app.model import model
from app.utils import draw_detection

app = FastAPI(title="Skin Lesion Detection API")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    processed_img = img.copy()

    results = model.predict(img, device="cpu", conf=0.05, verbose=False)

    r = results[0]
    detections = []
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            label = f"{model.names[cls]} {score:.2f}"
            processed_img = draw_detection(img, (x1, y1, x2, y2), label, score, model.names[cls])

            detections.append({
                "class": model.names[cls],
                "confidence": float(score),
            })

    _, buffer = cv2.imencode(".jpg", processed_img)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return JSONResponse({
        "filename": file.filename,
        "detections": detections,
        "annotated_image": img_base64
    })



app.get("/health")
def health_check():
    return {"msg": "OK"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(f"{FRONTEND_STORAGE}/index.html") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=8000)