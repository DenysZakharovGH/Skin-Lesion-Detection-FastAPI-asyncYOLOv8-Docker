import base64

from fastapi import FastAPI, UploadFile, File, APIRouter, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

from starlette.responses import HTMLResponse

from app.core.app_config import settings, FRONTEND_STORAGE
from app.model import model, predict_async
from app.utils import draw_detection
from slowapi import Limiter
from slowapi.util import get_remote_address


cnn_route = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@cnn_route.post("/predict")
@limiter.limit(settings.cnn.rate_limits)
async def predict_image(request: Request, file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    processed_img = img.copy()

    results = await predict_async(image)

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

cnn_route.get("/health")
@limiter.limit(settings.run.rate_limits)
def health_check(request: Request,):
    return {"msg": "OK"}

@cnn_route.get("/", response_class=HTMLResponse)
@limiter.limit(settings.run.rate_limits)
async def root(request: Request):
    with open(f"{FRONTEND_STORAGE}/index.html") as f:
        return f.read()
