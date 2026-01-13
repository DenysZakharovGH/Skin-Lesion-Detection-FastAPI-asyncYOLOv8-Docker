import uvicorn
from fastapi import FastAPI

from app.core.app_core import cnn_route
from app.core.app_config import settings
from app.core.middleware import RequestLoggingMiddleware

app = FastAPI(title="Skin Lesion Detection API")

app.include_router(cnn_route)
app.add_middleware(RequestLoggingMiddleware)

if __name__ == "__main__":
    uvicorn.run("main:app",
                workers=4,
                host=settings.run.host,
                port=settings.run.port)