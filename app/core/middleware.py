import time
import uuid
from fastapi import Request
from app.core.logs import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        logger.info(
            "[%s] %s %s %.3fs",
            request_id,
            request.method,
            request.url.path,
            duration,
        )
        response.headers["X-Request-ID"] = request_id
        return response