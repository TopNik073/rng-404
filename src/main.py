from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import env_config
from src.core.logger import get_logger
from src.presentation.middlewares.logging import RequestLoggingMiddleware

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(
    _app: FastAPI,
) -> AsyncGenerator[None]:
    base_url: str = f"http://{env_config.APP_HOST}:{env_config.APP_PORT}"
    logger.info(f"App started on {base_url}")
    logger.info(f"See Swagger for mode info: {base_url}/docs")
    yield
    logger.warning("Stopping app...")

app = FastAPI(title=env_config.APP_NAME, debug=env_config.DEBUG, lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=env_config.APP_HOST, port=env_config.APP_PORT, log_level=50)
