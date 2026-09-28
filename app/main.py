from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, status, predict
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(status.router)
    app.include_router(predict.router)
    return app

app = create_app()
