# app/api/__init__.py

from fastapi import APIRouter
from .endpoints import generate, status

api_router = APIRouter()
api_router.include_router(generate.router, prefix="/generate", tags=["generation"])
api_router.include_router(status.router, prefix="/status", tags=["status"])