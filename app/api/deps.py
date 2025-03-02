# app/api/deps.py

from typing import Generator
import logging

from ..core.gpu_manager import gpu_manager
from ..services.task_queue import task_queue

def get_task_queue() -> Generator:
    yield task_queue

def get_gpu_manager() -> Generator:
    yield gpu_manager

# app/api/__init__.py

from fastapi import APIRouter
from .endpoints import generate, status

api_router = APIRouter()
api_router.include_router(generate.router, prefix="/generate", tags=["generation"])
api_router.include_router(status.router, prefix="/status", tags=["status"])