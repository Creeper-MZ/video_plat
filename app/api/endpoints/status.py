# app/api/endpoints/status.py

import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict

from ...core.gpu_manager import gpu_manager
from ...services.task_queue import task_queue
from ...models.schemas import SystemStatus, GPUStatus, QueueStatus

router = APIRouter()
logger = logging.getLogger("videoGenPlatform")

@router.get("/gpu", response_model=List[Dict])
async def get_gpu_status():
    """
    Get status of all GPUs
    """
    return gpu_manager.get_gpu_status()

@router.get("/system", response_model=Dict)
async def get_system_status():
    """
    Get system status including GPUs and queue
    """
    try:
        # Ensure default values if retrieval fails
        gpu_status = []
        try:
            gpu_status = gpu_manager.get_gpu_status()
        except Exception as gpu_error:
            logger.error(f"Error retrieving GPU status: {str(gpu_error)}")
        
        queue_status = {}
        try:
            queue_status = task_queue.get_queue_status()
        except Exception as queue_error:
            logger.error(f"Error retrieving queue status: {str(queue_error)}")
        
        return {
            "gpus": gpu_status,
            "queue": queue_status
        }
    except Exception as e:
        logger.exception("Unexpected error in system status retrieval")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve system status: {str(e)}"
        )

@router.get("/config", response_model=Dict)
async def get_system_config():
    """
    Get system configuration including available models and parameters
    """
    from ...core.config import settings
    
    return {
        "models": {
            "t2v": {
                "name": "Wan2.1 T2V 14B",
                "resolutions": ["1280x720", "720x1280", "832x480", "480x832"]
            },
            "i2v": {
                "480p": {
                    "name": "Wan2.1 I2V 14B 480P",
                    "resolutions": ["832x480", "480x832"]
                },
                "720p": {
                    "name": "Wan2.1 I2V 14B 720P",
                    "resolutions": ["1280x720", "720x1280"]
                }
            }
        },
        "default_params": {
            "resolution": settings.default_resolution,
            "frame_num": settings.default_frame_num,
            "fps": settings.default_fps,
            "steps": settings.default_steps,
            "step_range": {
                "min": settings.min_steps,
                "max": settings.max_steps
            },
            "shift": settings.default_shift,
            "guide_scale": settings.default_guide_scale,
            "seed": settings.default_seed,
            "use_fp8": settings.default_use_fp8,
            "save_vram": settings.default_save_vram
        },
        "resolution_options": [
            {
                "id": res_id,
                "name": res_info["name"],
                "width": res_info["width"],
                "height": res_info["height"]
            }
            for res_id, res_info in settings.resolution_map.items()
        ]
    }