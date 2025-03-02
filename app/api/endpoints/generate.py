# app/api/endpoints/generate.py

import os
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import json

from ...models.schemas import T2VRequest, I2VRequest, GenerationResponse, ErrorResponse, TaskStatus
from ...services.task_queue import task_queue
from ...core.config import settings
from ...core.logging import setup_app_logger

router = APIRouter()
logger = logging.getLogger("videoGenPlatform")

@router.post("/t2v", response_model=GenerationResponse)
async def text_to_video(
    request: T2VRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a video from text prompt
    """
    try:
        # Validate parameters
        num_frames = request.basic.num_frames
        if num_frames % 4 != 1:
            # Round to nearest 4n+1
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            logger.warning(f"Adjusted num_frames to {num_frames} (must be 4n+1)")
        
        # Check if steps is within range
        steps = max(min(request.basic.steps, settings.max_steps), settings.min_steps)
        if steps != request.basic.steps:
            logger.warning(f"Adjusted steps to {steps} (min: {settings.min_steps}, max: {settings.max_steps})")
        
        # Parse resolution
        resolution = request.basic.resolution
        if resolution not in settings.resolution_map:
            raise HTTPException(status_code=400, detail=f"Invalid resolution: {resolution}")
        
        # Check if prompt is empty
        if not request.basic.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Create task parameters
        params = {
            "prompt": request.basic.prompt,
            "negative_prompt": request.basic.negative_prompt or settings.default_negative_prompt,
            "resolution": resolution,
            "num_frames": num_frames,
            "fps": request.basic.fps,
            "steps": steps,
            "shift": request.basic.shift or settings.default_shift,
            "guide_scale": request.basic.guide_scale,
            "seed": request.basic.seed,
            "use_fp8": request.basic.use_fp8,
        }
        
        # Add advanced parameters if provided
        if request.advanced:
            params.update({
                "save_vram": request.advanced.save_vram,
                "debug": request.advanced.debug
            })
        else:
            params.update({
                "save_vram": settings.default_save_vram,
                "debug": False
            })
        
        # Add task to queue
        debug_mode = request.advanced.debug if request.advanced else False
        task = task_queue.add_task("t2v", params, debug=debug_mode)
        
        return GenerationResponse(
            task_id=task.id,
            status="queued",
            message="Task added to queue"
        )
    
    except Exception as e:
        logger.exception(f"Error adding T2V task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create T2V task: {str(e)}"
        )

@router.post("/i2v", response_model=GenerationResponse)
async def image_to_video(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    resolution: str = Form("832x480"),
    num_frames: int = Form(81),
    fps: int = Form(20),
    steps: int = Form(40),
    shift: Optional[float] = Form(None),
    guide_scale: float = Form(5.0),
    seed: int = Form(-1),
    use_fp8: bool = Form(True),
    save_vram: bool = Form(False),
    debug: bool = Form(False),
    image: UploadFile = File(...)
):
    """
    Generate a video from image and text prompt
    """
    try:
        # Check if prompt is empty
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"Uploaded file is not an image. Got content-type: {image.content_type}")
            
        # Check file size (limit to 10MB)
        MAX_SIZE = 10 * 1024 * 1024  # 10MB
        
        # Read image data
        image_data = await image.read()
        if len(image_data) > MAX_SIZE:
            raise HTTPException(status_code=400, detail=f"Image file is too large (max 10MB). Got {len(image_data) // (1024*1024)}MB")
            
        # Validate parameters
        if num_frames % 4 != 1:
            # Round to nearest 4n+1
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            logger.warning(f"Adjusted num_frames to {num_frames} (must be 4n+1)")
        
        # Check if steps is within range
        steps = max(min(steps, settings.max_steps), settings.min_steps)
        
        # Parse resolution
        if resolution not in settings.resolution_map:
            raise HTTPException(status_code=400, detail=f"Invalid resolution: {resolution}")
        
        # Create task parameters
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or settings.default_negative_prompt,
            "resolution": resolution,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "shift": shift or settings.default_shift,
            "guide_scale": guide_scale,
            "seed": seed,
            "use_fp8": use_fp8,
            "save_vram": save_vram,
            "image_data": image_data
        }
        
        # Add task to queue
        task = task_queue.add_task("i2v", params, debug=debug)
        
        return GenerationResponse(
            task_id=task.id,
            status="queued",
            message="Task added to queue"
        )
    
    except Exception as e:
        logger.exception(f"Error adding I2V task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create I2V task: {str(e)}"
        )

@router.get("/task/{task_id}", response_model=dict)
async def get_task_status(task_id: str):
    """
    Get status of a task
    """
    task_info = task_queue.get_task_details(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return task_info

@router.delete("/task/{task_id}", response_model=dict)
async def cancel_task(task_id: str):
    """
    Cancel a running or queued task
    """
    success = task_queue.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")
    
    return {"status": "cancelled", "message": f"Task {task_id} cancelled successfully"}

@router.get("/queue", response_model=dict)
async def get_queue_status():
    """
    Get status of the task queue
    """
    return task_queue.get_queue_status()