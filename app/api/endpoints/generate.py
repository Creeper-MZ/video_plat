# app/api/endpoints/generate.py

import os
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
import json
from datetime import datetime

from ...models.schemas import T2VRequest, I2VRequest, GenerationResponse, ErrorResponse, TaskStatus
from ...services.task_queue import task_queue
from ...core.config import settings
from ...utils.helpers import ensure_directory_exists

router = APIRouter()
logger = logging.getLogger("videoGenPlatform")


# Add a dependency function to get the client_id from headers
def get_client_id(request: Request):
    client_id = request.headers.get("X-Client-ID")
    if not client_id:
        # Generate a random one if not provided
        client_id = str(uuid.uuid4())
    return client_id


def validate_resolution(resolution: str):
    """验证分辨率格式和有效性"""
    if resolution not in settings.resolution_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution: {resolution}. Available options: {', '.join(settings.resolution_map.keys())}"
        )
    return resolution


def validate_num_frames(num_frames: int):
    """验证并调整帧数为4n+1格式"""
    if num_frames < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Number of frames must be at least 5. Got {num_frames}"
        )

    if num_frames % 4 != 1:
        original = num_frames
        # 调整为最接近的4n+1
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        logger.warning(f"Adjusted num_frames from {original} to {num_frames} (must be 4n+1)")

    return num_frames


def validate_steps(steps: int):
    """验证并限制采样步数在允许范围内"""
    if steps < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Sampling steps must be positive. Got {steps}"
        )

    original = steps
    steps = max(min(steps, settings.max_steps), settings.min_steps)

    if steps != original:
        logger.warning(
            f"Adjusted steps from {original} to {steps} (min: {settings.min_steps}, max: {settings.max_steps})")

    return steps


def validate_fps(fps: int):
    """验证帧率在合理范围内"""
    if fps < 1 or fps > 60:
        raise HTTPException(
            status_code=400,
            detail=f"FPS must be between 1 and 60. Got {fps}"
        )
    return fps


def validate_prompt(prompt: str):
    """验证提示词不为空且长度合理"""
    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )

    if len(prompt) > 10000:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt is too long ({len(prompt)} chars). Maximum length is 10000 characters"
        )

    return prompt.strip()


@router.post("/t2v", response_model=GenerationResponse)
async def text_to_video(
        request: T2VRequest,
        background_tasks: BackgroundTasks,
        client_id: str = Depends(get_client_id)
):
    """
    Generate a video from text prompt
    """
    try:
        # 验证所有输入参数
        prompt = validate_prompt(request.basic.prompt)
        num_frames = validate_num_frames(request.basic.num_frames)
        steps = validate_steps(request.basic.steps)
        fps = validate_fps(request.basic.fps)
        resolution = validate_resolution(request.basic.resolution)

        # Create task parameters
        params = {
            "prompt": prompt,
            "negative_prompt": request.basic.negative_prompt or settings.default_negative_prompt,
            "resolution": resolution,
            "num_frames": num_frames,
            "fps": fps,
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

        # Add task to queue with client_id
        debug_mode = request.advanced.debug if request.advanced else False
        task = task_queue.add_task("t2v", params, debug=debug_mode, client_id=client_id)

        return GenerationResponse(
            task_id=task.id,
            status="queued",
            message="Task added to queue"
        )

    except ValueError as e:
        # Special handling for task limit errors
        if "maximum number of concurrent tasks" in str(e):
            raise HTTPException(
                status_code=429,
                detail=f"Task limit reached: {str(e)}"
            )
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # 直接传递HTTP异常
        raise
    except Exception as e:
        logger.exception(f"Error adding T2V task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create T2V task: {str(e)}"
        )


@router.get("/task/{task_id}/progress", response_model=dict)
async def get_task_progress(task_id: str, client_id: str = Depends(get_client_id)):
    """
    Get real-time progress of a task
    """
    task_info = task_queue.get_task_details(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Check if task belongs to the current client
    if task_info.get("client_id") and task_info.get("client_id") != client_id:
        raise HTTPException(status_code=403, detail="You do not have permission to view this task")

    return {
        "status": task_info["status"],
        "progress": task_info["progress"],
        "current_step": task_info.get("current_step", 0),
        "total_steps": task_info.get("total_steps", 0)
    }


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
        image: UploadFile = File(...),
        client_id: str = Depends(get_client_id)
):
    """
    Generate a video from image and text prompt
    """
    try:
        # Validate input parameters
        prompt = validate_prompt(prompt)
        num_frames = validate_num_frames(num_frames)
        steps = validate_steps(steps)
        fps = validate_fps(fps)
        resolution = validate_resolution(resolution)

        # Check if prompt is empty
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400,
                                detail=f"Uploaded file is not an image. Got content-type: {image.content_type}")

        # Check file size (limit to 10MB)
        MAX_SIZE = 10 * 1024 * 1024  # 10MB

        # Read image data
        image_data = await image.read()
        if len(image_data) > MAX_SIZE:
            raise HTTPException(status_code=400,
                                detail=f"Image file is too large (max 10MB). Got {len(image_data) // (1024 * 1024)}MB")

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

        # Add task to queue with client_id
        task = task_queue.add_task("i2v", params, debug=debug, client_id=client_id)

        return GenerationResponse(
            task_id=task.id,
            status="queued",
            message="Task added to queue"
        )

    except ValueError as e:
        # Special handling for task limit errors
        if "maximum number of concurrent tasks" in str(e):
            raise HTTPException(
                status_code=429,
                detail=f"Task limit reached: {str(e)}"
            )
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Pass HTTP exceptions directly
        raise
    except Exception as e:
        logger.exception(f"Error adding I2V task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create I2V task: {str(e)}"
        )


@router.get("/task/{task_id}", response_model=dict)
async def get_task_status(task_id: str, client_id: str = Depends(get_client_id)):
    """
    Get status of a task
    """
    task_info = task_queue.get_task_details(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Check if task belongs to the current client
    if task_info.get("client_id") and task_info.get("client_id") != client_id:
        raise HTTPException(status_code=403, detail="You do not have permission to view this task")

    return task_info


@router.delete("/task/{task_id}", response_model=dict)
async def cancel_task(task_id: str, client_id: str = Depends(get_client_id)):
    """
    Cancel a running or queued task
    """
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Check if task belongs to the current client
    if task.client_id and task.client_id != client_id:
        raise HTTPException(status_code=403, detail="You do not have permission to cancel this task")

    success = task_queue.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")

    return {"status": "cancelled", "message": f"Task {task_id} cancelled successfully"}


@router.get("/queue", response_model=dict)
async def get_queue_status(client_id: str = Depends(get_client_id)):
    """
    Get status of the task queue for the current client
    """
    return task_queue.get_client_tasks(client_id)