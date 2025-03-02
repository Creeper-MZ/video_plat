# app/models/schemas.py

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class GenerationParameters(BaseModel):
    """Basic parameters for video generation"""
    prompt: str = Field(..., description="Text prompt for generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt for generation")
    resolution: str = Field("832x480", description="Video resolution (width x height)")
    num_frames: int = Field(81, description="Number of frames to generate (must be 4n+1)")
    fps: int = Field(20, description="Frames per second")
    steps: int = Field(40, description="Number of sampling steps")
    shift: Optional[float] = Field(None, description="Sampling shift factor")
    guide_scale: float = Field(5.0, description="Classifier free guidance scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    use_fp8: bool = Field(True, description="Use FP8 quantization")

class AdvancedGenerationParameters(BaseModel):
    """Advanced parameters for video generation"""
    save_vram: bool = Field(False, description="Save VRAM by offloading parameters to CPU")
    debug: bool = Field(False, description="Enable debug logging")

class T2VRequest(BaseModel):
    """Request for text-to-video generation"""
    basic: GenerationParameters
    advanced: Optional[AdvancedGenerationParameters] = Field(None)

class I2VRequest(BaseModel):
    """Request for image-to-video generation"""
    basic: GenerationParameters
    advanced: Optional[AdvancedGenerationParameters] = Field(None)
    # Image will be uploaded separately as form data

class GenerationResponse(BaseModel):
    """Response for generation request"""
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    """Task status model"""
    id: str
    type: str
    status: str
    progress: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    gpu_id: Optional[int]
    result: Optional[Dict[str, Any]]
    error: Optional[str]

class TaskDetails(TaskStatus):
    """Detailed task information"""
    logs: List[Dict[str, Any]]
    params: Dict[str, Any]

class QueueStatus(BaseModel):
    """Queue status model"""
    queue_length: int
    running_tasks: int
    completed_tasks: int
    queue: List[TaskStatus]
    running: List[TaskStatus]
    recent_completed: List[TaskStatus]

class GPUStatus(BaseModel):
    """GPU status model"""
    device_id: int
    available: bool
    current_task: Optional[str]
    vram_used: Optional[int]
    vram_total: Optional[int]
    vram_free: Optional[int]
    utilization: Optional[float]

class SystemStatus(BaseModel):
    """System status model"""
    gpus: List[GPUStatus]
    queue: QueueStatus

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    details: Optional[str] = None