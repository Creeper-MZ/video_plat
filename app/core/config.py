# app/core/config.py

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

class ModelConfig(BaseModel):
    model_path: str
    type: str
    resolutions: List[str]

class GPUConfig(BaseModel):
    device_id: int
    vram: int
    available: bool = True
    current_task: Optional[str] = None

class Settings(BaseModel):
    debug: bool = False
    api_title: str = "Wan2.1 Video Generation Platform"
    api_description: str = "Generate videos from text or images using Wan2.1 models"
    api_version: str = "0.1.0"
    
    # Output directory for generated videos
    output_dir: str = "app/static/videos"
    
    # Model paths
    models: Dict[str, ModelConfig] = {
        "wan2.1-t2v-14b": ModelConfig(
            model_path="/home/ps/videoGen/models/Wan2.1-T2V-14B/",
            type="t2v",
            resolutions=["1280x720", "720x1280", "832x480", "480x832"]
        ),
        "wan2.1-i2v-14b-480p": ModelConfig(
            model_path="/home/ps/videoGen/models/Wan2.1-I2V-14B-480P/",
            type="i2v",
            resolutions=["832x480", "480x832"]
        ),
        "wan2.1-i2v-14b-720p": ModelConfig(
            model_path="/home/ps/videoGen/models/Wan2.1-I2V-14B-720P/",
            type="i2v",
            resolutions=["1280x720", "720x1280"]
        )
    }
    
    # Default parameters
    default_resolution: str = "832x480"  # 480P landscape
    default_frame_num: int = 81  # Must be 4n+1
    default_fps: int = 20
    default_steps: int = 40
    min_steps: int = 20
    max_steps: int = 60
    default_shift: float = 5.0
    default_guide_scale: float = 5.0
    default_use_fp8: bool = True
    default_seed: int = -1  # Random seed
    
    # Default negative prompt (extracted from your code examples)
    default_negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    # GPU management
    gpu_devices: List[GPUConfig] = [
        GPUConfig(device_id=0, vram=48),
        GPUConfig(device_id=1, vram=48),
        GPUConfig(device_id=2, vram=48),
        GPUConfig(device_id=3, vram=48)
    ]
    
    # Task queue settings
    max_queue_size: int = 100
    task_timeout: int = 1800  # 30 minutes in seconds
    polling_interval: int = 1  # Check task status every second
    
    # GPU memory optimization
    default_save_vram: bool = False
    
    # Mapping between resolution names and dimensions
    resolution_map: Dict[str, Dict[str, Union[int, str]]] = {
        "1280x720": {"width": 1280, "height": 720, "name": "720P Landscape"},
        "720x1280": {"width": 720, "height": 1280, "name": "720P Portrait"},
        "832x480": {"width": 832, "height": 480, "name": "480P Landscape"},
        "480x832": {"width": 480, "height": 832, "name": "480P Portrait"}
    }
    
settings = Settings()

# Ensure output directory exists
os.makedirs(settings.output_dir, exist_ok=True)