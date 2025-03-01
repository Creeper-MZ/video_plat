import os
import uuid
import time
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, Form, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

# 创建应用
app = FastAPI(title="视频生成平台", description="基于Wan2.1的文生视频和图生视频平台")

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库设置
DATABASE_URL = "sqlite:///./video_generation.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 创建文件夹
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# WebSocket连接管理
active_connections: Dict[str, WebSocket] = {}


# 枚举类型定义
class VideoType(str, Enum):
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"


class Resolution(str, Enum):
    RESOLUTION_720P = "720p"
    RESOLUTION_720P_VERTICAL = "720p_vertical"
    RESOLUTION_480P = "480p"
    RESOLUTION_480P_VERTICAL = "480p_vertical"


class ModelPrecision(str, Enum):
    FP16 = "fp16"
    FP8 = "fp8"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 数据模型
class VideoTask(Base):
    __tablename__ = "video_tasks"

    id = Column(String, primary_key=True, index=True)
    type = Column(String, index=True)
    prompt = Column(String)
    negative_prompt = Column(String, nullable=True)
    resolution = Column(String)
    frames = Column(Integer)
    fps = Column(Integer)
    steps = Column(Integer)
    seed = Column(Integer)
    status = Column(String, index=True)
    model_precision = Column(String)
    save_vram = Column(Boolean, default=False)
    image_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    gpu_id = Column(Integer, nullable=True)
    progress = Column(Float, default=0.0)
    error_message = Column(String, nullable=True)
    tiled = Column(Boolean, default=False)
    additional_params = Column(JSON, nullable=True)
    user_id = Column(String, nullable=True)


# 创建表
Base.metadata.create_all(bind=engine)


# 依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 请求模型
class VideoGenerationRequest(BaseModel):
    type: VideoType
    prompt: str
    negative_prompt: Optional[str] = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    resolution: Resolution = Resolution.RESOLUTION_480P
    frames: int = Field(100, ge=8, le=128, description="视频帧数")
    fps: int = Field(20, ge=10, le=60, description="视频帧率")
    steps: int = Field(40, ge=20, le=60, description="推理步数")
    seed: int = Field(0, ge=-1, le=2147483647, description="随机种子，-1为随机")
    model_precision: ModelPrecision = ModelPrecision.FP8
    save_vram: bool = False
    tiled: bool = True
    additional_params: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    id: str
    status: TaskStatus
    type: VideoType
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_url: Optional[str] = None
    error_message: Optional[str] = None
    stage: Optional[str] = None
    stage_progress: Optional[float] = None
    eta: Optional[float] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    logs: Optional[List[str]] = None
    additional_info: Optional[Dict[str, Any]] = None


# 队列管理
task_queue = []
gpu_status = {0: False, 1: False, 2: False, 3: False}  # False means available


# 路由定义
@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(
        background_tasks: BackgroundTasks,
        params: VideoGenerationRequest = None,
        image: Optional[UploadFile] = File(None),
        prompt: Optional[str] = Form(None),
        negative_prompt: Optional[str] = Form(None),
        resolution: Optional[str] = Form(None),
        frames: Optional[int] = Form(None),
        fps: Optional[int] = Form(None),
        steps: Optional[int] = Form(None),
        seed: Optional[int] = Form(None),
        model_precision: Optional[str] = Form(None),
        save_vram: Optional[bool] = Form(None),
        tiled: Optional[bool] = Form(None),
        video_type: Optional[str] = Form(None),
        db: Session = Depends(get_db)
):
    # 处理表单数据和JSON数据
    task_data = {}
    if params:
        task_data = params.dict()
    else:
        if video_type:
            task_data["type"] = video_type
        if prompt:
            task_data["prompt"] = prompt
        if negative_prompt:
            task_data["negative_prompt"] = negative_prompt
        if resolution:
            task_data["resolution"] = resolution
        if frames:
            task_data["frames"] = frames
        if fps:
            task_data["fps"] = fps
        if steps:
            task_data["steps"] = steps
        if seed is not None:
            task_data["seed"] = seed
        if model_precision:
            task_data["model_precision"] = model_precision
        if save_vram is not None:
            task_data["save_vram"] = save_vram
        if tiled is not None:
            task_data["tiled"] = tiled

    # 验证必要的参数
    if "type" not in task_data or "prompt" not in task_data:
        raise HTTPException(status_code=400, detail="缺少必要参数")

    # 处理图片上传
    image_path = None
    if task_data["type"] == VideoType.IMAGE_TO_VIDEO:
        if not image:
            raise HTTPException(status_code=400, detail="图生视频模式需要上传图片")
        image_filename = f"{uuid.uuid4()}{os.path.splitext(image.filename)[1]}"
        image_path = os.path.join("uploads", image_filename)
        with open(image_path, "wb") as f:
            f.write(await image.read())

    # 生成任务ID
    task_id = str(uuid.uuid4())
    output_filename = f"{task_id}.mp4"
    output_path = os.path.join("outputs", output_filename)

    # 创建任务记录
    task = VideoTask(
        id=task_id,
        type=task_data["type"],
        prompt=task_data["prompt"],
        negative_prompt=task_data.get("negative_prompt", ""),
        resolution=task_data.get("resolution", Resolution.RESOLUTION_720P),
        frames=task_data.get("frames", 16),
        fps=task_data.get("fps", 25),
        steps=task_data.get("steps", 50),
        seed=task_data.get("seed", 0),
        status=TaskStatus.QUEUED,
        model_precision=task_data.get("model_precision", ModelPrecision.FP16),
        save_vram=task_data.get("save_vram", False),
        image_path=image_path,
        output_path=output_path,
        tiled=task_data.get("tiled", True),
        additional_params=task_data.get("additional_params"),
    )

    db.add(task)
    db.commit()
    db.refresh(task)

    # 添加到队列并尝试启动处理
    task_queue.append(task_id)
    background_tasks.add_task(process_queue, db)

    return TaskResponse(
        id=task_id,
        status=TaskStatus.QUEUED,
        type=task_data["type"],
        created_at=task.created_at,
        output_url=f"/api/videos/{task_id}"
    )


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    additional_info = task.additional_params or {}

    return TaskResponse(
        id=task.id,
        status=task.status,
        type=task.type,
        progress=task.progress,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        output_url=f"/api/videos/{task_id}" if task.status == TaskStatus.COMPLETED else None,
        error_message=task.error_message,
        stage=additional_info.get("stage"),
        stage_progress=additional_info.get("stage_progress"),
        eta=additional_info.get("eta"),
        step=additional_info.get("step"),
        total_steps=additional_info.get("total_steps"),
        logs=additional_info.get("logs"),
        additional_info={
            k: v for k, v in additional_info.items()
            if k not in ["stage", "stage_progress", "eta", "step", "total_steps", "logs"]
        }
    )


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except:
        if task_id in active_connections:
            del active_connections[task_id]


async def notify_client(task_id: str, data: dict):
    if task_id in active_connections:
        try:
            await active_connections[task_id].send_json(data)
            logger.info(f"WebSocket 更新: 任务 {task_id} - 进度 {data.get('progress', 0) * 100:.1f}%")
        except Exception as e:
            logger.error(f"WebSocket 发送失败: {e}")
            if task_id in active_connections:
                del active_connections[task_id]


async def process_queue(db: Session):
    for gpu_id, is_busy in gpu_status.items():
        if not is_busy and task_queue:
            task_id = task_queue.pop(0)
            task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                task.gpu_id = gpu_id
                db.commit()
                gpu_status[gpu_id] = True
                await notify_client(task_id, {"status": TaskStatus.RUNNING, "progress": 0.0})


# 添加前端静态文件服务
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)