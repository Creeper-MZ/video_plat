import os
import uuid
import time
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, Form, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 配置日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

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
    INITIALIZING = "initializing"  # 初始化模型阶段
    RUNNING = "running"
    SAVING = "saving"  # 保存视频阶段
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
    tiled = Column(Boolean, default=True)
    additional_params = Column(JSON, nullable=True)
    user_id = Column(String, nullable=True)
    status_message = Column(String, nullable=True)  # 添加状态消息字段


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
    negative_prompt: Optional[
        str] = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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
    status_message: Optional[str] = None  # 添加状态消息
    current_step: Optional[int] = None  # 当前步骤
    total_steps: Optional[int] = None  # 总步骤数
    estimated_time: Optional[int] = None  # 预计剩余时间(秒)


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
        resolution=task_data.get("resolution", Resolution.RESOLUTION_480P),
        frames=task_data.get("frames", 100),
        fps=task_data.get("fps", 20),
        steps=task_data.get("steps", 40),
        seed=task_data.get("seed", 0),
        status=TaskStatus.QUEUED,
        model_precision=task_data.get("model_precision", ModelPrecision.FP8),
        save_vram=task_data.get("save_vram", False),
        image_path=image_path,
        output_path=output_path,
        tiled=task_data.get("tiled", True),
        additional_params=task_data.get("additional_params"),
        status_message="任务已创建，等待分配资源"
    )

    db.add(task)
    db.commit()
    db.refresh(task)

    logger.info(f"创建新任务: {task_id}, 类型: {task.type}, 分辨率: {task.resolution}")

    # 添加到队列并尝试启动处理
    task_queue.append(task_id)
    background_tasks.add_task(process_queue, db)

    return TaskResponse(
        id=task_id,
        status=TaskStatus.QUEUED,
        type=task_data["type"],
        created_at=task.created_at,
        output_url=f"/api/videos/{task_id}",
        status_message="任务已创建，等待分配资源",
        total_steps=task.steps
    )


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 读取进度文件获取最新状态（如果存在）
    try:
        progress_file = f"progress_{task_id}.json"
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress_data = json.load(f)

                # 仅当显示完成时检查视频文件是否存在
                if progress_data.get("status") == TaskStatus.COMPLETED:
                    # 检查视频文件是否真的存在
                    if not os.path.exists(task.output_path) or os.path.getsize(task.output_path) < 1000:
                        # 文件不存在或太小，可能是保存失败，强制状态为处理中
                        progress_data["status"] = TaskStatus.RUNNING
                        progress_data["status_message"] = "等待视频生成完成..."
                        progress_data["progress"] = 0.95

                # 更新任务状态
                if progress_data.get("status") and progress_data["status"] != task.status:
                    task.status = progress_data["status"]

                # 更新进度
                if "progress" in progress_data:
                    task.progress = progress_data["progress"]

                # 更新状态消息
                if "status_message" in progress_data:
                    task.status_message = progress_data["status_message"]

                # 提交更新
                db.commit()
    except Exception as e:
        logger.error(f"读取进度文件失败: {e}")

    # 计算预计时间
    estimated_time = None
    current_step = None
    total_steps = task.steps

    if task.progress > 0 and task.started_at and task.status in [TaskStatus.RUNNING, TaskStatus.INITIALIZING]:
        elapsed_seconds = (datetime.utcnow() - task.started_at).total_seconds()
        if task.progress > 0.05:  # 至少完成5%才计算
            estimated_time = int((elapsed_seconds / task.progress) * (1 - task.progress))

        current_step = int(task.progress * task.steps)

    # 获取状态消息
    status_message = task.status_message
    if not status_message and task.additional_params and 'status_message' in task.additional_params:
        status_message = task.additional_params.get('status_message')

    # 特殊处理COMPLETED状态 - 确保视频文件真的存在
    if task.status == TaskStatus.COMPLETED:
        if not os.path.exists(task.output_path) or os.path.getsize(task.output_path) < 1000:
            task.status = TaskStatus.RUNNING
            task.progress = 0.95
            status_message = "等待视频生成完成..."
            db.commit()

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
        status_message=status_message,
        current_step=current_step,
        total_steps=total_steps,
        estimated_time=estimated_time
    )


@app.get("/api/tasks", response_model=List[TaskResponse])
def list_tasks(
        status: Optional[TaskStatus] = None,
        limit: int = 10,
        offset: int = 0,
        db: Session = Depends(get_db)
):
    query = db.query(VideoTask)
    if status:
        query = query.filter(VideoTask.status == status)

    total = query.count()
    tasks = query.order_by(VideoTask.created_at.desc()).offset(offset).limit(limit).all()

    result = []
    for task in tasks:
        # 获取状态消息
        status_message = task.status_message
        if not status_message and task.additional_params and 'status_message' in task.additional_params:
            status_message = task.additional_params.get('status_message')

        # 特殊处理COMPLETED状态 - 确保视频文件真的存在
        task_status = task.status
        if task_status == TaskStatus.COMPLETED:
            if not os.path.exists(task.output_path) or os.path.getsize(task.output_path) < 1000:
                task_status = TaskStatus.RUNNING
                status_message = "等待视频生成完成..."

        result.append(TaskResponse(
            id=task.id,
            status=task_status,
            type=task.type,
            progress=task.progress,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            output_url=f"/api/videos/{task.id}" if task_status == TaskStatus.COMPLETED else None,
            error_message=task.error_message,
            status_message=status_message,
            total_steps=task.steps
        ))

    return result


@app.delete("/api/tasks/{task_id}")
def cancel_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status in [TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.INITIALIZING, TaskStatus.SAVING]:
        # 如果任务在队列中，从队列移除
        if task_id in task_queue:
            task_queue.remove(task_id)

        # 如果任务正在运行，标记为取消
        if task.status != TaskStatus.QUEUED and task.gpu_id is not None:
            # 在这里你可能需要一个机制来通知GPU worker停止处理
            # 这里简化处理，直接释放GPU
            gpu_status[task.gpu_id] = False

        task.status = TaskStatus.CANCELLED
        task.status_message = "任务已取消"
        db.commit()

        # 写入进度文件通知前端
        try:
            progress_file = f"progress_{task_id}.json"
            with open(progress_file, "w") as f:
                json.dump({
                    "status": TaskStatus.CANCELLED,
                    "status_message": "任务已取消",
                    "timestamp": time.time()
                }, f)
        except Exception as e:
            logger.error(f"写入进度文件失败: {e}")

    return {"status": "success", "message": "任务已取消"}


@app.get("/api/videos/{task_id}")
def get_video(task_id: str, db: Session = Depends(get_db)):
    task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
    if not task or not task.output_path:
        raise HTTPException(status_code=404, detail="视频不存在")

    if not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="视频文件不存在，可能仍在生成中")

    if task.status != TaskStatus.COMPLETED:
        # 检查视频文件是否真的存在且有效
        if os.path.exists(task.output_path) and os.path.getsize(task.output_path) > 1000:
            # 文件存在并且大小合理，更新任务状态为已完成
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            if not task.completed_at:
                task.completed_at = datetime.utcnow()
            db.commit()
        else:
            raise HTTPException(status_code=404, detail="视频仍在生成中")

    # 使用FileResponse发送文件，添加内容类型和下载文件名
    response = FileResponse(
        task.output_path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4"
    )

    # 添加必要的响应头，防止缓存问题
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    logger.info(f"WebSocket连接已建立: 任务 {task_id}")

    # 发送连接成功消息
    await websocket.send_json({"connection_status": "connected"})

    try:
        # 连接成功后，立即发送一次当前状态
        db = SessionLocal()
        task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        if task:
            # 获取状态消息
            status_message = task.status_message
            if not status_message and task.additional_params and 'status_message' in task.additional_params:
                status_message = task.additional_params.get('status_message')

            # 计算预计时间
            estimated_time = None
            current_step = None

            if task.progress > 0 and task.started_at and task.status in [TaskStatus.RUNNING, TaskStatus.INITIALIZING]:
                elapsed_seconds = (datetime.utcnow() - task.started_at).total_seconds()
                if task.progress > 0.05:  # 至少完成5%才计算
                    estimated_time = int((elapsed_seconds / task.progress) * (1 - task.progress))

                current_step = int(task.progress * task.steps)

            # 特殊处理COMPLETED状态 - 确保视频文件真的存在
            task_status = task.status
            if task_status == TaskStatus.COMPLETED:
                if not os.path.exists(task.output_path) or os.path.getsize(task.output_path) < 1000:
                    task_status = TaskStatus.RUNNING
                    status_message = "等待视频生成完成..."

            # 发送初始状态
            await websocket.send_json({
                "status": task_status,
                "progress": task.progress,
                "status_message": status_message,
                "current_step": current_step,
                "total_steps": task.steps,
                "estimated_time": estimated_time,
                "output_url": f"/api/videos/{task_id}" if task_status == TaskStatus.COMPLETED else None,
                "timestamp": time.time()
            })
        db.close()

        # 监听进度文件变化
        last_mtime = 0
        progress_file = f"progress_{task_id}.json"

        while True:
            # 检查进度文件是否更新
            try:
                if os.path.exists(progress_file):
                    mtime = os.path.getmtime(progress_file)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        with open(progress_file, "r") as f:
                            progress_data = json.load(f)

                            # 特殊处理COMPLETED状态 - 确保视频文件真的存在
                            if progress_data.get("status") == TaskStatus.COMPLETED:
                                db = SessionLocal()
                                task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
                                if task and (not os.path.exists(task.output_path) or os.path.getsize(
                                        task.output_path) < 1000):
                                    progress_data["status"] = TaskStatus.RUNNING
                                    progress_data["status_message"] = "等待视频生成完成..."
                                    progress_data["progress"] = 0.95
                                db.close()

                            # 添加时间戳以确保前端知道这是新消息
                            progress_data["timestamp"] = time.time()

                            await websocket.send_json(progress_data)
            except Exception as e:
                logger.error(f"读取或发送进度数据失败: {e}")

            # 接收客户端消息以保持连接活跃
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # 如果客户端发送了消息，可以在这里处理
                if msg == "ping":
                    await websocket.send_json({"pong": True, "timestamp": time.time()})
            except asyncio.TimeoutError:
                # 超时，继续循环
                pass
            except Exception as e:
                # 连接关闭或其他错误
                logger.error(f"WebSocket错误: {e}")
                break

            # 休眠一小段时间，避免CPU占用过高
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 连接关闭时移除
        if task_id in active_connections:
            del active_connections[task_id]
            logger.info(f"WebSocket连接已关闭: 任务 {task_id}")


# 检查GPU状态和分配任务
async def process_queue(db: Session):
    # 检查是否有可用GPU和等待中的任务
    for gpu_id, is_busy in gpu_status.items():
        if not is_busy and task_queue:
            # 获取下一个任务
            task_id = task_queue[0]
            task_queue.remove(task_id)

            # 更新任务状态
            task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.INITIALIZING
                task.started_at = datetime.utcnow()
                task.gpu_id = gpu_id
                task.status_message = f"已分配GPU {gpu_id}，正在初始化..."
                db.commit()

                # 更新进度文件
                try:
                    progress_file = f"progress_{task_id}.json"
                    with open(progress_file, "w") as f:
                        json.dump({
                            "status": TaskStatus.INITIALIZING,
                            "progress": 0.0,
                            "status_message": f"已分配GPU {gpu_id}，正在初始化...",
                            "timestamp": time.time()
                        }, f)
                except Exception as e:
                    logger.error(f"写入进度文件失败: {e}")

                # 标记GPU为忙碌
                gpu_status[gpu_id] = True

                logger.info(f"已将任务 {task_id} 分配给GPU {gpu_id}")


# 添加前端静态文件服务
@app.middleware("http")
async def check_if_static_file(request: Request, call_next):
    # 检查是否是API请求
    if request.url.path.startswith("/api/") or request.url.path.startswith("/ws/"):
        return await call_next(request)

    # 检查是否存在静态文件
    static_path = os.path.join("frontend/build", request.url.path.lstrip("/"))
    if os.path.exists(static_path) and os.path.isfile(static_path):
        return await call_next(request)

    # 如果不是API请求且没有找到静态文件，返回index.html（SPA模式）
    return FileResponse(os.path.join("frontend/build", "index.html"))


app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")


# 检查GPU状态的端点
@app.get("/api/system/status")
def get_system_status():
    # 获取GPU状态
    gpu_busy_count = sum(1 for is_busy in gpu_status.values() if is_busy)

    # 获取队列长度
    queue_length = len(task_queue)

    return {
        "total_gpus": len(gpu_status),
        "busy_gpus": gpu_busy_count,
        "queue_length": queue_length,
        "gpu_status": gpu_status,
        "timestamp": time.time()
    }


# 检查文件是否存在
@app.get("/api/files/check/{file_type}/{file_id}")
def check_file_exists(file_type: str, file_id: str):
    if file_type == "video":
        file_path = os.path.join("outputs", f"{file_id}.mp4")
        if os.path.exists(file_path):
            return {
                "exists": True,
                "size": os.path.getsize(file_path),
                "last_modified": os.path.getmtime(file_path)
            }
    elif file_type == "image":
        # 检查所有可能的图片扩展名
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            file_path = os.path.join("uploads", f"{file_id}{ext}")
            if os.path.exists(file_path):
                return {
                    "exists": True,
                    "size": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path)
                }

    return {"exists": False}


# 主入口
if __name__ == "__main__":
    import asyncio

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)