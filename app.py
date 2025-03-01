from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import redis
import json
import asyncio
import os
import uuid
import logging
from typing import Optional, List
import uvicorn
import time
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="Wan2.1视频生成平台")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis连接
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# 模型路径配置
MODEL_PATHS = {
    "t2v": "/home/ps/videoGen/models/Wan2.1-T2V-14B/",
    "i2v-480p": "/home/ps/videoGen/models/Wan2.1-I2V-14B-480P/",
    "i2v-720p": "/home/ps/videoGen/models/Wan2.1-I2V-14B-720P/",
}

# 默认负面提示词
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# 分辨率选项
class Resolution(str, Enum):
    RES_720P = "1280x720"
    RES_720P_PORTRAIT = "720x1280"
    RES_480P = "832x480"
    RES_480P_PORTRAIT = "480x832"

# 创建输出目录
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# 存储活跃WebSocket连接
active_connections = {}

# 存储任务取消标志
cancel_flags = {}

# 基本模型参数
class GenerationParams(BaseModel):
    task_type: str = Field(..., description="生成任务类型: 't2v' 或 'i2v'")
    prompt: str = Field(..., description="生成提示词")
    negative_prompt: str = Field(DEFAULT_NEGATIVE_PROMPT, description="负面提示词")
    resolution: Resolution = Field(Resolution.RES_480P, description="输出分辨率")
    num_frames: int = Field(100, description="帧数", ge=5, le=200)
    fps: int = Field(20, description="帧率", ge=5, le=60)
    num_inference_steps: int = Field(40, description="推理步数", ge=20, le=60)

    # 高级参数
    fp8: bool = Field(True, description="是否使用FP8精度")
    save_vram: bool = Field(False, description="是否启用节省显存模式")
    seed: Optional[int] = Field(None, description="随机种子")
    guidance_scale: float = Field(5.0, description="引导比例", ge=1.0, le=10.0)
    sample_shift: float = Field(5.0, description="采样偏移")

# 任务状态更新
class TaskUpdate(BaseModel):
    task_id: str
    status: str
    progress: float = 0
    message: str = ""
    output_path: Optional[str] = None

# 生成任务队列管理
@app.post("/api/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    task_type: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    resolution: Resolution = Form(Resolution.RES_480P),
    num_frames: int = Form(100),
    fps: int = Form(20),
    num_inference_steps: int = Form(40),
    fp8: bool = Form(True),
    save_vram: bool = Form(False),
    seed: Optional[int] = Form(None),
    guidance_scale: float = Form(5.0),
    sample_shift: float = Form(5.0),
    image: Optional[UploadFile] = File(None)
):
    # 验证参数
    if task_type == "i2v" and image is None:
        raise HTTPException(status_code=400, detail="图生视频需要上传图片")

    # 创建任务ID
    task_id = str(uuid.uuid4())

    # 保存上传的图片（如果有）
    image_path = None
    if image:
        image_path = f"{OUTPUT_DIR}/{task_id}_input.jpg"
        with open(image_path, "wb") as f:
            f.write(image.file.read())

    # 准备任务数据
    task_data = {
        "task_id": task_id,
        "task_type": task_type,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": resolution,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
        "fp8": fp8,
        "save_vram": save_vram,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "sample_shift": sample_shift,
        "image_path": image_path,
        "status": "pending",
        "created_at": time.time()
    }

    # 将任务加入队列
    redis_client.lpush("video_generation_queue", json.dumps(task_data))
    redis_client.hset("tasks", task_id, json.dumps(task_data))

    # 设置取消标志（初始为False）
    cancel_flags[task_id] = False

    logger.info(f"任务 {task_id} 已添加到队列")

    return {"task_id": task_id, "status": "pending"}

# 获取任务状态
@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task_data = redis_client.hget("tasks", task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")

    return json.loads(task_data)

# 取消任务
@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    task_data = redis_client.hget("tasks", task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_data)
    if task_data["status"] in ["completed", "failed", "cancelled"]:
        return {"status": "已完成或已取消的任务无法取消"}

    # 设置取消标志
    cancel_flags[task_id] = True

    # 更新任务状态
    task_data["status"] = "cancelling"
    redis_client.hset("tasks", task_id, json.dumps(task_data))

    logger.info(f"请求取消任务 {task_id}")

    return {"status": "cancelling"}

# 获取所有任务
@app.get("/api/tasks")
async def get_all_tasks():
    tasks = redis_client.hgetall("tasks")
    result = []

    for task_id, task_data in tasks.items():
        result.append(json.loads(task_data))

    # 按创建时间排序
    result.sort(key=lambda x: x.get("created_at", 0), reverse=True)

    return result

# WebSocket连接用于实时更新
@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()

    # 保存WebSocket连接
    active_connections[task_id] = websocket

    try:
        # 检查任务是否存在
        task_data = redis_client.hget("tasks", task_id)
        if not task_data:
            await websocket.send_json({"error": "任务不存在"})
            await websocket.close()
            return

        # 发送当前任务状态
        await websocket.send_json(json.loads(task_data))

        # 保持连接直到客户端断开
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info(f"客户端断开WebSocket连接: {task_id}")
    finally:
        # 移除连接
        if task_id in active_connections:
            del active_connections[task_id]

# 更新任务状态（供Worker调用）
@app.post("/api/internal/update_task")
async def update_task(update: TaskUpdate):
    task_id = update.task_id

    # 获取当前任务状态
    task_data = redis_client.hget("tasks", task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = json.loads(task_data)

    # 更新任务状态
    task_data["status"] = update.status
    task_data["progress"] = update.progress
    task_data["message"] = update.message

    if update.output_path:
        task_data["output_path"] = update.output_path

    # 保存更新后的任务状态
    redis_client.hset("tasks", task_id, json.dumps(task_data))

    # 通过WebSocket发送更新
    if task_id in active_connections:
        try:
            await active_connections[task_id].send_json(task_data)
        except Exception as e:
            logger.error(f"发送WebSocket更新失败: {e}")

    return {"success": True}

# 检查任务是否被取消（供Worker调用）
@app.get("/api/internal/check_cancel/{task_id}")
async def check_cancel(task_id: str):
    is_cancelled = cancel_flags.get(task_id, False)
    return {"cancelled": is_cancelled}

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)