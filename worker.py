import os
import time
import torch
import signal
import asyncio
import logging
from PIL import Image
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import VideoTask, TaskStatus, VideoType, Resolution, ModelPrecision, Base

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("worker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库设置
DATABASE_URL = "sqlite:///./video_generation.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 确保数据库表存在
Base.metadata.create_all(bind=engine)

# 全局变量
running_task = None
should_stop = False


# 严格遵循WanVideoPipeline进度的tqdm替代
class PipelineTqdm:
    def __init__(self, iterable, **kwargs):
        """只传递WanVideoPipeline的真实进度"""
        self.iterable = list(iterable)
        self.total = len(self.iterable)
        self.current = 0
        self.task_id = None
        self.callback_fn = None
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 5  # 每5秒记录一次日志

    def set_task(self, task_id, callback_fn):
        """设置任务ID和回调函数"""
        self.task_id = task_id
        self.callback_fn = callback_fn
        return self

    def __iter__(self):
        """迭代并传递真实进度"""
        for item in self.iterable:
            yield item
            self.current += 1

            # 计算真实进度
            progress = self.current / self.total

            # 计算ETA
            elapsed = time.time() - self.start_time
            eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0

            # 记录日志（每5秒或每10%记录一次）
            now = time.time()
            if (now - self.last_log_time > self.log_interval or
                    (self.current == 1 or self.current == self.total or
                     int(progress * 10) > int((self.current - 1) / self.total * 10))):
                logger.info(
                    f"WanVideoPipeline进度: {self.current}/{self.total} ({progress * 100:.1f}%) [ETA: {eta:.1f}s]")
                self.last_log_time = now

            # 传递真实的进度信息
            if self.callback_fn and self.task_id:
                # 构建日志消息
                log_message = f"去噪步骤: {self.current}/{self.total} ({progress * 100:.1f}%)"
                if eta > 0:
                    log_message += f", 预计剩余: {eta:.1f}秒"

                info = {
                    "task_id": self.task_id,
                    "status": "running",
                    "progress": progress,  # 进度完全基于Pipeline输出
                    "stage": "denoising",
                    "stage_progress": progress,
                    "step": self.current,
                    "total_steps": self.total,
                    "eta": eta,
                    "logs": [log_message]
                }

                try:
                    if asyncio.iscoroutinefunction(self.callback_fn):
                        asyncio.create_task(self.callback_fn(info))
                    else:
                        self.callback_fn(info)
                except Exception as e:
                    logger.error(f"进度回调失败: {e}")


class GPUWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_manager = None
        self.current_pipeline = None
        self.current_task = None

    async def initialize(self):
        logger.info(f"GPU Worker {self.gpu_id} 初始化中...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        try:
            # 导入依赖库
            from diffsynth import ModelManager
            self.model_manager = ModelManager(device="cpu")  # 初始时加载到CPU
            logger.info(f"GPU Worker {self.gpu_id} 初始化完成")
        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} 初始化失败: {e}")
            raise e

    async def load_model(self, model_type, resolution, precision):
        logger.info(f"GPU Worker {self.gpu_id} 加载模型: {model_type}, {resolution}, {precision}")

        try:
            from diffsynth import WanVideoPipeline

            # 记录日志但不影响进度
            log_message = f"开始加载{model_type}模型 ({resolution}, {precision})"
            await self.log_process_step(log_message)

            # 根据分辨率和类型选择模型路径
            model_path = None
            if model_type == VideoType.TEXT_TO_VIDEO:
                model_path = "/home/ps/videoGen/models/Wan2.1-T2V-14B/"
            elif model_type == VideoType.IMAGE_TO_VIDEO:
                if resolution in [Resolution.RESOLUTION_480P, Resolution.RESOLUTION_480P_VERTICAL]:
                    model_path = "/home/ps/videoGen/models/Wan2.1-I2V-14B-480P/"
                else:
                    model_path = "/home/ps/videoGen/models/Wan2.1-I2V-14B-720P/"

            if not model_path:
                raise ValueError(f"无效的模型类型或分辨率: {model_type}, {resolution}")

            # 设置模型精度
            torch_dtype = torch.bfloat16
            if precision == ModelPrecision.FP8:
                torch_dtype = torch.float8_e4m3fn

            # 加载模型文件
            model_files = []

            # 记录模型路径
            await self.log_process_step(f"使用模型路径: {model_path}")

            # 文生视频模型
            if model_type == VideoType.TEXT_TO_VIDEO:
                model_files = [
                    [
                        f"{model_path}diffusion_pytorch_model-00001-of-00006.safetensors",
                        f"{model_path}diffusion_pytorch_model-00002-of-00006.safetensors",
                        f"{model_path}diffusion_pytorch_model-00003-of-00006.safetensors",
                        f"{model_path}diffusion_pytorch_model-00004-of-00006.safetensors",
                        f"{model_path}diffusion_pytorch_model-00005-of-00006.safetensors",
                        f"{model_path}diffusion_pytorch_model-00006-of-00006.safetensors",
                    ],
                    f"{model_path}models_t5_umt5-xxl-enc-bf16.pth",
                    f"{model_path}Wan2.1_VAE.pth"
                ]
            # 图生视频模型
            elif model_type == VideoType.IMAGE_TO_VIDEO:
                if resolution in [Resolution.RESOLUTION_480P, Resolution.RESOLUTION_480P_VERTICAL]:
                    model_files = [
                        [
                            f"{model_path}diffusion_pytorch_model-00001-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00002-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00003-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00004-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00005-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00006-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00007-of-00007.safetensors",
                        ],
                        f"{model_path}models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        f"{model_path}models_t5_umt5-xxl-enc-bf16.pth",
                        f"{model_path}Wan2.1_VAE.pth"
                    ]
                else:
                    model_files = [
                        [
                            f"{model_path}diffusion_pytorch_model-00001-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00002-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00003-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00004-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00005-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00006-of-00007.safetensors",
                            f"{model_path}diffusion_pytorch_model-00007-of-00007.safetensors",
                        ],
                        f"{model_path}models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        f"{model_path}models_t5_umt5-xxl-enc-bf16.pth",
                        f"{model_path}Wan2.1_VAE.pth"
                    ]

            # 加载模型
            await self.log_process_step("加载模型权重中...")
            self.model_manager.load_models(
                model_files,
                torch_dtype=torch_dtype,
            )

            # 创建管道
            await self.log_process_step("创建推理管道...")
            self.current_pipeline = WanVideoPipeline.from_model_manager(
                self.model_manager,
                torch_dtype=torch.bfloat16,
                device=self.device
            )

            await self.log_process_step("模型加载完成")
            logger.info(f"GPU Worker {self.gpu_id} 模型加载完成")
            return True

        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} 加载模型失败: {e}")
            await self.log_process_step(f"模型加载失败: {str(e)}", "error")
            return False

    async def process_task(self, task):
        self.current_task = task
        db = SessionLocal()

        try:
            # 更新任务状态为运行中
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 0.0  # 进度从0开始

            # 初始化附加信息
            additional_info = task.additional_params or {}
            additional_info["logs"] = []
            additional_info["stage"] = "initializing"
            task.additional_params = additional_info
            db.commit()

            # 记录开始信息
            await self.log_process_step("初始化任务...")

            # 计算实际分辨率
            width, height = self._get_resolution_dimensions(task.resolution)

            # 添加基本信息但不影响进度
            additional_info = task.additional_params or {}
            additional_info["resolution"] = f"{width}x{height}"
            additional_info["gpu"] = f"GPU {self.gpu_id}"
            additional_info["task_type"] = "文生视频" if task.type == VideoType.TEXT_TO_VIDEO else "图生视频"
            additional_info["frames"] = task.frames
            additional_info["steps"] = task.steps
            task.additional_params = additional_info
            db.commit()

            # 加载模型
            await self.log_process_step("准备加载模型...")
            model_loaded = await self.load_model(task.type, task.resolution, task.model_precision)
            if not model_loaded:
                raise Exception("模型加载失败")

            # 设置显存管理配置
            await self.log_process_step("配置显存管理...")
            num_persistent_param = None if not task.save_vram else 0
            self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

            # 准备生成参数
            await self.log_process_step("准备生成参数...")
            generation_params = {
                "prompt": task.prompt,
                "negative_prompt": task.negative_prompt,
                "num_inference_steps": task.steps,
                "height": height,
                "width": width,
                "num_frames": task.frames,
                "seed": task.seed if task.seed >= 0 else None,
                "tiled": task.tiled,
            }

            # 如果是图生视频，加载输入图片
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                await self.log_process_step("加载输入图片...")
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image

            # 创建进度回调
            tqdm_obj = PipelineTqdm(range(task.steps))
            tqdm_obj.set_task(task.id, self.update_progress_callback)

            # 设置进度回调 - 这是关键
            generation_params["progress_bar_cmd"] = lambda x, **kwargs: tqdm_obj.set_task(task.id,
                                                                                          self.update_progress_callback).__iter__(
                x)

            # 开始生成视频
            await self.log_process_step("开始生成视频...")
            video = self.current_pipeline(**generation_params)

            # 保存视频
            await self.log_process_step("视频生成完成，正在保存...")
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 更新任务状态
            await self.log_process_step("视频保存完成")
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0  # 最终进度为100%
            task.completed_at = datetime.utcnow()

            # 更新最终状态
            additional_info = task.additional_params or {}
            additional_info["stage"] = "completed"
            additional_info["stage_progress"] = 1.0
            task.additional_params = additional_info
            db.commit()

            # 发送最终完成通知
            await self.send_update({
                "task_id": task.id,
                "status": "completed",
                "progress": 1.0,
                "stage": "completed",
                "stage_progress": 1.0,
                "logs": ["视频生成和保存已完成"]
            })

            # 释放资源
            self.current_pipeline = None
            torch.cuda.empty_cache()

            logger.info(f"视频生成完成 任务ID: {task.id}")
            return True

        except Exception as e:
            logger.error(f"生成视频失败 任务ID: {task.id}, 错误: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)

            # 添加错误信息
            additional_info = task.additional_params or {}
            additional_info["error_detail"] = str(e)
            additional_info["stage"] = "failed"
            task.additional_params = additional_info

            db.commit()

            # 发送失败通知
            await self.send_update({
                "task_id": task.id,
                "status": "failed",
                "progress": 0.0,
                "stage": "failed",
                "error": str(e),
                "logs": [f"生成失败: {str(e)}"]
            })

            return False

        finally:
            db.close()
            self.current_task = None

    async def log_process_step(self, message, level="info"):
        """记录处理步骤，不影响进度"""
        if not self.current_task:
            return

        logger.info(f"[任务 {self.current_task.id}] {message}")

        db = SessionLocal()
        try:
            task = db.query(VideoTask).filter(VideoTask.id == self.current_task.id).first()
            if task:
                # 仅添加日志，不修改进度
                additional_info = task.additional_params or {}
                logs = additional_info.get("logs", [])
                logs.append(message)

                # 保留最新的10条日志
                if len(logs) > 10:
                    logs = logs[-10:]

                additional_info["logs"] = logs
                additional_info["last_message"] = message
                task.additional_params = additional_info
                db.commit()

                # 通知前端日志更新
                await self.send_update({
                    "task_id": task.id,
                    "logs": [message],
                    "last_message": message
                })

        except Exception as e:
            logger.error(f"记录处理步骤失败: {e}")
        finally:
            db.close()

    async def update_progress_callback(self, info):
        """接收并传递WanVideoPipeline的进度"""
        if not self.current_task:
            return

        db = SessionLocal()
        try:
            task = db.query(VideoTask).filter(VideoTask.id == info["task_id"]).first()
            if task:
                # 直接使用Pipeline传来的进度值
                task.progress = info["progress"]

                # 更新附加信息
                additional_info = task.additional_params or {}
                additional_info["stage"] = info["stage"]
                additional_info["stage_progress"] = info["stage_progress"]
                additional_info["step"] = info["step"]
                additional_info["total_steps"] = info["total_steps"]
                additional_info["eta"] = info.get("eta", 0)

                if "logs" in info and info["logs"]:
                    logs = additional_info.get("logs", [])
                    logs.extend(info["logs"])
                    # 保留最新的10条日志
                    additional_info["logs"] = logs[-10:]

                task.additional_params = additional_info
                db.commit()

                # 通过WebSocket通知前端
                await self.send_update(info)

        except Exception as e:
            logger.error(f"更新进度失败: {e}")
        finally:
            db.close()

    async def send_update(self, info):
        """发送更新到前端"""
        try:
            from app import notify_client
            await notify_client(info["task_id"], info)
        except Exception as e:
            logger.error(f"发送更新失败: {e}")

    def _get_resolution_dimensions(self, resolution):
        if resolution == Resolution.RESOLUTION_720P:
            return 1280, 720
        elif resolution == Resolution.RESOLUTION_720P_VERTICAL:
            return 720, 1280
        elif resolution == Resolution.RESOLUTION_480P:
            return 854, 480
        elif resolution == Resolution.RESOLUTION_480P_VERTICAL:
            return 480, 854
        else:
            return 854, 480  # 默认为480P

    def stop_current_task(self):
        if self.current_pipeline:
            logger.info(f"尝试停止GPU {self.gpu_id}上的当前任务")
        return True


async def worker_main(gpu_id):
    global running_task, should_stop

    worker = GPUWorker(gpu_id)
    await worker.initialize()

    logger.info(f"GPU Worker {gpu_id} 开始监听任务...")

    while not should_stop:
        try:
            # 查找分配给此GPU的任务
            db = SessionLocal()
            task = db.query(VideoTask).filter(
                VideoTask.status == TaskStatus.RUNNING,
                VideoTask.gpu_id == gpu_id
            ).first()

            if task:
                running_task = task.id
                logger.info(f"GPU {gpu_id} 开始处理任务 {task.id}")

                # 处理任务
                await worker.process_task(task)

                running_task = None

            db.close()

            # 等待一段时间再检查新任务
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Worker循环出错: {e}")
            await asyncio.sleep(10)

    logger.info(f"GPU Worker {gpu_id} 退出")


def signal_handler(sig, frame):
    global should_stop
    logger.info("收到停止信号，准备优雅退出...")
    should_stop = True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python worker.py <gpu_id>")
        sys.exit(1)

    gpu_id = int(sys.argv[1])
    if gpu_id not in [0, 1, 2, 3]:
        print("GPU ID必须是0-3之间的整数")
        sys.exit(1)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(worker_main(gpu_id))
    except KeyboardInterrupt:
        logger.info("收到Keyboard Interrupt，退出...")
    except Exception as e:
        logger.error(f"Worker主循环出错: {e}")