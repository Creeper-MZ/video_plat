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


# 直接替换tqdm的简单类
class SimpleTqdm:
    def __init__(self, iterable=None, total=None, desc="Processing"):
        self.iterable = list(iterable) if iterable is not None else None
        self.total = total if total is not None else (len(self.iterable) if self.iterable is not None else None)
        self.desc = desc
        self.n = 0
        self.start_time = time.time()
        self.task_id = None
        self.callback_fn = None
        self.last_print_time = time.time()
        self.print_interval = 0.2  # 200ms更新间隔

    def set_callback(self, task_id, callback_fn):
        self.task_id = task_id
        self.callback_fn = callback_fn
        return self

    def __iter__(self):
        if self.iterable is None:
            return self

        try:
            # 发送初始进度
            self._send_progress(0)

            for obj in self.iterable:
                yield obj
                self.n += 1

                # 控制进度更新频率
                now = time.time()
                if now - self.last_print_time > self.print_interval or self.n == self.total:
                    self._send_progress(self.n)
                    self.last_print_time = now

        finally:
            # 确保发送最终进度
            self._send_progress(self.total or self.n)

    def _send_progress(self, current):
        if self.total:
            progress = current / self.total
        else:
            progress = 0

        # 计算ETA和速度
        elapsed = time.time() - self.start_time
        if current > 0:
            rate = current / elapsed
            eta = (self.total - current) / rate if self.total else 0
        else:
            rate = 0
            eta = 0

        # 每10%或首尾打印一次日志
        if (int(progress * 10) > int(
                (current - 1) / self.total * 10) if self.total else False) or current == 1 or current == self.total:
            logger.info(f"{self.desc}: {current}/{self.total} [{progress * 100:.1f}%] - ETA: {eta:.1f}s")

        # 发送回调
        if self.callback_fn and self.task_id:
            info = {
                "task_id": self.task_id,
                "progress": 0.1 + progress * 0.8,  # 占总进度的80%
                "stage": "denoising",
                "stage_progress": progress,
                "step": current,
                "total_steps": self.total,
                "eta": eta,
                "logs": [f"去噪步骤: {current}/{self.total} ({progress * 100:.1f}%), ETA: {eta:.1f}秒"]
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

            # 发送模型加载进度
            await self.update_task_status(0.05, "loading_models", 0.2,
                                          f"开始加载{model_type}模型 ({resolution}, {precision})")

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

            # 获取模型文件列表
            await self.update_task_status(0.05, "loading_models", 0.4, f"加载模型文件: {model_path}")

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
            await self.update_task_status(0.05, "loading_models", 0.6, "加载模型权重")
            self.model_manager.load_models(
                model_files,
                torch_dtype=torch_dtype,
            )

            # 创建管道
            await self.update_task_status(0.05, "loading_models", 0.8, "初始化模型推理管道")
            self.current_pipeline = WanVideoPipeline.from_model_manager(
                self.model_manager,
                torch_dtype=torch.bfloat16,
                device=self.device
            )

            await self.update_task_status(0.05, "loading_models", 1.0, "模型加载完成")
            logger.info(f"GPU Worker {self.gpu_id} 模型加载完成")
            return True

        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} 加载模型失败: {e}")
            await self.update_task_status(0.05, "failed", 0, f"模型加载失败: {str(e)}")
            return False

    async def process_task(self, task):
        self.current_task = task
        db = SessionLocal()

        try:
            # 更新任务状态为运行中
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 0.01
            db.commit()

            # 初始化阶段
            await self.update_task_status(0.01, "initializing", 0.2, "初始化任务参数")

            # 计算实际分辨率
            width, height = self._get_resolution_dimensions(task.resolution)

            # 添加基本信息
            additional_info = task.additional_params or {}
            additional_info["resolution"] = f"{width}x{height}"
            additional_info["gpu"] = f"GPU {self.gpu_id}"
            additional_info["task_type"] = "文生视频" if task.type == VideoType.TEXT_TO_VIDEO else "图生视频"
            additional_info["frames"] = task.frames
            additional_info["steps"] = task.steps
            task.additional_params = additional_info
            db.commit()

            # 加载模型
            await self.update_task_status(0.03, "initializing", 0.6, "准备加载模型")
            model_loaded = await self.load_model(task.type, task.resolution, task.model_precision)
            if not model_loaded:
                raise Exception("模型加载失败")

            # 设置显存管理配置
            await self.update_task_status(0.09, "initializing", 0.8, "配置显存管理")
            num_persistent_param = None if not task.save_vram else 0
            self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

            # 准备生成参数
            await self.update_task_status(0.09, "initializing", 1.0, "准备生成参数")
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
                await self.update_task_status(0.1, "encoding_image", 0.5, "加载输入图片")
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image
                await self.update_task_status(0.1, "encoding_image", 1.0, "图片加载完成")

            # 自定义tqdm替代
            tqdm_obj = SimpleTqdm()
            tqdm_obj.set_callback(task.id, self.update_progress_callback)
            generation_params["progress_bar_cmd"] = lambda x, **kwargs: tqdm_obj.set_callback(task.id,
                                                                                              self.update_progress_callback).__iter__(
                x)

            # 生成视频
            logger.info(f"开始生成视频 任务ID: {task.id}")
            video = self.current_pipeline(**generation_params)

            # 保存视频
            await self.update_task_status(0.9, "saving", 0.3, "视频生成完成，正在保存")
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 更新任务状态
            await self.update_task_status(0.9, "saving", 1.0, "视频保存完成")
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            db.commit()

            # 释放资源
            self.current_pipeline = None
            torch.cuda.empty_cache()

            logger.info(f"视频生成完成 任务ID: {task.id}")
            return True

        except Exception as e:
            logger.error(f"生成视频失败 任务ID: {task.id}, 错误: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)

            # 添加错误信息到additional_params
            additional_info = task.additional_params or {}
            additional_info["error_detail"] = str(e)
            additional_info["stage"] = "failed"
            task.additional_params = additional_info

            db.commit()
            return False

        finally:
            db.close()
            self.current_task = None

    async def update_task_status(self, overall_progress, stage, stage_progress, message):
        """更新任务状态"""
        if not self.current_task:
            return

        task_id = self.current_task.id
        db = SessionLocal()

        try:
            task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
            if not task:
                return

            # 更新进度
            task.progress = overall_progress + stage_progress * 0.1  # 假设每个阶段占总进度的10%

            # 更新额外信息
            additional_info = task.additional_params or {}
            additional_info["stage"] = stage
            additional_info["stage_progress"] = stage_progress
            additional_info["last_message"] = message
            additional_info["logs"] = additional_info.get("logs", [])
            additional_info["logs"].append(f"[{stage}] {message}")

            # 保留最新的10条日志
            if len(additional_info["logs"]) > 10:
                additional_info["logs"] = additional_info["logs"][-10:]

            task.additional_params = additional_info
            db.commit()

            # 通过WebSocket通知前端
            message_data = {
                "status": task.status,
                "progress": task.progress,
                "stage": stage,
                "stage_progress": stage_progress,
                "message": message,
                "logs": additional_info["logs"]
            }

            # 导入通知函数
            from app import notify_client
            await notify_client(task_id, message_data)

        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")
        finally:
            db.close()

    async def update_progress_callback(self, info):
        """更新任务进度的回调函数"""
        if not self.current_task:
            return

        db = SessionLocal()
        try:
            task = db.query(VideoTask).filter(VideoTask.id == info["task_id"]).first()
            if task:
                task.progress = info["progress"]

                # 添加额外信息到JSON字段
                additional_info = task.additional_params or {}
                additional_info["stage"] = info["stage"]
                additional_info["stage_progress"] = info["stage_progress"]
                additional_info["step"] = info["step"]
                additional_info["total_steps"] = info["total_steps"]
                additional_info["eta"] = info["eta"]

                if "logs" in info:
                    # 保留最新的10条日志
                    logs = additional_info.get("logs", [])
                    logs.extend(info["logs"])
                    additional_info["logs"] = logs[-10:]

                task.additional_params = additional_info
                db.commit()

                # 通过WebSocket通知前端
                from app import notify_client
                await notify_client(task.id, info)

        except Exception as e:
            logger.error(f"更新任务进度失败: {e}")
        finally:
            db.close()

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
            # 在实际情况下，这里应该调用Wan2.1的中断方法
            # 现在只能简单记录
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
            await asyncio.sleep(10)  # 错误后等待更长时间

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