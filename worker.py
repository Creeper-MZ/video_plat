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


# 直接透明传递WanVideoPipeline进度的tqdm替代
class RawTqdm:
    """完全透明的进度传递，直接传递WanVideoPipeline的实际进度"""

    def __init__(self, iterable):
        self.iterable = list(iterable) if hasattr(iterable, '__len__') else list(iterable)
        self.total = len(self.iterable)
        self.task_id = None
        self.last_log_time = time.time()
        self.log_interval = 2  # 日志记录间隔（秒）

    def set_task_id(self, task_id):
        self.task_id = task_id
        return self

    def __iter__(self):
        current = 0
        start_time = time.time()
        for item in self.iterable:
            yield item
            current += 1

            # 计算纯粹的原始进度值
            raw_progress = current / self.total

            # 计算ETA
            elapsed = time.time() - start_time
            eta = elapsed / current * (self.total - current) if current > 0 else 0

            # 记录日志
            now = time.time()
            if now - self.last_log_time >= self.log_interval or current == 1 or current == self.total:
                logger.info(f"去噪进度: {current}/{self.total} ({raw_progress * 100:.1f}%) [ETA: {eta:.1f}s]")
                self.last_log_time = now

            # 原子化更新数据库和发送通知
            if self.task_id:
                asyncio.create_task(self._update_progress(raw_progress, current, self.total, eta))

    async def _update_progress(self, progress, current, total, eta):
        """更新进度到数据库并发送WebSocket通知"""
        try:
            # 更新数据库
            db = SessionLocal()
            try:
                task = db.query(VideoTask).filter(VideoTask.id == self.task_id).first()
                if task:
                    # 直接更新原始进度值
                    task.progress = progress

                    # 更新其他有用信息
                    additional_info = task.additional_params or {}
                    additional_info["step"] = current
                    additional_info["total_steps"] = total
                    additional_info["eta"] = eta
                    task.additional_params = additional_info

                    db.commit()
            except Exception as e:
                logger.error(f"更新数据库进度失败: {e}")
            finally:
                db.close()

            # 发送WebSocket通知
            try:
                from app import notify_client
                await notify_client(self.task_id, {
                    "status": "running",
                    "progress": progress,  # 直接传递原始进度
                    "step": current,
                    "total_steps": total,
                    "eta": eta
                })
            except Exception as e:
                logger.error(f"发送WebSocket通知失败: {e}")

        except Exception as e:
            logger.error(f"进度更新失败: {e}")


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
            self.model_manager.load_models(
                model_files,
                torch_dtype=torch_dtype,
            )

            # 创建管道
            self.current_pipeline = WanVideoPipeline.from_model_manager(
                self.model_manager,
                torch_dtype=torch.bfloat16,
                device=self.device
            )

            logger.info(f"GPU Worker {self.gpu_id} 模型加载完成")
            return True

        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} 加载模型失败: {e}")
            return False

    async def process_task(self, task):
        self.current_task = task
        db = SessionLocal()

        try:
            # 更新任务状态为运行中
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 0.0  # 初始进度为0
            db.commit()

            # 初始化通知
            from app import notify_client
            await notify_client(task.id, {
                "status": "running",
                "progress": 0.0,
                "message": "开始加载模型"
            })

            # 加载模型
            logger.info(f"开始加载模型: {task.id}")
            model_loaded = await self.load_model(task.type, task.resolution, task.model_precision)
            if not model_loaded:
                raise Exception("模型加载失败")

            # 计算实际分辨率
            width, height = self._get_resolution_dimensions(task.resolution)

            # 设置显存管理配置
            num_persistent_param = None if not task.save_vram else 0
            self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

            # 准备生成参数
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
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image

            # 使用纯净的进度传递
            generation_params["progress_bar_cmd"] = lambda x: RawTqdm(x).set_task_id(task.id)

            # 生成视频
            logger.info(f"开始生成视频: {task.id}")
            video = self.current_pipeline(**generation_params)

            # 保存视频
            logger.info(f"保存视频: {task.id}")
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 更新任务状态为完成
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            db.commit()

            # 完成通知
            await notify_client(task.id, {
                "status": "completed",
                "progress": 1.0,
                "output_url": f"/api/videos/{task.id}"
            })

            # 释放资源
            self.current_pipeline = None
            torch.cuda.empty_cache()

            logger.info(f"视频生成完成: {task.id}")
            return True

        except Exception as e:
            logger.error(f"生成视频失败: {task.id}, 错误: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()

            # 失败通知
            from app import notify_client
            await notify_client(task.id, {
                "status": "failed",
                "error": str(e)
            })

            return False

        finally:
            db.close()
            self.current_task = None

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