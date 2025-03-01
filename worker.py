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


# 修改后的 RawTqdm：根据传入的 timestep 数值计算进度（如果可解析），否则回退到按迭代次数计算
class RawTqdm:
    """完全透明的进度传递，同步更新进度，并根据 timestep 数值计算进度百分比"""
    def __init__(self, iterable):
        self.iterable = list(iterable) if hasattr(iterable, '__len__') else list(iterable)
        self.total = len(self.iterable)
        self.task_id = None
        self.last_log_time = time.time()
        self.log_interval = 2  # 日志记录间隔（秒）
        self.first_value = None
        self.last_value = None
        # 尝试解析 iterable 中的第一个和最后一个 timestep 数值
        try:
            first_item = self.iterable[0]
            last_item = self.iterable[-1]
            if hasattr(first_item, 'item'):
                self.first_value = float(first_item.item())
            else:
                self.first_value = float(first_item)
            if hasattr(last_item, 'item'):
                self.last_value = float(last_item.item())
            else:
                self.last_value = float(last_item)
        except Exception as e:
            logger.error(f"无法解析 timesteps 数值: {e}")
            self.first_value = None
            self.last_value = None

    def set_task_id(self, task_id):
        self.task_id = task_id
        return self

    def __iter__(self):
        current_index = 0
        start_time = time.time()
        for item in self.iterable:
            yield item
            current_index += 1

            # 尝试根据 timestep 数值计算进度
            progress = current_index / self.total  # 默认值
            if self.first_value is not None and self.last_value is not None:
                try:
                    if hasattr(item, 'item'):
                        current_val = float(item.item())
                    else:
                        current_val = float(item)
                    # 假设 timesteps 为降序排列：初始值大，最终值小
                    progress = (self.first_value - current_val) / (self.first_value - self.last_value)
                    progress = max(0, min(progress, 1))
                except Exception as e:
                    logger.error(f"计算进度时出错: {e}")

            elapsed = time.time() - start_time
            eta = elapsed / current_index * (self.total - current_index) if current_index > 0 else 0

            now = time.time()
            if now - self.last_log_time >= self.log_interval or current_index == 1 or current_index == self.total:
                logger.info(f"去噪进度: {current_index}/{self.total} ({progress * 100:.1f}%) [ETA: {eta:.1f}s]")
                self.last_log_time = now

            if self.task_id:
                self.update_progress_sync(progress, current_index, self.total, eta)

    def update_progress_sync(self, progress, current, total, eta):
        try:
            # 更新数据库中的进度
            db = SessionLocal()
            try:
                task = db.query(VideoTask).filter(VideoTask.id == self.task_id).first()
                if task:
                    task.progress = progress
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
            # 同步发送 WebSocket 通知
            try:
                from app import notify_client
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(notify_client(self.task_id, {
                    "status": "running",
                    "progress": progress,
                    "step": current,
                    "total_steps": total,
                    "eta": eta
                }))
                loop.close()
            except Exception as e:
                logger.error(f"发送 WebSocket 通知失败: {e}")
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
            torch_dtype = torch.bfloat16
            if precision == ModelPrecision.FP8:
                torch_dtype = torch.float8_e4m3fn
            model_files = []
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
            self.model_manager.load_models(
                model_files,
                torch_dtype=torch_dtype,
            )
            # 注意：WanVideoPipeline 为库代码，不能修改
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
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 0.0
            db.commit()
            from app import notify_client
            await notify_client(task.id, {
                "status": "running",
                "progress": 0.0,
                "message": "开始加载模型"
            })
            logger.info(f"开始加载模型: {task.id}")
            model_loaded = await self.load_model(task.type, task.resolution, task.model_precision)
            if not model_loaded:
                raise Exception("模型加载失败")
            width, height = self._get_resolution_dimensions(task.resolution)
            num_persistent_param = None if not task.save_vram else 0
            self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)
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
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                from PIL import Image
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image
            generation_params["progress_bar_cmd"] = lambda x: RawTqdm(x).set_task_id(task.id)
            logger.info(f"开始生成视频: {task.id}")
            video = self.current_pipeline(**generation_params)
            logger.info(f"保存视频: {task.id}")
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            db.commit()
            await notify_client(task.id, {
                "status": "completed",
                "progress": 1.0,
                "output_url": f"/api/videos/{task.id}"
            })
            self.current_pipeline = None
            torch.cuda.empty_cache()
            logger.info(f"视频生成完成: {task.id}")
            return True
        except Exception as e:
            logger.error(f"生成视频失败: {task.id}, 错误: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()
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
            return 854, 480

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
            db = SessionLocal()
            task = db.query(VideoTask).filter(
                VideoTask.status == TaskStatus.RUNNING,
                VideoTask.gpu_id == gpu_id
            ).first()
            if task:
                running_task = task.id
                logger.info(f"GPU {gpu_id} 开始处理任务 {task.id}")
                await worker.process_task(task)
                running_task = None
            db.close()
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
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        asyncio.run(worker_main(gpu_id))
    except KeyboardInterrupt:
        logger.info("收到Keyboard Interrupt，退出...")
    except Exception as e:
        logger.error(f"Worker主循环出错: {e}")
