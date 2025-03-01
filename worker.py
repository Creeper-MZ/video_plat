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


# 透明传递Pipeline进度的极简类
class RawProgressTqdm:
    def __init__(self, iterable):
        self.iterable = iterable
        self.total = len(iterable)
        self.task_id = None

    def __iter__(self):
        current = 0
        for item in self.iterable:
            yield item
            current += 1

            # 直接计算原始进度
            raw_progress = current / self.total

            # 直接更新数据库
            self._update_db(raw_progress, current, self.total)

    def set_task_id(self, task_id):
        self.task_id = task_id
        return self

    def _update_db(self, progress, current, total):
        # 完全透明的进度更新
        if not self.task_id:
            return

        try:
            db = SessionLocal()
            task = db.query(VideoTask).filter(VideoTask.id == self.task_id).first()
            if task:
                # 直接设置原始进度，不做任何转换
                task.progress = progress

                # 记录当前步骤
                additional_info = task.additional_params or {}
                additional_info["step"] = current
                additional_info["total_steps"] = total
                additional_info["logs"] = additional_info.get("logs", [])
                if len(additional_info["logs"]) < 10:
                    additional_info["logs"].append(f"去噪步骤: {current}/{total}")
                task.additional_params = additional_info

                db.commit()

                # 通知websocket
                asyncio.create_task(self._notify_progress(task.id, progress, current, total))
        except Exception as e:
            logger.error(f"更新进度失败: {e}")
        finally:
            db.close()

    async def _notify_progress(self, task_id, progress, current, total):
        try:
            # 导入通知函数
            from app import notify_client

            # 发送原始进度
            await notify_client(task_id, {
                "status": "running",
                "progress": progress,  # 直接传递原始进度
                "step": current,
                "total_steps": total
            })
        except Exception as e:
            logger.error(f"通知进度失败: {e}")


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
            self.model_manager = ModelManager(device="cpu")
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

            # 进度设为0，等待实际Pipeline更新
            task.progress = 0

            # 只记录一些基本信息
            additional_info = task.additional_params or {}
            additional_info["gpu"] = f"GPU {self.gpu_id}"
            additional_info["logs"] = ["开始处理..."]
            task.additional_params = additional_info
            db.commit()

            # 加载模型 - 仅做日志，不影响进度
            logger.info(f"开始加载模型，任务ID: {task.id}")
            model_loaded = await self.load_model(task.type, task.resolution, task.model_precision)
            if not model_loaded:
                raise Exception("模型加载失败")

            # 计算尺寸
            width, height = self._get_resolution_dimensions(task.resolution)

            # 配置显存管理
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

            # 加载图片
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image

            # 创建进度回调对象 - 只传递原始进度
            tqdm_func = lambda x: RawProgressTqdm(x).set_task_id(task.id)
            generation_params["progress_bar_cmd"] = tqdm_func

            # 生成视频
            logger.info(f"开始生成视频 任务ID: {task.id}")
            video = self.current_pipeline(**generation_params)

            # 保存视频
            logger.info(f"开始保存视频 任务ID: {task.id}")
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0  # 最终进度为100%
            task.completed_at = datetime.utcnow()
            db.commit()

            # 通知前端完成
            from app import notify_client
            await notify_client(task.id, {
                "status": "completed",
                "progress": 1.0,
                "output_url": f"/api/videos/{task.id}"
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
            additional_info["error"] = str(e)
            task.additional_params = additional_info

            db.commit()

            # 通知前端失败
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