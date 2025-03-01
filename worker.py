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
from progress_tracker import ProgressTracker, StageType

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


class GPUWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_manager = None
        self.current_pipeline = None
        self.current_task = None
        self.progress_tracker = None

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
        if self.progress_tracker:
            self.progress_tracker.set_stage(StageType.LOADING_MODELS,
                                            f"加载{model_type}模型，分辨率：{resolution}，精度：{precision}")

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
            t5_model_file = ""
            vae_model_file = ""

            if self.progress_tracker:
                self.progress_tracker.log(f"正在加载模型，路径: {model_path}")

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
            if self.progress_tracker:
                self.progress_tracker.update_stage_progress(0.2, message="开始加载模型权重")

            self.model_manager.load_models(
                model_files,
                torch_dtype=torch_dtype,
            )

            if self.progress_tracker:
                self.progress_tracker.update_stage_progress(0.6, message="模型权重加载完成，创建推理管道")

            # 创建管道
            self.current_pipeline = WanVideoPipeline.from_model_manager(
                self.model_manager,
                torch_dtype=torch.bfloat16,
                device=self.device
            )

            if self.progress_tracker:
                self.progress_tracker.update_stage_progress(1.0, message="推理管道创建完成")

            logger.info(f"GPU Worker {self.gpu_id} 模型加载完成")
            return True

        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} 加载模型失败: {e}")
            if self.progress_tracker:
                self.progress_tracker.log(f"模型加载失败: {str(e)}", level="error")
            return False

    async def process_task(self, task):
        self.current_task = task
        db = SessionLocal()

        try:
            # 设置进度跟踪器
            self.progress_tracker = ProgressTracker(task.id, callback=self.update_progress_callback)
            self.progress_tracker.set_stage(StageType.INITIALIZING, "准备开始视频生成")

            # 计算实际分辨率
            width, height = self._get_resolution_dimensions(task.resolution)
            self.progress_tracker.add_info("resolution", f"{width}x{height}")

            # 设置显存管理配置
            self.progress_tracker.update_stage_progress(0.3, message="配置显存管理")
            num_persistent_param = None if not task.save_vram else 0
            self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

            # 准备生成参数
            self.progress_tracker.update_stage_progress(0.5, message="准备生成参数")
            generation_params = {
                "prompt": task.prompt,
                "negative_prompt": task.negative_prompt,
                "num_inference_steps": task.steps,
                "num_frames": task.frames,
                "height": height,
                "width": width,
                "seed": task.seed if task.seed >= 0 else None,
                "tiled": task.tiled,
            }

            if task.type == VideoType.TEXT_TO_VIDEO:
                self.progress_tracker.add_info("type", "文生视频")
            else:
                self.progress_tracker.add_info("type", "图生视频")

            self.progress_tracker.add_info("steps", task.steps)
            self.progress_tracker.add_info("frames", task.frames)
            self.progress_tracker.add_info("gpu", f"GPU {self.gpu_id}")

            # 如果是图生视频，加载输入图片
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                self.progress_tracker.update_stage_progress(0.7, message="加载输入图片")
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image

            # 注册自定义进度回调
            self.progress_tracker.update_stage_progress(1.0, message="初始化完成")

            # 加载模型
            model_loaded = await self.load_model(task.type, task.resolution, task.model_precision)
            if not model_loaded:
                raise Exception("模型加载失败")

            # 开始生成视频
            self.progress_tracker.set_stage(StageType.DENOISING, "开始生成视频")
            logger.info(f"开始生成视频 任务ID: {task.id}, 参数: {generation_params}")

            # 定义转换tqdm回调的函数
            def progress_callback(iterator):
                return self.progress_tracker.tqdm_callback(iterator, desc="正在去噪")

            # 添加进度回调
            generation_params["progress_bar_cmd"] = progress_callback

            # 生成视频
            video = self.current_pipeline(**generation_params)

            # 保存视频
            self.progress_tracker.set_stage(StageType.SAVING, "正在保存视频")
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            db.commit()

            # 释放资源
            self.progress_tracker.log("释放模型资源")
            self.current_pipeline = None
            torch.cuda.empty_cache()

            self.progress_tracker.complete(True, "视频生成成功")
            logger.info(f"视频生成完成 任务ID: {task.id}")
            return True

        except Exception as e:
            logger.error(f"生成视频失败 任务ID: {task.id}, 错误: {e}")
            if self.progress_tracker:
                self.progress_tracker.complete(False, f"视频生成失败: {str(e)}")

            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()
            return False

        finally:
            db.close()
            self.current_task = None
            self.progress_tracker = None
            torch.cuda.empty_cache()  # 确保释放GPU内存

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

    async def update_progress_callback(self, progress_info):
        """更新任务进度的回调函数"""
        if not self.current_task:
            return

        db = SessionLocal()
        try:
            task = db.query(VideoTask).filter(VideoTask.id == progress_info["task_id"]).first()
            if task:
                task.progress = progress_info["progress"]

                # 添加额外信息到JSON字段
                additional_info = task.additional_params or {}
                additional_info["stage"] = progress_info["stage"]
                additional_info["stage_progress"] = progress_info["stage_progress"]
                additional_info["elapsed"] = progress_info["elapsed"]
                additional_info["eta"] = progress_info["eta"]
                additional_info["step"] = progress_info["step"]
                additional_info["total_steps"] = progress_info["total_steps"]
                additional_info["logs"] = progress_info["logs"]

                if "type" in progress_info:
                    additional_info["type"] = progress_info["type"]
                if "resolution" in progress_info:
                    additional_info["resolution"] = progress_info["resolution"]
                if "steps" in progress_info:
                    additional_info["steps"] = progress_info["steps"]
                if "frames" in progress_info:
                    additional_info["frames"] = progress_info["frames"]
                if "gpu" in progress_info:
                    additional_info["gpu"] = progress_info["gpu"]

                task.additional_params = additional_info
                db.commit()

                # 通过WebSocket通知前端
                message = {
                    "status": task.status,
                    "progress": task.progress,
                    "stage": progress_info["stage"],
                    "stage_progress": progress_info["stage_progress"],
                    "eta": progress_info["eta"],
                    "step": progress_info["step"],
                    "total_steps": progress_info["total_steps"],
                    "logs": progress_info["logs"],
                }

                # 调用app.py中的notify_client函数
                # 因为这个函数在另一个模块中，我们需要导入它
                from app import notify_client
                await notify_client(task.id, message)

        except Exception as e:
            logger.error(f"更新任务进度失败: {e}")
        finally:
            db.close()

    def stop_current_task(self):
        if self.current_pipeline:
            # 在实际情况下，这里应该调用Wan2.1的中断方法
            # 现在只能简单记录
            logger.info(f"尝试停止GPU {self.gpu_id}上的当前任务")

            if self.progress_tracker:
                self.progress_tracker.complete(False, "任务被用户取消")
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

                # 加载模型
                model_loaded = await worker.load_model(task.type, task.resolution, task.model_precision)

                if model_loaded:
                    # 处理任务
                    await worker.process_task(task)
                else:
                    # 模型加载失败
                    task.status = TaskStatus.FAILED
                    task.error_message = "模型加载失败"
                    db.commit()

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