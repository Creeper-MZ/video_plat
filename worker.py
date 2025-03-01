import os
import time
import torch
import signal
import asyncio
import logging
import json
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


# 自定义进度条类，用于跟踪WanVideoPipeline进度
class ProgressBar:
    def __init__(self, task_id, db_session, total_steps, model_name="dit"):
        self.task_id = task_id
        self.db_session = db_session
        self.total_steps = total_steps
        self.model_name = model_name
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = 0

    def __call__(self, iterable):
        for item in iterable:
            self.current_step += 1

            # 限制更新频率，避免数据库负担过重
            current_time = time.time()
            if current_time - self.last_update_time > 0.5:  # 最多每0.5秒更新一次
                # 计算进度百分比
                progress = self.current_step / self.total_steps

                # 更新数据库
                try:
                    task = self.db_session.query(VideoTask).filter(VideoTask.id == self.task_id).first()
                    if task and task.status in [TaskStatus.RUNNING, TaskStatus.INITIALIZING]:
                        task.progress = progress

                        # 计算预计剩余时间
                        elapsed_time = current_time - self.start_time
                        if progress > 0.01:  # 至少完成1%才计算
                            remaining_time = int((elapsed_time / progress) * (1 - progress))
                            # 保存状态消息
                            status_message = f"正在生成: 步骤 {self.current_step}/{self.total_steps}, 预计剩余 {remaining_time} 秒"

                            if not hasattr(task, 'status_message') or task.status_message is None:
                                if task.additional_params is None:
                                    task.additional_params = {}
                                task.additional_params['status_message'] = status_message
                            else:
                                task.status_message = status_message

                        self.db_session.commit()

                        # 写入进度文件，以便API服务器读取
                        self._save_progress_file(progress)

                        # 记录日志
                        if self.current_step % 5 == 0 or self.current_step == self.total_steps:
                            logger.info(
                                f"[{self.model_name}] 任务 {self.task_id} 进度: {progress:.1%}, 步骤 {self.current_step}/{self.total_steps}")

                    self.last_update_time = current_time
                except Exception as e:
                    logger.error(f"更新进度失败: {e}", exc_info=True)

            yield item

        # 完成所有步骤后，确保进度为100%
        try:
            task = self.db_session.query(VideoTask).filter(VideoTask.id == self.task_id).first()
            if task and task.status in [TaskStatus.RUNNING, TaskStatus.INITIALIZING]:
                task.progress = 1.0
                self.db_session.commit()
                self._save_progress_file(1.0, final=True)
        except Exception as e:
            logger.error(f"完成时更新进度失败: {e}", exc_info=True)

    def _save_progress_file(self, progress, final=False):
        """保存进度信息到文件，供API服务器读取"""
        try:
            progress_data = {
                "status": TaskStatus.RUNNING,
                "progress": float(progress),
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "status_message": f"正在生成: 步骤 {self.current_step}/{self.total_steps}"
            }

            if final:
                progress_data["status_message"] = f"生成完成，共 {self.total_steps} 步"

            # 添加时间戳避免缓存
            progress_data["timestamp"] = time.time()

            progress_file = f"progress_{self.task_id}.json"
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)

            return True
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}", exc_info=True)
            return False


class GPUWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_manager = None
        self.current_pipeline = None
        self.current_task = None
        self.is_processing = False

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
        logger.info(f"GPU Worker {self.gpu_id} 加载模型开始: {model_type}, {resolution}, {precision}")

        try:
            from diffsynth import WanVideoPipeline

            # 根据分辨率和类型选择模型路径
            model_path = None
            if model_type == VideoType.TEXT_TO_VIDEO:
                model_path = "/home/ps/videoGen/models/Wan2.1-T2V-14B/"
                logger.info(f"选择文生视频模型: {model_path}")
            elif model_type == VideoType.IMAGE_TO_VIDEO:
                if resolution in [Resolution.RESOLUTION_480P, Resolution.RESOLUTION_480P_VERTICAL]:
                    model_path = "/home/ps/videoGen/models/Wan2.1-I2V-14B-480P/"
                    logger.info(f"选择图生视频480P模型: {model_path}")
                else:
                    model_path = "/home/ps/videoGen/models/Wan2.1-I2V-14B-720P/"
                    logger.info(f"选择图生视频720P模型: {model_path}")

            if not model_path:
                logger.error(f"无效的模型类型或分辨率: {model_type}, {resolution}")
                raise ValueError(f"无效的模型类型或分辨率: {model_type}, {resolution}")

            # 设置模型精度
            torch_dtype = torch.bfloat16
            if precision == ModelPrecision.FP8:
                torch_dtype = torch.float8_e4m3fn
                logger.info("使用FP8精度加载模型")
            else:
                logger.info("使用FP16精度加载模型")

            logger.info(f"开始加载模型文件: {model_path}")

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
                logger.info("已准备文生视频模型文件列表")
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
                    logger.info("已准备图生视频480P模型文件列表")
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
                    logger.info("已准备图生视频720P模型文件列表")

            # 加载模型
            logger.info("开始加载模型到内存...")
            self.model_manager.load_models(
                model_files,
                torch_dtype=torch_dtype,
            )
            logger.info("模型加载到内存完成")

            # 创建管道
            logger.info(f"创建模型管道，设备: {self.device}")
            self.current_pipeline = WanVideoPipeline.from_model_manager(
                self.model_manager,
                torch_dtype=torch.bfloat16,
                device=self.device
            )

            # 清理内存
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(
                f"GPU Worker {self.gpu_id} 模型加载完成，当前GPU显存使用: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f}GB")

            # 尝试输出可用显存信息
            try:
                total_mem = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3
                reserved_mem = torch.cuda.memory_reserved(self.device) / 1024 ** 3
                allocated_mem = torch.cuda.memory_allocated(self.device) / 1024 ** 3
                free_mem = total_mem - allocated_mem

                logger.info(
                    f"GPU {self.gpu_id} 内存状态: 总计 {total_mem:.2f}GB, 已分配 {allocated_mem:.2f}GB, 已预留 {reserved_mem:.2f}GB, 剩余 {free_mem:.2f}GB")
            except Exception as e:
                logger.warning(f"获取GPU内存信息失败: {e}")

            return True

        except Exception as e:
            logger.error(f"GPU Worker {self.gpu_id} 加载模型失败: {e}", exc_info=True)

            # 尝试清理内存
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("已尝试清理CUDA内存")
            except Exception as cleanup_error:
                logger.error(f"清理内存失败: {cleanup_error}")

            return False

    async def process_task(self, task):
        self.current_task = task
        self.is_processing = True
        db = SessionLocal()

        try:
            # 更新任务开始信息
            logger.info(f"开始处理任务 {task.id}, 类型: {task.type}, 分辨率: {task.resolution}")
            logger.info(
                f"任务参数: 帧数={task.frames}, 帧率={task.fps}, 步数={task.steps}, 精度={task.model_precision}")

            # 计算实际分辨率
            width, height = self._get_resolution_dimensions(task.resolution)
            logger.info(f"实际分辨率: {width}x{height}")

            # 设置显存管理配置
            num_persistent_param = None if not task.save_vram else 0
            logger.info(f"显存节省模式: {'开启' if task.save_vram else '关闭'}")

            # 更新任务状态为"初始化模型"
            await self._update_task_status(task.id, db, TaskStatus.INITIALIZING, "正在初始化模型和加载资源...")

            # 准备生成参数
            generation_params = {
                "prompt": task.prompt,
                "negative_prompt": task.negative_prompt,
                "height": height,
                "width": width,
                "num_frames": task.frames,
                "num_inference_steps": task.steps,
                "cfg_scale": 5.0,  # 使用默认值
                "seed": task.seed if task.seed >= 0 else None,
                "tiled": task.tiled,
            }

            # 如果是图生视频，加载输入图片
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                logger.info(f"加载输入图片: {task.image_path}")
                input_image = Image.open(task.image_path).convert("RGB")
                generation_params["input_image"] = input_image

            # 启用显存管理 (必须在加载模型后调用)
            if self.current_pipeline:
                self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

            # 更新任务状态为"开始生成"
            await self._update_task_status(task.id, db, TaskStatus.RUNNING, "正在生成视频...")

            # 创建进度条对象
            progress_bar = ProgressBar(task.id, db, total_steps=task.steps)

            # 设置生成进度回调
            generation_params["progress_bar_cmd"] = progress_bar

            # 实时记录生成开始时间
            start_time = time.time()

            # 生成视频
            logger.info(f"开始生成视频 任务ID: {task.id}")
            video = self.current_pipeline(**generation_params)

            # 更新保存状态
            logger.info(f"视频生成完成，正在保存: {task.output_path}")
            await self._update_task_status(task.id, db, TaskStatus.SAVING, "视频生成完成，正在保存...")

            # 避免快速变化任务状态导致前端混淆，等待一小段时间
            await asyncio.sleep(1)

            # 保存视频
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 确认视频文件已保存成功
            if not os.path.exists(task.output_path):
                raise FileNotFoundError(f"视频文件未成功保存到 {task.output_path}")

            # 获取视频文件大小
            video_size = os.path.getsize(task.output_path) / (1024 * 1024)  # MB
            logger.info(f"视频已保存: {task.output_path}, 大小: {video_size:.2f}MB")

            # 再次等待一小段时间，确保视频文件完全写入
            await asyncio.sleep(1)

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            if task.additional_params is None:
                task.additional_params = {}
            task.additional_params['video_size'] = f"{video_size:.2f}MB"
            db.commit()

            # 更新进度文件标记完成
            self._write_progress_file(task.id, {
                "status": TaskStatus.COMPLETED,
                "progress": 1.0,
                "status_message": "生成已完成",
                "output_url": f"/api/videos/{task.id}"
            })

            # 计算总耗时
            total_time = time.time() - start_time
            logger.info(f"视频生成和保存完成 任务ID: {task.id}, 总耗时: {total_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"生成视频失败 任务ID: {task.id}, 错误: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()

            # 更新进度文件标记失败
            self._write_progress_file(task.id, {
                "status": TaskStatus.FAILED,
                "error": str(e)
            })

            return False

        finally:
            # 释放资源
            self.is_processing = False
            if self.current_pipeline:
                try:
                    # 尝试清理CUDA内存
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"已清理任务 {task.id} 的CUDA内存")
                except Exception as e:
                    logger.error(f"清理内存失败: {e}")

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

    async def _update_task_status(self, task_id, db, status=None, status_message=None):
        """更新任务状态和状态消息"""
        task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        if not task:
            logger.error(f"任务 {task_id} 不存在，无法更新状态")
            return False

        if status:
            task.status = status

        if status_message:
            if not hasattr(task, 'status_message') or task.status_message is None:
                # 如果status_message字段不存在，我们通过additional_params存储
                if not task.additional_params:
                    task.additional_params = {}
                task.additional_params['status_message'] = status_message
            else:
                task.status_message = status_message

        db.commit()

        # 写入进度文件通知前端
        self._write_progress_file(task_id, {
            'status': status if status else task.status,
            'progress': task.progress,
            'status_message': status_message
        })

        return True

    def _write_progress_file(self, task_id, data):
        """写入进度文件，以便API服务器读取"""
        try:
            progress_file = f"progress_{task_id}.json"
            # 添加时间戳避免缓存
            data["timestamp"] = time.time()

            with open(progress_file, "w") as f:
                json.dump(data, f)

            return True
        except Exception as e:
            logger.error(f"写入进度文件失败: {e}")
            return False

    def stop_current_task(self):
        if not self.current_pipeline or not self.current_task:
            logger.warning(f"GPU {self.gpu_id} 没有正在运行的任务，无法停止")
            return False

        if not self.is_processing:
            logger.warning(f"GPU {self.gpu_id} 任务已不在处理中，无需停止")
            return True

        try:
            logger.info(f"正在停止 GPU {self.gpu_id} 上的任务 {self.current_task.id}")
            # 在实际实现中，这里可能需要设置一个停止标志或信号
            # 对于WanVideoPipeline，可能没有直接的中断方法

            # 尝试释放资源
            import gc
            torch.cuda.empty_cache()
            gc.collect()

            # 标记处理已停止
            self.is_processing = False
            return True
        except Exception as e:
            logger.error(f"停止任务失败: {e}")
            return False


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
                VideoTask.status.in_([TaskStatus.QUEUED, TaskStatus.INITIALIZING]),
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
                    task.error_message = "模型加载失败，请检查日志了解详情"
                    db.commit()

                    # 写入进度文件通知前端
                    worker._write_progress_file(task.id, {
                        'status': TaskStatus.FAILED,
                        'error': "模型加载失败，请检查日志了解详情"
                    })

                running_task = None

            db.close()

            # 等待一段时间再检查新任务
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Worker循环出错: {e}", exc_info=True)
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