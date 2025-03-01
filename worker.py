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
            t5_model_file = ""
            vae_model_file = ""

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
            self.current_pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

            # 准备生成参数
            generation_params = {
                "prompt": task.prompt,
                "negative_prompt": task.negative_prompt,
                "num_inference_steps": task.steps,
                "seed": task.seed if task.seed >= 0 else None,
                "tiled": task.tiled,
                # "width": width,
                # "height": height,
            }

            # 如果是图生视频，加载输入图片
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                logger.info(f"加载输入图片: {task.image_path}")
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image

            # 更新任务状态为"初始化模型"
            self._update_task_status(task.id, db, "initializing", "正在初始化模型...")

            # 注册回调以更新进度
            self._register_progress_callback(task.id, db)

            # 生成视频
            logger.info(f"开始生成视频 任务ID: {task.id}")
            self._update_task_status(task.id, db, "generating", "开始生成过程...")

            # 实时记录生成开始时间，用于预估剩余时间
            start_time = time.time()

            # 定期更新进度的协程
            async def update_progress():
                prev_step = 0
                while task.status == TaskStatus.RUNNING:
                    # 查询当前任务最新状态
                    current_task = db.query(VideoTask).filter(VideoTask.id == task.id).first()
                    if not current_task or current_task.status != TaskStatus.RUNNING:
                        break

                    elapsed_time = time.time() - start_time
                    if current_task.progress > 0:
                        estimated_total_time = elapsed_time / current_task.progress
                        remaining_time = estimated_total_time - elapsed_time

                        # 更新详细进度信息
                        current_step = int(current_task.progress * task.steps)
                        if current_step > prev_step:
                            prev_step = current_step
                            logger.info(
                                f"任务 {task.id} 进度: {current_task.progress:.1%}, 步骤 {current_step}/{task.steps}, 预计剩余时间: {remaining_time:.1f}秒")

                            # 更新详细状态信息
                            status_message = f"正在生成: 步骤 {current_step}/{task.steps}, 预计剩余 {int(remaining_time)}秒"
                            self._update_task_status(task.id, db, None, status_message)

                    await asyncio.sleep(2)  # 每2秒更新一次

            # 启动进度更新协程
            progress_task = asyncio.create_task(update_progress())

            # 使用回调进行进度跟踪
            video = await self._generate_video_with_progress(task, generation_params, db)

            # 取消进度更新协程
            progress_task.cancel()

            # 更新任务状态
            logger.info(f"视频生成完成，正在保存: {task.output_path}")
            self._update_task_status(task.id, db, "saving", "视频生成完成，正在保存...")

            # 保存视频
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            db.commit()

            # 计算总耗时
            total_time = time.time() - start_time
            logger.info(f"视频生成和保存完成 任务ID: {task.id}, 总耗时: {total_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"生成视频失败 任务ID: {task.id}, 错误: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()
            return False

        finally:
            db.close()
            self.current_task = None

    async def _generate_video_with_progress(self, task, generation_params, db):
        """带有进度更新的视频生成过程"""
        # 这里应该根据实际情况调用Wan2.1的API
        # 注意：在实际实现中，你需要检查Wan2.1是否提供回调机制

        # 模拟生成过程，用于演示
        total_steps = task.steps
        video = None

        try:
            # 实际执行视频生成
            logger.info(f"执行视频生成: 类型={task.type}, 总步数={total_steps}")
            video = self.current_pipeline(**generation_params)
            return video

        except Exception as e:
            logger.error(f"视频生成过程中出错: {e}", exc_info=True)
            raise e

    def _update_task_status(self, task_id, db, status=None, status_message=None):
        """更新任务状态和状态消息"""
        task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
        if not task:
            return False

        if status:
            task.status = status

        if status_message:
            if not hasattr(task, 'status_message'):
                # 如果status_message字段不存在，我们通过additional_params存储
                if not task.additional_params:
                    task.additional_params = {}
                task.additional_params['status_message'] = status_message
            else:
                task.status_message = status_message

        db.commit()

        # 通过WebSocket通知前端状态变化
        asyncio.create_task(self._notify_progress(task_id, {
            'status': task.status,
            'progress': task.progress,
            'status_message': status_message
        }))

        return True

    async def _notify_progress(self, task_id, data):
        """通知API服务器更新任务进度"""
        try:
            # 通过本地JSON文件或Redis等方式与API服务器通信
            # 这里简化实现，将信息写入临时文件
            import json
            progress_file = f"progress_{task_id}.json"
            with open(progress_file, "w") as f:
                json.dump(data, f)

            logger.debug(f"更新进度信息: {data}")
            return True
        except Exception as e:
            logger.error(f"通知进度更新失败: {e}")
            return False

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
            return 1280, 720  # 默认为720P

    def _register_progress_callback(self, task_id, db):
        try:
            # 在实际情况下，这里应该注册Wan2.1模型的回调函数
            # 例如，如果Wan2.1提供了进度回调，则应在此处注册

            # 自定义进度回调函数
            def progress_callback(step, total_steps, latents=None):
                progress = step / total_steps

                # 更新数据库
                task = db.query(VideoTask).filter(VideoTask.id == task_id).first()
                if task and task.status == TaskStatus.RUNNING:
                    task.progress = progress
                    db.commit()

                    # 每5步输出一次日志
                    if step % 5 == 0 or step == total_steps - 1:
                        logger.info(f"任务 {task_id} 进度: {progress:.1%}, 步骤 {step}/{total_steps}")

                    # 通知客户端进度更新
                    asyncio.create_task(self._notify_progress(task_id, {
                        'status': TaskStatus.RUNNING,
                        'progress': progress,
                        'current_step': step,
                        'total_steps': total_steps,
                        'status_message': f"正在生成: 步骤 {step}/{total_steps}"
                    }))

            # 尝试在pipeline中注册回调
            # self.current_pipeline.register_callback(progress_callback)

            # 如果无法直接注册回调，也可以在模型代码中添加钩子
            # 或者修改模型代码直接调用回调函数

            # 示例：为了演示，我们模拟一个伪回调注册
            setattr(self.current_pipeline, "_progress_callback", progress_callback)

            logger.info(f"已注册进度回调函数: 任务ID {task_id}")

        except Exception as e:
            logger.error(f"注册进度回调失败: {e}", exc_info=True)

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