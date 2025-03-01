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
                "seed": task.seed if task.seed >= 0 else None,
                "tiled": task.tiled,
                # "width": width,
                # "height": height,
            }
            
            # 如果是图生视频，加载输入图片
            if task.type == VideoType.IMAGE_TO_VIDEO and task.image_path:
                input_image = Image.open(task.image_path)
                generation_params["input_image"] = input_image
            
            # 注册回调以更新进度
            self._register_progress_callback(task.id, db)
            
            # 生成视频
            logger.info(f"开始生成视频 任务ID: {task.id}")
            video = self.current_pipeline(**generation_params)
            
            # 保存视频
            from diffsynth import save_video
            save_video(video, task.output_path, fps=task.fps, quality=5)
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"视频生成完成 任务ID: {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"生成视频失败 任务ID: {task.id}, 错误: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()
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
            return 1280, 720  # 默认为720P
    
    def _register_progress_callback(self, task_id, db):
        try:
            # 在实际情况下，这里应该注册Wan2.1模型的回调函数
            # 例如，如果Wan2.1提供了进度回调，则应在此处注册
            # 这里仅作为示例
            pass
        except Exception as e:
            logger.error(f"注册进度回调失败: {e}")
    
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