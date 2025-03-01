import os
import json
import time
import redis
import requests
import torch
import logging
import random
import sys
from PIL import Image
import argparse
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 自定义进度条类，用于将进度发送到API
class APIProgressBar(tqdm):
    def __init__(self, *args, task_id=None, api_url=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = task_id
        self.api_url = api_url
        self.last_update_time = time.time()

    def update(self, n=1):
        super().update(n)
        current_time = time.time()

        # 限制更新频率（每0.5秒最多一次）
        if current_time - self.last_update_time >= 0.5:
            self.last_update_time = current_time
            if self.task_id and self.api_url:
                progress = self.n / self.total if self.total else 0
                message = f"步骤 {self.n}/{self.total}"
                try:
                    requests.post(
                        f"{self.api_url}/api/internal/update_task",
                        json={
                            "task_id": self.task_id,
                            "status": "processing",
                            "progress": progress,
                            "message": message
                        }
                    )
                except Exception as e:
                    logger.error(f"更新进度失败: {e}")

# 检查任务是否被取消
def check_if_cancelled(task_id, api_url):
    try:
        response = requests.get(f"{api_url}/api/internal/check_cancel/{task_id}")
        if response.status_code == 200:
            return response.json().get("cancelled", False)
    except Exception as e:
        logger.error(f"检查任务取消状态失败: {e}")
    return False

def update_task_status(task_id, api_url, status, progress=None, message=None, output_path=None):
    data = {
        "task_id": task_id,
        "status": status
    }

    if progress is not None:
        data["progress"] = progress

    if message is not None:
        data["message"] = message

    if output_path is not None:
        data["output_path"] = output_path

    try:
        requests.post(f"{api_url}/api/internal/update_task", json=data)
    except Exception as e:
        logger.error(f"更新任务状态失败: {e}")

def generate_video(task, gpu_id, api_url):
    task_id = task["task_id"]
    task_type = task["task_type"]
    prompt = task["prompt"]
    negative_prompt = task["negative_prompt"]
    resolution = task["resolution"]
    num_frames = task["num_frames"]
    fps = task["fps"]
    num_inference_steps = task["num_inference_steps"]
    fp8 = task["fp8"]
    save_vram = task["save_vram"]
    seed = task["seed"] if task["seed"] is not None else random.randint(0, 2147483647)
    guidance_scale = task["guidance_scale"]
    sample_shift = task["sample_shift"]
    image_path = task.get("image_path")

    # 更新任务状态
    update_task_status(
        task_id=task_id,
        api_url=api_url,
        status="processing",
        progress=0.0,
        message="准备环境和模型..."
    )

    output_file = f"outputs/{task_id}_output.mp4"

    try:
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 导入wan相关模块（放在这里避免GPU冲突）
        import torch
        from diffsynth import ModelManager, WanVideoPipeline, save_video

        # 解析分辨率
        width, height = map(int, task["resolution"].split("x"))

        # 确定模型路径
        if task_type == "t2v":
            model_path = "/home/ps/videoGen/models/Wan2.1-T2V-14B/"
        elif task_type == "i2v":
            if max(width, height) > 480:
                model_path = "/home/ps/videoGen/models/Wan2.1-I2V-14B-720P/"
            else:
                model_path = "/home/ps/videoGen/models/Wan2.1-I2V-14B-480P/"
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        # 更新状态
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="processing",
            progress=0.05,
            message=f"加载模型: {model_path}"
        )

        # 使用正确的torch数据类型
        torch_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16

        # 加载模型
        model_manager = ModelManager(device="cpu")

        if task_type == "t2v":
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
                f"{model_path}Wan2.1_VAE.pth",
            ]
        else:  # i2v
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
                f"{model_path}Wan2.1_VAE.pth",
            ]

        # 加载模型
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="processing",
            progress=0.1,
            message="加载模型文件..."
        )
        model_manager.load_models(model_files, torch_dtype=torch_dtype)

        # 创建pipeline
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="processing",
            progress=0.3,
            message="准备生成pipeline..."
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

        # 配置显存管理
        num_persistent_param = 0 if save_vram else None
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param)

        # 创建进度条回调
        progress_bar = APIProgressBar(
            total=num_inference_steps,
            task_id=task_id,
            api_url=api_url,
            desc="生成中"
        )

        # 准备图片（如果是I2V）
        input_image = None
        if task_type == "i2v" and image_path:
            input_image = Image.open(image_path).convert("RGB")

        # 开始生成
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="processing",
            progress=0.4,
            message="开始生成视频..."
        )

        # 计算帧数（必须是4n+1）
        adjusted_num_frames = ((num_frames - 1) // 4) * 4 + 1
        if adjusted_num_frames != num_frames:
            logger.info(f"调整帧数: {num_frames} -> {adjusted_num_frames}")
            num_frames = adjusted_num_frames

        # 生成视频
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            sigma_shift=sample_shift,
            seed=seed,
            tiled=True,
            progress_bar_cmd=progress_bar,
        )

        # 保存视频
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="processing",
            progress=0.95,
            message="保存视频..."
        )
        save_video(video, output_file, fps=fps, quality=5)

        # 更新任务状态为完成
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="completed",
            progress=1.0,
            message="视频生成完成",
            output_path=f"/outputs/{task_id}_output.mp4"
        )

        return True

    except Exception as e:
        logger.exception(f"生成视频失败: {e}")
        update_task_status(
            task_id=task_id,
            api_url=api_url,
            status="failed",
            message=f"生成失败: {str(e)}"
        )
        return False

    finally:
        # 清理临时文件
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

        # 清理CUDA缓存
        torch.cuda.empty_cache()

def worker_main(gpu_id, api_url):
    logger.info(f"Worker启动, 使用GPU: {gpu_id}")

    # 连接到Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    while True:
        try:
            # 从队列中获取任务
            task_data = redis_client.rpop("video_generation_queue")

            if task_data:
                task = json.loads(task_data)
                task_id = task["task_id"]
                logger.info(f"处理任务: {task_id}")

                # 更新任务状态
                update_task_status(
                    task_id=task_id,
                    api_url=api_url,
                    status="processing",
                    progress=0,
                    message=f"在GPU #{gpu_id}上开始处理"
                )

                # 执行视频生成
                generate_video(task, gpu_id, api_url)
            else:
                # 没有任务，休息一下
                time.sleep(1)

        except Exception as e:
            logger.exception(f"Worker异常: {e}")
            time.sleep(5)  # 出错后休息一下再继续

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频生成Worker")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="API URL")

    args = parser.parse_args()
    worker_main(args.gpu, args.api)