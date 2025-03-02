# app/services/generator.py

import os
import logging
import sys
import time
import subprocess
import uuid
import signal
from pathlib import Path
import tempfile
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from PIL import Image

from ..core.config import settings


# Progress callback handler
class ProgressHandler:
    def __init__(self, task):
        self.task = task
        self.current_step = 0
        self.total_steps = 0
        self.status = "initializing"
        self.last_progress_time = time.time()
        self.start_time = time.time()

    def update(self, step, total):
        self.current_step = step
        self.total_steps = total

        # Update task progress
        self.task.update_progress(step, total)

        # Calculate ETA
        current_time = time.time()
        elapsed = current_time - self.start_time

        if step > 0:
            steps_per_second = step / elapsed
            remaining_steps = total - step
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        else:
            eta_str = "unknown"

        # Log progress less frequently (every 5% or at least 2 seconds)
        if current_time - self.last_progress_time >= 2.0 or step == 0 or step >= total or step % max(1,
                                                                                                     int(total * 0.05)) == 0:
            progress_percent = int(step / total * 100)
            self.task.logger.info(f"Generation progress: {step}/{total} steps ({progress_percent}%) - ETA: {eta_str}")
            self.last_progress_time = current_time


class VideoGenerator:
    """
    Handles video generation using Wan models
    """

    def __init__(self, task):
        self.task = task
        self.logger = task.logger
        self.params = task.params
        self.gpu_id = task.gpu_id
        self.output_path = None
        self.temp_dir = None

    def generate(self):
        """
        Generate a video or image based on task parameters

        Returns:
            Path to the generated video/image
        """
        self.logger.info(f"Starting generation for task {self.task.id}")

        # Set up environment variables for GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.logger.info(f"Using GPU {self.gpu_id}")

        # Create a unique filename for the output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.task.type}_{self.task.id}_{timestamp}"
        if self.task.type == "t2v" or self.task.type == "i2v":
            extension = ".mp4"
        else:  # t2i
            extension = ".png"

        output_file = filename + extension
        self.output_path = os.path.join(settings.output_dir, output_file)
        self.logger.info(f"Output will be saved to {self.output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Save input image if present
        input_image_path = None
        if "image_data" in self.params and self.params["image_data"]:
            try:
                # Create temp directory for input files
                self.temp_dir = Path(tempfile.mkdtemp())
                input_image_path = self.temp_dir / "input_image.jpg"

                # Save image data to file
                with open(input_image_path, "wb") as f:
                    f.write(self.params["image_data"])

                self.logger.info(f"Saved input image to {input_image_path}")
            except Exception as e:
                self.logger.error(f"Error saving input image: {str(e)}")
                raise

        # Determine which model to use based on task type and resolution
        model_config = self._get_model_config()
        self.logger.info(f"Using model: {model_config['model_path']}")

        # Parse resolution
        resolution = self.params.get("resolution", settings.default_resolution)
        width, height = [int(dim) for dim in resolution.split("x")]

        # Prepare generation command
        try:
            if self.task.type == "t2v":
                self._generate_t2v(model_config, width, height)
            elif self.task.type == "i2v":
                self._generate_i2v(model_config, input_image_path, width, height)
            else:
                raise ValueError(f"Unsupported task type: {self.task.type}")

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise
        finally:
            # Clean up temp files
            self._cleanup_temp_files()

        self.logger.info(f"Generation completed successfully")
        return {
            "file_path": self.output_path,
            "file_url": f"/static/videos/{os.path.basename(self.output_path)}"
        }

    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {str(e)}")

    def _get_model_config(self):
        """Determine which model to use based on task parameters"""
        task_type = self.task.type
        resolution = self.params.get("resolution", settings.default_resolution)

        # Check resolution dimensions
        width, height = [int(dim) for dim in resolution.split("x")]

        if task_type == "t2v":
            # For T2V, we use the same model regardless of resolution
            return {
                "model_path": settings.models["wan2.1-t2v-14b"].model_path,
                "model_type": "t2v"
            }
        elif task_type == "i2v":
            # For I2V, we use different models based on resolution
            if max(width, height) <= 480:
                return {
                    "model_path": settings.models["wan2.1-i2v-14b-480p"].model_path,
                    "model_type": "i2v"
                }
            else:
                return {
                    "model_path": settings.models["wan2.1-i2v-14b-720p"].model_path,
                    "model_type": "i2v"
                }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _generate_t2v(self, model_config, width, height):
        """Generate video from text"""
        # Initialize progress handler
        progress_handler = ProgressHandler(self.task)

        # 添加更多详细日志
        self.logger.info("Starting text-to-video generation")
        self.logger.info(f"Loading models from {model_config['model_path']}")
        self.logger.info(
            f"Generation parameters: Resolution: {width}x{height}, Frames: {self.params.get('num_frames')}, Steps: {self.params.get('steps')}")

        # Parameters
        prompt = self.params.get("prompt", "")
        negative_prompt = self.params.get("negative_prompt", settings.default_negative_prompt)
        num_frames = self.params.get("num_frames", settings.default_frame_num)
        fps = self.params.get("fps", settings.default_fps)
        steps = self.params.get("steps", settings.default_steps)
        shift = self.params.get("shift", settings.default_shift)
        guide_scale = self.params.get("guide_scale", settings.default_guide_scale)
        seed = self.params.get("seed", settings.default_seed)
        use_fp8 = self.params.get("use_fp8", settings.default_use_fp8)
        save_vram = self.params.get("save_vram", settings.default_save_vram)

        try:
            self.logger.info("Initializing ModelManager")

            # Create a Python script to run the generation
            if self.temp_dir is None:
                self.temp_dir = Path(tempfile.mkdtemp())
            temp_script = self.temp_dir / "generate_script.py"

            # Create model files list
            model_files_list = [
                f"{model_config['model_path']}/diffusion_pytorch_model-00001-of-00006.safetensors",
                f"{model_config['model_path']}/diffusion_pytorch_model-00002-of-00006.safetensors",
                f"{model_config['model_path']}/diffusion_pytorch_model-00003-of-00006.safetensors",
                f"{model_config['model_path']}/diffusion_pytorch_model-00004-of-00006.safetensors",
                f"{model_config['model_path']}/diffusion_pytorch_model-00005-of-00006.safetensors",
                f"{model_config['model_path']}/diffusion_pytorch_model-00006-of-00006.safetensors"
            ]
            model_files = "[" + ", ".join([f'"{file}"' for file in model_files_list]) + "]"

            # 处理prompts需格外小心引号和换行符
            prompt_escaped = prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            negative_prompt_escaped = negative_prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            # 使用列表构建脚本，避免三引号问题
            script_parts = [
                'import torch',
                'import os',
                'import sys',
                'from tqdm import tqdm',
                'import traceback',
                'from diffsynth import ModelManager, WanVideoPipeline, save_video',
                '',
                '# Set up progress callback',
                'class TqdmCallback(tqdm):',
                '    def __init__(self, *args, **kwargs):',
                '        super().__init__(*args, **kwargs)',
                '        self.last_reported = 0',
                '    ',
                '    def update(self, n=1):',
                '        super().update(n)',
                '        current = self.n',
                '        total = self.total',
                '        if current != self.last_reported:',
                '            print(f"PROGRESS:{current}:{total}", file=sys.stderr, flush=True)',
                '            self.last_reported = current',
                '',
                'try:',
                '    # Load models',
                '    print("Initializing ModelManager...")',
                '    model_manager = ModelManager(device="cpu")',
                '    ',
                '    print("Loading models...")',
                f'    model_manager.load_models(',
                f'        [',
                f'            {model_files},',
                f'            f"{model_config["model_path"]}/models_t5_umt5-xxl-enc-bf16.pth",',
                f'            f"{model_config["model_path"]}/Wan2.1_VAE.pth",',
                f'        ],',
                f'        torch_dtype=torch.float8_e4m3fn if {use_fp8} else torch.bfloat16,',
                f'    )',
                '    ',
                '    print("Initializing pipeline...")',
                '    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")'
            ]

            # 添加是否开启VRAM优化的代码
            if save_vram:
                script_parts.append('    pipe.enable_vram_management(num_persistent_param_in_dit=0)')
            else:
                script_parts.append('    pipe.enable_vram_management()')

            # 添加生成和保存代码
            script_parts.extend([
                '    ',
                '    print("Starting video generation...")',
                f'    video = pipe(',
                f'        prompt="{prompt_escaped}",',
                f'        negative_prompt="{negative_prompt_escaped}",',
                f'        num_inference_steps={steps},',
                f'        num_frames={num_frames},',
                f'        width={width},',
                f'        height={height},',
                f'        seed={seed},',
                f'        cfg_scale={guide_scale},',
                f'        sigma_shift={shift},',
                f'        progress_bar_cmd=TqdmCallback',
                f'    )',
                '    ',
                '    print("Saving video...")',
                f'    save_video(video, "{self.output_path}", fps={fps}, quality=5)',
                '    print("Video generation completed successfully!")',
                'except Exception as e:',
                '    print(f"ERROR: {str(e)}", file=sys.stderr)',
                '    print(traceback.format_exc(), file=sys.stderr)',
                '    sys.exit(1)'
            ])

            # 将所有脚本部分连接成完整脚本
            script_content = '\n'.join(script_parts)

            # 验证生成的脚本
            try:
                compile(script_content, '<string>', 'exec')
                self.logger.info("Generated script passed syntax check")
            except SyntaxError as e:
                self.logger.error(f"Generated script has syntax error: {str(e)}")
                # 将脚本内容输出到日志以便调试
                self.logger.error(f"Script content with error: {script_content}")
                raise RuntimeError(f"Generated script has syntax error: {str(e)}")

            with open(temp_script, "w") as f:
                f.write(script_content)

            self.logger.info(f"Created generation script at {temp_script}")

            # 保存脚本副本到日志目录
            script_copy_path = Path("logs") / f"script_{self.task.id}.py"
            try:
                import shutil
                shutil.copy(temp_script, script_copy_path)
                self.logger.info(f"Saved script copy to {script_copy_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save script copy: {str(e)}")

            # Run the script as a subprocess to isolate it
            cmd = [sys.executable, str(temp_script)]
            self.logger.info(f"Executing command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.task.process = process

            # 收集所有stdout和stderr输出
            stdout_lines = []
            stderr_lines = []

            # Track progress by parsing output
            for line in iter(process.stderr.readline, ''):
                stderr_lines.append(line)

                if line.startswith("PROGRESS:"):
                    try:
                        _, current, total = line.strip().split(":")
                        progress_handler.update(int(current), int(total))
                    except Exception as e:
                        self.logger.error(f"Error parsing progress: {str(e)}")
                else:
                    # 记录所有stderr输出，对调试很有用
                    self.logger.info(f"Process stderr: {line.strip()}")

            # 收集stdout输出
            for line in iter(process.stdout.readline, ''):
                stdout_lines.append(line)
                self.logger.info(f"Process stdout: {line.strip()}")

            # 等待进程完成
            return_code = process.wait()
            if return_code != 0:
                self.logger.error(f"Process failed with return code {return_code}")
                stderr_content = ''.join(stderr_lines)
                stdout_content = ''.join(stdout_lines)

                # 记录所有输出以便调试
                if stdout_content:
                    self.logger.error(f"Process stdout: {stdout_content}")
                if stderr_content:
                    self.logger.error(f"Process stderr: {stderr_content}")

                raise RuntimeError(f"Video generation process failed with return code {return_code}")

            self.logger.info("Text-to-video generation completed")

        except Exception as e:
            self.logger.error(f"Error in T2V generation: {str(e)}")
            raise

    def _generate_i2v(self, model_config, input_image_path, width, height):
        """Generate video from image and text"""
        # Initialize progress handler
        progress_handler = ProgressHandler(self.task)

        # Parameters
        prompt = self.params.get("prompt", "")
        negative_prompt = self.params.get("negative_prompt", settings.default_negative_prompt)
        num_frames = self.params.get("num_frames", settings.default_frame_num)
        fps = self.params.get("fps", settings.default_fps)
        steps = self.params.get("steps", settings.default_steps)
        shift = self.params.get("shift", settings.default_shift)
        guide_scale = self.params.get("guide_scale", settings.default_guide_scale)
        seed = self.params.get("seed", settings.default_seed)
        use_fp8 = self.params.get("use_fp8", settings.default_use_fp8)
        save_vram = self.params.get("save_vram", settings.default_save_vram)

        try:
            self.logger.info("Initializing ModelManager for I2V")
            self.logger.info(f"Loading models from {model_config['model_path']}")
            self.logger.info(
                f"Generation parameters: Resolution: {width}x{height}, Frames: {num_frames}, Steps: {steps}")

            # Create a Python script to run the generation
            if self.temp_dir is None:
                self.temp_dir = Path(tempfile.mkdtemp())
            temp_script = self.temp_dir / "generate_i2v_script.py"

            # Determine which model files to load based on resolution
            if max(width, height) <= 480:
                model_files_list = [
                    f"{model_config['model_path']}/diffusion_pytorch_model-00001-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00002-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00003-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00004-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00005-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00006-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00007-of-00007.safetensors"
                ]
            else:
                model_files_list = [
                    f"{model_config['model_path']}/diffusion_pytorch_model-00001-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00002-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00003-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00004-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00005-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00006-of-00007.safetensors",
                    f"{model_config['model_path']}/diffusion_pytorch_model-00007-of-00007.safetensors"
                ]
            model_files = "[" + ", ".join([f'"{file}"' for file in model_files_list]) + "]"

            # 处理提示词中的特殊字符
            prompt_escaped = prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            negative_prompt_escaped = negative_prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            # 使用列表构建脚本，避免三引号问题
            script_parts = [
                'import torch',
                'import os',
                'import sys',
                'import traceback',
                'from tqdm import tqdm',
                'from diffsynth import ModelManager, WanVideoPipeline, save_video',
                'from PIL import Image',
                '',
                'try:',
                '    # Set up progress callback',
                '    class TqdmCallback(tqdm):',
                '        def __init__(self, *args, **kwargs):',
                '            super().__init__(*args, **kwargs)',
                '            self.last_reported = 0',
                '        ',
                '        def update(self, n=1):',
                '            super().update(n)',
                '            current = self.n',
                '            total = self.total',
                '            if current != self.last_reported:',
                '                print(f"PROGRESS:{current}:{total}", file=sys.stderr, flush=True)',
                '                self.last_reported = current',
                '    ',
                '    # Load input image',
                '    print("Loading input image...")',
                f'    image = Image.open("{input_image_path}").convert("RGB")',
                '    ',
                '    # Load models',
                '    print("Initializing ModelManager...")',
                '    model_manager = ModelManager(device="cpu")',
                '    ',
                '    print("Loading models...")',
                f'    model_manager.load_models(',
                f'        [',
                f'            {model_files},',
                f'            f"{model_config["model_path"]}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",',
                f'            f"{model_config["model_path"]}/models_t5_umt5-xxl-enc-bf16.pth",',
                f'            f"{model_config["model_path"]}/Wan2.1_VAE.pth",',
                f'        ],',
                f'        torch_dtype=torch.float8_e4m3fn if {use_fp8} else torch.bfloat16,',
                f'    )',
                '    ',
                '    print("Initializing pipeline...")',
                '    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")'
            ]

            # 添加是否开启VRAM优化的代码
            if save_vram:
                script_parts.append('    pipe.enable_vram_management(num_persistent_param_in_dit=0)')
            else:
                script_parts.append('    pipe.enable_vram_management()')

            # 添加生成和保存代码
            script_parts.extend([
                '    ',
                '    print("Starting image-to-video generation...")',
                f'    video = pipe(',
                f'        prompt="{prompt_escaped}",',
                f'        negative_prompt="{negative_prompt_escaped}",',
                f'        input_image=image,',
                f'        num_inference_steps={steps},',
                f'        num_frames={num_frames},',
                f'        seed={seed},',
                f'        cfg_scale={guide_scale},',
                f'        sigma_shift={shift},',
                f'        tiled=True,',
                f'        progress_bar_cmd=TqdmCallback',
                f'    )',
                '    ',
                '    print("Saving video...")',
                f'    save_video(video, "{self.output_path}", fps={fps}, quality=5)',
                '    print("Video generation completed successfully!")',
                'except Exception as e:',
                '    print(f"ERROR: {str(e)}", file=sys.stderr)',
                '    print(traceback.format_exc(), file=sys.stderr)',
                '    sys.exit(1)'
            ])

            # 将所有脚本部分连接成完整脚本
            script_content = '\n'.join(script_parts)

            # 验证生成的脚本
            try:
                compile(script_content, '<string>', 'exec')
                self.logger.info("Generated script passed syntax check")
            except SyntaxError as e:
                self.logger.error(f"Generated script has syntax error: {str(e)}")
                # 将脚本内容输出到日志以便调试
                self.logger.error(f"Script content with error: {script_content}")
                raise RuntimeError(f"Generated script has syntax error: {str(e)}")

            with open(temp_script, "w") as f:
                f.write(script_content)

            self.logger.info(f"Created I2V generation script at {temp_script}")

            # 保存脚本副本到日志目录
            script_copy_path = Path("logs") / f"script_{self.task.id}.py"
            try:
                import shutil
                shutil.copy(temp_script, script_copy_path)
                self.logger.info(f"Saved script copy to {script_copy_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save script copy: {str(e)}")

            # 子进程执行脚本
            cmd = [sys.executable, str(temp_script)]
            self.logger.info(f"Executing command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.task.process = process

            # 收集所有stdout和stderr输出
            stdout_lines = []
            stderr_lines = []

            # 跟踪进度
            for line in iter(process.stderr.readline, ''):
                stderr_lines.append(line)

                if line.startswith("PROGRESS:"):
                    try:
                        _, current, total = line.strip().split(":")
                        progress_handler.update(int(current), int(total))
                    except Exception as e:
                        self.logger.error(f"Error parsing progress: {str(e)}")
                else:
                    # 记录所有stderr输出，对调试很有用
                    self.logger.info(f"Process stderr: {line.strip()}")

            # 收集stdout输出
            for line in iter(process.stdout.readline, ''):
                stdout_lines.append(line)
                self.logger.info(f"Process stdout: {line.strip()}")

            # 获取返回码
            return_code = process.wait()
            if return_code != 0:
                self.logger.error(f"Process failed with return code {return_code}")
                stderr_content = ''.join(stderr_lines)
                stdout_content = ''.join(stdout_lines)

                # 记录所有输出以便调试
                if stdout_content:
                    self.logger.error(f"Process stdout: {stdout_content}")
                if stderr_content:
                    self.logger.error(f"Process stderr: {stderr_content}")

                raise RuntimeError(f"Image-to-video generation process failed with return code {return_code}")

            self.logger.info("Image-to-video generation completed")

        except Exception as e:
            self.logger.error(f"Error in I2V generation: {str(e)}")
            raise