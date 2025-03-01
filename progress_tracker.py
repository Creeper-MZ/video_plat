import time
import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class StageType(str, Enum):
    LOADING_MODELS = "loading_models"
    INITIALIZING = "initializing"
    ENCODING_PROMPT = "encoding_prompt"
    ENCODING_IMAGE = "encoding_image"
    DENOISING = "denoising"
    DECODING = "decoding"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressTracker:
    """跟踪视频生成进度的工具类"""

    def __init__(self, task_id: str, callback: Optional[Callable] = None):
        self.task_id = task_id
        self.callback = callback
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.current_stage = StageType.INITIALIZING
        self.progress = 0.0
        self.stage_progress = 0.0
        self.stage_weight = {
            StageType.LOADING_MODELS: 0.05,
            StageType.INITIALIZING: 0.05,
            StageType.ENCODING_PROMPT: 0.05,
            StageType.ENCODING_IMAGE: 0.05,
            StageType.DENOISING: 0.70,
            StageType.DECODING: 0.05,
            StageType.SAVING: 0.05,
            StageType.COMPLETED: 0.0,
            StageType.FAILED: 0.0,
        }
        self.logs = []
        self.total_steps = 0
        self.current_step = 0
        self.eta_seconds = 0
        self.additional_info = {}

    def set_stage(self, stage: StageType, message: str = ""):
        """设置当前阶段"""
        self.stage_start_time = time.time()
        self.current_stage = stage
        self.stage_progress = 0.0
        self.current_step = 0

        elapsed = time.time() - self.start_time
        log_message = f"[{self.task_id}] 阶段: {stage} - {message} (总耗时: {elapsed:.2f}s)"
        logger.info(log_message)
        self.logs.append(log_message)

        # 计算总进度
        total_progress = 0.0
        for s, w in self.stage_weight.items():
            if s == stage:
                break
            total_progress += w
        self.progress = total_progress

        self._notify_progress()

    def update_stage_progress(self, progress: float, step: int = None, total_steps: int = None, message: str = ""):
        """更新当前阶段的进度"""
        self.stage_progress = progress

        if step is not None:
            self.current_step = step
        if total_steps is not None:
            self.total_steps = total_steps

        # 计算总进度
        stage_contribution = self.stage_weight[self.current_stage] * progress
        total_progress = 0.0
        for s, w in self.stage_weight.items():
            if s == self.current_stage:
                total_progress += stage_contribution
                break
            total_progress += w

        self.progress = total_progress

        # 计算ETA
        if self.current_step > 0 and self.total_steps > 0:
            elapsed = time.time() - self.stage_start_time
            steps_per_second = self.current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - self.current_step
            self.eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0

            if message:
                logger.info(
                    f"[{self.task_id}] 步骤: {self.current_step}/{self.total_steps} - {message} - ETA: {self.eta_seconds:.1f}s")

        self._notify_progress()

    def add_info(self, key: str, value: Any):
        """添加额外信息"""
        self.additional_info[key] = value
        self._notify_progress()

    def log(self, message: str, level: str = "info"):
        """记录日志"""
        log_message = f"[{self.task_id}] {message}"
        if level == "info":
            logger.info(log_message)
        elif level == "warning":
            logger.warning(log_message)
        elif level == "error":
            logger.error(log_message)
        self.logs.append(log_message)

    def complete(self, success: bool = True, message: str = ""):
        """完成跟踪"""
        elapsed = time.time() - self.start_time
        if success:
            self.set_stage(StageType.COMPLETED)
            self.progress = 1.0
            logger.info(f"[{self.task_id}] 任务完成 - 总耗时: {elapsed:.2f}s - {message}")
        else:
            self.set_stage(StageType.FAILED)
            logger.error(f"[{self.task_id}] 任务失败 - 总耗时: {elapsed:.2f}s - {message}")

        self._notify_progress()

    def _notify_progress(self):
        """通知进度更新"""
        if self.callback:
            info = {
                "task_id": self.task_id,
                "progress": self.progress,
                "stage": self.current_stage,
                "stage_progress": self.stage_progress,
                "elapsed": time.time() - self.start_time,
                "eta": self.eta_seconds,
                "step": self.current_step,
                "total_steps": self.total_steps,
                "logs": self.logs[-5:] if len(self.logs) > 5 else self.logs,
                **self.additional_info
            }

            if asyncio.iscoroutinefunction(self.callback):
                asyncio.create_task(self.callback(info))
            else:
                self.callback(info)

    def tqdm_callback(self, iterable, total=None, desc=None):
        """创建一个tqdm风格的进度回调，用于模型内部的迭代过程"""
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None

        self.total_steps = total
        self.current_step = 0

        for item in iterable:
            yield item
            self.current_step += 1
            progress = self.current_step / total if total else 0
            self.update_stage_progress(progress, self.current_step, total, desc or "")