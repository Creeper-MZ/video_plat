import time
import logging
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)


class CustomTQDM:
    """
    自定义的TQDM替代类，专为WanVideoPipeline设计的进度回调
    """

    def __init__(
            self,
            iterable: Iterable[T],
            callback: Callable[[int, int, float], None],
            desc: str = "",
            total: Optional[int] = None,
    ):
        """
        初始化自定义进度条

        参数:
            iterable: 要迭代的序列
            callback: 进度回调函数，接收参数(current_step, total_steps, progress_percentage)
            desc: 描述文本
            total: 总步数，如果为None则尝试从iterable获取
        """
        self.iterable = iterable
        self.callback = callback
        self.desc = desc

        if total is None:
            try:
                self.total = len(iterable)
            except (TypeError, AttributeError):
                self.total = 0
        else:
            self.total = total

        self.current = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_interval = 0.1  # 更新频率限制，秒

        # 立即发送0%进度
        self._update_progress()

    def __iter__(self):
        """迭代器实现"""
        try:
            for item in self.iterable:
                yield item
                self.current += 1

                # 控制更新频率，避免过于频繁的回调
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval or self.current == self.total:
                    self._update_progress()
                    self.last_update_time = current_time

        finally:
            # 确保完成时发送100%进度
            if self.current != self.total:
                self.current = self.total
                self._update_progress()

    def _update_progress(self):
        """更新进度回调"""
        if self.total > 0:
            progress = self.current / self.total
        else:
            progress = 0

        # 计算ETA
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0

        # 计算速度
        if elapsed > 0:
            rate = self.current / elapsed
        else:
            rate = 0

        # 记录日志 (每10%记录一次或第一次和最后一次)
        should_log = (
                (int(progress * 10) > int((self.current - 1) / self.total * 10)) or
                self.current == 1 or
                self.current == self.total
        )

        if should_log:
            log_msg = f"{self.desc}: {self.current}/{self.total} ({progress * 100:.1f}%) [速度:{rate:.2f}it/s, 已用:{elapsed:.1f}s, 剩余:{eta:.1f}s]"
            logger.info(log_msg)

        # 调用回调函数
        try:
            self.callback(self.current, self.total, progress)
        except Exception as e:
            logger.error(f"进度回调失败: {e}")


def create_pipeline_callback(task_id: str, callback_fn: Callable):
    """
    创建适用于WanVideoPipeline的进度回调函数

    参数:
        task_id: 任务ID
        callback_fn: 回调函数，接收任务信息字典

    返回:
        progress_bar_cmd: 自定义的进度条包装器
    """

    def progress_callback(current: int, total: int, progress: float):
        """处理进度更新的回调"""
        info = {
            "task_id": task_id,
            "stage": "denoising",  # 这里是固定的denoising阶段
            "stage_progress": progress,
            "step": current,
            "total_steps": total,
            "progress": 0.1 + progress * 0.8,  # 调整总进度，denoising占80%
            "logs": [f"去噪步骤: {current}/{total} ({progress * 100:.1f}%)"]
        }

        callback_fn(info)

    def progress_bar_cmd(iterable, total=None, desc="去噪中"):
        """返回自定义进度条包装器"""
        return CustomTQDM(iterable, progress_callback, desc=desc, total=total)

    return progress_bar_cmd