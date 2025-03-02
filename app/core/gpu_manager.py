# app/core/gpu_manager.py

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
import subprocess
import re

from .config import settings

logger = logging.getLogger("videoGenPlatform")

class GPUManager:
    """
    Manages GPU allocation and monitoring for video generation tasks
    """
    
    def __init__(self):
        self.devices = settings.gpu_devices
        self.lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background thread to monitor GPU status"""
        if self._monitoring_thread is None:
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_gpus, daemon=True)
            self._monitoring_thread.start()
            logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop the GPU monitoring thread"""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None
            logger.info("GPU monitoring stopped")

    def _monitor_gpus(self):
        """Background thread that periodically checks GPU utilization and memory"""
        logger.info("GPU monitoring thread started")
        while not self._stop_monitoring.is_set():
            try:
                gpu_stats = self._get_gpu_stats()
                with self.lock:
                    for i, device in enumerate(self.devices):
                        if i < len(gpu_stats):
                            # Update device stats safely using getattr with defaults
                            setattr(device, "vram_used", gpu_stats[i].get("memory_used", 0))
                            setattr(device, "vram_total", gpu_stats[i].get("memory_total", 49140))
                            setattr(device, "utilization", gpu_stats[i].get("utilization", 0.0))

                    # Log the updated stats
                    logger.debug(
                        f"Updated GPU stats: {[{d.device_id: {'used': getattr(d, 'vram_used', 0), 'total': getattr(d, 'vram_total', 0), 'util': getattr(d, 'utilization', 0)}} for d in self.devices]}")
            except Exception as e:
                logger.error(f"Error monitoring GPUs: {str(e)}")

            # Sleep for 5 seconds before next update
            time.sleep(5)

    def _get_gpu_stats(self) -> List[Dict]:
        """
        获取 GPU 统计信息的改进版本
        """
        try:
            # 运行 nvidia-smi 获取 GPU 统计信息
            logger.debug("Running nvidia-smi to get GPU stats")
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )

            # 解析输出
            gpu_stats = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) >= 7:  # 现在我们有 7 个参数
                    try:
                        gpu_stats.append({
                            "index": int(parts[0]),
                            "memory_used": int(parts[1]),
                            "memory_total": int(parts[2]),
                            "utilization": float(parts[3]),
                            "temperature": float(parts[4]),
                            "power_draw": float(parts[5]),
                            "power_limit": float(parts[6])
                        })
                        logger.debug(
                            f"GPU {parts[0]} stats: {parts[1]}MB/{parts[2]}MB, util: {parts[3]}%, temp: {parts[4]}°C")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing GPU stat line: {line}. Error: {e}")

            return gpu_stats
        except Exception as e:
            logger.error(f"Error getting GPU stats: {str(e)}")
            return []
    
    def allocate_gpu(self, task_id: str, required_memory: int = 40000) -> Optional[int]:
        """
        Allocate a GPU for a task
        
        Args:
            task_id: Unique identifier for the task
            required_memory: Required memory in MB (default: 40GB)
            
        Returns:
            device_id of allocated GPU or None if no GPU is available
        """
        with self.lock:
            # Find available GPU with enough free memory
            for device in self.devices:
                if device.available and not device.current_task:
                    # Check if we need to verify memory first
                    if hasattr(device, 'vram_used') and hasattr(device, 'vram_total'):
                        free_memory = device.vram_total - device.vram_used
                        if free_memory < required_memory:
                            logger.warning(f"GPU {device.device_id} has insufficient memory: {free_memory}MB < {required_memory}MB")
                            continue
                    
                    # Allocate the GPU
                    device.available = False
                    device.current_task = task_id
                    logger.info(f"Allocated GPU {device.device_id} for task {task_id}")
                    return device.device_id
            
            # No GPU available
            logger.warning(f"No GPU available for task {task_id}")
            return None
    
    def release_gpu(self, device_id: int, task_id: str) -> bool:
        """
        Release a GPU that was allocated to a task
        
        Args:
            device_id: ID of the GPU to release
            task_id: Task ID to verify ownership
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            for device in self.devices:
                if device.device_id == device_id:
                    if device.current_task == task_id:
                        device.available = True
                        device.current_task = None
                        logger.info(f"Released GPU {device_id} from task {task_id}")
                        return True
                    else:
                        logger.warning(f"Task {task_id} tried to release GPU {device_id} allocated to {device.current_task}")
            
            logger.error(f"GPU {device_id} not found or not allocated to task {task_id}")
            return False

    def get_gpu_status(self) -> List[Dict]:
        """
        Get the current status of all GPUs

        Returns:
            List of dictionaries with GPU status information
        """
        with self.lock:
            status = []
            for device in self.devices:
                device_status = {
                    "device_id": device.device_id,
                    "available": device.available,
                    "current_task": device.current_task
                }

                # Add utilization info with safe defaults
                device_status.update({
                    "vram_used": getattr(device, 'vram_used', 0),
                    "vram_total": getattr(device, 'vram_total', 49140),
                    "vram_free": max(0, getattr(device, 'vram_total', 49140) - getattr(device, 'vram_used', 0)),
                    "utilization": getattr(device, 'utilization', 0.0)
                })

                status.append(device_status)

            return status

# Create a singleton instance
gpu_manager = GPUManager()