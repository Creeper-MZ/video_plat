# app/services/task_queue.py

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import json
import os
from pathlib import Path

from ..core.config import settings
from ..core.gpu_manager import gpu_manager
from ..core.logging import TaskLogger

logger = logging.getLogger("videoGenPlatform")


class Task:
    """
    Represents a video generation task
    """

    def __init__(self, task_type: str, params: Dict[str, Any], debug_mode: bool = False, client_id: str = None):
        self.id = str(uuid.uuid4())
        self.type = task_type  # 't2v' or 'i2v'
        self.params = params
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.status = "queued"  # queued, running, completed, failed, cancelled
        self.result = None
        self.error = None
        self.gpu_id = None
        self.progress = 0
        self.callback = None
        self.process = None
        self.client_id = client_id  # Add client_id field to track task ownership
        # 传递debug_mode而不是debug
        self.logger = TaskLogger(self.id, debug_mode=debug_mode)

        # Save task information to disk
        self._save_task_info()

    def _save_task_info(self):
        """Save task information to disk"""
        tasks_dir = Path("tasks")
        tasks_dir.mkdir(exist_ok=True)

        task_info = {
            "id": self.id,
            "type": self.type,
            "params": {k: str(v) if isinstance(v, (Path, bytes)) else v for k, v in self.params.items()},
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "progress": self.progress,
            "gpu_id": self.gpu_id,
            "client_id": self.client_id  # Save client_id with task info
        }

        with open(tasks_dir / f"{self.id}.json", "w") as f:
            json.dump(task_info, f, indent=2)

    def update_status(self, status: str):
        """Update task status"""
        self.status = status
        self.logger.update_status(status)

        if status == "running" and self.started_at is None:
            self.started_at = datetime.now()
        elif status in ["completed", "failed", "cancelled"] and self.completed_at is None:
            self.completed_at = datetime.now()

        self._save_task_info()

    # 在 Task 类中
    def update_progress(self, current_step: int, total_steps: int):
        """Update task progress"""
        self.progress = self.logger.update_progress(current_step, total_steps)
        # 自己保存任务信息
        self._save_task_info()
        return self.progress

    def assign_gpu(self, gpu_id: int):
        """Assign a GPU to this task"""
        self.gpu_id = gpu_id
        self.logger.info(f"Task assigned to GPU {gpu_id}")
        self._save_task_info()

    def cancel(self):
        """Cancel the task"""
        if self.status in ["queued", "running"]:
            self.update_status("cancelled")
            self.logger.warning("Task cancelled by user")

            # Kill process if running
            if self.process:
                try:
                    self.process.terminate()
                    self.logger.info("Process terminated")
                except Exception as e:
                    self.logger.error(f"Failed to terminate process: {str(e)}")

            # Release GPU if allocated
            if self.gpu_id is not None:
                gpu_manager.release_gpu(self.gpu_id, self.id)
                self.logger.info(f"Released GPU {self.gpu_id}")

    def set_result(self, result):
        """Set task result"""
        self.result = result
        self.update_status("completed")
        self.logger.info("Task completed successfully")

        # Call callback if set
        if self.callback:
            try:
                self.callback(self)
            except Exception as e:
                self.logger.error(f"Callback error: {str(e)}")

    def set_error(self, error):
        """Set task error"""
        self.error = str(error)
        self.update_status("failed")
        self.logger.error(f"Task failed: {self.error}")

        # Call callback if set
        if self.callback:
            try:
                self.callback(self)
            except Exception as e:
                self.logger.error(f"Callback error: {str(e)}")

    def to_dict(self):
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "gpu_id": self.gpu_id,
            "result": self.result,
            "error": self.error,
            "client_id": self.client_id,
            "params": {k: str(v) if isinstance(v, (Path, bytes)) else v for k, v in self.params.items() if
                       k != "image_data"}
        }


class TaskQueue:
    """
    Manages the queue of video generation tasks
    """

    def __init__(self, max_size: int = 100, max_tasks_per_client: int = 2):
        self.max_size = max_size
        self.max_tasks_per_client = max_tasks_per_client  # Limit tasks per client
        self.queue: List[Task] = []
        self.running: Dict[str, Task] = {}
        self.completed: Dict[str, Task] = {}
        self.lock = threading.Lock()
        self.event = threading.Event()
        self._worker_thread = None
        self._stop_worker = threading.Event()
        self.start_worker()

    def start_worker(self):
        """Start the worker thread"""
        if self._worker_thread is None:
            self._stop_worker.clear()
            self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._worker_thread.start()
            logger.info("Task queue worker started")

    def stop_worker(self):
        """Stop the worker thread"""
        if self._worker_thread:
            self._stop_worker.set()
            self.event.set()  # Wake up the worker
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
            logger.info("Task queue worker stopped")

    def add_task(self, task_type: str, params: Dict[str, Any], callback: Optional[Callable] = None,
                 debug: bool = False, client_id: str = None) -> Task:
        """
        Add a task to the queue

        Args:
            task_type: Type of task ('t2v' or 'i2v')
            params: Parameters for the task
            callback: Optional callback function to call when task completes
            debug: Enable debug logging for this task
            client_id: ID of the client who created the task

        Returns:
            The created Task object
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                raise ValueError(f"Queue is full (max size: {self.max_size})")

            # Check if client has reached their task limit
            if client_id:
                # Count running and queued tasks for this client
                running_tasks = sum(1 for task in self.running.values() if task.client_id == client_id)
                queued_tasks = sum(1 for task in self.queue if task.client_id == client_id)
                total_tasks = running_tasks + queued_tasks

                if total_tasks >= self.max_tasks_per_client:
                    raise ValueError(
                        f"Client has reached the maximum number of concurrent tasks ({self.max_tasks_per_client})")

            # Create task with client_id
            task = Task(task_type, params, debug_mode=debug, client_id=client_id)
            if callback:
                task.callback = callback

            self.queue.append(task)
            logger.info(f"Added task {task.id} to queue (position: {len(self.queue)})")

            # Wake up the worker thread
            self.event.set()

            return task

    def get_client_tasks(self, client_id: str) -> Dict:
        """
        Get tasks for a specific client

        Args:
            client_id: ID of the client

        Returns:
            Dictionary with client's tasks
        """
        with self.lock:
            client_queue = [task.to_dict() for task in self.queue if task.client_id == client_id]
            client_running = [task.to_dict() for task in self.running.values() if task.client_id == client_id]
            client_completed = [task.to_dict() for task in self.completed.values() if task.client_id == client_id]

            return {
                "queue_length": len(client_queue),
                "running_tasks": len(client_running),
                "completed_tasks": len(client_completed),
                "queue": client_queue,
                "running": client_running,
                "recent_completed": client_completed[-10:] if client_completed else []
            }

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID

        Args:
            task_id: ID of the task to get

        Returns:
            Task object or None if not found
        """
        with self.lock:
            # Check running tasks
            if task_id in self.running:
                return self.running[task_id]

            # Check completed tasks
            if task_id in self.completed:
                return self.completed[task_id]

            # Check queued tasks
            for task in self.queue:
                if task.id == task_id:
                    return task

            return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was cancelled, False otherwise
        """
        task = self.get_task(task_id)
        if task:
            task.cancel()

            # Remove from queue if queued
            with self.lock:
                if task in self.queue:
                    self.queue.remove(task)
                    logger.info(f"Removed task {task_id} from queue")

                # Move from running to completed if running
                if task_id in self.running:
                    del self.running[task_id]
                    self.completed[task_id] = task
                    logger.info(f"Moved cancelled task {task_id} to completed")

            return True

        return False

    def get_queue_status(self) -> Dict:
        """
        Get the current status of the queue

        Returns:
            Dictionary with queue status information
        """
        with self.lock:
            return {
                "queue_length": len(self.queue),
                "running_tasks": len(self.running),
                "completed_tasks": len(self.completed),
                "queue": [task.to_dict() for task in self.queue],
                "running": [task.to_dict() for task in self.running.values()],
                "recent_completed": [task.to_dict() for task in list(self.completed.values())[-10:]]
            }

    def get_task_details(self, task_id: str) -> Optional[Dict]:
        """Get detailed information about a task"""
        task = self.get_task(task_id)
        if task:
            task_info = task.to_dict()
            task_info["logs"] = task.logger.get_logs(max_entries=50)
            return task_info
        return None

    def _process_queue(self):
        """Worker thread to process tasks in the queue"""
        logger.info("Task queue processor started")

        while not self._stop_worker.is_set():
            # Wait for tasks or timeout after 1 second
            self.event.wait(timeout=1)
            self.event.clear()

            task_to_process = None

            # Check if there are tasks in the queue and GPUs available
            with self.lock:
                if self.queue:
                    # Get first task from queue
                    task = self.queue[0]

                    # Try to allocate a GPU
                    gpu_id = gpu_manager.allocate_gpu(task.id)
                    if gpu_id is not None:
                        # Remove task from queue and add to running
                        task_to_process = self.queue.pop(0)
                        task_to_process.assign_gpu(gpu_id)
                        self.running[task_to_process.id] = task_to_process
                        logger.info(f"Processing task {task_to_process.id} on GPU {gpu_id}")

            # Process the task if one was allocated
            if task_to_process:
                # Start task in a separate thread
                thread = threading.Thread(
                    target=self._run_task,
                    args=(task_to_process,),
                    daemon=True
                )
                thread.start()

            # Check status of running tasks and clean up completed ones
            with self.lock:
                completed_tasks = []
                for task_id, task in list(self.running.items()):
                    if task.status in ["completed", "failed", "cancelled"]:
                        completed_tasks.append(task_id)

                # Move completed tasks to completed dict
                for task_id in completed_tasks:
                    task = self.running.pop(task_id)
                    self.completed[task_id] = task

                    # Limit completed tasks history (keep last 100)
                    if len(self.completed) > 100:
                        # Get oldest key by completion time
                        oldest_task_id = None
                        oldest_time = None

                        for tid, t in self.completed.items():
                            if oldest_time is None or (t.completed_at and t.completed_at < oldest_time):
                                oldest_time = t.completed_at
                                oldest_task_id = tid

                        if oldest_task_id:
                            # Clean up task files
                            try:
                                task_file = Path("tasks") / f"{oldest_task_id}.json"
                                if task_file.exists():
                                    task_file.unlink()
                            except Exception as e:
                                logger.error(f"Error cleaning up task file for {oldest_task_id}: {str(e)}")

                            del self.completed[oldest_task_id]
                            logger.debug(f"Removed oldest completed task {oldest_task_id} from history")

    def _run_task(self, task: Task):
        """Run a task on its assigned GPU"""
        from .generator import VideoGenerator

        try:
            # Update task status
            task.update_status("running")

            # Create generator
            generator = VideoGenerator(task)

            # Run the generation
            result = generator.generate()

            # Set result
            task.set_result(result)

        except Exception as e:
            logger.exception(f"Error processing task {task.id}: {str(e)}")
            task.set_error(str(e))
        finally:
            # Release GPU
            if task.gpu_id is not None:
                gpu_manager.release_gpu(task.gpu_id, task.id)
                task.logger.info(f"Released GPU {task.gpu_id}")


# Create a singleton instance
task_queue = TaskQueue(max_size=settings.max_queue_size)