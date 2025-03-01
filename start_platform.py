#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("platform.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("platform-starter")

# 全局变量
processes = {}
BASE_DIR = Path(__file__).resolve().parent
should_stop = False


def signal_handler(sig, frame):
    global should_stop
    logger.info("收到终止信号，准备关闭所有进程...")
    should_stop = True
    stop_all_processes()
    sys.exit(0)


def start_api_server(port=8000):
    """启动API服务器"""
    logger.info(f"启动API服务器，端口: {port}")

    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(BASE_DIR)
    )

    processes["api_server"] = proc
    logger.info(f"API服务器已启动，PID: {proc.pid}")
    return proc.pid


def start_gpu_worker(gpu_id):
    """启动GPU工作器进程"""
    logger.info(f"启动GPU {gpu_id}工作器")

    cmd = [sys.executable, "worker.py", str(gpu_id)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(BASE_DIR)
    )

    processes[f"gpu_worker_{gpu_id}"] = proc
    logger.info(f"GPU {gpu_id}工作器已启动，PID: {proc.pid}")
    return proc.pid


def stop_process(name):
    """停止特定进程"""
    if name in processes:
        logger.info(f"正在停止 {name}...")
        try:
            processes[name].terminate()
            # 等待进程终止
            try:
                processes[name].wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} 没有及时终止，强制杀死")
                processes[name].kill()

            logger.info(f"{name} 已停止")

        except Exception as e:
            logger.error(f"停止 {name} 时出错: {e}")
        finally:
            del processes[name]


def stop_all_processes():
    """停止所有进程"""
    logger.info("正在停止所有进程...")

    # 首先停止工作器进程
    for name in list(processes.keys()):
        if name.startswith("gpu_worker_"):
            stop_process(name)

    # 然后停止API服务器
    if "api_server" in processes:
        stop_process("api_server")

    logger.info("所有进程已停止")


def monitor_processes():
    """监控进程状态，如果有进程意外退出则重启"""
    logger.info("开始监控进程...")

    while not should_stop:
        for name, proc in list(processes.items()):
            returncode = proc.poll()
            if returncode is not None:
                logger.warning(f"{name} 意外退出，返回码: {returncode}")

                # 记录日志
                stdout, stderr = proc.communicate()
                logger.info(f"{name} 标准输出: {stdout}")
                logger.error(f"{name} 标准错误: {stderr}")

                # 重启进程
                logger.info(f"正在重启 {name}...")
                if name == "api_server":
                    start_api_server()
                elif name.startswith("gpu_worker_"):
                    gpu_id = int(name.split("_")[-1])
                    start_gpu_worker(gpu_id)

        # 休眠一段时间再检查
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="视频生成平台启动脚本")
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="要使用的GPU ID列表，用逗号分隔")
    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 确保必要的目录存在
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        # 启动API服务器
        start_api_server(args.port)

        # 等待API服务器启动
        time.sleep(2)

        # 启动GPU工作器
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
        for gpu_id in gpu_ids:
            start_gpu_worker(gpu_id)

        # 监控进程
        monitor_processes()

    except KeyboardInterrupt:
        logger.info("收到终止信号")
        stop_all_processes()
    except Exception as e:
        logger.error(f"发生错误: {e}")
        stop_all_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()