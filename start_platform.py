#!/usr/bin/env python3
"""
视频生成平台启动脚本
用法: python start_service.py [--port PORT] [--gpus GPU_IDS]
"""

import os
import sys
import time
import signal
import argparse
import subprocess
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
    """处理终止信号"""
    global should_stop
    logger.info("收到终止信号，准备关闭所有进程...")
    should_stop = True
    stop_all_processes()
    sys.exit(0)


def start_api_server(port=8000):
    """启动API服务器"""
    logger.info(f"启动API服务器，端口: {port}")

    # 确保环境变量设置
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(BASE_DIR)
    )

    processes["api_server"] = proc
    logger.info(f"API服务器已启动，PID: {proc.pid}")
    return proc.pid


def start_gpu_worker(gpu_id):
    """启动GPU工作器进程"""
    logger.info(f"启动GPU {gpu_id}工作器")

    # 确保环境变量设置
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "worker.py", str(gpu_id)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
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
            # 先尝试优雅地终止进程
            processes[name].terminate()

            # 等待进程终止，最多等待5秒
            try:
                processes[name].wait(timeout=5)
                logger.info(f"{name} 已优雅终止")
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} 没有及时终止，强制杀死")
                processes[name].kill()
                processes[name].wait(timeout=2)

            # 捕获任何输出
            stdout, stderr = processes[name].communicate()
            if stdout:
                logger.info(f"{name} 标准输出: {stdout.strip()}")
            if stderr:
                logger.error(f"{name} 标准错误: {stderr.strip()}")

        except Exception as e:
            logger.error(f"停止 {name} 时出错: {e}")
        finally:
            del processes[name]
            logger.info(f"{name} 已停止")


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

    # 清理临时文件
    try:
        for file in Path(BASE_DIR).glob("progress_*.json"):
            file.unlink()
        logger.info("已清理临时进度文件")
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")

    logger.info("所有进程已停止")


def monitor_processes():
    """监控进程状态，如果有进程意外退出则重启"""
    logger.info("开始监控进程...")

    while not should_stop:
        for name, proc in list(processes.items()):
            returncode = proc.poll()
            if returncode is not None:
                logger.warning(f"{name} 意外退出，返回码: {returncode}")

                # 捕获输出
                stdout, stderr = proc.communicate()
                if stdout:
                    logger.info(f"{name} 标准输出: {stdout.strip()}")
                if stderr:
                    logger.error(f"{name} 标准错误: {stderr.strip()}")

                # 重启进程
                logger.info(f"正在重启 {name}...")
                if name == "api_server":
                    start_api_server()
                elif name.startswith("gpu_worker_"):
                    gpu_id = int(name.split("_")[-1])
                    start_gpu_worker(gpu_id)

        # 休眠一段时间再检查
        time.sleep(5)


def verify_environment():
    """验证运行环境"""
    logger.info("检查运行环境...")
    missing_packages = []

    # 检查必要的Python包
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")

    try:
        import fastapi
    except ImportError:
        missing_packages.append("fastapi")

    try:
        import uvicorn
    except ImportError:
        missing_packages.append("uvicorn")

    try:
        import sqlalchemy
    except ImportError:
        missing_packages.append("sqlalchemy")

    # 检查GPU是否可用
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            logger.info(f"检测到 {gpu_count} 个GPU:")
            for i, name in enumerate(gpu_names):
                logger.info(f"  GPU {i}: {name}")
        else:
            logger.warning("未检测到可用GPU!")
    except:
        logger.warning("检查GPU状态失败!")

    # 检查必要的文件和目录
    required_files = ["app.py", "worker.py"]
    for file in required_files:
        if not Path(file).exists():
            logger.error(f"缺少必要文件: {file}")
            sys.exit(1)

    # 创建必要的目录
    dirs_to_create = ["uploads", "outputs", "frontend/build"]
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 如果缺少必要的包，提示安装
    if missing_packages:
        logger.warning(f"缺少以下Python包: {', '.join(missing_packages)}")
        logger.warning("请使用以下命令安装:")
        logger.warning(f"pip install {' '.join(missing_packages)}")

        response = input("是否继续启动? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    return gpu_available


def main():
    parser = argparse.ArgumentParser(description="视频生成平台启动脚本")
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="要使用的GPU ID列表，用逗号分隔")
    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 验证环境
        gpu_available = verify_environment()
        if not gpu_available:
            logger.warning("未检测到GPU，生成性能可能受限!")
            response = input("是否继续启动? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)

        # 确保必要的目录存在
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("frontend/build", exist_ok=True)

        # 清理旧的进度文件
        for file in Path().glob("progress_*.json"):
            file.unlink()

        # 启动API服务器
        start_api_server(args.port)

        # 等待API服务器启动
        logger.info("等待API服务器启动...")
        time.sleep(3)

        # 启动GPU工作器
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
        for gpu_id in gpu_ids:
            start_gpu_worker(gpu_id)
            # 间隔启动，避免资源竞争
            time.sleep(1)

        logger.info("所有服务已启动，按Ctrl+C停止...")

        # 监控进程
        monitor_processes()

    except KeyboardInterrupt:
        logger.info("收到终止信号")
        stop_all_processes()
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        stop_all_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()