#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import signal
import sys

# 定义进程信息
processes = {
    "api": None,
    "workers": []
}

def start_api_server():
    """启动API服务器"""
    print("启动API服务器...")
    process = subprocess.Popen(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes["api"] = process
    print(f"API服务器已启动 (PID: {process.pid})")

def start_worker(gpu_id, api_url):
    """启动Worker进程"""
    print(f"启动Worker {gpu_id}...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    process = subprocess.Popen(
        ["python", "worker.py", "--gpu", str(gpu_id), "--api", api_url],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes["workers"].append(process)
    print(f"Worker {gpu_id} 已启动 (PID: {process.pid})")

def start_all(num_gpus, api_url):
    """启动所有服务"""
    # 首先启动API服务器
    start_api_server()

    # 等待API服务器启动完成
    time.sleep(3)

    # 然后启动Worker进程
    for gpu_id in range(num_gpus):
        start_worker(gpu_id, api_url)
        time.sleep(1)  # 稍微延迟，避免并发加载模型

def stop_all():
    """停止所有服务"""
    print("正在停止所有服务...")

    # 停止Worker进程
    for worker in processes["workers"]:
        if worker and worker.poll() is None:
            worker.terminate()
            print(f"已停止Worker进程 (PID: {worker.pid})")

    # 停止API服务器
    if processes["api"] and processes["api"].poll() is None:
        processes["api"].terminate()
        print(f"已停止API服务器 (PID: {processes['api'].pid})")

    print("所有服务已停止")

def monitor_processes():
    """监控进程状态"""
    try:
        while True:
            # 检查API服务器
            if processes["api"] and processes["api"].poll() is not None:
                print("API服务器已意外终止，正在重启...")
                start_api_server()

            # 检查Worker进程
            for i, worker in enumerate(processes["workers"]):
                if worker and worker.poll() is not None:
                    print(f"Worker {i} 已意外终止，正在重启...")
                    processes["workers"][i] = None
                    start_worker(i, args.api_url)

            time.sleep(5)
    except KeyboardInterrupt:
        print("\n接收到退出信号")
        stop_all()
        sys.exit(0)

def handle_signal(sig, frame):
    """处理信号"""
    print("\n接收到退出信号")
    stop_all()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频生成平台启动脚本")
    parser.add_argument("--gpus", type=int, default=4, help="GPU数量")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API URL")

    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"正在启动视频生成平台，使用 {args.gpus} 个GPU")

    # 启动所有服务
    start_all(args.gpus, args.api_url)

    # 监控进程状态
    monitor_processes()