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
    "frontend": None,
    "workers": []
}

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 尝试使用socket获取IP
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接一个外部地址，不需要真正建立连接
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        # 失败时使用hostname
        try:
            return subprocess.check_output("hostname -I", shell=True).decode().split()[0]
        except:
            return "localhost"

def start_api_server(api_port):
    """启动API服务器"""
    print("启动API服务器...")
    process = subprocess.Popen(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes["api"] = process
    server_ip = get_local_ip()
    print(f"API服务器已启动 (PID: {process.pid})")
    print(f"API地址: http://{server_ip}:{api_port}")

def start_frontend_server(web_port):
    """启动前端静态文件服务器"""
    # 检查构建目录
    if os.path.exists("frontend/build"):
        static_dir = "frontend/build"
    elif os.path.exists("static"):
        static_dir = "static"
    else:
        print("未找到前端构建目录 (frontend/build 或 static)，跳过前端服务启动")
        return

    print(f"启动前端服务器，提供目录: {static_dir}...")

    # 启动前端服务
    process = subprocess.Popen(
        ["python", "-m", "http.server", str(web_port), "--bind", "0.0.0.0", "--directory", static_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes["frontend"] = process
    server_ip = get_local_ip()
    print(f"前端服务器已启动 (PID: {process.pid})")
    print(f"前端访问地址: http://{server_ip}:{web_port}")

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

def start_all(num_gpus, api_url, api_port, web_port):
    """启动所有服务"""
    # 首先启动API服务器
    start_api_server(api_port)

    # 启动前端服务器
    start_frontend_server(web_port)

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
            try:
                worker.terminate()
                print(f"已停止Worker进程 (PID: {worker.pid})")
            except:
                print(f"停止Worker进程失败 (PID: {worker.pid})")

    # 停止前端服务器
    if processes["frontend"] and processes["frontend"].poll() is None:
        try:
            processes["frontend"].terminate()
            print(f"已停止前端服务器 (PID: {processes['frontend'].pid})")
        except:
            print(f"停止前端服务器失败 (PID: {processes['frontend'].pid})")

    # 停止API服务器
    if processes["api"] and processes["api"].poll() is None:
        try:
            processes["api"].terminate()
            print(f"已停止API服务器 (PID: {processes['api'].pid})")
        except:
            print(f"停止API服务器失败 (PID: {processes['api'].pid})")

    print("所有服务已停止")

def monitor_processes(api_url, api_port, web_port):
    """监控进程状态"""
    try:
        while True:
            # 检查API服务器
            if processes["api"] and processes["api"].poll() is not None:
                print("API服务器已意外终止，正在重启...")
                start_api_server(api_port)

            # 检查前端服务器
            if processes["frontend"] and processes["frontend"].poll() is not None:
                print("前端服务器已意外终止，正在重启...")
                start_frontend_server(web_port)

            # 检查Worker进程
            for i, worker in enumerate(processes["workers"]):
                if worker and worker.poll() is not None:
                    print(f"Worker {i} 已意外终止，正在重启...")
                    processes["workers"][i] = None
                    start_worker(i, api_url)

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
    parser.add_argument("--api-port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--web-port", type=int, default=3000, help="前端Web服务器端口")
    parser.add_argument("--api-url", type=str, default=None, help="API URL (默认根据api-port自动生成)")

    args = parser.parse_args()

    # 如果没有指定API URL，则根据本机IP和端口自动生成
    if not args.api_url:
        server_ip = get_local_ip()
        args.api_url = f"http://{server_ip}:{args.api_port}"

    # 注册信号处理
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"正在启动视频生成平台，使用 {args.gpus} 个GPU")
    print(f"API端口: {args.api_port}")
    print(f"Web端口: {args.web_port}")
    print(f"API URL: {args.api_url}")

    # 启动所有服务
    start_all(args.gpus, args.api_url, args.api_port, args.web_port)

    # 监控进程状态
    monitor_processes(args.api_url, args.api_port, args.web_port)