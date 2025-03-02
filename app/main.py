# app/main.py

import logging
import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 获取项目关键路径
def get_project_paths():
    """获取项目的关键路径"""
    # 确定当前文件的目录
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果在app目录下直接运行，current_file_dir就是app目录
    app_dir = current_file_dir
    
    # 项目根目录是app目录的父目录
    project_root = os.path.dirname(app_dir)
    
    # 静态文件目录
    static_dir = os.path.join(app_dir, "static")
    
    # 确保静态目录存在
    os.makedirs(static_dir, exist_ok=True)
    
    # 静态文件的完整路径
    index_html_path = os.path.join(static_dir, "index.html")
    
    # 视频输出目录
    videos_dir = os.path.join(static_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # 日志和任务目录
    logs_dir = os.path.join(project_root, "logs")
    tasks_dir = os.path.join(project_root, "tasks")
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tasks_dir, exist_ok=True)
    
    return {
        "app_dir": app_dir,
        "project_root": project_root,
        "static_dir": static_dir,
        "index_html_path": index_html_path,
        "videos_dir": videos_dir,
        "logs_dir": logs_dir, 
        "tasks_dir": tasks_dir
    }

# 使用绝对导入
from app.api import api_router
from app.core.config import settings
from app.core.logging import setup_app_logger
from app.core.gpu_manager import gpu_manager
from app.services.task_queue import task_queue

def create_app(debug=False):
    # 获取路径
    paths = get_project_paths()
    
    # Setup logging
    logger = setup_app_logger(debug=debug)
    logger.info("Initializing Wan2.1 Video Generation Platform")
    logger.info(f"Project paths: {paths}")
    
    # 确保必要的目录存在
    for path_name, path in paths.items():
        if path_name.endswith('_dir'):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Ensured directory exists: {path}")
    
    # 更新settings中的输出目录
    settings.output_dir = paths["videos_dir"]
    logger.info(f"Output directory set to: {settings.output_dir}")
    
    # 检查index.html文件
    if not os.path.exists(paths["index_html_path"]):
        logger.warning(f"Index file not found at expected path: {paths['index_html_path']}")
        
        # 尝试在源码目录中找到示例index.html并复制过去
        source_html = os.path.join(current_dir, "static", "index.html.example")
        if os.path.exists(source_html):
            import shutil
            shutil.copy(source_html, paths["index_html_path"])
            logger.info(f"Copied example index.html to {paths['index_html_path']}")
        else:
            logger.warning("No example index.html found, will generate basic HTML page on request")
    else:
        logger.info(f"Found index.html at: {paths['index_html_path']}")
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        debug=debug
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for now
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"]
    )
    
    # Error handling
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception(f"Global exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"}
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    
    # Cleanup old files
    @app.on_event("startup")
    def cleanup_old_files():
        try:
            # Clean up old video files (keep last 100)
            video_dir = Path(settings.output_dir)
            if video_dir.exists():
                video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.png"))
                video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                if len(video_files) > 100:
                    for old_file in video_files[100:]:
                        try:
                            old_file.unlink()
                            logger.debug(f"Cleaned up old output file: {old_file}")
                        except Exception as e:
                            logger.error(f"Failed to delete old file {old_file}: {str(e)}")
            
            logger.info("Startup cleanup completed")
        except Exception as e:
            logger.error(f"Error during startup cleanup: {str(e)}")
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=paths["static_dir"]), name="static")
    
    # Serve index.html for root path
    @app.get("/")
    async def read_index():
        if not os.path.exists(paths["index_html_path"]):
            logger.error(f"Index file not found at {paths['index_html_path']}")
            
            # 如果找不到，返回一个简单的HTML页面
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Wan2.1 Video Generation Platform</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body class="bg-light">
                <div class="container mt-5">
                    <div class="row">
                        <div class="col-md-8 offset-md-2">
                            <div class="card">
                                <div class="card-header bg-primary text-white">
                                    <h2>Wan2.1 Video Generation Platform</h2>
                                </div>
                                <div class="card-body">
                                    <div class="alert alert-warning">
                                        <strong>Warning:</strong> The index.html file was not found at the expected location.
                                    </div>
                                    <p>The API is still available. You can:</p>
                                    <ul>
                                        <li>Access the API documentation at <a href="/docs">/docs</a></li>
                                        <li>Use the API endpoints directly</li>
                                    </ul>
                                    <hr>
                                    <h4>API Endpoints:</h4>
                                    <ul>
                                        <li><code>POST /api/generate/t2v</code> - Text to Video generation</li>
                                        <li><code>POST /api/generate/i2v</code> - Image to Video generation</li>
                                        <li><code>GET /api/generate/task/{task_id}</code> - Get task status</li>
                                        <li><code>GET /api/status/system</code> - Get system status</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        
        return FileResponse(paths["index_html_path"])
    
    # Include API routes
    app.include_router(api_router, prefix="/api")
    
    # Shutdown event
    @app.on_event("shutdown")
    def shutdown_event():
        logger.info("Shutting down application")
        task_queue.stop_worker()
        gpu_manager.stop_monitoring()
    
    return app

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.1 Video Generation Platform")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 直接运行app实例，不使用工厂函数
    app = create_app(debug=args.debug)
    
    # 输出启动信息
    paths = get_project_paths()
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Static files directory: {paths['static_dir']}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    
    # 检查index.html是否存在
    if not os.path.exists(paths["index_html_path"]):
        print(f"Warning: index.html not found at {paths['index_html_path']}")
        print("A basic HTML page will be generated instead.")
    
    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )