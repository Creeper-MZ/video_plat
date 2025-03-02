# app/main.py

import logging
import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 改为绝对导入
from app.api import api_router
from app.core.config import settings
from app.core.logging import setup_app_logger
from app.core.gpu_manager import gpu_manager
from app.services.task_queue import task_queue

def create_app(debug=False):
    # Setup logging
    logger = setup_app_logger(debug=debug)
    logger.info("Initializing Wan2.1 Video Generation Platform")
    
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
            
            # Create necessary directories
            os.makedirs(settings.output_dir, exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("tasks", exist_ok=True)
            
            logger.info("Startup cleanup completed")
        except Exception as e:
            logger.error(f"Error during startup cleanup: {str(e)}")
    
    # Mount static files
    static_dir = Path("app/static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Serve index.html for root path
    @app.get("/")
    async def read_index():
        return FileResponse("app/static/index.html")
    
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
    
    # Run server directly without factory function for easier command-line usage
    app = create_app(debug=args.debug)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )