# app/utils/helpers.py

import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

def generate_unique_filename(prefix: str, extension: str) -> str:
    """
    生成唯一的文件名
    
    Args:
        prefix: 文件名前缀
        extension: 文件扩展名（不包含点号）
        
    Returns:
        唯一文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{random_id}.{extension}"

def ensure_directory_exists(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)

def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小（字节）
    """
    return os.path.getsize(file_path)

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小为人类可读形式
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化后的文件大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def clean_old_files(directory: str, pattern: str, max_files: int = 100) -> int:
    """
    清理目录中的旧文件，只保留最新的max_files个文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        max_files: 保留的最大文件数量
        
    Returns:
        删除的文件数量
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0
        
    files = list(dir_path.glob(pattern))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    deleted_count = 0
    if len(files) > max_files:
        for file_path in files[max_files:]:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception:
                pass
    
    return deleted_count