import React, { useState } from 'react';
import VideoPreview from './VideoPreview';

const TaskCard = ({ task, onCancel }) => {
    const [showPreview, setShowPreview] = useState(false);

    // 获取任务状态
    const getStatusText = (status) => {
        const statusMap = {
            'pending': '等待中',
            'processing': '处理中',
            'completed': '已完成',
            'failed': '失败',
            'cancelled': '已取消',
            'cancelling': '取消中'
        };
        return statusMap[status] || status;
    };

    // 获取状态类名
    const getStatusClass = (status) => {
        const classMap = {
            'pending': 'status-badge pending',
            'processing': 'status-badge processing',
            'completed': 'status-badge completed',
            'failed': 'status-badge failed',
            'cancelled': 'status-badge cancelled',
            'cancelling': 'status-badge cancelled'
        };
        return classMap[status] || 'status-badge';
    };

    // 渲染进度条
    const renderProgressBar = (progress) => {
        const percentage = Math.round(progress * 100);
        return (
            <div className="progress-bar">
                <div
                    className="progress-bar-value"
                    style={{ width: `${percentage}%` }}
                ></div>
            </div>
        );
    };

    // 打开视频预览
    const handleOpenPreview = () => {
        setShowPreview(true);
    };

    // 关闭视频预览
    const handleClosePreview = () => {
        setShowPreview(false);
    };

    return (
        <div className="task-card border rounded-lg bg-white shadow-sm p-4 mb-4">
            <div className="flex justify-between items-start mb-2">
                <div>
          <span className="font-medium mr-2">
            {task.task_type === 't2v' ? '文本生成视频' : '图像生成视频'}
          </span>
                    <span className={getStatusClass(task.status)}>
            {getStatusText(task.status)}
          </span>
                </div>
                <div className="text-sm text-gray-500">
                    {new Date(task.created_at * 1000).toLocaleString()}
                </div>
            </div>

            <div className="mb-3">
                <p className="text-sm text-gray-700 mb-1 line-clamp-2" title={task.prompt}>
                    {task.prompt}
                </p>
            </div>

            {task.status === 'processing' && (
                <div className="mb-3">
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                        <span>{task.message || '处理中...'}</span>
                        <span>{Math.round((task.progress || 0) * 100)}%</span>
                    </div>
                    {renderProgressBar(task.progress || 0)}
                </div>
            )}

            {task.status === 'failed' && (
                <p className="text-sm text-red-600 mb-3">{task.message}</p>
            )}

            <div className="flex flex-wrap gap-2 mb-3 text-xs text-gray-500">
                <span>分辨率: {task.resolution}</span>
                <span>帧数: {task.num_frames}</span>
                <span>帧率: {task.fps}</span>
                <span>步数: {task.num_inference_steps}</span>
                {task.seed && <span>种子: {task.seed}</span>}
            </div>

            <div className="flex justify-end gap-2">
                {['pending', 'processing'].includes(task.status) && (
                    <button
                        onClick={() => onCancel(task.task_id)}
                        className="btn-danger text-sm"
                    >
                        取消
                    </button>
                )}

                {task.status === 'completed' && task.output_path && (
                    <>
                        <button
                            onClick={handleOpenPreview}
                            className="text-blue-500 hover:text-blue-700 text-sm"
                        >
                            预览
                        </button>
                        <a
                            href={task.output_path}
                            download
                            className="text-green-500 hover:text-green-700 text-sm ml-2"
                        >
                            下载
                        </a>
                    </>
                )}
            </div>

            {showPreview && task.output_path && (
                <VideoPreview
                    videoUrl={task.output_path}
                    onClose={handleClosePreview}
                />
            )}
        </div>
    );
};

export default TaskCard;