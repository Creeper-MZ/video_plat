import React, { useState, useEffect } from 'react';

const VideoPreview = ({ videoUrl, onClose }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    // 获取完整URL
    const getFullUrl = (url) => {
        if (!url) return '';
        // URL已经是绝对URL
        if (url.startsWith('http://') || url.startsWith('https://')) {
            return url;
        }

        // 相对URL，添加域名
        const baseUrl = window.location.protocol + '//' + window.location.hostname + ':8000';
        return `${baseUrl}${url.startsWith('/') ? '' : '/'}${url}`;
    };

    const fullVideoUrl = getFullUrl(videoUrl);

    useEffect(() => {
        // 重置状态
        setIsLoading(true);
        setError(null);
        console.log('加载视频:', fullVideoUrl);
    }, [fullVideoUrl]);

    const handleVideoLoad = () => {
        console.log('视频加载成功');
        setIsLoading(false);
    };

    const handleVideoError = (e) => {
        console.error('视频加载失败:', e);
        setIsLoading(false);
        setError('视频加载失败，请稍后再试。可能的原因：视频还未完全生成或URL错误。');
    };

    // 键盘事件处理
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [onClose]);

    return (
        <div
            className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
            onClick={onClose}  // 点击背景关闭
        >
            <div
                className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col"
                onClick={(e) => e.stopPropagation()}  // 阻止冒泡，避免点击内容区域关闭
            >
                <div className="flex justify-between items-center p-4 border-b">
                    <h3 className="text-xl font-semibold">视频预览</h3>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700"
                        aria-label="关闭"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-gray-100">
                    {isLoading && (
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
                            <p className="mt-2 text-gray-600">视频加载中，请稍候...</p>
                        </div>
                    )}

                    {error && (
                        <div className="text-center text-red-500 p-4">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-red-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <p className="mb-2">{error}</p>
                            <div className="flex justify-center gap-2 mt-4">
                                <button
                                    onClick={() => window.open(fullVideoUrl, '_blank')}
                                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                                >
                                    在新窗口打开
                                </button>
                                <button
                                    onClick={() => { setIsLoading(true); setError(null); }}
                                    className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                                >
                                    重试
                                </button>
                            </div>
                        </div>
                    )}

                    <video
                        key={fullVideoUrl} // 当URL改变时强制重新加载
                        src={fullVideoUrl}
                        className={`max-h-[70vh] max-w-full rounded ${isLoading ? 'hidden' : 'block'}`}
                        controls
                        autoPlay
                        playsInline
                        preload="auto"
                        onLoadedData={handleVideoLoad}
                        onError={handleVideoError}
                    />
                </div>

                <div className="p-4 border-t flex justify-between">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
                    >
                        关闭
                    </button>

                    <a
                        href={fullVideoUrl}
                        download
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                        下载视频
                    </a>
                </div>
            </div>
        </div>
    );
};

export default VideoPreview;