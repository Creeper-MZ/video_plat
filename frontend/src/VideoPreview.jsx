import React, { useState, useEffect } from 'react';

const VideoPreview = ({ videoUrl, onClose }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        // 重置状态
        setIsLoading(true);
        setError(null);
    }, [videoUrl]);

    const handleVideoLoad = () => {
        setIsLoading(false);
    };

    const handleVideoError = () => {
        setIsLoading(false);
        setError('视频加载失败，请稍后再试。');
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
                <div className="flex justify-between items-center p-4 border-b">
                    <h3 className="text-xl font-semibold">视频预览</h3>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700"
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
                            <p className="mt-2 text-gray-600">加载中...</p>
                        </div>
                    )}

                    {error && (
                        <div className="text-center text-red-500">
                            <p>{error}</p>
                            <button
                                onClick={() => window.open(videoUrl, '_blank')}
                                className="mt-2 text-blue-500 hover:underline"
                            >
                                尝试在新窗口打开
                            </button>
                        </div>
                    )}

                    <video
                        src={videoUrl}
                        className={`max-h-[70vh] max-w-full rounded ${isLoading ? 'hidden' : 'block'}`}
                        controls
                        autoPlay
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
                        href={videoUrl}
                        download
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