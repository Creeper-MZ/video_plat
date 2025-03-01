import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import TaskCard from './TaskCard';
import VideoPreview from './VideoPreview';

// 关键修复：动态API地址，适应不同主机环境
const API_URL = window.location.protocol + '//' + window.location.hostname + ':8000';
console.log('使用API地址:', API_URL);

function App() {
    // 状态变量
    const [taskType, setTaskType] = useState('t2v');
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走');
    const [resolution, setResolution] = useState('832x480');
    const [numFrames, setNumFrames] = useState(100);
    const [fps, setFps] = useState(20);
    const [numInferenceSteps, setNumInferenceSteps] = useState(40);
    const [selectedImage, setSelectedImage] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [fp8, setFp8] = useState(true);
    const [saveVram, setSaveVram] = useState(false);
    const [seed, setSeed] = useState('');
    const [guidanceScale, setGuidanceScale] = useState(5.0);
    const [sampleShift, setSampleShift] = useState(5.0);

    // 任务相关状态
    const [tasks, setTasks] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('form'); // 'form' 或 'tasks'

    // WebSocket引用
    const websockets = useRef({});
    const fileInputRef = useRef(null);

    // 获取所有任务
    const fetchTasks = async () => {
        try {
            console.log('获取任务列表:', `${API_URL}/api/tasks`);
            const response = await fetch(`${API_URL}/api/tasks`);
            if (response.ok) {
                const data = await response.json();
                console.log('获取到任务:', data.length);
                setTasks(data);

                // 为新任务创建WebSocket连接
                data.forEach(task => {
                    if (['pending', 'processing'].includes(task.status) && !websockets.current[task.task_id]) {
                        connectWebSocket(task.task_id);
                    }
                });
            } else {
                console.error('获取任务失败:', response.status, response.statusText);
            }
        } catch (err) {
            console.error('获取任务失败:', err);
        }
    };

    // 创建WebSocket连接
    const connectWebSocket = (taskId) => {
        if (websockets.current[taskId]) return;

        // 修复：正确的WebSocket URL构建
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/ws/${taskId}`;

        console.log('连接WebSocket:', wsUrl);

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log(`WebSocket连接已建立: ${taskId}`);
        };

        ws.onmessage = (event) => {
            console.log(`收到WebSocket消息: ${taskId}`, event.data);
            const data = JSON.parse(event.data);

            setTasks(prevTasks => {
                const updatedTasks = prevTasks.map(task =>
                    task.task_id === taskId ? { ...task, ...data } : task
                );
                return updatedTasks;
            });

            // 如果任务完成或失败，关闭WebSocket
            if (['completed', 'failed', 'cancelled'].includes(data.status)) {
                ws.close();
                delete websockets.current[taskId];
            }
        };

        ws.onerror = (error) => {
            console.error(`WebSocket错误: ${taskId}`, error);
        };

        ws.onclose = () => {
            console.log(`WebSocket连接已关闭: ${taskId}`);
            delete websockets.current[taskId];
        };

        websockets.current[taskId] = ws;
    };

    // 处理图片选择
    const handleImageChange = (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            setSelectedImage(file);

            // 创建预览
            const reader = new FileReader();
            reader.onload = (e) => {
                setImagePreview(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    };

    // 打开文件选择器
    const handleImageClick = () => {
        fileInputRef.current.click();
    };

    // 处理拖放上传
    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('active');
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('active');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('active');

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                setSelectedImage(file);

                const reader = new FileReader();
                reader.onload = (e) => {
                    setImagePreview(e.target.result);
                };
                reader.readAsDataURL(file);
            }
        }
    };

    // 提交表单
    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        setError(null);

        try {
            // 验证必填字段
            if (!prompt) {
                throw new Error('请输入提示词');
            }

            if (taskType === 'i2v' && !selectedImage) {
                throw new Error('图生视频需要上传图片');
            }

            // 创建表单数据
            const formData = new FormData();
            formData.append('task_type', taskType);
            formData.append('prompt', prompt);
            formData.append('negative_prompt', negativePrompt);
            formData.append('resolution', resolution);
            formData.append('num_frames', numFrames);
            formData.append('fps', fps);
            formData.append('num_inference_steps', numInferenceSteps);
            formData.append('fp8', fp8);
            formData.append('save_vram', saveVram);

            if (seed) {
                formData.append('seed', parseInt(seed));
            }

            formData.append('guidance_scale', guidanceScale);
            formData.append('sample_shift', sampleShift);

            if (selectedImage) {
                formData.append('image', selectedImage);
            }

            // 增加调试日志
            console.log('提交任务到:', `${API_URL}/api/generate`);
            console.log('表单数据:', Object.fromEntries(formData.entries()));

            // 发送请求
            const response = await fetch(`${API_URL}/api/generate`, {
                method: 'POST',
                body: formData,
            });

            console.log('响应状态:', response.status, response.statusText);

            if (!response.ok) {
                let errorDetail = '';
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || '';
                } catch (e) {
                    errorDetail = await response.text() || '未知错误';
                }
                throw new Error(`提交任务失败: ${response.status} ${response.statusText} ${errorDetail}`);
            }

            const data = await response.json();
            console.log('任务提交成功:', data);

            // 任务提交成功后刷新任务列表
            await fetchTasks();

            // 连接WebSocket获取实时进度
            connectWebSocket(data.task_id);

            // 重置表单字段
            if (taskType === 'i2v') {
                setSelectedImage(null);
                setImagePreview(null);
            }

            // 切换到任务列表标签
            setActiveTab('tasks');

        } catch (err) {
            console.error('提交表单错误:', err);
            setError(err.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    // 取消任务
    const handleCancelTask = async (taskId) => {
        try {
            console.log('取消任务:', taskId);
            const response = await fetch(`${API_URL}/api/tasks/${taskId}/cancel`, {
                method: 'POST',
            });

            if (response.ok) {
                console.log('任务取消成功');
                // 取消请求成功后刷新任务列表
                await fetchTasks();
            } else {
                console.error('取消任务失败:', response.status, response.statusText);
            }
        } catch (err) {
            console.error('取消任务错误:', err);
        }
    };

    // 随机种子
    const handleRandomSeed = () => {
        setSeed(Math.floor(Math.random() * 1000000));
    };

    // 初始化和定期获取任务列表
    useEffect(() => {
        console.log('初始化应用...');
        fetchTasks();

        // 每10秒更新任务列表
        const interval = setInterval(fetchTasks, 10000);

        return () => {
            clearInterval(interval);

            // 关闭所有WebSocket连接
            Object.values(websockets.current).forEach(ws => {
                ws.close();
            });
        };
    }, []);

    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow">
                <div className="container mx-auto px-4 py-6">
                    <h1 className="text-3xl font-bold text-center text-gray-800">Wan2.1 视频生成平台</h1>
                </div>
            </header>

            <main className="container mx-auto px-4 py-8">
                {/* 移动端标签切换 */}
                <div className="md:hidden mb-6">
                    <div className="flex border rounded overflow-hidden">
                        <button
                            className={`flex-1 py-2 ${activeTab === 'form' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
                            onClick={() => setActiveTab('form')}
                        >
                            创建任务
                        </button>
                        <button
                            className={`flex-1 py-2 ${activeTab === 'tasks' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
                            onClick={() => setActiveTab('tasks')}
                        >
                            任务列表
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* 左侧表单 */}
                    <div className={activeTab === 'form' ? 'block' : 'hidden md:block'}>
                        <div className="bg-white rounded-lg shadow p-6">
                            <h2 className="text-xl font-semibold mb-6">创建新任务</h2>

                            {error && (
                                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                                    {error}
                                </div>
                            )}

                            <form onSubmit={handleSubmit}>
                                {/* 任务类型 */}
                                <div className="mb-6">
                                    <label className="form-label">任务类型</label>
                                    <div className="flex">
                                        <button
                                            type="button"
                                            className={`flex-1 px-4 py-2 rounded-l ${taskType === 't2v' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                                            onClick={() => setTaskType('t2v')}
                                        >
                                            文本生成视频
                                        </button>
                                        <button
                                            type="button"
                                            className={`flex-1 px-4 py-2 rounded-r ${taskType === 'i2v' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                                            onClick={() => setTaskType('i2v')}
                                        >
                                            图像生成视频
                                        </button>
                                    </div>
                                </div>

                                {/* 提示词 */}
                                <div className="mb-6">
                                    <label className="form-label">
                                        提示词 <span className="text-red-500">*</span>
                                    </label>
                                    <textarea
                                        className="form-input"
                                        rows="4"
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                        placeholder="描述您想要生成的视频内容..."
                                        required
                                    ></textarea>
                                </div>

                                {/* 负面提示词 */}
                                <div className="mb-6">
                                    <label className="form-label">负面提示词</label>
                                    <textarea
                                        className="form-input"
                                        rows="2"
                                        value={negativePrompt}
                                        onChange={(e) => setNegativePrompt(e.target.value)}
                                    ></textarea>
                                </div>

                                {/* 图片上传 (I2V) */}
                                {taskType === 'i2v' && (
                                    <div className="mb-6">
                                        <label className="form-label">
                                            上传图片 <span className="text-red-500">*</span>
                                        </label>
                                        <div
                                            className={`upload-area ${imagePreview ? 'bg-gray-50' : ''}`}
                                            onClick={handleImageClick}
                                            onDragOver={handleDragOver}
                                            onDragLeave={handleDragLeave}
                                            onDrop={handleDrop}
                                        >
                                            <input
                                                type="file"
                                                ref={fileInputRef}
                                                accept="image/*"
                                                onChange={handleImageChange}
                                                className="hidden"
                                            />

                                            {imagePreview ? (
                                                <div className="flex flex-col items-center">
                                                    <img
                                                        src={imagePreview}
                                                        alt="预览"
                                                        className="image-preview mb-2"
                                                    />
                                                    <p className="text-sm text-gray-500">点击或拖放更换图片</p>
                                                </div>
                                            ) : (
                                                <div className="flex flex-col items-center">
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                                    </svg>
                                                    <p className="text-gray-500">点击或拖放上传图片</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* 分辨率 */}
                                <div className="mb-6">
                                    <label className="form-label">分辨率</label>
                                    <select
                                        className="form-input"
                                        value={resolution}
                                        onChange={(e) => setResolution(e.target.value)}
                                    >
                                        <option value="832x480">480P (832x480)</option>
                                        <option value="480x832">480P 竖屏 (480x832)</option>
                                        <option value="1280x720">720P (1280x720)</option>
                                        <option value="720x1280">720P 竖屏 (720x1280)</option>
                                    </select>
                                </div>

                                {/* 基本参数 */}
                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                                    <div>
                                        <label className="form-label">帧数</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={numFrames}
                                            onChange={(e) => setNumFrames(parseInt(e.target.value))}
                                            min="5"
                                            max="200"
                                        />
                                    </div>
                                    <div>
                                        <label className="form-label">帧率</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={fps}
                                            onChange={(e) => setFps(parseInt(e.target.value))}
                                            min="5"
                                            max="60"
                                        />
                                    </div>
                                    <div>
                                        <label className="form-label">推理步数</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={numInferenceSteps}
                                            onChange={(e) => setNumInferenceSteps(parseInt(e.target.value))}
                                            min="20"
                                            max="60"
                                        />
                                    </div>
                                </div>

                                {/* 高级参数开关 */}
                                <div className="mb-4">
                                    <button
                                        type="button"
                                        className="text-blue-500 hover:underline flex items-center"
                                        onClick={() => setShowAdvanced(!showAdvanced)}
                                    >
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            className={`h-4 w-4 mr-1 transition-transform ${showAdvanced ? 'rotate-90' : ''}`}
                                            fill="none"
                                            viewBox="0 0 24 24"
                                            stroke="currentColor"
                                        >
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                        {showAdvanced ? '隐藏高级参数' : '显示高级参数'}
                                    </button>
                                </div>

                                {/* 高级参数 */}
                                {showAdvanced && (
                                    <div className="mb-6 p-4 border rounded bg-gray-50">
                                        <h3 className="font-medium mb-4">高级参数</h3>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                                            <div>
                                                <label className="flex items-center">
                                                    <input
                                                        type="checkbox"
                                                        checked={fp8}
                                                        onChange={(e) => setFp8(e.target.checked)}
                                                        className="mr-2 h-4 w-4 text-blue-600"
                                                    />
                                                    <span>使用FP8精度</span>
                                                    <span className="ml-1 text-xs text-gray-500" title="FP8精度可以显著减少显存使用，但可能略微影响生成质量">ⓘ</span>
                                                </label>
                                            </div>
                                            <div>
                                                <label className="flex items-center">
                                                    <input
                                                        type="checkbox"
                                                        checked={saveVram}
                                                        onChange={(e) => setSaveVram(e.target.checked)}
                                                        className="mr-2 h-4 w-4 text-blue-600"
                                                    />
                                                    <span>节省显存模式</span>
                                                    <span className="ml-1 text-xs text-gray-500" title="节省显存模式会更频繁地将模型参数卸载到CPU，减少GPU内存占用，但会降低生成速度">ⓘ</span>
                                                </label>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                                            <div>
                                                <label className="form-label">种子</label>
                                                <div className="flex">
                                                    <input
                                                        type="number"
                                                        className="form-input rounded-r-none"
                                                        value={seed}
                                                        onChange={(e) => setSeed(e.target.value)}
                                                        placeholder="随机"
                                                    />
                                                    <button
                                                        type="button"
                                                        className="bg-gray-200 px-2 rounded-r hover:bg-gray-300"
                                                        onClick={handleRandomSeed}
                                                        title="随机种子"
                                                    >
                                                        🎲
                                                    </button>
                                                </div>
                                            </div>
                                            <div>
                                                <label className="form-label">引导比例</label>
                                                <input
                                                    type="number"
                                                    step="0.1"
                                                    className="form-input"
                                                    value={guidanceScale}
                                                    onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                                                    min="1"
                                                    max="10"
                                                />
                                            </div>
                                            <div>
                                                <label className="form-label">采样偏移</label>
                                                <input
                                                    type="number"
                                                    step="0.1"
                                                    className="form-input"
                                                    value={sampleShift}
                                                    onChange={(e) => setSampleShift(parseFloat(e.target.value))}
                                                    min="1"
                                                    max="10"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* 提交按钮 */}
                                <div className="mt-6">
                                    <button
                                        type="submit"
                                        className="w-full btn-primary py-3 text-lg font-medium disabled:bg-gray-400 disabled:cursor-not-allowed"
                                        disabled={isSubmitting}
                                    >
                                        {isSubmitting ? (
                                            <span className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        提交中...
                      </span>
                                        ) : '开始生成'}
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>

                    {/* 右侧任务列表 */}
                    <div className={activeTab === 'tasks' ? 'block' : 'hidden md:block'}>
                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-xl font-semibold">任务列表</h2>
                                <button
                                    onClick={fetchTasks}
                                    className="text-blue-500 hover:text-blue-700"
                                    title="刷新任务列表"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                </button>
                            </div>

                            {tasks.length === 0 ? (
                                <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                    </svg>
                                    <p className="text-gray-500">暂无任务</p>
                                    <button
                                        onClick={() => setActiveTab('form')}
                                        className="mt-4 text-blue-500 hover:text-blue-700 md:hidden"
                                    >
                                        创建新任务
                                    </button>
                                </div>
                            ) : (
                                <div className="space-y-4 max-h-[800px] overflow-y-auto pr-1">
                                    {tasks.map((task) => (
                                        <TaskCard
                                            key={task.task_id}
                                            task={task}
                                            onCancel={handleCancelTask}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            <footer className="bg-white border-t mt-12 py-6">
                <div className="container mx-auto px-4 text-center text-gray-600 text-sm">
                    <p>Wan2.1 视频生成平台 &copy; {new Date().getFullYear()}</p>
                    <p className="text-xs mt-1 text-gray-400">API地址: {API_URL}</p>
                </div>
            </footer>
        </div>
    );
}

export default App;