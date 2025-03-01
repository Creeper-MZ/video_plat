import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Badge,
  IconButton,
  Text,
  HStack,
  VStack,
  Heading,
  Progress,
  Spinner,
  Flex,
  Button,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure
} from '@chakra-ui/react';
import { ViewIcon, DeleteIcon, RepeatIcon, DownloadIcon, CheckIcon, WarningIcon } from '@chakra-ui/icons';

// 格式化日期
const formatDate = (dateString) => {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  }).format(date);
};

// 状态徽章配置
const statusConfig = {
  queued: { color: 'yellow', label: '排队中' },
  initializing: { color: 'orange', label: '初始化中' },
  running: { color: 'blue', label: '生成中' },
  saving: { color: 'teal', label: '保存中' },
  completed: { color: 'green', label: '已完成' },
  failed: { color: 'red', label: '失败' },
  cancelled: { color: 'gray', label: '已取消' }
};

export const TaskDetails = ({ taskId, onCancelTask }) => {
  const [task, setTask] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  const socketRef = useRef(null);
  const videoRef = useRef(null);
  const pingIntervalRef = useRef(null);
  const fetchIntervalRef = useRef(null);
  const toast = useToast();

  const { isOpen, onOpen, onClose } = useDisclosure(); // 用于视频预览模态框

  // 获取任务详情
  const fetchTaskDetails = async () => {
    try {
      const response = await fetch(`/api/tasks/${taskId}`);
      if (response.ok) {
        const data = await response.json();

        // 避免状态回退
        setTask(prevTask => {
          if (!prevTask) return data;

          // 确保不会把正在运行的任务更新为已完成，除非真的完成了
          if (
            prevTask.status === 'running' &&
            data.status === 'completed' &&
            !videoLoaded
          ) {
            console.log("防止状态回退: 保持运行状态直到视频加载完成");
            return { ...data, status: 'running', progress: 0.95 };
          }

          // 如果收到的数据比当前状态旧（时间戳更早），忽略它
          if (
            prevTask.timestamp &&
            data.timestamp &&
            prevTask.timestamp > data.timestamp
          ) {
            console.log("忽略旧数据");
            return prevTask;
          }

          return data;
        });
      } else {
        throw new Error('获取任务详情失败');
      }
    } catch (error) {
      console.error("获取任务失败:", error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // 初始化 WebSocket 连接
  const initializeWebSocket = () => {
    // 关闭之前的连接
    if (socketRef.current) {
      socketRef.current.close();
    }

    // 创建新连接
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/${taskId}`);

    socket.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`);
      setConnectionStatus('connected');

      // 设置定时ping保持连接
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }

      pingIntervalRef.current = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send("ping");
        }
      }, 15000); // 每15秒ping一次
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("WebSocket received data:", data);

        // 忽略pong响应
        if (data.pong) return;

        // 更新任务状态
        setTask(prevTask => {
          // 如果收到的数据比当前状态旧（时间戳更早），忽略它
          if (
            prevTask &&
            prevTask.timestamp &&
            data.timestamp &&
            prevTask.timestamp > data.timestamp
          ) {
            console.log("忽略旧的WebSocket数据");
            return prevTask;
          }

          // 如果当前视频已加载但接收到的状态不是completed，保持completed状态
          if (
            prevTask &&
            prevTask.status === 'completed' &&
            videoLoaded &&
            data.status !== 'completed'
          ) {
            return {
              ...prevTask,
              ...data,
              status: 'completed',
              progress: 1.0,
              timestamp: data.timestamp
            };
          }

          return {
            ...(prevTask || {}),
            ...data,
            timestamp: data.timestamp
          };
        });
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    socket.onclose = (event) => {
      console.log(`WebSocket closed for task ${taskId}:`, event.code, event.reason);
      setConnectionStatus('disconnected');

      // 清除ping interval
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }

      // 如果任务仍在进行中，尝试重新连接
      if (task && ['queued', 'running', 'initializing', 'saving'].includes(task.status)) {
        console.log("Reconnecting WebSocket in 2 seconds...");
        setTimeout(initializeWebSocket, 2000);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    socketRef.current = socket;
  };

  // 初始化和清理
  useEffect(() => {
    if (taskId) {
      fetchTaskDetails();
      initializeWebSocket();

      // 定时刷新任务状态 - 对于运行中的任务更频繁刷新
      fetchIntervalRef.current = setInterval(() => {
        if (task && ['running', 'initializing', 'saving'].includes(task.status)) {
          fetchTaskDetails();
        }
      }, 2000); // 每2秒刷新一次

      return () => {
        // 清理定时器
        if (fetchIntervalRef.current) {
          clearInterval(fetchIntervalRef.current);
        }

        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }

        // 关闭WebSocket连接
        if (socketRef.current) {
          socketRef.current.close();
        }
      };
    }
  }, [taskId]);

  // 当状态变为completed时，检查视频是否可用
  useEffect(() => {
    if (task && task.status === 'completed' && !videoLoaded && !videoError) {
      checkVideoAvailability();
    }
  }, [task?.status]);

  // 检查视频是否可用
  const checkVideoAvailability = async () => {
    if (!task || !task.output_url) return;

    try {
      // 检查文件是否存在
      const response = await fetch(`/api/files/check/video/${taskId}`);
      const data = await response.json();

      if (data.exists && data.size > 1000) {
        console.log("视频文件已确认存在:", data);
        setVideoError(false);
      } else {
        console.log("视频文件不存在或太小:", data);
        setVideoError(true);

        // 如果文件不存在但状态是completed，更新状态
        if (task.status === 'completed') {
          setTask(prev => ({
            ...prev,
            status: 'running',
            progress: 0.95,
            status_message: "等待视频生成完成..."
          }));
        }
      }
    } catch (error) {
      console.error("检查视频可用性失败:", error);
      setVideoError(true);
    }
  };

  // 重新加载视频
  const handleReloadVideo = () => {
    setVideoLoaded(false);
    setVideoError(false);
    setRetryCount(prev => prev + 1);

    if (videoRef.current) {
      videoRef.current.load();
    }

    // 同时刷新任务状态
    fetchTaskDetails();
  };

  // 下载视频
  const handleDownloadVideo = () => {
    if (task && task.output_url) {
      // 创建一个临时链接
      const link = document.createElement('a');
      link.href = task.output_url;
      link.download = `${taskId}.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  if (isLoading) {
    return (
      <Flex justify="center" align="center" height="200px">
        <Spinner />
      </Flex>
    );
  }

  if (error) {
    return (
      <Box textAlign="center" py={8}>
        <Alert status="error">
          <AlertIcon />
          <AlertTitle>获取任务失败</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button mt={4} onClick={fetchTaskDetails}>重试</Button>
      </Box>
    );
  }

  if (!task) {
    return (
      <Box textAlign="center" py={8}>
        <Text>未找到任务信息</Text>
      </Box>
    );
  }

  const statusColor = statusConfig[task.status]?.color || 'gray';
  const statusLabel = statusConfig[task.status]?.label || task.status;
  const taskTypeLabel = task.type === 'text_to_video' ? '文本生成视频' : '图片生成视频';

  return (
    <Box>
      <Heading size="md" mb={4}>任务详情</Heading>

      <VStack spacing={4} align="stretch">
        <HStack justify="space-between">
          <Badge colorScheme={statusColor} fontSize="md" px={2} py={1}>
            {statusLabel}
          </Badge>
          <HStack>
            <Text fontSize="sm">{taskTypeLabel}</Text>
            <Badge colorScheme={connectionStatus === 'connected' ? 'green' : 'gray'} size="sm">
              {connectionStatus === 'connected' ? '实时连接' : '检查连接'}
            </Badge>
          </HStack>
        </HStack>

        {/* 进度条和状态信息 */}
        {(task.status === 'running' || task.status === 'queued' || task.status === 'initializing' || task.status === 'saving') && (
          <Box>
            <HStack justify="space-between" mb={1}>
              <Text fontSize="sm">进度</Text>
              <Text fontSize="sm">{Math.round(task.progress * 100)}%</Text>
            </HStack>
            <Progress
              value={task.progress * 100}
              size="sm"
              colorScheme={
                task.status === 'initializing' ? "orange" :
                task.status === 'saving' ? "teal" :
                "blue"
              }
              borderRadius="md"
              hasStripe
              isAnimated={task.status !== 'queued'}
            />

            {/* 详细状态信息 */}
            {task.status_message && (
              <Text fontSize="sm" mt={1} color="gray.600">
                {task.status_message}
              </Text>
            )}

            {/* 步骤信息 */}
            {task.current_step !== undefined && task.total_steps !== undefined && (
              <Text fontSize="sm" mt={1} color="gray.600">
                步骤: {task.current_step}/{task.total_steps}
              </Text>
            )}

            {/* 预计剩余时间 */}
            {task.estimated_time !== undefined && task.estimated_time > 0 && (
              <Text fontSize="sm" mt={1} color="gray.600">
                预计剩余: {task.estimated_time > 60
                  ? `${Math.floor(task.estimated_time / 60)}分${task.estimated_time % 60}秒`
                  : `${task.estimated_time}秒`}
              </Text>
            )}
          </Box>
        )}

        {/* 错误信息 */}
        {task.error_message && (
          <Alert status="error" variant="left-accent">
            <AlertIcon />
            <Box>
              <AlertTitle>生成失败</AlertTitle>
              <AlertDescription fontSize="sm">{task.error_message}</AlertDescription>
            </Box>
          </Alert>
        )}

        {/* 提示词显示 */}
        <Box>
          <Text fontWeight="semibold" mb={1}>提示词</Text>
          <Text fontSize="sm" whiteSpace="pre-wrap" bg="gray.50" p={2} borderRadius="md">
            {task.prompt}
          </Text>
        </Box>

        {/* 负面提示词显示 */}
        {task.negative_prompt && (
          <Box>
            <Text fontWeight="semibold" mb={1}>负面提示词</Text>
            <Text fontSize="sm" whiteSpace="pre-wrap" bg="gray.50" p={2} borderRadius="md" color="gray.600">
              {task.negative_prompt}
            </Text>
          </Box>
        )}

        {/* 视频预览 */}
        {task.status === 'completed' && task.output_url && (
          <Box mt={2}>
            <HStack justify="space-between" mb={2}>
              <Text fontWeight="semibold">生成结果</Text>
              <HStack>
                <IconButton
                  aria-label="在弹窗中预览"
                  icon={<ViewIcon />}
                  size="sm"
                  onClick={onOpen}
                  title="在弹窗中预览"
                />
                <IconButton
                  aria-label="重新加载视频"
                  icon={<RepeatIcon />}
                  size="sm"
                  onClick={handleReloadVideo}
                  title="重新加载视频"
                />
                <IconButton
                  aria-label="下载视频"
                  icon={<DownloadIcon />}
                  size="sm"
                  colorScheme="blue"
                  onClick={handleDownloadVideo}
                  title="下载视频"
                />
              </HStack>
            </HStack>

            {/* 视频预览区域 */}
            <Box borderWidth={1} borderRadius="md" overflow="hidden" position="relative">
              {!videoLoaded && (
                <Flex
                  position="absolute"
                  top="0"
                  left="0"
                  right="0"
                  bottom="0"
                  bg="gray.100"
                  zIndex="1"
                  justify="center"
                  align="center"
                  direction="column"
                >
                  <Spinner size="xl" color="blue.500" mb={2} />
                  <Text color="gray.600">加载视频中...</Text>
                </Flex>
              )}

              {videoError && (
                <Alert status="warning" variant="subtle" flexDirection="column" alignItems="center" justifyContent="center" textAlign="center" height="200px">
                  <AlertIcon boxSize="40px" mr={0} />
                  <AlertTitle mt={4} mb={1} fontSize="lg">视频加载失败</AlertTitle>
                  <AlertDescription maxWidth="sm">
                    视频可能仍在处理中，请稍后再试
                    <Button
                      leftIcon={<RepeatIcon />}
                      mt={4}
                      colorScheme="blue"
                      onClick={handleReloadVideo}
                    >
                      重新加载
                    </Button>
                  </AlertDescription>
                </Alert>
              )}

              <video
                ref={videoRef}
                controls
                width="100%"
                height="auto"
                src={`${task.output_url}?v=${retryCount}`} // 添加版本参数避免缓存
                style={{ display: videoError ? 'none' : 'block' }}
                onLoadedData={() => {
                  console.log("视频加载完成");
                  setVideoLoaded(true);
                  setVideoError(false);

                  // 确保任务状态为已完成
                  if (task.status !== 'completed') {
                    setTask(prev => ({
                      ...prev,
                      status: 'completed',
                      progress: 1.0
                    }));
                  }
                }}
                onError={(e) => {
                  console.error("视频加载失败:", e);
                  setVideoLoaded(false);
                  setVideoError(true);

                  // 如果是COMPLETED状态但视频无法加载，可能是误报完成
                  if (task.status === 'completed') {
                    console.log("视频未完全生成，重置状态");
                    setTask(prev => ({
                      ...prev,
                      status: 'running',
                      progress: 0.95,
                      status_message: "等待视频生成完成..."
                    }));

                    // 刷新任务状态
                    fetchTaskDetails();
                  }
                }}
              >
                您的浏览器不支持视频标签
              </video>
            </Box>

            {/* 视频预览模态框 */}
            <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered>
              <ModalOverlay />
              <ModalContent>
                <ModalHeader>视频预览</ModalHeader>
                <ModalCloseButton />
                <ModalBody>
                  <video
                    controls
                    autoPlay
                    width="100%"
                    height="auto"
                    src={`${task.output_url}?v=${retryCount}`}
                  >
                    您的浏览器不支持视频标签
                  </video>
                </ModalBody>
                <ModalFooter>
                  <Button colorScheme="blue" mr={3} leftIcon={<DownloadIcon />} onClick={handleDownloadVideo}>
                    下载视频
                  </Button>
                  <Button variant="ghost" onClick={onClose}>关闭</Button>
                </ModalFooter>
              </ModalContent>
            </Modal>
          </Box>
        )}

        {/* 任务元数据信息 */}
        <VStack align="stretch" spacing={2} mt={2}>
          <Box>
            <Text fontSize="xs" color="gray.500">创建时间</Text>
            <Text fontSize="sm">{formatDate(task.created_at)}</Text>
          </Box>

          {task.started_at && (
            <Box>
              <Text fontSize="xs" color="gray.500">开始时间</Text>
              <Text fontSize="sm">{formatDate(task.started_at)}</Text>
            </Box>
          )}

          {task.completed_at && (
            <Box>
              <Text fontSize="xs" color="gray.500">完成时间</Text>
              <Text fontSize="sm">{formatDate(task.completed_at)}</Text>
            </Box>
          )}

          <Box>
            <Text fontSize="xs" color="gray.500">任务ID</Text>
            <Text fontSize="sm" fontFamily="mono">{task.id}</Text>
          </Box>

          {/* 任务参数 */}
          <Box>
            <Text fontSize="xs" color="gray.500">参数设置</Text>
            <HStack fontSize="sm" mt={1} flexWrap="wrap" spacing={2}>
              <Badge>{task.resolution}</Badge>
              <Badge>{task.frames}帧</Badge>
              <Badge>{task.fps}fps</Badge>
              <Badge>{task.steps}步</Badge>
              <Badge>{task.model_precision}</Badge>
              {task.save_vram && <Badge colorScheme="purple">节省显存</Badge>}
              {task.tiled && <Badge colorScheme="teal">平铺模式</Badge>}
            </HStack>
          </Box>
        </VStack>

        {/* 取消任务按钮 */}
        {['queued', 'running', 'initializing', 'saving'].includes(task.status) && (
          <Box mt={4}>
            <Button
              leftIcon={<DeleteIcon />}
              colorScheme="red"
              variant="outline"
              width="full"
              onClick={() => {
                onCancelTask(task.id);
                toast({
                  title: "正在取消任务",
                  description: "请稍等，任务正在取消中...",
                  status: "info",
                  duration: 3000,
                  isClosable: true,
                });
              }}
            >
              取消任务
            </Button>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default TaskDetails;