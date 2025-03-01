import React, { useState, useEffect, useRef } from 'react';

const TaskDetails = ({ taskId, onCancelTask }) => {
  const [task, setTask] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const socketRef = useRef(null);
  const videoRef = useRef(null);

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
          if (prevTask.status === 'running' && data.status === 'completed') {
            const isTrulyCompleted = data.progress >= 0.99;
            if (!isTrulyCompleted) {
              console.log("防止状态回退: 保持运行状态直到真正完成");
              return { ...data, status: 'running' };
            }
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
    const socket = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/${taskId}`);

    socket.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`);
      setConnectionStatus('connected');
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("WebSocket received data:", data);

        // 更新任务状态
        setTask(prevTask => ({
          ...prevTask,
          status: data.status || prevTask.status,
          progress: data.progress !== undefined ? data.progress : prevTask.progress,
          error_message: data.error || prevTask.error_message,
          output_url: data.output_url || prevTask.output_url,
          status_message: data.status_message || prevTask.status_message,
          current_step: data.current_step || prevTask.current_step,
          total_steps: data.total_steps || prevTask.total_steps,
          estimated_time: data.estimated_time || prevTask.estimated_time
        }));

        // 保持连接活跃
        socket.send("ping");
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    socket.onclose = (event) => {
      console.log(`WebSocket closed for task ${taskId}:`, event.code, event.reason);
      setConnectionStatus('disconnected');

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
      const interval = setInterval(() => {
        if (task && ['running', 'initializing', 'saving'].includes(task.status)) {
          fetchTaskDetails();
        }
      }, 1000); // 每秒刷新一次

      // 对于所有任务的常规更新
      const regularInterval = setInterval(fetchTaskDetails, 5000); // 每5秒刷新一次

      return () => {
        clearInterval(interval);
        clearInterval(regularInterval);
        if (socketRef.current) {
          socketRef.current.close();
        }
      };
    }
  }, [taskId]);

  // 重新加载视频
  const handleReloadVideo = () => {
    if (videoRef.current) {
      videoRef.current.load();
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
        <Text color="red.500">{error}</Text>
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
          <Text fontSize="sm">{taskTypeLabel}</Text>
        </HStack>

        {(task.status === 'running' || task.status === 'queued' || task.status === 'initializing' || task.status === 'saving') && (
          <Box>
            <HStack justify="space-between" mb={1}>
              <Text fontSize="sm">进度</Text>
              <Text fontSize="sm">{Math.round(task.progress * 100)}%</Text>
            </HStack>
            <Progress
              value={task.progress * 100}
              size="sm"
              colorScheme={task.status === 'initializing' ? "yellow" : task.status === 'saving' ? "green" : "blue"}
              borderRadius="md"
              hasStripe
              isAnimated={task.status !== 'queued'}
            />

            {/* 添加详细状态信息 */}
            {task.status_message && (
              <Text fontSize="sm" mt={1} color="gray.600">
                {task.status_message}
              </Text>
            )}

            {/* 添加步骤信息 */}
            {task.current_step !== undefined && task.total_steps !== undefined && (
              <Text fontSize="sm" mt={1} color="gray.600">
                步骤: {task.current_step}/{task.total_steps}
              </Text>
            )}

            {/* 添加预计剩余时间 */}
            {task.estimated_time !== undefined && task.estimated_time > 0 && (
              <Text fontSize="sm" mt={1} color="gray.600">
                预计剩余: {task.estimated_time > 60
                  ? `${Math.floor(task.estimated_time / 60)}分${task.estimated_time % 60}秒`
                  : `${task.estimated_time}秒`}
              </Text>
            )}
          </Box>
        )}

        {task.error_message && (
          <Box bg="red.50" p={3} borderRadius="md" borderLeft="4px" borderColor="red.500">
            <Text fontSize="sm" color="red.600">{task.error_message}</Text>
          </Box>
        )}

        <Box>
          <Text fontWeight="semibold" mb={1}>提示词</Text>
          <Text fontSize="sm" whiteSpace="pre-wrap" bg="gray.50" p={2} borderRadius="md">
            {task.prompt}
          </Text>
        </Box>

        {task.status === 'completed' && task.output_url && (
          <Box mt={2}>
            <HStack justify="space-between" mb={2}>
              <Text fontWeight="semibold">生成结果</Text>
              <HStack>
                <IconButton
                  aria-label="重新加载视频"
                  icon={<ViewIcon />}
                  size="xs"
                  onClick={handleReloadVideo}
                />
                <Badge colorScheme={connectionStatus === 'connected' ? 'green' : 'gray'} size="sm">
                  {connectionStatus === 'connected' ? '实时连接' : '检查连接'}
                </Badge>
              </HStack>
            </HStack>
            <Box borderWidth={1} borderRadius="md" overflow="hidden">
              <video
                ref={videoRef}
                controls
                autoPlay
                width="100%"
                height="auto"
                src={task.output_url}
                onError={(e) => {
                  console.error("视频加载失败:", e);
                  // 如果是COMPLETED状态但视频无法加载，可能是误报完成
                  if (task.progress < 0.99) {
                    console.log("视频未完全生成，重置状态");
                    setTask(prev => ({...prev, status: 'running'}));
                  }
                }}
              >
                您的浏览器不支持视频标签
              </video>
            </Box>
          </Box>
        )}

        <VStack align="stretch" spacing={2}>
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
        </VStack>

        {['queued', 'running'].includes(task.status) && (
          <Box mt={2}>
            <IconButton
              icon={<DeleteIcon />}
              colorScheme="red"
              variant="outline"
              width="full"
              onClick={() => onCancelTask(task.id)}
            >
              取消任务
            </IconButton>
          </Box>
        )}
      </VStack>
    </Box>
  );
};
export default TaskDetails;