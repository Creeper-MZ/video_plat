import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Heading,
  VStack,
  HStack,
  Badge,
  Text,
  Progress
} from '@chakra-ui/react';

export const TaskDetails = ({ taskId, onCancelTask }) => {
  const [task, setTask] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    if (taskId) {
      fetchTaskDetails();
      initializeWebSocket();

      const interval = setInterval(fetchTaskDetails, 5000);

      return () => {
        clearInterval(interval);
        if (socketRef.current) {
          socketRef.current.close();
        }
      };
    }
  }, [taskId]);

  const fetchTaskDetails = async () => {
    try {
      const response = await fetch(`/api/tasks/${taskId}`);
      if (response.ok) {
        const data = await response.json();
        setTask(data);
      } else {
        throw new Error('获取任务详情失败');
      }
    } catch (error) {
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const initializeWebSocket = () => {
    if (socketRef.current) {
      socketRef.current.close();
    }

    const socket = new WebSocket(`ws://localhost:8000/ws/${taskId}`);

    socket.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTask(prevTask => ({
        ...prevTask,
        status: data.status || prevTask.status,
        progress: data.progress !== undefined ? data.progress : prevTask.progress,
        output_url: data.output_url || prevTask.output_url,
        step: data.step !== undefined ? data.step : prevTask.step,
        total_steps: data.total_steps !== undefined ? data.total_steps : prevTask.total_steps,
        message: data.message || prevTask.message
      }));
    };

    socket.onclose = () => {
      console.log(`WebSocket closed for task ${taskId}`);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    socketRef.current = socket;
  };

  if (isLoading) return <Text>加载中...</Text>;
  if (error) return <Text color="red.500">错误: {error}</Text>;
  if (!task) return <Text>任务未找到</Text>;

  const statusColor = {
    queued: 'yellow',
    running: 'blue',
    completed: 'green',
    failed: 'red',
    cancelled: 'gray'
  }[task.status] || 'gray';

  const statusLabel = {
    queued: '排队中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消'
  }[task.status] || task.status;

  const taskTypeLabel = task.type === 'text_to_video' ? '文生视频' : '图生视频';

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

        {(task.status === 'running' || task.status === 'queued') && (
          <Box>
            <HStack justify="space-between" mb={1}>
              <Text fontSize="sm">生成进度</Text>
              <Text fontSize="sm">{Math.round(task.progress * 100)}%</Text>
            </HStack>
            <Progress
              value={task.progress * 100}
              size="sm"
              colorScheme="blue"
              borderRadius="md"
              hasStripe={false}
              isAnimated={false}
            />

            {task.step !== undefined && task.total_steps !== undefined && (
              <HStack justify="space-between" fontSize="xs" color="gray.500" mt={1}>
                <Text>步骤: {task.step}/{task.total_steps}</Text>
                {task.message && <Text>{task.message}</Text>}
              </HStack>
            )}
          </Box>
        )}

        {task.status === 'completed' && task.output_url && (
          <Box>
            <Text fontSize="sm" mb={2}>生成结果:</Text>
            <video controls width="100%">
              <source src={task.output_url} type="video/mp4" />
              您的浏览器不支持视频播放。
            </video>
          </Box>
        )}

        {task.error_message && (
          <Text color="red.500" fontSize="sm">错误信息: {task.error_message}</Text>
        )}
      </VStack>
    </Box>
  );
};
export default TaskDetails;