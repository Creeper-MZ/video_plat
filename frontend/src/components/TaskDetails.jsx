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
  Flex
} from '@chakra-ui/react';
import { ViewIcon, DeleteIcon } from '@chakra-ui/icons';

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
  running: { color: 'blue', label: '生成中' },
  completed: { color: 'green', label: '已完成' },
  failed: { color: 'red', label: '失败' },
  cancelled: { color: 'gray', label: '已取消' }
};

export const TaskDetails = ({ taskId, onCancelTask }) => {
  const [task, setTask] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const videoRef = useRef(null);

  // 获取任务详情
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
    };
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // 更新任务状态
      setTask(prevTask => ({
        ...prevTask,
        status: data.status || prevTask.status,
        progress: data.progress !== undefined ? data.progress : prevTask.progress,
        error_message: data.error || prevTask.error_message,
        output_url: data.output_url || prevTask.output_url
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

  // 初始化和清理
  useEffect(() => {
    if (taskId) {
      fetchTaskDetails();
      initializeWebSocket();
      
      // 定时刷新任务状态
      const interval = setInterval(fetchTaskDetails, 3000);
      
      return () => {
        clearInterval(interval);
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
        
        {(task.status === 'running' || task.status === 'queued') && (
          <Box>
            <HStack justify="space-between" mb={1}>
              <Text fontSize="sm">进度</Text>
              <Text fontSize="sm">{Math.round(task.progress * 100)}%</Text>
            </HStack>
            <Progress
              value={task.progress * 100}
              size="sm"
              colorScheme="blue"
              borderRadius="md"
              hasStripe
              isAnimated={task.status === 'running'}
            />
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
              <IconButton
                aria-label="重新加载视频"
                icon={<ViewIcon />}
                size="xs"
                onClick={handleReloadVideo}
              />
            </HStack>
            <Box borderWidth={1} borderRadius="md" overflow="hidden">
              <video 
                ref={videoRef}
                controls
                autoPlay
                width="100%"
                height="auto"
                src={task.output_url}
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