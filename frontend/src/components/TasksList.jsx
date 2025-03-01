import React from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  IconButton,
  Text,
  HStack,
  Heading,
  TableContainer,
  Button,
  Flex,
  Spinner,
  useToast
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
  initializing: { color: 'orange', label: '初始化中' },
  running: { color: 'blue', label: '生成中' },
  saving: { color: 'teal', label: '保存中' },
  completed: { color: 'green', label: '已完成' },
  failed: { color: 'red', label: '失败' },
  cancelled: { color: 'gray', label: '已取消' }
};

const TasksList = ({ tasks, onSelectTask, onCancelTask, activeTaskId }) => {
  const toast = useToast();

  // 处理任务类型显示
  const getTaskTypeLabel = (type) => {
    return type === 'text_to_video' ? '文生视频' : '图生视频';
  };

  // 处理取消任务
  const handleCancelTask = (taskId) => {
    onCancelTask(taskId);
    toast({
      title: "正在取消任务",
      description: "请稍等，任务正在取消中...",
      status: "info",
      duration: 3000,
      isClosable: true,
    });
  };

  // 如果没有任务，显示空状态
  if (!tasks || tasks.length === 0) {
    return (
      <Box>
        <Heading size="md" mb={4}>任务列表</Heading>
        <Flex
          direction="column"
          align="center"
          justify="center"
          py={10}
          bg="gray.50"
          borderRadius="md"
        >
          <Text color="gray.500" mb={4}>暂无任务记录</Text>
          <Button colorScheme="blue" onClick={() => window.location.reload()}>
            刷新列表
          </Button>
        </Flex>
      </Box>
    );
  }

  return (
    <Box>
      <Heading size="md" mb={4}>任务列表</Heading>

      <TableContainer>
        <Table variant="simple" size="sm">
          <Thead>
            <Tr>
              <Th>ID</Th>
              <Th>类型</Th>
              <Th>状态</Th>
              <Th>进度</Th>
              <Th>创建时间</Th>
              <Th>操作</Th>
            </Tr>
          </Thead>
          <Tbody>
            {tasks.map((task) => (
              <Tr
                key={task.id}
                bg={task.id === activeTaskId ? "blue.50" : undefined}
                _hover={{ bg: "gray.50" }}
                cursor="pointer"
                onClick={() => onSelectTask(task.id)}
              >
                <Td fontSize="xs" fontFamily="mono">{task.id.substring(0, 8)}...</Td>
                <Td>{getTaskTypeLabel(task.type)}</Td>
                <Td>
                  <Badge colorScheme={statusConfig[task.status]?.color || 'gray'}>
                    {statusConfig[task.status]?.label || task.status}
                  </Badge>
                </Td>
                <Td>
                  {task.status === 'completed' ? (
                    '100%'
                  ) : (
                    `${Math.round(task.progress * 100)}%`
                  )}
                </Td>
                <Td fontSize="xs">{formatDate(task.created_at)}</Td>
                <Td>
                  <HStack spacing={2} onClick={(e) => e.stopPropagation()}>
                    <IconButton
                      aria-label="查看详情"
                      icon={<ViewIcon />}
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectTask(task.id);
                      }}
                    />
                    {['queued', 'running', 'initializing', 'saving'].includes(task.status) && (
                      <IconButton
                        aria-label="取消任务"
                        icon={<DeleteIcon />}
                        size="sm"
                        colorScheme="red"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCancelTask(task.id);
                        }}
                      />
                    )}
                  </HStack>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default TasksList;