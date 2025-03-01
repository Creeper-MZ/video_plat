import React, { useState, useEffect } from 'react';
import { 
  ChakraProvider,
  Box,
  Flex,
  Heading,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Container,
  Button,
  Badge,
  HStack,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  useColorModeValue,
  theme
} from '@chakra-ui/react';
import GenerationForm from './components/GenerationForm';
import TasksList from './components/TasksList';
import TaskDetails from './components/TaskDetails';

const App = () => {
  const [activeTask, setActiveTask] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingTasks, setIsLoadingTasks] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const [error, setError] = useState(null);
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const toast = useToast();

  // 加载任务列表
  const fetchTasks = async () => {
    try {
      setIsLoadingTasks(true);
      const response = await fetch('/api/tasks');
      if (response.ok) {
        const data = await response.json();
        setTasks(data);
        setError(null);
      } else {
        throw new Error('获取任务列表失败');
      }
    } catch (error) {
      console.error('获取任务列表失败:', error);
      setError('加载任务失败，请刷新页面重试');
    } finally {
      setIsLoadingTasks(false);
    }
  };

  // 加载系统状态
  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/system/status');
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
      }
    } catch (error) {
      console.error('获取系统状态失败:', error);
    }
  };

  // 初始加载和定时刷新
  useEffect(() => {
    // 初始加载
    fetchTasks();
    fetchSystemStatus();

    // 定时刷新任务列表
    const tasksInterval = setInterval(() => {
      fetchTasks();
    }, 5000);

    // 定时刷新系统状态
    const statusInterval = setInterval(() => {
      fetchSystemStatus();
    }, 10000);

    return () => {
      clearInterval(tasksInterval);
      clearInterval(statusInterval);
    };
  }, []);

  // 提交新任务
  const handleSubmit = async (formData) => {
    setIsLoading(true);

    try {
      const response = await fetch('/api/tasks', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        toast({
          title: '任务创建成功',
          description: `任务ID: ${result.id}`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });

        // 刷新任务列表并显示新任务详情
        await fetchTasks();
        setActiveTask(result.id);
        setActiveTabIndex(1); // 切换到任务列表标签页
      } else {
        const error = await response.json();
        throw new Error(error.detail || '创建任务失败');
      }
    } catch (error) {
      toast({
        title: '创建任务失败',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // 取消任务
  const handleCancelTask = async (taskId) => {
    try {
      const response = await fetch(`/api/tasks/${taskId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        toast({
          title: '任务已取消',
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
        fetchTasks();
      } else {
        throw new Error('取消任务失败');
      }
    } catch (error) {
      toast({
        title: '取消任务失败',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // 渲染系统状态信息
  const renderSystemStatus = () => {
    if (!systemStatus) return null;

    const { total_gpus, busy_gpus, queue_length } = systemStatus;

    return (
      <HStack spacing={4} my={2}>
        <Badge colorScheme="purple">{total_gpus}个GPU</Badge>
        <Badge colorScheme={busy_gpus === total_gpus ? "red" : "green"}>
          {busy_gpus}个工作中
        </Badge>
        <Badge colorScheme={queue_length > 0 ? "yellow" : "gray"}>
          队列长度: {queue_length}
        </Badge>
      </HStack>
    );
  };

  return (
    <ChakraProvider theme={theme}>
      <Box bg="gray.50" minH="100vh">
        <Box bg="blue.600" color="white" p={4} shadow="md">
          <Container maxW="container.xl">
            <Heading size="lg">视频生成平台</Heading>
            <Text mt={1}>基于Wan2.1的文生视频与图生视频系统</Text>
            {renderSystemStatus()}
          </Container>
        </Box>

        <Container maxW="container.xl" py={6}>
          {error && (
            <Alert status="error" mb={4}>
              <AlertIcon />
              <AlertTitle mr={2}>错误</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Flex direction={{ base: 'column', lg: 'row' }} gap={6}>
            <Box flex="1" bg="white" p={5} borderRadius="md" shadow="sm">
              <Tabs
                isFitted
                colorScheme="blue"
                index={activeTabIndex}
                onChange={(index) => setActiveTabIndex(index)}
              >
                <TabList>
                  <Tab>创建任务</Tab>
                  <Tab>任务列表 {tasks.length > 0 && `(${tasks.length})`}</Tab>
                </TabList>

                <TabPanels>
                  <TabPanel>
                    <GenerationForm
                      onSubmit={handleSubmit}
                      isLoading={isLoading}
                    />
                  </TabPanel>
                  <TabPanel>
                    {isLoadingTasks ? (
                      <Flex justify="center" align="center" height="200px">
                        <Spinner size="xl" />
                      </Flex>
                    ) : (
                      <TasksList
                        tasks={tasks}
                        onSelectTask={(taskId) => {
                          setActiveTask(taskId);
                          // 在移动设备上，任务详情会显示在下方，需要滚动到可见区域
                          if (window.innerWidth < 992) {
                            setTimeout(() => {
                              document.getElementById('task-details-section')?.scrollIntoView({
                                behavior: 'smooth'
                              });
                            }, 100);
                          }
                        }}
                        onCancelTask={handleCancelTask}
                        activeTaskId={activeTask}
                      />
                    )}
                  </TabPanel>
                </TabPanels>
              </Tabs>
            </Box>

            {activeTask && (
              <Box
                id="task-details-section"
                w={{ base: '100%', lg: '500px' }}
                bg="white"
                p={5}
                borderRadius="md"
                shadow="sm"
              >
                <TaskDetails
                  taskId={activeTask}
                  onCancelTask={handleCancelTask}
                />
              </Box>
            )}
          </Flex>

          <Box mt={10} textAlign="center" fontSize="sm" color="gray.500">
            <Text>视频生成平台 &copy; {new Date().getFullYear()}</Text>
            <Text mt={1}>基于Wan2.1模型 | 支持文生视频和图生视频</Text>
          </Box>
        </Container>
      </Box>
    </ChakraProvider>
  );
};

export default App;