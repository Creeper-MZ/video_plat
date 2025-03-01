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
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  CloseButton
} from '@chakra-ui/react';
import GenerationForm from './components/GenerationForm';
import TasksList from './components/TasksList';
import { TaskDetails } from './components/TaskDetails';
import ErrorBoundary from './components/ErrorBoundary';

const App = () => {
  const [activeTask, setActiveTask] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const toast = useToast();

  // 加载任务列表
  const fetchTasks = async () => {
    try {
      const response = await fetch('/api/tasks');
      if (response.ok) {
        const data = await response.json();
        setTasks(data);
      }
    } catch (error) {
      console.error('获取任务列表失败:', error);
    }
  };

  // 初始加载和定时刷新
  useEffect(() => {
    fetchTasks();

    // 定时刷新任务列表
    const interval = setInterval(() => {
      fetchTasks();
    }, 5000);

    return () => clearInterval(interval);
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
        fetchTasks();
        setActiveTask(result.id);
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

  return (
    <ChakraProvider>
      <ErrorBoundary>
        <Box bg="gray.50" minH="100vh">
          <Box bg="blue.600" color="white" p={4} shadow="md">
            <Container maxW="container.xl">
              <Heading size="lg">视频生成平台</Heading>
              <Text mt={1}>基于Wan2.1的文生视频与图生视频系统</Text>
            </Container>
          </Box>

          <Container maxW="container.xl" py={6}>
            {errorMessage && (
              <Alert status="error" mb={4}>
                <AlertIcon />
                <AlertTitle mr={2}>发生错误!</AlertTitle>
                <AlertDescription>{errorMessage}</AlertDescription>
                <CloseButton
                  position="absolute"
                  right="8px"
                  top="8px"
                  onClick={() => setErrorMessage(null)}
                />
              </Alert>
            )}

            <Flex direction={{ base: 'column', lg: 'row' }} gap={6}>
              <Box flex="1" bg="white" p={5} borderRadius="md" shadow="sm">
                <Tabs isFitted colorScheme="blue">
                  <TabList>
                    <Tab>创建任务</Tab>
                    <Tab>任务列表</Tab>
                  </TabList>

                  <TabPanels>
                    <TabPanel>
                      <GenerationForm
                        onSubmit={handleSubmit}
                        isLoading={isLoading}
                      />
                    </TabPanel>
                    <TabPanel>
                      <TasksList
                        tasks={tasks}
                        onSelectTask={(taskId) => {
                          try {
                            setActiveTask(taskId);
                          } catch (err) {
                            console.error("选择任务时出错:", err);
                            setErrorMessage("显示任务详情时发生错误，请刷新页面重试");
                          }
                        }}
                        onCancelTask={handleCancelTask}
                        activeTaskId={activeTask}
                      />
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </Box>

              {activeTask && (
                <Box
                  w={{ base: '100%', lg: '500px' }}
                  bg="white"
                  p={5}
                  borderRadius="md"
                  shadow="sm"
                >
                  <ErrorBoundary>
                    <TaskDetails
                      taskId={activeTask}
                      onCancelTask={handleCancelTask}
                      onError={(err) => {
                        console.error("任务详情组件错误:", err);
                        setErrorMessage("显示任务详情时出错: " + (err.message || "未知错误"));
                      }}
                    />
                  </ErrorBoundary>
                </Box>
              )}
            </Flex>
          </Container>
        </Box>
      </ErrorBoundary>
    </ChakraProvider>
  );
};

export default App;