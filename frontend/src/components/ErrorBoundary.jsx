import React, { Component } from 'react';
import {
  Box,
  Heading,
  Text,
  Button,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Container,
  VStack,
  Code,
  Divider
} from '@chakra-ui/react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    // 更新状态，下次渲染时显示回退UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // 记录错误信息
    console.error('React错误边界捕获到错误:', error, errorInfo);
    this.setState({ errorInfo });

    // 记录到服务器或其他日志系统（可选）
    // logErrorToService(error, errorInfo);
  }

  handleReload = () => {
    // 重新加载页面
    window.location.reload();
  }

  handleGoHome = () => {
    // 回到首页
    window.location.href = '/';
  }

  render() {
    if (this.state.hasError) {
      // 渲染自定义回退UI
      return (
        <Container maxW="container.md" py={10}>
          <Alert
            status="error"
            variant="subtle"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            textAlign="center"
            borderRadius="md"
            py={6}
            mb={6}
          >
            <AlertIcon boxSize="40px" mr={0} />
            <AlertTitle mt={4} mb={1} fontSize="lg">
              应用程序发生错误
            </AlertTitle>
            <AlertDescription maxWidth="sm">
              我们很抱歉，应用程序出现了意外问题。请尝试刷新页面或返回首页。
            </AlertDescription>
          </Alert>

          <VStack spacing={4} align="stretch">
            <Box>
              <Heading as="h4" size="md" mb={2}>错误详情</Heading>
              <Text color="red.500" fontWeight="medium">
                {this.state.error && this.state.error.toString()}
              </Text>
            </Box>

            {this.state.errorInfo && (
              <Box>
                <Heading as="h4" size="md" mb={2}>组件堆栈</Heading>
                <Box
                  bg="gray.50"
                  p={3}
                  borderRadius="md"
                  overflowX="auto"
                  maxH="200px"
                  overflowY="auto"
                >
                  <Code colorScheme="red" whiteSpace="pre-wrap">
                    {this.state.errorInfo.componentStack}
                  </Code>
                </Box>
              </Box>
            )}

            <Divider my={2} />

            <Box>
              <Heading as="h4" size="md" mb={4}>您可以尝试:</Heading>
              <Button
                colorScheme="blue"
                mr={4}
                onClick={this.handleReload}
              >
                刷新页面
              </Button>
              <Button
                variant="outline"
                onClick={this.handleGoHome}
              >
                返回首页
              </Button>
            </Box>
          </VStack>
        </Container>
      );
    }

    // 没有错误时正常渲染子组件
    return this.props.children;
  }
}
export {ErrorBoundary}
export default ErrorBoundary;