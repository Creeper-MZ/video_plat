import React from 'react';
import { Box, Heading, Text, Button, Code, Alert, AlertIcon, AlertTitle, AlertDescription } from '@chakra-ui/react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // 更新 state 使下一次渲染能够显示降级 UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // 你同样可以将错误日志上报给服务器
    console.error("组件错误:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      // 你可以自定义降级 UI
      return (
        <Box p={5} maxW="800px" mx="auto" mt={10} borderWidth={1} borderRadius="lg" boxShadow="md">
          <Alert status="error" mb={5} borderRadius="md">
            <AlertIcon />
            <AlertTitle mr={2}>页面渲染错误</AlertTitle>
            <AlertDescription>应用遇到了一个错误，请尝试刷新页面</AlertDescription>
          </Alert>

          <Heading size="md" mb={3}>错误信息:</Heading>
          <Text mb={4} color="red.500">{this.state.error && this.state.error.toString()}</Text>

          <Heading size="md" mb={3}>组件堆栈:</Heading>
          <Box bg="gray.50" p={3} borderRadius="md" mb={5} overflowX="auto">
            <Code whiteSpace="pre-wrap" display="block">
              {this.state.errorInfo && this.state.errorInfo.componentStack}
            </Code>
          </Box>

          <Button
            colorScheme="blue"
            onClick={() => window.location.reload()}
            mr={3}
          >
            刷新页面
          </Button>

          <Button
            variant="outline"
            onClick={() => window.location.href = '/'}
          >
            返回首页
          </Button>
        </Box>
      );
    }

    return this.props.children;
  }
}
export { ErrorBoundary }
export default ErrorBoundary;