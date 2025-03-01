import React, { useState } from 'react';
import {
  Box,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Select,
  Button,
  Heading,
  RadioGroup,
  Radio,
  Stack,
  FormHelperText,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Checkbox,
  Collapse,
  InputGroup,
  InputRightAddon,
  useDisclosure,
  VStack,
  HStack,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Image,
  Text,
  useToast
} from '@chakra-ui/react';
import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';

const GenerationForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    type: 'text_to_video',
    prompt: '',
    negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
    resolution: '480p',
    frames: 100,
    fps: 20,
    steps: 40,
    seed: 0,
    model_precision: 'fp8',
    save_vram: false,
    tiled: true,
  });

  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const [formErrors, setFormErrors] = useState({});
  const toast = useToast();

  const { isOpen: isAdvancedOpen, onToggle: onAdvancedToggle } = useDisclosure();

  // 处理表单字段变化
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value,
    });

    // 清除该字段的错误
    if (formErrors[name]) {
      setFormErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  // 处理数值输入的变化
  const handleNumberChange = (name, value) => {
    setFormData({
      ...formData,
      [name]: value,
    });

    // 清除该字段的错误
    if (formErrors[name]) {
      setFormErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  // 处理图片文件选择
  const handleImageChange = (e) => {
    const file = e.target.files[0];

    if (file) {
      // 检查文件大小（限制为10MB）
      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "文件过大",
          description: "图片大小不能超过10MB",
          status: "error",
          duration: 5000,
          isClosable: true,
        });
        return;
      }

      // 检查文件类型
      const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
      if (!validTypes.includes(file.type)) {
        toast({
          title: "文件类型不支持",
          description: "请上传JPG、PNG、WEBP或GIF格式的图片",
          status: "error",
          duration: 5000,
          isClosable: true,
        });
        return;
      }

      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);

      // 清除图片相关的错误
      if (formErrors.image) {
        setFormErrors(prev => ({ ...prev, image: null }));
      }
    }
  };

  // 验证表单
  const validateForm = () => {
    const errors = {};

    // 验证提示词
    if (!formData.prompt.trim()) {
      errors.prompt = "提示词不能为空";
    } else if (formData.prompt.trim().length < 5) {
      errors.prompt = "提示词太短，请提供更详细的描述";
    }

    // 如果是图生视频，验证是否上传了图片
    if (formData.type === 'image_to_video' && !imageFile) {
      errors.image = "请上传参考图片";
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // 处理表单提交
  const handleSubmit = (e) => {
    e.preventDefault();

    // 表单验证
    if (!validateForm()) {
      toast({
        title: "表单验证失败",
        description: "请检查表单中的错误并修正",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    const submitData = new FormData();

    // 添加所有表单字段
    for (const [key, value] of Object.entries(formData)) {
      submitData.append(key, value);
    }

    // 添加视频类型
    submitData.append('video_type', formData.type);

    // 如果是图生视频，添加图片文件
    if (formData.type === 'image_to_video' && imageFile) {
      submitData.append('image', imageFile);
    }

    onSubmit(submitData);
  };

  return (
    <Box as="form" onSubmit={handleSubmit}>
      <Heading size="md" mb={4}>创建新的视频生成任务</Heading>

      <FormControl isRequired mb={4} isInvalid={formErrors.type}>
        <FormLabel>生成类型</FormLabel>
        <RadioGroup
          value={formData.type}
          onChange={(value) => handleChange({ target: { name: 'type', value } })}
        >
          <Stack direction="row">
            <Radio value="text_to_video">文本生成视频</Radio>
            <Radio value="image_to_video">图片生成视频</Radio>
          </Stack>
        </RadioGroup>
      </FormControl>

      {formData.type === 'image_to_video' && (
        <FormControl isRequired mb={4} isInvalid={formErrors.image}>
          <FormLabel>上传参考图片</FormLabel>
          <Input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            p={1}
          />
          <FormHelperText>选择一张图片作为视频的起始参考（最大10MB）</FormHelperText>

          {formErrors.image && (
            <Alert status="error" mt={2} size="sm">
              <AlertIcon />
              {formErrors.image}
            </Alert>
          )}

          {imagePreview && (
            <Box mt={2} borderWidth={1} borderRadius="md" overflow="hidden">
              <Image
                src={imagePreview}
                alt="Preview"
                maxH="200px"
                maxW="100%"
                objectFit="contain"
              />
            </Box>
          )}
        </FormControl>
      )}

      <FormControl isRequired mb={4} isInvalid={formErrors.prompt}>
        <FormLabel>提示词</FormLabel>
        <Textarea
          name="prompt"
          value={formData.prompt}
          onChange={handleChange}
          placeholder="详细描述你想生成的视频内容..."
          rows={4}
        />
        <FormHelperText>详细描述希望在视频中呈现的内容、风格和场景</FormHelperText>
        {formErrors.prompt && (
          <Alert status="error" mt={2} size="sm">
            <AlertIcon />
            {formErrors.prompt}
          </Alert>
        )}
      </FormControl>

      <FormControl mb={4}>
        <FormLabel>负面提示词</FormLabel>
        <Textarea
          name="negative_prompt"
          value={formData.negative_prompt}
          onChange={handleChange}
          placeholder="描述你不希望在视频中出现的元素..."
          rows={2}
        />
        <FormHelperText>描述需要避免的内容、风格或缺陷</FormHelperText>
      </FormControl>

      <FormControl isRequired mb={4}>
        <FormLabel>分辨率</FormLabel>
        <Select name="resolution" value={formData.resolution} onChange={handleChange}>
          <option value="480p">480P (854×480)</option>
          <option value="480p_vertical">480P 竖屏 (480×854)</option>
          <option value="720p">720P (1280×720)</option>
          <option value="720p_vertical">720P 竖屏 (720×1280)</option>
        </Select>
      </FormControl>

      <Button
        rightIcon={isAdvancedOpen ? <ChevronUpIcon /> : <ChevronDownIcon />}
        onClick={onAdvancedToggle}
        variant="outline"
        mb={4}
        size="sm"
      >
        高级选项
      </Button>

      <Collapse in={isAdvancedOpen} animateOpacity>
        <VStack spacing={4} p={4} bg="gray.50" borderRadius="md" align="stretch" mb={4}>
          <HStack spacing={6}>
            <FormControl>
              <FormLabel>视频帧数</FormLabel>
              <NumberInput
                min={8}
                max={128}
                value={formData.frames}
                onChange={(_, value) => handleNumberChange('frames', value)}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>视频总帧数 (8-128)</FormHelperText>
            </FormControl>

            <FormControl>
              <FormLabel>帧率 (FPS)</FormLabel>
              <NumberInput
                min={10}
                max={60}
                value={formData.fps}
                onChange={(_, value) => handleNumberChange('fps', value)}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>每秒帧数 (10-60)</FormHelperText>
            </FormControl>
          </HStack>

          <FormControl>
            <FormLabel>推理步数: {formData.steps}</FormLabel>
            <Slider
              min={20}
              max={60}
              step={1}
              value={formData.steps}
              onChange={(value) => handleNumberChange('steps', value)}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
            <FormHelperText>增加步数可提高质量但会延长生成时间</FormHelperText>
          </FormControl>

          <FormControl>
            <FormLabel>随机种子</FormLabel>
            <InputGroup>
              <NumberInput
                min={-1}
                max={2147483647}
                value={formData.seed}
                onChange={(_, value) => handleNumberChange('seed', value)}
                width="full"
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <InputRightAddon>
                <Button
                  size="xs"
                  onClick={() => handleNumberChange('seed', -1)}
                >
                  随机
                </Button>
              </InputRightAddon>
            </InputGroup>
            <FormHelperText>使用特定种子可以复现结果，使用-1表示随机种子</FormHelperText>
          </FormControl>

          <FormControl>
            <FormLabel>模型精度</FormLabel>
            <Select name="model_precision" value={formData.model_precision} onChange={handleChange}>
              <option value="fp8">FP8 (低精度，节省显存)</option>
              <option value="fp16">FP16 (高精度)</option>
            </Select>
            <FormHelperText>FP8可以节省显存但可能降低质量</FormHelperText>
          </FormControl>
          
          <FormControl>
            <Checkbox 
              name="save_vram"
              isChecked={formData.save_vram} 
              onChange={handleChange}
            >
              启用显存节省模式
            </Checkbox>
            <FormHelperText>适用于显存紧张的情况 (设置 num_persistent_param_in_dit=0)</FormHelperText>
          </FormControl>
          
          <FormControl>
            <Checkbox 
              name="tiled"
              isChecked={formData.tiled} 
              onChange={handleChange}
            >
              使用平铺模式
            </Checkbox>
            <FormHelperText>平铺模式可以帮助处理更大尺寸的图像</FormHelperText>
          </FormControl>
        </VStack>
      </Collapse>
      
      <Button 
        type="submit" 
        colorScheme="blue" 
        size="lg" 
        width="full" 
        mt={4}
        isLoading={isLoading}
        loadingText="提交中"
        isDisabled={(formData.type === 'image_to_video' && !imageFile)}
      >
        创建任务
      </Button>
    </Box>
  );
};

export default GenerationForm;