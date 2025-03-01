// ChakraCompatibility.jsx
import React from 'react';
import {
  Field,
  FieldRoot,
  FieldLabel,
  FieldHelperText,
  FieldErrorText,
  FieldRequiredIndicator,
  RadioGroupRoot,
  RadioGroupItem,
  RadioGroupItemText,
  RadioGroupItemControl,
  RadioGroupItemIndicator,
  CheckboxRoot,
  CheckboxControl,
  CheckboxLabel,
  SliderRoot,
  SliderTrack,
  SliderThumb,
  NumberInputRoot,
  NumberInputControl,
  NumberInputInput,
  NumberInputIncrementTrigger,
  NumberInputDecrementTrigger,
  SelectRoot,
  SelectTrigger,
  SelectContent,
  SelectItem,
  TextareaPropsProvider
} from '@chakra-ui/react';

// FormControl 兼容层
export const FormControl = ({ children, isRequired, isInvalid, ...props }) => (
  <Field {...props}>
    <FieldRoot isRequired={isRequired} isInvalid={isInvalid}>
      {children}
    </FieldRoot>
  </Field>
);

// FormLabel 兼容层
export const FormLabel = ({ children, ...props }) => (
  <FieldLabel {...props}>
    {children}
    {props.isRequired && <FieldRequiredIndicator />}
  </FieldLabel>
);

// FormHelperText 兼容层
export const FormHelperText = (props) => <FieldHelperText {...props} />;

// FormErrorMessage 兼容层
export const FormErrorMessage = (props) => <FieldErrorText {...props} />;

// RadioGroup 兼容层
export const RadioGroup = ({ children, value, onChange, ...props }) => (
  <RadioGroupRoot value={value} onChange={onChange} {...props}>
    {children}
  </RadioGroupRoot>
);

// Radio 兼容层
export const Radio = ({ children, value, ...props }) => (
  <RadioGroupItem value={value} {...props}>
    <RadioGroupItemControl>
      <RadioGroupItemIndicator />
    </RadioGroupItemControl>
    <RadioGroupItemText>{children}</RadioGroupItemText>
  </RadioGroupItem>
);

// Checkbox 兼容层
export const CheckboxComponent = ({ children, isChecked, onChange, ...props }) => (
  <CheckboxRoot isChecked={isChecked} onChange={onChange} {...props}>
    <CheckboxControl>
      <CheckboxIndicator />
    </CheckboxControl>
    <CheckboxLabel>{children}</CheckboxLabel>
  </CheckboxRoot>
);

// Slider 兼容层
export const SliderComponent = ({ value, onChange, min, max, ...props }) => (
  <SliderRoot value={value} onChange={onChange} min={min} max={max} {...props}>
    <SliderTrack />
    <SliderThumb />
  </SliderRoot>
);

// NumberInput 兼容层
export const NumberInputComponent = ({ value, onChange, min, max, ...props }) => (
  <NumberInputRoot value={value} onChange={onChange} min={min} max={max} {...props}>
    <NumberInputControl>
      <NumberInputInput />
      <NumberInputIncrementTrigger />
      <NumberInputDecrementTrigger />
    </NumberInputControl>
  </NumberInputRoot>
);

// Select 兼容层
export const SelectComponent = ({ children, value, onChange, placeholder, ...props }) => (
  <SelectRoot value={value} onChange={onChange} {...props}>
    <SelectTrigger>
      <SelectValueText>{placeholder}</SelectValueText>
    </SelectTrigger>
    <SelectContent>
      {children}
    </SelectContent>
  </SelectRoot>
);

// Option 兼容层
export const Option = ({ children, value, ...props }) => (
  <SelectItem value={value} {...props}>{children}</SelectItem>
);

// Textarea 兼容层
export const TextareaComponent = (props) => (
  <Textarea {...props} />
);