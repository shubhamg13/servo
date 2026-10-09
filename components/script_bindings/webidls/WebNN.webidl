/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// Source: Web Neural Network API (https://www.w3.org/TR/webnn/#navigatorml)

// skip-unless CARGO_FEATURE_WEBNN

// https://www.w3.org/TR/webnn/#navigatorml
interface mixin NavigatorML {
  [SecureContext, SameObject, Pref="dom_webnn_enabled"] readonly attribute ML ml;
};
Navigator includes NavigatorML;
WorkerNavigator includes NavigatorML;

enum MLPowerPreference {
  "default",
  "high-performance",
  "low-power"
};

// https://www.w3.org/TR/webnn/#enumdef-mldevicetype
enum MLDeviceType {
  "cpu",
  "gpu",
  "npu"
};

dictionary MLContextOptions {
  MLDeviceType deviceType = "cpu";
  MLPowerPreference powerPreference = "default";
  boolean accelerated = true;
};

// https://www.w3.org/TR/webnn/#ml
[Exposed=(Window, Worker), SecureContext, Pref="dom_webnn_enabled"]
interface ML {
  Promise<MLContext> createContext(optional MLContextOptions options = {});
};

typedef record<USVString, MLTensor> MLNamedTensors;

dictionary MLContextLostInfo {
  DOMString message;
};

// https://www.w3.org/TR/webnn/#api-mlcontext
[Exposed=(Window, Worker), SecureContext, Pref="dom_webnn_enabled"]
interface MLContext {
  [Throws] undefined dispatch(MLGraph graph, MLNamedTensors inputs, MLNamedTensors outputs);
  Promise<MLTensor> createTensor(MLTensorDescriptor descriptor);
  Promise<MLTensor> createConstantTensor(MLOperandDescriptor descriptor, [AllowShared] BufferSource inputData);
  Promise<ArrayBuffer> readTensor(MLTensor tensor);
  Promise<undefined> readTensor(MLTensor tensor, [AllowShared] BufferSource outputData);
  [Throws] undefined writeTensor(MLTensor tensor, [AllowShared] BufferSource inputData);
  MLOpSupportLimits opSupportLimits();
  undefined destroy();
  readonly attribute boolean accelerated;
  readonly attribute Promise<MLContextLostInfo> lost;
};

dictionary MLRankRange {
  unsigned long min;
  unsigned long max;
};

typedef sequence<MLOperandDataType> MLDataTypeList;

dictionary MLTensorLimits {
  MLDataTypeList dataTypes;
  MLRankRange rankRange;
};

dictionary MLSingleInputSupportLimits {
  MLTensorLimits input;
  MLTensorLimits output;
};

dictionary MLOpSupportLimits {
  MLInputOperandLayout preferredInputLayout;
  [EnforceRange] unsigned long long maxTensorByteLength;
  MLTensorLimits input;
  MLTensorLimits constant;
  MLTensorLimits output;
  MLSingleInputSupportLimits add;
};

// https://www.w3.org/TR/webnn/#mlgraph
[Exposed=(Window, Worker), SecureContext, Pref="dom_webnn_enabled"]
interface MLGraph {
  undefined destroy();
};

enum MLInputOperandLayout {
  "nchw",
  "nhwc"
};

enum MLOperandDataType {
  "float32",
  "float16",
  "int32",
  "uint32",
  "int64",
  "uint64",
  "int8",
  "uint8"
};

dictionary MLOperandDescriptor {
  required MLOperandDataType dataType;
  required sequence<[EnforceRange] unsigned long> shape;
};

// https://www.w3.org/TR/webnn/#mloperand
[Exposed=(Window, Worker), SecureContext, Pref="dom_webnn_enabled"]
interface MLOperand {
  readonly attribute MLOperandDataType dataType;
  readonly attribute any shape;
};

dictionary MLOperatorOptions {
  USVString label = "";
};

dictionary MLTensorDescriptor : MLOperandDescriptor {
  boolean readable = false;
  boolean writable = false;
};

// https://www.w3.org/TR/webnn/#mltensor
[Exposed=(Window, Worker), SecureContext, Pref="dom_webnn_enabled"]
interface MLTensor {
  readonly attribute MLOperandDataType dataType;
  readonly attribute any shape;
  readonly attribute boolean readable;
  readonly attribute boolean writable;
  readonly attribute boolean constant;
  undefined destroy();
};

typedef record<USVString, MLOperand> MLNamedOperands;

// https://www.w3.org/TR/webnn/#mlgraphbuilder
[Exposed=(Window, Worker), SecureContext, Pref="dom_webnn_enabled"]
interface MLGraphBuilder {
  [Throws] constructor(MLContext context);
  [Throws] MLOperand input(USVString name, MLOperandDescriptor descriptor);
  [Throws] MLOperand constant(MLOperandDescriptor descriptor, [AllowShared] BufferSource buffer);
  [Throws] Promise<MLGraph> build(MLNamedOperands outputs);
};

partial interface MLGraphBuilder {
  [Throws] MLOperand add(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
};

enum MLConv2dFilterOperandLayout {
  "oihw",
  "hwio",
  "ohwi",
  "ihwo"
};

enum MLConvTranspose2dFilterOperandLayout {
  "iohw",
  "hwoi",
  "ohwi"
};

enum MLRoundingType {
  "floor",
  "ceil"
};

enum MLInterpolationMode {
  "nearest-neighbor",
  "linear"
};

enum MLPaddingMode {
  "constant",
  "edge",
  "reflection",
  "symmetric"
};

dictionary MLConv2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations;
  [EnforceRange] unsigned long groups = 1;
  MLInputOperandLayout inputLayout = "nchw";
  MLConv2dFilterOperandLayout filterLayout = "oihw";
  MLOperand bias;
};

dictionary MLPool2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> windowDimensions;
  sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations;
  MLInputOperandLayout layout = "nchw";
  MLRoundingType outputShapeRounding = "floor";
  sequence<[EnforceRange] unsigned long> outputSizes;
};

dictionary MLResample2dOptions : MLOperatorOptions {
  MLInterpolationMode mode = "nearest-neighbor";
  sequence<float> scales;
  sequence<[EnforceRange] unsigned long> sizes;
  sequence<[EnforceRange] unsigned long> axes;
};

dictionary MLTransposeOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> permutation;
};

dictionary MLSliceOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> strides;
};

dictionary MLSplitOptions : MLOperatorOptions {
  [EnforceRange] unsigned long axis = 0;
};

dictionary MLReduceOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> axes;
  boolean keepDimensions = false;
};

dictionary MLPadOptions : MLOperatorOptions {
  MLPaddingMode mode = "constant";
  float value = 0;
};

dictionary MLLeakyReluOptions : MLOperatorOptions {
  float alpha = 0.01;
};

dictionary MLGemmOptions : MLOperatorOptions {
  MLOperand c;
  float alpha = 1.0;
  float beta = 1.0;
  boolean aTranspose = false;
  boolean bTranspose = false;
};

dictionary MLConvTranspose2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations;
  sequence<[EnforceRange] unsigned long> outputPadding;
  sequence<[EnforceRange] unsigned long> outputSizes;
  [EnforceRange] unsigned long groups = 1;
  MLInputOperandLayout inputLayout = "nchw";
  MLConvTranspose2dFilterOperandLayout filterLayout = "iohw";
  MLOperand bias;
};

partial interface MLGraphBuilder {
  [Throws] MLOperand sub(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand mul(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand div(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand prelu(MLOperand input, MLOperand slope, optional MLOperatorOptions options = {});
  [Throws] MLOperand relu(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sigmoid(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand cast(MLOperand input, MLOperandDataType outputDataType, optional MLOperatorOptions options = {});
  [Throws] MLOperand concat(sequence<MLOperand> inputs, [EnforceRange] unsigned long axis, optional MLOperatorOptions options = {});
  [Throws] MLOperand conv2d(MLOperand input, MLOperand filter, optional MLConv2dOptions options = {});
  [Throws] MLOperand maxPool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand averagePool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand pad(MLOperand input, sequence<[EnforceRange] unsigned long> beginningPadding, sequence<[EnforceRange] unsigned long> endingPadding, optional MLPadOptions options = {});
  [Throws] MLOperand reshape(MLOperand input, sequence<[EnforceRange] long> newShape, optional MLOperatorOptions options = {});
  [Throws] MLOperand resample2d(MLOperand input, optional MLResample2dOptions options = {});
  [Throws] MLOperand slice(MLOperand input, sequence<[EnforceRange] unsigned long> starts, sequence<[EnforceRange] unsigned long> sizes, optional MLSliceOptions options = {});
  [Throws] MLOperand softmax(MLOperand input, [EnforceRange] unsigned long axis, optional MLOperatorOptions options = {});
  [Throws] sequence<MLOperand> split(MLOperand input, ([EnforceRange] unsigned long or sequence<[EnforceRange] unsigned long>) splits, optional MLSplitOptions options = {});
  [Throws] MLOperand transpose(MLOperand input, optional MLTransposeOptions options = {});
  [Throws] MLOperand reduceSum(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMean(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand sqrt(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand reciprocal(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand gelu(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand leakyRelu(MLOperand input, optional MLLeakyReluOptions options = {});
  [Throws] MLOperand min(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand max(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand matmul(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand gemm(MLOperand a, MLOperand b, optional MLGemmOptions options = {});
  [Throws] MLOperand convTranspose2d(MLOperand input, MLOperand filter, optional MLConvTranspose2dOptions options = {});
};
