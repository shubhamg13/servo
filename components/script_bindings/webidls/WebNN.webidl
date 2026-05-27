/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// skip-unless CARGO_FEATURE_WEBNN
// Source: Web Neural Network API (https://webmachinelearning.github.io/webnn/)

interface mixin NavigatorML {
  [SecureContext, SameObject] readonly attribute ML ml;
};
Navigator includes NavigatorML;
WorkerNavigator includes NavigatorML;

enum MLPowerPreference { "default", "high-performance", "low-power" };
dictionary MLContextOptions { MLPowerPreference powerPreference = "default"; boolean accelerated = true; };

[SecureContext, Exposed=(Window, Worker)]
interface ML { Promise<MLContext> createContext(optional MLContextOptions options = {}); };

typedef record<USVString, MLTensor> MLNamedTensors;
dictionary MLContextLostInfo { DOMString message; };

[SecureContext, Exposed=(Window, Worker)]
interface MLContext {
  undefined dispatch(MLGraph graph, MLNamedTensors inputs, MLNamedTensors outputs);
  Promise<MLTensor> createTensor(MLTensorDescriptor descriptor);
  Promise<MLTensor> createConstantTensor(MLOperandDescriptor descriptor, BufferSource inputData);
  Promise<ArrayBuffer> readTensor(MLTensor tensor);
  Promise<undefined> readTensor(MLTensor tensor, BufferSource outputData);
  undefined writeTensor(MLTensor tensor, BufferSource inputData);
  MLOpSupportLimits opSupportLimits();
  undefined destroy();
  readonly attribute boolean accelerated;
  readonly attribute Promise<MLContextLostInfo> lost;
};

dictionary MLRankRange { unsigned long min; unsigned long max; };
typedef sequence<MLOperandDataType> MLDataTypeList;
dictionary MLTensorLimits { MLDataTypeList dataTypes; MLRankRange rankRange; };
dictionary MLSingleInputSupportLimits { MLTensorLimits input; MLTensorLimits output; };
dictionary MLBinarySupportLimits { MLTensorLimits a; MLTensorLimits b; MLTensorLimits output; };

dictionary MLOpSupportLimits {
  MLInputOperandLayout preferredInputLayout;
  [EnforceRange] unsigned long long maxTensorByteLength;
  MLTensorLimits input; MLTensorLimits constant; MLTensorLimits output;
  MLSingleInputSupportLimits abs; MLSingleInputSupportLimits add; MLSingleInputSupportLimits argMax;
  MLSingleInputSupportLimits argMin; MLSingleInputSupportLimits averagePool2d; MLSingleInputSupportLimits batchNormalization;
  MLSingleInputSupportLimits cast; MLSingleInputSupportLimits ceil; MLSingleInputSupportLimits clamp;
  MLSingleInputSupportLimits concat; MLSingleInputSupportLimits conv2d; MLSingleInputSupportLimits convTranspose2d;
  MLSingleInputSupportLimits cos; MLSingleInputSupportLimits cumulativeSum; MLSingleInputSupportLimits dequantizeLinear;
  MLSingleInputSupportLimits div; MLSingleInputSupportLimits elu; MLSingleInputSupportLimits equal;
  MLSingleInputSupportLimits erf; MLSingleInputSupportLimits exp; MLSingleInputSupportLimits expand;
  MLSingleInputSupportLimits floor; MLSingleInputSupportLimits gather; MLSingleInputSupportLimits gatherElements;
  MLSingleInputSupportLimits gatherND; MLSingleInputSupportLimits gelu; MLSingleInputSupportLimits gemm;
  MLSingleInputSupportLimits greater; MLSingleInputSupportLimits greaterOrEqual; MLSingleInputSupportLimits hardSigmoid;
  MLSingleInputSupportLimits hardSwish; MLSingleInputSupportLimits identity; MLSingleInputSupportLimits instanceNormalization;
  MLSingleInputSupportLimits l2Pool2d; MLSingleInputSupportLimits layerNormalization; MLSingleInputSupportLimits leakyRelu;
  MLSingleInputSupportLimits lesser; MLSingleInputSupportLimits lesserOrEqual; MLSingleInputSupportLimits linear;
  MLSingleInputSupportLimits log; MLSingleInputSupportLimits logicalAnd; MLSingleInputSupportLimits logicalNot;
  MLSingleInputSupportLimits logicalOr; MLSingleInputSupportLimits logicalXor; MLSingleInputSupportLimits matmul;
  MLSingleInputSupportLimits max; MLSingleInputSupportLimits maxPool2d; MLSingleInputSupportLimits min;
  MLSingleInputSupportLimits mul; MLSingleInputSupportLimits neg; MLSingleInputSupportLimits notEqual;
  MLSingleInputSupportLimits pad; MLSingleInputSupportLimits pow; MLSingleInputSupportLimits prelu;
  MLSingleInputSupportLimits quantizeLinear; MLSingleInputSupportLimits reciprocal; MLSingleInputSupportLimits reduceL1;
  MLSingleInputSupportLimits reduceL2; MLSingleInputSupportLimits reduceLogSum; MLSingleInputSupportLimits reduceLogSumExp;
  MLSingleInputSupportLimits reduceMax; MLSingleInputSupportLimits reduceMean; MLSingleInputSupportLimits reduceMin;
  MLSingleInputSupportLimits reduceProduct; MLSingleInputSupportLimits reduceSum; MLSingleInputSupportLimits reduceSumSquare;
  MLSingleInputSupportLimits relu; MLSingleInputSupportLimits resample2d; MLSingleInputSupportLimits reshape;
  MLSingleInputSupportLimits reverse; MLSingleInputSupportLimits scatterElements; MLSingleInputSupportLimits scatterND;
  MLSingleInputSupportLimits sigmoid; MLSingleInputSupportLimits sin; MLSingleInputSupportLimits slice;
  MLSingleInputSupportLimits softmax; MLSingleInputSupportLimits softplus; MLSingleInputSupportLimits softsign;
  MLSingleInputSupportLimits sqrt; MLSingleInputSupportLimits sub; MLSingleInputSupportLimits tan;
  MLSingleInputSupportLimits tanh; MLSingleInputSupportLimits tile; MLSingleInputSupportLimits transpose;
  MLSingleInputSupportLimits triangular;
};

[SecureContext, Exposed=(Window, Worker)]
interface MLGraph { undefined destroy(); };

enum MLInputOperandLayout { "nchw", "nhwc" };
enum MLOperandDataType { "float32", "float16", "int32", "uint32", "int64", "uint64", "int8", "uint8" };

dictionary MLOperandDescriptor { required MLOperandDataType dataType; required sequence<[EnforceRange] unsigned long> shape; };

[SecureContext, Exposed=(Window, Worker)]
interface MLOperand { readonly attribute MLOperandDataType dataType; readonly attribute any shape; };

dictionary MLOperatorOptions { USVString label = ""; };
dictionary MLTensorDescriptor : MLOperandDescriptor { boolean readable = false; boolean writable = false; };

[SecureContext, Exposed=(Window, Worker)]
interface MLTensor {
  readonly attribute MLOperandDataType dataType; readonly attribute any shape;
  readonly attribute boolean readable; readonly attribute boolean writable; readonly attribute boolean constant;
  undefined destroy();
};

typedef record<USVString, MLOperand> MLNamedOperands;

[SecureContext, Exposed=(Window, Worker)]
interface MLGraphBuilder {
  constructor(MLContext context);
  [Throws] MLOperand input(USVString name, MLOperandDescriptor descriptor);
  [Throws] MLOperand constant(MLOperandDescriptor descriptor, BufferSource buffer);
  [Throws] Promise<MLGraph> build(MLNamedOperands outputs);
};

// Binary
partial interface MLGraphBuilder {
  [Throws] MLOperand add(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand sub(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand mul(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand div(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand max(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand min(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand pow(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
};

// Logical
partial interface MLGraphBuilder {
  [Throws] MLOperand equal(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand notEqual(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand greater(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand greaterOrEqual(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand lesser(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand lesserOrEqual(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalNot(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalAnd(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalOr(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalXor(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
};

// Unary
partial interface MLGraphBuilder {
  [Throws] MLOperand abs(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand ceil(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand cos(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand erf(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand exp(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand floor(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand identity(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand log(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand neg(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand reciprocal(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sin(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sqrt(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand tan(MLOperand input, optional MLOperatorOptions options = {});
};

// Activation
partial interface MLGraphBuilder {
  [Throws] MLOperand relu(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sigmoid(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand tanh(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand softmax(MLOperand input, [EnforceRange] unsigned long axis, optional MLOperatorOptions options = {});
  [Throws] MLOperand gelu(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand hardSigmoid(MLOperand input, optional MLHardSigmoidOptions options = {});
  [Throws] MLOperand hardSwish(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand elu(MLOperand input, optional MLEluOptions options = {});
  [Throws] MLOperand leakyRelu(MLOperand input, optional MLLeakyReluOptions options = {});
  [Throws] MLOperand linear(MLOperand input, optional MLLinearOptions options = {});
  [Throws] MLOperand softplus(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand softsign(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand clamp(MLOperand input, optional MLClampOptions options = {});
};

// Layout
partial interface MLGraphBuilder {
  [Throws] MLOperand reshape(MLOperand input, sequence<[EnforceRange] unsigned long> newShape, optional MLOperatorOptions options = {});
  [Throws] MLOperand transpose(MLOperand input, optional MLTransposeOptions options = {});
  [Throws] MLOperand concat(sequence<MLOperand> inputs, [EnforceRange] unsigned long axis, optional MLOperatorOptions options = {});
  [Throws] MLOperand slice(MLOperand input, sequence<[EnforceRange] unsigned long> starts, sequence<[EnforceRange] unsigned long> sizes, optional MLSliceOptions options = {});
  [Throws] sequence<MLOperand> split(MLOperand input, ([EnforceRange] unsigned long or sequence<[EnforceRange] unsigned long>) splits, optional MLSplitOptions options = {});
  [Throws] MLOperand pad(MLOperand input, sequence<[EnforceRange] unsigned long> beginningPadding, sequence<[EnforceRange] unsigned long> endingPadding, optional MLPadOptions options = {});
  [Throws] MLOperand tile(MLOperand input, sequence<unsigned long> repetitions, optional MLOperatorOptions options = {});
  [Throws] MLOperand reverse(MLOperand input, optional MLReverseOptions options = {});
  [Throws] MLOperand expand(MLOperand input, sequence<[EnforceRange] unsigned long> newShape, optional MLOperatorOptions options = {});
  [Throws] MLOperand gather(MLOperand input, MLOperand indices, optional MLGatherOptions options = {});
  [Throws] MLOperand gatherElements(MLOperand input, MLOperand indices, optional MLGatherOptions options = {});
  [Throws] MLOperand gatherND(MLOperand input, MLOperand indices, optional MLOperatorOptions options = {});
  [Throws] MLOperand scatterElements(MLOperand input, MLOperand indices, MLOperand updates, optional MLScatterOptions options = {});
  [Throws] MLOperand scatterND(MLOperand input, MLOperand indices, MLOperand updates, optional MLOperatorOptions options = {});
};

// Matrix
partial interface MLGraphBuilder {
  [Throws] MLOperand matmul(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand gemm(MLOperand a, MLOperand b, optional MLGemmOptions options = {});
};

// Pooling
partial interface MLGraphBuilder {
  [Throws] MLOperand averagePool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand maxPool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand l2Pool2d(MLOperand input, optional MLPool2dOptions options = {});
};

// Conv
partial interface MLGraphBuilder {
  [Throws] MLOperand conv2d(MLOperand input, MLOperand filter, optional MLConv2dOptions options = {});
  [Throws] MLOperand convTranspose2d(MLOperand input, MLOperand filter, optional MLConvTranspose2dOptions options = {});
};

// Reduction
partial interface MLGraphBuilder {
  [Throws] MLOperand reduceL1(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceL2(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceLogSum(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceLogSumExp(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMax(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMean(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMin(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceProduct(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceSum(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceSumSquare(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand argMin(MLOperand input, [EnforceRange] unsigned long axis, optional MLArgMinMaxOptions options = {});
  [Throws] MLOperand argMax(MLOperand input, [EnforceRange] unsigned long axis, optional MLArgMinMaxOptions options = {});
};

// Normalization
partial interface MLGraphBuilder {
  [Throws] MLOperand batchNormalization(MLOperand input, MLOperand mean, MLOperand variance, optional MLBatchNormalizationOptions options = {});
  [Throws] MLOperand layerNormalization(MLOperand input, optional MLLayerNormalizationOptions options = {});
  [Throws] MLOperand instanceNormalization(MLOperand input, optional MLInstanceNormalizationOptions options = {});
};

// Quantization
partial interface MLGraphBuilder {
  [Throws] MLOperand cast(MLOperand input, MLOperandDataType dataType, optional MLOperatorOptions options = {});
  [Throws] MLOperand dequantizeLinear(MLOperand input, MLOperand scale, MLOperand zeroPoint, optional MLOperatorOptions options = {});
  [Throws] MLOperand quantizeLinear(MLOperand input, MLOperand scale, MLOperand zeroPoint, optional MLOperatorOptions options = {});
};

// Misc
partial interface MLGraphBuilder {
  [Throws] MLOperand prelu(MLOperand input, MLOperand slope, optional MLOperatorOptions options = {});
  [Throws] MLOperand where(MLOperand condition, MLOperand trueValue, MLOperand falseValue, optional MLOperatorOptions options = {});
  [Throws] MLOperand triangular(MLOperand input, optional MLTriangularOptions options = {});
  [Throws] MLOperand cumulativeSum(MLOperand input, unsigned long axis, optional MLCumulativeSumOptions options = {});
  [Throws] MLOperand resample2d(MLOperand input, optional MLResample2dOptions options = {});
};

// Dictionaries
dictionary MLConv2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> padding; sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations; [EnforceRange] unsigned long groups = 1;
  MLInputOperandLayout inputLayout = "nchw"; MLConv2dFilterOperandLayout filterLayout = "oihw"; MLOperand bias;
};

enum MLConv2dFilterOperandLayout { "oihw", "hwio", "ohwi", "ihwo" };
enum MLRoundingType { "floor", "ceil" };

dictionary MLConvTranspose2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> padding; sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations; sequence<[EnforceRange] unsigned long> outputPadding;
  sequence<[EnforceRange] unsigned long> outputSizes; [EnforceRange] unsigned long groups = 1;
  MLInputOperandLayout inputLayout = "nchw"; MLConvTranspose2dFilterOperandLayout filterLayout = "iohw"; MLOperand bias;
};
enum MLConvTranspose2dFilterOperandLayout { "iohw", "hwoi", "ohwi" };

dictionary MLPool2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> windowDimensions; sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides; sequence<[EnforceRange] unsigned long> dilations;
  MLInputOperandLayout layout = "nchw"; MLRoundingType outputShapeRounding = "floor";
  sequence<[EnforceRange] unsigned long> outputSizes;
};

dictionary MLTransposeOptions : MLOperatorOptions { sequence<[EnforceRange] unsigned long> permutation; };
dictionary MLGemmOptions : MLOperatorOptions { MLOperand c; double alpha = 1.0; double beta = 1.0; boolean aTranspose = false; boolean bTranspose = false; };
dictionary MLArgMinMaxOptions : MLOperatorOptions { boolean keepDimensions = false; MLOperandDataType outputDataType = "int32"; };
dictionary MLSliceOptions : MLOperatorOptions { sequence<[EnforceRange] unsigned long> strides; };
dictionary MLSplitOptions : MLOperatorOptions { [EnforceRange] unsigned long axis = 0; };
dictionary MLPadOptions : MLOperatorOptions { MLPaddingMode mode = "constant"; double value = 0; };
enum MLPaddingMode { "constant", "edge", "reflection" };
dictionary MLReverseOptions : MLOperatorOptions { sequence<[EnforceRange] unsigned long> axes; };
dictionary MLGatherOptions : MLOperatorOptions { [EnforceRange] unsigned long axis = 0; };
dictionary MLScatterOptions : MLOperatorOptions { [EnforceRange] unsigned long axis = 0; };
dictionary MLReduceOptions : MLOperatorOptions { sequence<[EnforceRange] unsigned long> axes; boolean keepDimensions = false; };
dictionary MLCumulativeSumOptions : MLOperatorOptions { boolean exclusive = false; boolean reversed = false; };
dictionary MLTriangularOptions : MLOperatorOptions { boolean upper = true; [EnforceRange] long diagonal = 0; };
dictionary MLResample2dOptions : MLOperatorOptions { MLInterpolationMode mode = "nearest-neighbor"; sequence<float> scales; sequence<[EnforceRange] unsigned long> sizes; sequence<[EnforceRange] unsigned long> axes; };
enum MLInterpolationMode { "nearest-neighbor", "linear" };

dictionary MLBatchNormalizationOptions : MLOperatorOptions { MLOperand scale; MLOperand bias; [EnforceRange] unsigned long axis = 1; double epsilon = 1e-5; };
dictionary MLLayerNormalizationOptions : MLOperatorOptions { MLOperand scale; MLOperand bias; sequence<[EnforceRange] unsigned long> axes; double epsilon = 1e-5; };
dictionary MLInstanceNormalizationOptions : MLOperatorOptions { MLOperand scale; MLOperand bias; double epsilon = 1e-5; MLInputOperandLayout layout = "nchw"; };
dictionary MLEluOptions : MLOperatorOptions { double alpha = 1; };
dictionary MLLeakyReluOptions : MLOperatorOptions { double alpha = 0.01; };
dictionary MLLinearOptions : MLOperatorOptions { double alpha = 1; double beta = 0; };
dictionary MLHardSigmoidOptions : MLOperatorOptions { double alpha = 0.2; double beta = 0.5; };
dictionary MLClampOptions : MLOperatorOptions { double minValue; double maxValue; };
