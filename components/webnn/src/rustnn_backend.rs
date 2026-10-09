/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraph, MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::{
    MLConv2dFilterOperandLayout, MLConvTranspose2dFilterOperandLayout, MLInputOperandLayout,
    MLOperandDataType,
};
use rustnn::operator_options::{
    MLConv2dOptions, MLConvTranspose2dOptions, MLDimension, MLGemmOptions, MLLeakyReluOptions,
    MLOperatorOptions, MLPadOptions, MLPool2dOptions, MLReduceOptions, MLResample2dOptions,
    MLSliceOptions, MLSplitOptions, MLTransposeOptions,
};
use rustnn::operators::Operation;

use crate::{
    Backend, BackendOptions, BackendPowerPreference, BuilderId, GraphId, OperandId, Operator,
};

// rustnn models the layout options as enums (rustnn#260); servo's operator IR
// still carries the string form produced by the WebIDL layer, so translate at
// the boundary. An unrecognised string falls back to the WebNN default, which
// is what an empty string has always meant here.
fn input_operand_layout(layout: &str) -> MLInputOperandLayout {
    match layout {
        "nhwc" => MLInputOperandLayout::Nhwc,
        _ => MLInputOperandLayout::Nchw,
    }
}

fn conv2d_filter_operand_layout(layout: &str) -> MLConv2dFilterOperandLayout {
    match layout {
        "hwio" => MLConv2dFilterOperandLayout::Hwio,
        "ohwi" => MLConv2dFilterOperandLayout::Ohwi,
        "ihwo" => MLConv2dFilterOperandLayout::Ihwo,
        _ => MLConv2dFilterOperandLayout::Oihw,
    }
}

fn conv_transpose2d_filter_operand_layout(layout: &str) -> MLConvTranspose2dFilterOperandLayout {
    match layout {
        "hwoi" => MLConvTranspose2dFilterOperandLayout::Hwoi,
        "ohwi" => MLConvTranspose2dFilterOperandLayout::Ohwi,
        _ => MLConvTranspose2dFilterOperandLayout::Iohw,
    }
}

/// Maps Servo's `MLOperandDataType` discriminant (WebIDL enum order) to rustnn's
/// graph `DataType`.
fn to_dtype(data_type: u32) -> DataType {
    match data_type {
        0 => DataType::Float32,
        1 => DataType::Float16,
        2 => DataType::Int32,
        3 => DataType::Uint32,
        4 => DataType::Int64,
        5 => DataType::Uint64,
        6 => DataType::Int8,
        7 => DataType::Uint8,
        _ => DataType::Float32,
    }
}

/// Maps Servo's `MLOperandDataType` discriminant (WebIDL enum order) to rustnn's
/// `MLOperandDataType` enum.
fn to_ml_operand_data_type(data_type: u32) -> MLOperandDataType {
    match data_type {
        0 => MLOperandDataType::Float32,
        1 => MLOperandDataType::Float16,
        2 => MLOperandDataType::Int32,
        3 => MLOperandDataType::Uint32,
        4 => MLOperandDataType::Int64,
        5 => MLOperandDataType::Uint64,
        6 => MLOperandDataType::Int8,
        7 => MLOperandDataType::Uint8,
        _ => MLOperandDataType::Float32,
    }
}

/// Builds a rustnn `Operation` from a typed [`Operator`] and dense operand
/// indices.
fn make_op(
    operator: &Operator,
    inputs: &[u32],
    outputs: &[u32],
    index_of: &HashMap<OperandId, u32>,
    label: &str,
) -> Result<Operation, String> {
    let label = label.to_string();
    let op_options = || MLOperatorOptions {
        label: label.clone(),
    };
    Ok(match operator {
        Operator::Add => Operation::Add {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Sub => Operation::Sub {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Mul => Operation::Mul {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Div => Operation::Div {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Max => Operation::Max {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Min => Operation::Min {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Matmul => Operation::Matmul {
            a: inputs[0],
            b: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Gemm(options) => Operation::Gemm {
            a: inputs[0],
            b: inputs[1],
            options: Some(MLGemmOptions {
                label: label.clone(),
                c: options.c.map(|id| index_of[&id]),
                alpha: options.alpha as f64,
                beta: options.beta as f64,
                a_transpose: options.a_transpose,
                b_transpose: options.b_transpose,
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Sigmoid => Operation::Sigmoid {
            input: inputs[0],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Relu => Operation::Relu {
            input: inputs[0],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Sqrt => Operation::Sqrt {
            input: inputs[0],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Reciprocal => Operation::Reciprocal {
            input: inputs[0],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Gelu => Operation::Gelu {
            input: inputs[0],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::LeakyRelu { alpha } => Operation::LeakyRelu {
            input: inputs[0],
            options: Some(MLLeakyReluOptions {
                label: label.clone(),
                alpha: *alpha as f64,
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Prelu => Operation::Prelu {
            input: inputs[0],
            slope: inputs[1],
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Cast { output_data_type } => Operation::Cast {
            input: inputs[0],
            data_type: to_ml_operand_data_type(*output_data_type),
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Concat { axis } => Operation::Concat {
            inputs: inputs.to_vec(),
            axis: *axis,
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Softmax { axis } => Operation::Softmax {
            input: inputs[0],
            axis: *axis,
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Reshape { new_shape } => Operation::Reshape {
            input: inputs[0],
            new_shape: new_shape
                .iter()
                .map(|&dimension| MLDimension::Static(dimension))
                .collect(),
            options: Some(op_options()),
            outputs: outputs.to_vec(),
        },
        Operator::Transpose { permutation } => Operation::Transpose {
            input: inputs[0],
            options: Some(MLTransposeOptions {
                label: label.clone(),
                permutation: permutation.clone(),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Slice {
            starts,
            sizes,
            strides,
        } => Operation::Slice {
            input: inputs[0],
            starts: starts.clone(),
            sizes: sizes
                .iter()
                .map(|&dimension| MLDimension::Static(dimension))
                .collect(),
            options: Some(MLSliceOptions {
                label: label.clone(),
                strides: strides.clone(),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Pad {
            beginning_padding,
            ending_padding,
            mode,
            value,
        } => Operation::Pad {
            input: inputs[0],
            beginning_padding: beginning_padding.clone(),
            ending_padding: ending_padding.clone(),
            options: Some(MLPadOptions {
                label: label.clone(),
                mode: mode.clone(),
                value: Some(serde_json::json!(value)),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Split { splits, axis } => Operation::Split {
            input: inputs[0],
            splits: splits.clone(),
            split_equal_parts: None,
            options: Some(MLSplitOptions {
                label: label.clone(),
                axis: *axis,
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Conv2d(options) => Operation::Conv2d {
            input: inputs[0],
            filter: inputs[1],
            options: Some(MLConv2dOptions {
                label: label.clone(),
                padding: options.padding.clone(),
                strides: options.strides.clone(),
                dilations: options.dilations.clone(),
                groups: options.groups,
                input_layout: input_operand_layout(&options.input_layout),
                filter_layout: conv2d_filter_operand_layout(&options.filter_layout),
                bias: options.bias.map(|id| index_of[&id]),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::ConvTranspose2d(options) => Operation::ConvTranspose2d {
            input: inputs[0],
            filter: inputs[1],
            options: Some(MLConvTranspose2dOptions {
                label: label.clone(),
                padding: options.padding.clone(),
                strides: options.strides.clone(),
                dilations: options.dilations.clone(),
                output_padding: options.output_padding.clone(),
                output_sizes: options.output_sizes.clone(),
                groups: options.groups,
                input_layout: input_operand_layout(&options.input_layout),
                filter_layout: conv_transpose2d_filter_operand_layout(&options.filter_layout),
                bias: options.bias.map(|id| index_of[&id]),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::MaxPool2d(options) => Operation::MaxPool2d {
            input: inputs[0],
            options: Some(MLPool2dOptions {
                label: label.clone(),
                window_dimensions: if options.window_dimensions.is_empty() {
                    None
                } else {
                    Some(options.window_dimensions.clone())
                },
                padding: options.padding.clone(),
                strides: options.strides.clone(),
                dilations: options.dilations.clone(),
                layout: options.layout.clone(),
                output_shape_rounding: options.output_shape_rounding.clone(),
                output_sizes: options.output_sizes.clone(),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::AveragePool2d(options) => Operation::AveragePool2d {
            input: inputs[0],
            options: Some(MLPool2dOptions {
                label: label.clone(),
                window_dimensions: if options.window_dimensions.is_empty() {
                    None
                } else {
                    Some(options.window_dimensions.clone())
                },
                padding: options.padding.clone(),
                strides: options.strides.clone(),
                dilations: options.dilations.clone(),
                layout: options.layout.clone(),
                output_shape_rounding: options.output_shape_rounding.clone(),
                output_sizes: options.output_sizes.clone(),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::Resample2d(options) => Operation::Resample2d {
            input: inputs[0],
            options: Some(MLResample2dOptions {
                label: label.clone(),
                mode: options.mode.clone(),
                scales: options.scales.clone(),
                sizes: options.sizes.clone(),
                axes: options.axes.clone(),
            }),
            outputs: outputs.to_vec(),
        },
        Operator::ReduceSum(options) => Operation::ReduceSum {
            input: inputs[0],
            options: Some(MLReduceOptions {
                label: label.clone(),
                axes: if options.axes.is_empty() {
                    None
                } else {
                    Some(options.axes.clone())
                },
                keep_dimensions: options.keep_dimensions,
            }),
            outputs: outputs.to_vec(),
        },
        Operator::ReduceMean(options) => Operation::ReduceMean {
            input: inputs[0],
            options: Some(MLReduceOptions {
                label: label.clone(),
                axes: if options.axes.is_empty() {
                    None
                } else {
                    Some(options.axes.clone())
                },
                keep_dimensions: options.keep_dimensions,
            }),
            outputs: outputs.to_vec(),
        },
    })
}

/// A single operand accumulated by the DOM layer.
struct OperandEntry {
    name: Option<String>,
    data_type: DataType,
    shape: Vec<u32>,
    data: Option<Vec<u8>>,
}

/// A pending operator accumulated by the DOM layer.
struct PendingOp {
    operator: Operator,
    inputs: Vec<OperandId>,
    outputs: Vec<OperandId>,
    label: String,
}

/// Per-builder state accumulated from `add_input` / `add_constant` /
/// `add_operator` and converted to a `GraphInfo` on `build`.
struct BuilderState {
    operands: HashMap<OperandId, OperandEntry>,
    operations: Vec<PendingOp>,
    input_names: HashMap<String, OperandId>,
}

impl BuilderState {
    fn new() -> Self {
        Self {
            operands: HashMap::new(),
            operations: Vec::new(),
            input_names: HashMap::new(),
        }
    }

    fn into_graph_info(self, outputs: &[(String, OperandId)]) -> Result<GraphInfo, String> {
        // Assign a dense operand index to every operand id in a stable order.
        let mut ids: Vec<OperandId> = self.operands.keys().copied().collect();
        ids.sort_unstable();
        let index_of: HashMap<OperandId, u32> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();

        let output_names: HashMap<OperandId, String> = outputs
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        let output_set: HashSet<OperandId> = outputs.iter().map(|&(_, id)| id).collect();
        let input_set: HashSet<OperandId> = self.input_names.values().copied().collect();

        let mut operands = Vec::with_capacity(ids.len());
        for &id in &ids {
            let entry = &self.operands[&id];
            let kind = if input_set.contains(&id) {
                OperandKind::Input
            } else if entry.data.is_some() {
                OperandKind::Constant
            } else if output_set.contains(&id) {
                OperandKind::Output
            } else {
                OperandKind::Intermediate
            };
            let name = if kind == OperandKind::Output {
                output_names.get(&id).cloned()
            } else {
                entry.name.clone()
            };
            operands.push(Operand {
                kind,
                name,
                descriptor: OperandDescriptor {
                    data_type: entry.data_type,
                    shape: entry
                        .shape
                        .iter()
                        .map(|&dimension| Dimension::Static(dimension))
                        .collect(),
                    pending_permutation: Vec::new(),
                },
            });
        }

        let mut operations = Vec::new();
        for pending in &self.operations {
            let out_indices: Vec<u32> = pending
                .outputs
                .iter()
                .map(|output| index_of[output])
                .collect();
            let in_indices: Vec<u32> = pending.inputs.iter().map(|input| index_of[input]).collect();
            operations.push(make_op(
                &pending.operator,
                &in_indices,
                &out_indices,
                &index_of,
                &pending.label,
            )?);
        }

        let mut input_operands: Vec<u32> =
            self.input_names.values().map(|&id| index_of[&id]).collect();
        input_operands.sort_unstable();
        // Outputs must be in ascending dense-operand-index order to satisfy
        // rustnn's `validate_io_operand_lists`; the `build()` output list is in
        // JS `Record` order, which is not guaranteed to be ascending.
        let mut output_operands: Vec<u32> = outputs.iter().map(|&(_, id)| index_of[&id]).collect();
        output_operands.sort_unstable();

        let mut constant_operand_ids_to_handles = HashMap::new();
        for &id in &ids {
            if let Some(data) = self.operands[&id].data.as_ref() {
                constant_operand_ids_to_handles.insert(
                    index_of[&id],
                    ConstantData {
                        data: data.clone(),
                        label: None,
                    },
                );
            }
        }

        Ok(GraphInfo {
            operands,
            input_operands,
            output_operands,
            operations,
            constant_operand_ids_to_handles,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        })
    }
}

/// A device-resident tensor, keyed by the Servo-side tensor id.
///
/// The tensor is created once (on `CreateTensor`) and reused across dispatches,
/// so a tensor produced by one graph can be consumed by another without a host
/// round-trip.
struct DeviceTensor {
    tensor: MLTensor,
    /// Servo's `MLOperandDataType` discriminant, used to dispatch typed
    /// `write`/`read` operations.
    data_type: u32,
}

/// Backend that delegates to the rustnn crate.
///
/// Each `RustnnBackend` instance owns a single rustnn `MLContext` (one per
/// Servo `MLContext`) and a set of compiled graphs. Graph compilation and
/// compute both happen on the shared WebNN backend thread.
pub struct RustnnBackend {
    context: Mutex<MLContext<'static>>,
    builders: Mutex<HashMap<BuilderId, BuilderState>>,
    graphs: Mutex<HashMap<GraphId, MLGraph<'static>>>,
    tensors: Mutex<HashMap<u32, DeviceTensor>>,
    next_builder_id: AtomicUsize,
    next_graph_id: AtomicUsize,
}

impl RustnnBackend {
    /// Creates a backend for the requested WebNN device type.
    ///
    /// Which backend actually runs is rustnn's call: it resolves the power
    /// preference and `accelerated` flag against the backends compiled in for
    /// this target (`rustnn::backend_selection`). The one case Servo has to
    /// speak up for is CANN, which rustnn only selects when explicitly hinted —
    /// so an `npu` request on OpenHarmony pins it, and every other request is
    /// passed through as-is.
    pub fn new(options: &BackendOptions) -> Result<Self, String> {
        let context_options = ml_context_options(options);

        #[cfg(target_env = "ohos")]
        let context_options = match options.device_type {
            crate::BackendDeviceType::Npu => {
                use rustnn::mlcontext::{BackendDevice, DeviceType};
                context_options.with_rustnn_device_hint(BackendDevice::Cann {
                    device_type: DeviceType::Npu,
                })
            },
            _ => context_options,
        };

        Self::from_options(context_options)
    }

    fn from_options(options: MLContextOptions) -> Result<Self, String> {
        let context = MLContext::create(&options).map_err(|e| e.to_string())?;
        Ok(Self {
            context: Mutex::new(context),
            builders: Mutex::new(HashMap::new()),
            graphs: Mutex::new(HashMap::new()),
            tensors: Mutex::new(HashMap::new()),
            next_builder_id: AtomicUsize::new(1),
            next_graph_id: AtomicUsize::new(1),
        })
    }
}

/// Maps Servo's backend options to rustnn's `MLContextOptions`.
fn ml_context_options(options: &BackendOptions) -> MLContextOptions {
    let power_preference = match options.power_preference {
        BackendPowerPreference::Default => MLPowerPreference::Default,
        BackendPowerPreference::HighPerformance => MLPowerPreference::HighPerformance,
        BackendPowerPreference::LowPower => MLPowerPreference::LowPower,
    };
    MLContextOptions::new(power_preference, options.accelerated)
}

/// Creates a read/write tensor descriptor from a Servo-side data type
/// discriminant and shape.
fn tensor_descriptor_from(data_type: u32, shape: &[u32]) -> MLTensorDescriptor {
    let data_type = to_ml_operand_data_type(data_type);
    let shape: Vec<u64> = shape.iter().map(|&dimension| dimension as u64).collect();
    let mut tensor_descriptor = MLTensorDescriptor::new(data_type, shape);
    tensor_descriptor.set_readable(true);
    tensor_descriptor.set_writable(true);
    tensor_descriptor
}

/// Writes raw little-endian input bytes into a tensor, interpreting them
/// according to `data_type`.
fn write_input(
    context: &mut MLContext<'static>,
    tensor: &MLTensor,
    data: &[u8],
    data_type: DataType,
) -> Result<(), String> {
    match data_type {
        DataType::Float32 => write_typed::<f32>(context, tensor, data),
        DataType::Float16 => write_typed::<half::f16>(context, tensor, data),
        DataType::Int32 => write_typed::<i32>(context, tensor, data),
        DataType::Uint32 => write_typed::<u32>(context, tensor, data),
        DataType::Int64 => write_typed::<i64>(context, tensor, data),
        DataType::Uint64 => write_typed::<u64>(context, tensor, data),
        DataType::Int8 => write_typed::<i8>(context, tensor, data),
        DataType::Uint8 => write_typed::<u8>(context, tensor, data),
        DataType::Int4 | DataType::Uint4 => Err("4-bit data types are not supported".to_string()),
    }
}

/// Reads a tensor back into raw little-endian bytes according to `data_type`.
fn read_output(
    context: &mut MLContext<'static>,
    tensor: &MLTensor,
    data_type: DataType,
) -> Result<Vec<u8>, String> {
    match data_type {
        DataType::Float32 => read_typed::<f32>(context, tensor),
        DataType::Float16 => read_typed::<half::f16>(context, tensor),
        DataType::Int32 => read_typed::<i32>(context, tensor),
        DataType::Uint32 => read_typed::<u32>(context, tensor),
        DataType::Int64 => read_typed::<i64>(context, tensor),
        DataType::Uint64 => read_typed::<u64>(context, tensor),
        DataType::Int8 => read_typed::<i8>(context, tensor),
        DataType::Uint8 => read_typed::<u8>(context, tensor),
        DataType::Int4 | DataType::Uint4 => Err("4-bit data types are not supported".to_string()),
    }
}

/// Writes a typed tensor, reinterpreting `data` as little-endian `T` values.
fn write_typed<T: bytemuck::Pod>(
    context: &mut MLContext<'static>,
    tensor: &MLTensor,
    data: &[u8],
) -> Result<(), String> {
    let size = std::mem::size_of::<T>();
    if data.len() % size != 0 {
        return Err(format!(
            "input byte length {} is not a multiple of {size}",
            data.len()
        ));
    }
    // Zero-copy reinterpret instead of a per-element copy. `data` always comes
    // from a freshly allocated `Vec<u8>` in the backend tensor store, so it is
    // allocator-aligned (≥ 8 bytes) and safe to view as `[T]` for any WebNN
    // scalar type (max alignment 8).
    let values: &[T] = bytemuck::cast_slice(data);
    context
        .write_tensor(tensor, values)
        .map_err(|e| e.to_string())
}

/// Reads a typed tensor and returns its raw little-endian bytes, zero-copy:
/// the buffer is allocated as bytes and reinterpreted as `[T]` in place, so the
/// backend reads straight into the returned `Vec<u8>` (no `to_vec` copy).
fn read_typed<T: bytemuck::Pod>(
    context: &mut MLContext<'static>,
    tensor: &MLTensor,
) -> Result<Vec<u8>, String> {
    let byte_len = tensor.rustnn_required_bytes();
    // Allocate WITHOUT zeroing: `read_tensor` overwrites the entire buffer (the
    // full tensor byte size), so the memset is wasted work. The capacity is
    // allocated and `set_len` marks the buffer usable; `read_tensor` writes
    // exactly `byte_len` bytes before this Vec is ever read.
    let mut bytes = Vec::<u8>::with_capacity(byte_len);
    // SAFETY: `byte_len` bytes are written by `read_tensor` (the MLContext
    // wrapper checks `rustnn_required_bytes() == size_of_val(array)` and the
    // backend copies exactly that many bytes), so no uninitialized byte is read.
    unsafe { bytes.set_len(byte_len) };
    // `bytes` is freshly allocated, so it is allocator-aligned (≥ 8 bytes) and
    // safe to view as `[T]` for any WebNN scalar type (max alignment 8).
    let values: &mut [T] = bytemuck::cast_slice_mut::<u8, T>(&mut bytes);
    context
        .read_tensor(tensor, values)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

impl Backend for RustnnBackend {
    fn name(&self) -> &str {
        "rustnn"
    }

    fn create_builder(&self) -> BuilderId {
        let id = self.next_builder_id.fetch_add(1, Ordering::Relaxed);
        self.builders
            .lock()
            .unwrap()
            .insert(id, BuilderState::new());
        id
    }

    fn add_input(
        &self,
        builder_id: BuilderId,
        operand_id: OperandId,
        name: &str,
        data_type: u32,
        shape: &[u32],
    ) {
        if let Some(state) = self.builders.lock().unwrap().get_mut(&builder_id) {
            state.input_names.insert(name.to_string(), operand_id);
            state.operands.insert(
                operand_id,
                OperandEntry {
                    name: Some(name.to_string()),
                    data_type: to_dtype(data_type),
                    shape: shape.to_vec(),
                    data: None,
                },
            );
        }
    }

    fn add_constant(
        &self,
        builder_id: BuilderId,
        operand_id: OperandId,
        data_type: u32,
        shape: &[u32],
        data: &[u8],
    ) {
        if let Some(state) = self.builders.lock().unwrap().get_mut(&builder_id) {
            state.operands.insert(
                operand_id,
                OperandEntry {
                    name: None,
                    data_type: to_dtype(data_type),
                    shape: shape.to_vec(),
                    data: Some(data.to_vec()),
                },
            );
        }
    }

    fn add_operator(
        &self,
        builder_id: BuilderId,
        operator: &Operator,
        inputs: &[OperandId],
        outputs: &[(OperandId, u32, Vec<u32>)],
        label: &str,
    ) {
        if let Some(state) = self.builders.lock().unwrap().get_mut(&builder_id) {
            let mut output_ids = Vec::with_capacity(outputs.len());
            for (operand_id, data_type, shape) in outputs {
                state.operands.insert(
                    *operand_id,
                    OperandEntry {
                        name: None,
                        data_type: to_dtype(*data_type),
                        shape: shape.clone(),
                        data: None,
                    },
                );
                output_ids.push(*operand_id);
            }
            state.operations.push(PendingOp {
                operator: operator.clone(),
                inputs: inputs.to_vec(),
                outputs: output_ids,
                label: label.to_string(),
            });
        }
    }

    fn create_tensor(&self, tensor_id: u32, data_type: u32, shape: &[u32]) -> Result<(), String> {
        let desc = tensor_descriptor_from(data_type, shape);
        let tensor = self
            .context
            .lock()
            .unwrap()
            .create_tensor(&desc)
            .map_err(|e| e.to_string())?;
        self.tensors
            .lock()
            .unwrap()
            .insert(tensor_id, DeviceTensor { tensor, data_type });
        log::error!(
            "[webnn-leak] backend create_tensor {tensor_id} tensors={}",
            self.tensors.lock().unwrap().len()
        );
        Ok(())
    }

    fn write_tensor(&self, tensor_id: u32, bytes: &[u8]) -> Result<(), String> {
        let mut context = self.context.lock().unwrap();
        let tensors = self.tensors.lock().unwrap();
        let device_tensor = tensors
            .get(&tensor_id)
            .ok_or_else(|| format!("unknown tensor {tensor_id}"))?;
        write_input(
            &mut context,
            &device_tensor.tensor,
            bytes,
            to_dtype(device_tensor.data_type),
        )
    }

    fn read_tensor(&self, tensor_id: u32) -> Result<Vec<u8>, String> {
        let mut context = self.context.lock().unwrap();
        let tensors = self.tensors.lock().unwrap();
        let device_tensor = tensors
            .get(&tensor_id)
            .ok_or_else(|| format!("unknown tensor {tensor_id}"))?;
        read_output(
            &mut context,
            &device_tensor.tensor,
            to_dtype(device_tensor.data_type),
        )
    }

    fn destroy_tensor(&self, tensor_id: u32) {
        self.tensors.lock().unwrap().remove(&tensor_id);
        log::error!(
            "[webnn-leak] backend destroy_tensor {tensor_id} tensors={}",
            self.tensors.lock().unwrap().len()
        );
    }

    fn build(
        &self,
        builder_id: BuilderId,
        outputs: &[(String, OperandId)],
    ) -> Result<GraphId, String> {
        let builder_state = self
            .builders
            .lock()
            .unwrap()
            .remove(&builder_id)
            .ok_or_else(|| format!("unknown builder {builder_id}"))?;
        let graph_info = builder_state.into_graph_info(outputs)?;
        let graph = {
            let mut context = self.context.lock().unwrap();
            let mut builder = MLGraphBuilder::new(&mut context).map_err(|e| e.to_string())?;
            builder
                .build_graph_info(graph_info)
                .map_err(|e| e.to_string())?
        };
        let graph_id = self.next_graph_id.fetch_add(1, Ordering::Relaxed);
        self.graphs.lock().unwrap().insert(graph_id, graph);
        log::error!(
            "[webnn-leak] backend build graph {graph_id} graphs={}",
            self.graphs.lock().unwrap().len()
        );
        Ok(graph_id)
    }

    fn run(
        &self,
        graph_id: GraphId,
        inputs: &[(String, u32)],
        outputs: &[(String, u32)],
    ) -> Result<(), String> {
        let mut context = self.context.lock().unwrap();
        let mut graphs = self.graphs.lock().unwrap();
        let graph = graphs
            .get_mut(&graph_id)
            .ok_or_else(|| format!("graph {graph_id} not found"))?;
        let tensors = self.tensors.lock().unwrap();

        // Resolve Servo tensor ids to device tensors. A tensor produced by one
        // graph and consumed by another resolves to the same device tensor, so
        // intermediate data never round-trips the host.
        let input_map: BTreeMap<&str, &MLTensor> = inputs
            .iter()
            .map(|(name, id)| {
                let device_tensor = tensors
                    .get(id)
                    .ok_or_else(|| format!("unknown input tensor {id}"))?;
                Ok((name.as_str(), &device_tensor.tensor))
            })
            .collect::<Result<_, String>>()?;
        let output_map: BTreeMap<&str, &MLTensor> = outputs
            .iter()
            .map(|(name, id)| {
                let device_tensor = tensors
                    .get(id)
                    .ok_or_else(|| format!("unknown output tensor {id}"))?;
                Ok((name.as_str(), &device_tensor.tensor))
            })
            .collect::<Result<_, String>>()?;

        context
            .dispatch(graph, &input_map, &output_map)
            .map_err(|e| e.to_string())
    }

    fn destroy_graph(&self, graph_id: GraphId) {
        self.graphs.lock().unwrap().remove(&graph_id);
        log::error!(
            "[webnn-leak] backend destroy_graph {graph_id} graphs={}",
            self.graphs.lock().unwrap().len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_operand(
        state: &mut BuilderState,
        id: OperandId,
        name: Option<&str>,
        data_type: u32,
        shape: &[u32],
        data: Option<Vec<u8>>,
    ) {
        state.operands.insert(
            id,
            OperandEntry {
                name: name.map(str::to_string),
                data_type: to_dtype(data_type),
                shape: shape.to_vec(),
                data,
            },
        );
    }

    #[test]
    fn test_to_dtype_mapping() {
        assert_eq!(to_dtype(0), DataType::Float32);
        assert_eq!(to_dtype(1), DataType::Float16);
        assert_eq!(to_dtype(2), DataType::Int32);
        assert_eq!(to_dtype(3), DataType::Uint32);
        assert_eq!(to_dtype(4), DataType::Int64);
        assert_eq!(to_dtype(5), DataType::Uint64);
        assert_eq!(to_dtype(6), DataType::Int8);
        assert_eq!(to_dtype(7), DataType::Uint8);
        // Unknown values fall back to float32.
        assert_eq!(to_dtype(99), DataType::Float32);
    }

    #[test]
    fn test_make_op_add() {
        let op = make_op(&Operator::Add, &[0, 1], &[2], &HashMap::new(), "").unwrap();
        assert!(matches!(
            op,
            Operation::Add { a: 0, b: 1, options: Some(_), outputs } if outputs == vec![2]
        ));
    }

    #[test]
    fn test_make_op_cast() {
        let op = make_op(
            &Operator::Cast {
                output_data_type: 2,
            },
            &[0],
            &[1],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::Cast { input: 0, data_type: MLOperandDataType::Int32, options: Some(_), outputs }
                if outputs == vec![1]
        ));
    }

    #[test]
    fn test_make_op_div() {
        let op = make_op(&Operator::Div, &[0, 1], &[2], &HashMap::new(), "").unwrap();
        assert!(matches!(
            op,
            Operation::Div { a: 0, b: 1, options: Some(_), outputs } if outputs == vec![2]
        ));
    }

    #[test]
    fn test_make_op_prelu() {
        let op = make_op(&Operator::Prelu, &[0, 1], &[2], &HashMap::new(), "").unwrap();
        assert!(matches!(
            op,
            Operation::Prelu { input: 0, slope: 1, options: Some(_), outputs } if outputs == vec![2]
        ));
    }

    #[test]
    fn test_make_op_relu() {
        let op = make_op(&Operator::Relu, &[0], &[1], &HashMap::new(), "").unwrap();
        assert!(matches!(
            op,
            Operation::Relu { input: 0, options: Some(_), outputs } if outputs == vec![1]
        ));
    }

    #[test]
    fn test_make_op_pad() {
        let op = make_op(
            &Operator::Pad {
                beginning_padding: vec![0, 0, 0, 0],
                ending_padding: vec![0, 4, 0, 0],
                mode: "constant".to_string(),
                value: 0.0,
            },
            &[0],
            &[1],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::Pad {
                input: 0,
                beginning_padding,
                ending_padding,
                options: Some(_),
                outputs,
            } if beginning_padding == vec![0, 0, 0, 0]
                && ending_padding == vec![0, 4, 0, 0]
                && outputs == vec![1]
        ));
    }

    #[test]
    fn test_make_op_reduce_sum() {
        let op = make_op(
            &Operator::ReduceSum(crate::ReduceOptions {
                axes: vec![2, 3],
                keep_dimensions: true,
            }),
            &[0],
            &[1],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::ReduceSum { input: 0, options: Some(options), outputs }
                if options.axes == Some(vec![2, 3]) && options.keep_dimensions && outputs == vec![1]
        ));
    }

    #[test]
    fn test_make_op_reduce_mean() {
        let op = make_op(
            &Operator::ReduceMean(crate::ReduceOptions {
                axes: vec![0, 2],
                keep_dimensions: false,
            }),
            &[0],
            &[1],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::ReduceMean { input: 0, options: Some(options), outputs }
                if options.axes == Some(vec![0, 2]) && !options.keep_dimensions && outputs == vec![1]
        ));
    }

    #[test]
    fn test_make_op_unary_ops() {
        for (operator, expect) in [
            (Operator::Sqrt, "sqrt"),
            (Operator::Reciprocal, "reciprocal"),
            (Operator::Gelu, "gelu"),
        ] {
            let op = make_op(&operator, &[0], &[1], &HashMap::new(), "").unwrap();
            match op {
                Operation::Sqrt {
                    input: 0,
                    options: Some(_),
                    outputs,
                } if outputs == vec![1] => {
                    assert_eq!(expect, "sqrt");
                },
                Operation::Reciprocal {
                    input: 0,
                    options: Some(_),
                    outputs,
                } if outputs == vec![1] => {
                    assert_eq!(expect, "reciprocal");
                },
                Operation::Gelu {
                    input: 0,
                    options: Some(_),
                    outputs,
                } if outputs == vec![1] => {
                    assert_eq!(expect, "gelu");
                },
                _ => panic!("unexpected op for {expect}"),
            }
        }
    }

    #[test]
    fn test_make_op_leaky_relu() {
        let op = make_op(
            &Operator::LeakyRelu { alpha: 0.2 },
            &[0],
            &[1],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::LeakyRelu { input: 0, options: Some(options), outputs }
                if (options.alpha - 0.2).abs() < 1e-6 && outputs == vec![1]
        ));
    }

    #[test]
    fn test_make_op_min_max_matmul() {
        for (operator, expect) in [
            (Operator::Min, "min"),
            (Operator::Max, "max"),
            (Operator::Matmul, "matmul"),
        ] {
            let op = make_op(&operator, &[0, 1], &[2], &HashMap::new(), "").unwrap();
            match op {
                Operation::Min {
                    a: 0,
                    b: 1,
                    options: Some(_),
                    outputs,
                } if outputs == vec![2] => {
                    assert_eq!(expect, "min");
                },
                Operation::Max {
                    a: 0,
                    b: 1,
                    options: Some(_),
                    outputs,
                } if outputs == vec![2] => {
                    assert_eq!(expect, "max");
                },
                Operation::Matmul {
                    a: 0,
                    b: 1,
                    options: Some(_),
                    outputs,
                } if outputs == vec![2] => {
                    assert_eq!(expect, "matmul");
                },
                _ => panic!("unexpected op for {expect}"),
            }
        }
    }

    #[test]
    fn test_make_op_gemm() {
        let mut index_of = HashMap::new();
        index_of.insert(10, 0);
        index_of.insert(11, 1);
        index_of.insert(12, 2);
        let op = make_op(
            &Operator::Gemm(crate::GemmOptions {
                c: Some(12),
                alpha: 1.0,
                beta: 1.0,
                a_transpose: false,
                b_transpose: true,
            }),
            &[0, 1],
            &[3],
            &index_of,
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::Gemm { a: 0, b: 1, options: Some(options), outputs }
                if options.c == Some(2) && options.b_transpose && !options.a_transpose && outputs == vec![3]
        ));
    }

    #[test]
    fn test_make_op_conv_transpose2d() {
        let op = make_op(
            &Operator::ConvTranspose2d(crate::ConvTranspose2dOptions {
                padding: vec![0, 0, 0, 0],
                strides: vec![4, 4],
                dilations: vec![1, 1],
                output_padding: vec![0, 0],
                output_sizes: None,
                groups: 1,
                input_layout: "nhwc".to_string(),
                filter_layout: "ohwi".to_string(),
                bias: None,
            }),
            &[0, 1],
            &[2],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::ConvTranspose2d { input: 0, filter: 1, options: Some(options), outputs }
                if options.strides == vec![4, 4]
                    && options.input_layout == MLInputOperandLayout::Nhwc
                    && options.filter_layout == MLConvTranspose2dFilterOperandLayout::Ohwi
                    && outputs == vec![2]
        ));
    }

    #[test]
    fn test_make_op_split_multi_output() {
        let op = make_op(
            &Operator::Split {
                splits: vec![2, 2],
                axis: 0,
            },
            &[0],
            &[1, 2],
            &HashMap::new(),
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::Split { splits, split_equal_parts: None, options: Some(_), outputs, .. }
                if splits == vec![2, 2] && outputs == vec![1, 2]
        ));
    }

    #[test]
    fn test_make_op_conv2d_with_bias() {
        // input = 10, filter = 11, bias = 12, output = 13.
        let mut index_of = HashMap::new();
        index_of.insert(10, 0);
        index_of.insert(11, 1);
        index_of.insert(12, 2);
        index_of.insert(13, 3);
        let op = make_op(
            &Operator::Conv2d(crate::Conv2dOptions {
                padding: vec![1, 1, 1, 1],
                strides: vec![2, 2],
                dilations: vec![1, 1],
                groups: 1,
                input_layout: "nchw".to_string(),
                filter_layout: "oihw".to_string(),
                bias: Some(12),
            }),
            &[0, 1],
            &[3],
            &index_of,
            "",
        )
        .unwrap();
        assert!(matches!(
            op,
            Operation::Conv2d { input: 0, filter: 1, options: Some(options), outputs }
                if options.padding == vec![1, 1, 1, 1]
                    && options.bias == Some(2)
                    && outputs == vec![3]
        ));
    }

    #[test]
    fn test_into_graph_info_add() {
        let mut state = BuilderState::new();
        push_operand(&mut state, 1, Some("a"), 0, &[2, 2], None);
        state.input_names.insert("a".to_string(), 1);
        push_operand(&mut state, 2, Some("b"), 0, &[2, 2], None);
        state.input_names.insert("b".to_string(), 2);
        push_operand(&mut state, 3, None, 0, &[2, 2], None);
        state.operations.push(PendingOp {
            operator: Operator::Add,
            inputs: vec![1, 2],
            outputs: vec![3],
            label: String::new(),
        });

        let info = state
            .into_graph_info(&[("sum".to_string(), 3)])
            .expect("graph info");
        assert_eq!(info.operands.len(), 3);
        assert_eq!(info.input_operands, vec![0, 1]);
        assert_eq!(info.output_operands, vec![2]);
        assert_eq!(info.operations.len(), 1);
        assert_eq!(info.operands[0].name.as_deref(), Some("a"));
        assert_eq!(info.operands[0].kind, OperandKind::Input);
        assert_eq!(info.operands[1].name.as_deref(), Some("b"));
        assert_eq!(info.operands[1].kind, OperandKind::Input);
        assert_eq!(info.operands[2].name.as_deref(), Some("sum"));
        assert_eq!(info.operands[2].kind, OperandKind::Output);
    }

    #[test]
    fn test_into_graph_info_with_constant() {
        let mut state = BuilderState::new();
        push_operand(&mut state, 1, Some("a"), 0, &[2], None);
        state.input_names.insert("a".to_string(), 1);
        push_operand(&mut state, 2, None, 0, &[2], Some(vec![0, 0, 0, 0]));
        push_operand(&mut state, 3, None, 0, &[2], None);
        state.operations.push(PendingOp {
            operator: Operator::Add,
            inputs: vec![1, 2],
            outputs: vec![3],
            label: String::new(),
        });

        let info = state
            .into_graph_info(&[("sum".to_string(), 3)])
            .expect("graph info");
        assert_eq!(info.operands.len(), 3);
        assert_eq!(info.input_operands, vec![0]);
        assert_eq!(info.output_operands, vec![2]);
        // The constant operand is tagged Constant and carried in the handles map.
        assert_eq!(info.operands[1].kind, OperandKind::Constant);
        assert!(info.constant_operand_ids_to_handles.contains_key(&1));
        // Constants are not operations.
        assert_eq!(info.operations.len(), 1);
    }

    #[test]
    fn test_into_graph_info_outputs_sorted() {
        let mut state = BuilderState::new();
        push_operand(&mut state, 1, Some("a"), 0, &[2], None);
        state.input_names.insert("a".to_string(), 1);
        push_operand(&mut state, 3, None, 0, &[2], None);
        push_operand(&mut state, 5, None, 0, &[2], None);

        // Declare outputs in non-ascending operand-id order; `output_operands`
        // must still be ascending in dense-index order to satisfy rustnn's
        // `validate_io_operand_lists`.
        let info = state
            .into_graph_info(&[("out_high".to_string(), 5), ("out_low".to_string(), 3)])
            .expect("graph info");
        // Dense indices: 1 -> 0, 3 -> 1, 5 -> 2.
        assert_eq!(info.output_operands, vec![1, 2]);
        assert_eq!(info.operands[1].kind, OperandKind::Output);
        assert_eq!(info.operands[1].name.as_deref(), Some("out_low"));
        assert_eq!(info.operands[2].kind, OperandKind::Output);
        assert_eq!(info.operands[2].name.as_deref(), Some("out_high"));
    }
}
