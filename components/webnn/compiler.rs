use std::collections::{HashMap, HashSet};

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, WIPOffset};

use crate::backend::{DataType, GraphNode, TensorDesc};

// ── TFLite schema enums ──

#[allow(dead_code)]
mod tfl {
    pub const PLACEHOLDER_FOR_GREATER_OP_CODES: i32 = 127;

    pub mod op {
        use super::PLACEHOLDER_FOR_GREATER_OP_CODES;
        pub const ADD: i32 = 0;
        pub const AVERAGE_POOL_2D: i32 = 1;
        pub const CONCATENATION: i32 = 2;
        pub const CONV_2D: i32 = 3;
        pub const DEPTHWISE_CONV_2D: i32 = 4;
        pub const DEQUANTIZE: i32 = 6;
        pub const FLOOR: i32 = 8;
        pub const FULLY_CONNECTED: i32 = 9;
        pub const L2_POOL_2D: i32 = 12;
        pub const LOGISTIC: i32 = 14;
        pub const MAX_POOL_2D: i32 = 17;
        pub const MUL: i32 = 18;
        pub const RELU: i32 = 19;
        pub const RELU6: i32 = 21;
        pub const RESHAPE: i32 = 22;
        pub const RESIZE_BILINEAR: i32 = 23;
        pub const SOFTMAX: i32 = 25;
        pub const TANH: i32 = 28;
        pub const PAD: i32 = 34;
        pub const GATHER: i32 = 36;
        pub const TRANSPOSE: i32 = 39;
        pub const MEAN: i32 = 40;
        pub const SUB: i32 = 41;
        pub const DIV: i32 = 42;
        pub const SQUEEZE: i32 = 43;
        pub const EXP: i32 = 47;
        pub const TOPK_V2: i32 = 48;
        pub const SPLIT: i32 = 49;
        pub const SPLIT_V: i32 = 102;
        pub const CAST: i32 = 53;
        pub const PRELU: i32 = 54;
        pub const MAXIMUM: i32 = 55;
        pub const ARG_MAX: i32 = 56;
        pub const MINIMUM: i32 = 57;
        pub const LESS: i32 = 58;
        pub const NEG: i32 = 59;
        pub const GREATER: i32 = 61;
        pub const GREATER_EQUAL: i32 = 62;
        pub const LESS_EQUAL: i32 = 63;
        pub const SELECT: i32 = 64;
        pub const SLICE: i32 = 65;
        pub const SIN: i32 = 66;
        pub const TRANSPOSE_CONV: i32 = 67;
        pub const TILE: i32 = 69;
        pub const EXPAND_DIMS: i32 = 70;
        pub const EQUAL: i32 = 71;
        pub const NOT_EQUAL: i32 = 72;
        pub const LOG: i32 = 73;
        pub const SUM: i32 = 74;
        pub const SQRT: i32 = 75;
        pub const RSQRT: i32 = 76;
        pub const POW: i32 = 78;
        pub const ARG_MIN: i32 = 79;
        pub const REDUCE_PROD: i32 = 81;
        pub const REDUCE_MAX: i32 = 82;
        pub const LOGICAL_OR: i32 = 84;
        pub const LOGICAL_AND: i32 = 86;
        pub const LOGICAL_NOT: i32 = 87;
        pub const REDUCE_MIN: i32 = 89;
        pub const RESIZE_NEAREST_NEIGHBOR: i32 = 97;
        pub const LEAKY_RELU: i32 = 98;
        pub const ABS: i32 = 101;
        pub const CEIL: i32 = 104;
        pub const REVERSE_V2: i32 = 105;
        pub const GATHER_ND: i32 = 107;
        pub const COS: i32 = 108;
        pub const WHERE: i32 = 109;
        pub const ELU: i32 = 111;
        pub const QUANTIZE: i32 = 114;
        pub const HARD_SWISH: i32 = 117;
        pub const HARD_SIGMOID: i32 = 119;
        pub const SCATTER_ND: i32 = 122;
        pub const BATCH_MATMUL: i32 = 126;
        pub const CUM_SUM: i32 = 128;
        pub const BROADCAST_TO: i32 = 130;
        pub const SOFTPLUS: i32 = 94;
        pub const SOFTSIGN: i32 = 96;
        pub const SQUARE: i32 = 103;
        pub const GELU: i32 = 150;
        pub const SIGN: i32 = 158;

        pub fn deprecated(code: i32) -> i32 {
            std::cmp::min(code, PLACEHOLDER_FOR_GREATER_OP_CODES)
        }
    }

    pub mod padding {
        pub const SAME: u8 = 0;
        pub const VALID: u8 = 1;
        pub const EXPLICIT: u8 = 2;
    }

    pub mod activation {
        pub const NONE: u8 = 0;
        pub const RELU: u8 = 1;
        pub const RELU6: u8 = 3;
        pub const TANH: u8 = 4;
    }

    pub mod builtin_options {
        pub const NONE: u8 = 0;
        pub const CONV_2D: u8 = 1;
        pub const DEPTHWISE_CONV_2D: u8 = 2;
        pub const POOL_2D: u8 = 5;
        pub const FULLY_CONNECTED: u8 = 8;
        pub const SOFTMAX: u8 = 9;
        pub const CONCATENATION: u8 = 10;
        pub const RESIZE_BILINEAR: u8 = 15;
        pub const RESHAPE: u8 = 17;
        pub const PAD: u8 = 22;
        pub const GATHER: u8 = 23;
        pub const TRANSPOSE: u8 = 26;
        pub const REDUCER: u8 = 27;
        pub const SQUEEZE: u8 = 30;
        pub const TOPK_V2: u8 = 34;
        pub const SPLIT: u8 = 35;
        pub const CAST: u8 = 37;
        pub const DEQUANTIZE: u8 = 38;
        pub const ARG_MAX: u8 = 40;
        pub const SELECT: u8 = 47;
        pub const SLICE: u8 = 48;
        pub const TRANSPOSE_CONV: u8 = 49;
        pub const EXPAND_DIMS: u8 = 52;
        pub const POW: u8 = 56;
        pub const ARG_MIN: u8 = 57;
        pub const RESIZE_NEAREST_NEIGHBOR: u8 = 74;
        pub const LEAKY_RELU: u8 = 75;
        pub const REVERSE_V2: u8 = 81;
        pub const WHERE: u8 = 85;
        pub const HARD_SWISH: u8 = 91;
        pub const SCATTER_ND: u8 = 97;
        pub const BATCH_MATMUL: u8 = 101;
        pub const CUM_SUM: u8 = 102;
        pub const BROADCAST_TO: u8 = 104;
        pub const SPLIT_V: u8 = 79;
        pub const QUANTIZE: u8 = 89;
    }
}

fn webnn_type_to_tflite(dt: DataType) -> u8 {
    match dt {
        DataType::Float32 => 0,
        DataType::Float16 => 1,
        DataType::Int32 => 2,
        DataType::Uint8 => 3,
        DataType::Int64 => 4,
        DataType::Int8 => 9,
        DataType::Uint64 => 12,
        DataType::Uint32 => 13,
    }
}

/// Transpose filter data from OIHW → OHWI (for regular conv2d).
fn transpose_oihw_to_ohwi(data: &[u8], o: usize, i: usize, h: usize, w: usize) -> Vec<u8> {
    let esz = 4;
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for ii in 0..i {
            for hh in 0..h {
                for ww in 0..w {
                    let src = ((oo * i + ii) * h + hh) * w + ww;
                    let dst = ((oo * h + hh) * w + ww) * i + ii;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

/// Transpose depthwise filter from OIHW [O, 1, H, W] → TFLite [1, H, W, O].
fn transpose_oihw_to_depthwise(data: &[u8], o: usize, h: usize, w: usize) -> Vec<u8> {
    let esz = 4;
    let mut out = vec![0u8; data.len()];
    for oo in 0..o {
        for hh in 0..h {
            for ww in 0..w {
                let src = (oo * h + hh) * w + ww;
                let dst = (hh * w + ww) * o + oo;
                out[dst * esz..(dst + 1) * esz]
                    .copy_from_slice(&data[src * esz..(src + 1) * esz]);
            }
        }
    }
    out
}

/// Transpose convTranspose2d filter from IOHW [C_in, C_out, H, W] → OHWI [C_out, H, W, C_in].
fn transpose_iohw_to_ohwi(data: &[u8], c_in: usize, c_out: usize, h: usize, w: usize) -> Vec<u8> {
    let esz = 4;
    let mut out = vec![0u8; data.len()];
    for ci in 0..c_in {
        for co in 0..c_out {
            for hh in 0..h {
                for ww in 0..w {
                    let src = ((ci * c_out + co) * h + hh) * w + ww;
                    let dst = ((co * h + hh) * w + ww) * c_in + ci;
                    out[dst * esz..(dst + 1) * esz]
                        .copy_from_slice(&data[src * esz..(src + 1) * esz]);
                }
            }
        }
    }
    out
}

// ── TFLite schema field offsets (from tensorflow/lite/schema/schema.fbs) ──
//
// OperatorCode: deprecated_builtin_code=0(byte), custom_code=2(string),
//               version=4(int), builtin_code=6(int32)
// Tensor:       shape=0([int]), type=2(byte), buffer=4(uint), name=6(string),
//               quantization=8, is_variable=10(bool), sparsity=12,
//               shape_signature=14, has_rank=16(bool)
// Buffer:       data=0([byte])
// Operator:     opcode_index=0(uint), inputs=1([int]), outputs=2([int]),
//               builtin_options_type=3(u8), builtin_options=4(uoffset)
// SubGraph:     tensors=0([Tensor]), inputs=2([int]), outputs=4([int]),
//               operators=6([Operator]), name=8(string)
// Model:        version=0(uint), operator_codes=2([OperatorCode]),
//               subgraphs=4([SubGraph]), description=6(string),
//               buffers=8([Buffer])
//
// Conv2DOptions:   padding=0(u8), stride_w=2(int), stride_h=4(int),
//                  fused_activation=6(u8), dilation_w=8(int), dilation_h=10(int)
// Pool2DOptions:   padding=0(u8), stride_w=2(int), stride_h=4(int),
//                  filter_w=6(int), filter_h=8(int), fused_activation=10(u8)
// TransposeConv:   padding=0(u8), stride_w=2(int), stride_h=4(int),
//                  fused_activation=6(u8)
// ResizeBilinear:  align_corners=0(bool), half_pixel_centers=2(bool)
// ResizeNearest:   align_corners=0(bool), half_pixel_centers=2(bool)
// BatchMatMul:     adj_x=0(bool), adj_y=2(bool)
// Softmax:         beta=0(float)
// Concatenation:   axis=0(int)
// Pad:             (no fields)

fn build_operator_code(
    fbb: &mut FlatBufferBuilder,
    builtin_code: i32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let dep = tfl::op::deprecated(builtin_code);
    let t = fbb.start_table();
    fbb.push_slot::<u8>(4, dep as u8, 0);
    fbb.push_slot::<i32>(8, 1, 1); // version = 1
    fbb.push_slot::<i32>(10, builtin_code, 0);
    fbb.end_table(t)
}

fn build_tensor(
    fbb: &mut FlatBufferBuilder,
    shape: &[u32],
    dtype: u8,
    buffer: u32,
    name: &str,
) -> WIPOffset<TableFinishedWIPOffset> {
    let shape_off = fbb.create_vector(&shape.iter().map(|&d| d as i32).collect::<Vec<_>>());
    let name_off = fbb.create_string(name);

    let t = fbb.start_table();
    fbb.push_slot_always(4, shape_off);
    fbb.push_slot::<u8>(6, dtype, 0);
    fbb.push_slot::<u32>(8, buffer, 0);
    fbb.push_slot_always(10, name_off);
    fbb.push_slot::<bool>(14, false, false);
    fbb.end_table(t)
}

fn build_buffer(fbb: &mut FlatBufferBuilder, data: &[u8]) -> WIPOffset<TableFinishedWIPOffset> {
    let data_off = if !data.is_empty() {
        Some(fbb.create_vector(data))
    } else {
        None
    };
    let t = fbb.start_table();
    if let Some(off) = data_off {
        fbb.push_slot_always(4, off);
    }
    fbb.end_table(t)
}

// ── Option table builders ──

fn build_conv2d_options(
    fbb: &mut FlatBufferBuilder,
    padding: u8,
    stride_w: i32,
    stride_h: i32,
    dilation_w: i32,
    dilation_h: i32,
    activation: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot_always::<u8>(4, padding);
    fbb.push_slot_always::<i32>(6, stride_w);
    fbb.push_slot_always::<i32>(8, stride_h);
    fbb.push_slot_always::<u8>(10, activation);
    fbb.push_slot_always::<i32>(12, dilation_w);
    fbb.push_slot_always::<i32>(14, dilation_h);
    fbb.end_table(t)
}

fn build_depthwise_conv2d_options(
    fbb: &mut FlatBufferBuilder,
    padding: u8,
    stride_w: i32,
    stride_h: i32,
    dilation_w: i32,
    dilation_h: i32,
    activation: u8,
    depth_multiplier: i32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot_always::<u8>(4, padding);
    fbb.push_slot_always::<i32>(6, stride_w);
    fbb.push_slot_always::<i32>(8, stride_h);
    fbb.push_slot_always::<i32>(10, depth_multiplier);
    fbb.push_slot_always::<u8>(12, activation);
    fbb.push_slot_always::<i32>(14, dilation_w);
    fbb.push_slot_always::<i32>(16, dilation_h);
    fbb.end_table(t)
}

fn build_pool2d_options(
    fbb: &mut FlatBufferBuilder,
    padding: u8,
    stride_w: i32,
    stride_h: i32,
    filter_w: i32,
    filter_h: i32,
    activation: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot_always::<u8>(4, padding);
    fbb.push_slot_always::<i32>(6, stride_w);
    fbb.push_slot_always::<i32>(8, stride_h);
    fbb.push_slot_always::<i32>(10, filter_w);
    fbb.push_slot_always::<i32>(12, filter_h);
    fbb.push_slot::<u8>(14, activation, tfl::activation::NONE);
    fbb.end_table(t)
}

fn build_transpose_conv_options(
    fbb: &mut FlatBufferBuilder,
    padding: u8,
    stride_w: i32,
    stride_h: i32,
    activation: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot_always::<u8>(4, padding);
    fbb.push_slot_always::<i32>(6, stride_w);
    fbb.push_slot_always::<i32>(8, stride_h);
    fbb.push_slot_always::<u8>(10, activation);
    fbb.end_table(t)
}

fn build_softmax_options(
    fbb: &mut FlatBufferBuilder,
    beta: f32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot_always::<f32>(4, beta);
    fbb.end_table(t)
}

fn build_concatenation_options(
    fbb: &mut FlatBufferBuilder,
    axis: i32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<i32>(4, axis, 0);
    fbb.end_table(t)
}

fn build_resize_bilinear_options(
    fbb: &mut FlatBufferBuilder,
    align_corners: bool,
    half_pixel_centers: bool,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<bool>(4, align_corners, false);
    fbb.push_slot::<bool>(6, half_pixel_centers, false);
    fbb.end_table(t)
}

fn build_batch_matmul_options(
    fbb: &mut FlatBufferBuilder,
    adj_x: bool,
    adj_y: bool,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<bool>(4, adj_x, false);
    fbb.push_slot::<bool>(6, adj_y, false);
    fbb.end_table(t)
}

fn build_pad_options(fbb: &mut FlatBufferBuilder) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.end_table(t)
}

fn build_reshape_options(
    fbb: &mut FlatBufferBuilder,
    new_shape: &[i32],
) -> WIPOffset<TableFinishedWIPOffset> {
    let shape_off = fbb.create_vector(new_shape);
    let t = fbb.start_table();
    fbb.push_slot_always(4, shape_off);
    fbb.end_table(t)
}

fn build_fully_connected_options(
    fbb: &mut FlatBufferBuilder,
    fused_activation: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<u8>(4, fused_activation, tfl::activation::NONE);
    fbb.end_table(t)
}

fn build_squeeze_options(
    fbb: &mut FlatBufferBuilder,
    squeeze_dims: &[i32],
) -> WIPOffset<TableFinishedWIPOffset> {
    let dims_off = fbb.create_vector(squeeze_dims);
    let t = fbb.start_table();
    fbb.push_slot_always(4, dims_off);
    fbb.end_table(t)
}

fn build_split_options(
    fbb: &mut FlatBufferBuilder,
    num_splits: i32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<i32>(4, num_splits, 1);
    fbb.end_table(t)
}

fn build_leaky_relu_options(
    fbb: &mut FlatBufferBuilder,
    alpha: f32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<f32>(4, alpha, 0.0);
    fbb.end_table(t)
}

fn build_cast_options(
    fbb: &mut FlatBufferBuilder,
    in_data_type: u8,
    out_data_type: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<u8>(4, in_data_type, 0);
    fbb.push_slot::<u8>(6, out_data_type, 0);
    fbb.end_table(t)
}

fn build_arg_max_options(
    fbb: &mut FlatBufferBuilder,
    output_type: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<u8>(4, output_type, 2);
    fbb.end_table(t)
}

fn build_arg_min_options(
    fbb: &mut FlatBufferBuilder,
    output_type: u8,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<u8>(4, output_type, 2);
    fbb.end_table(t)
}

fn build_cumsum_options(
    fbb: &mut FlatBufferBuilder,
    exclusive: bool,
    reverse: bool,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<bool>(4, exclusive, false);
    fbb.push_slot::<bool>(6, reverse, false);
    fbb.end_table(t)
}

fn build_reducer_options(
    fbb: &mut FlatBufferBuilder,
    keep_dims: bool,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<bool>(4, keep_dims, false);
    fbb.end_table(t)
}

fn build_gather_options(
    fbb: &mut FlatBufferBuilder,
    axis: i32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    fbb.push_slot::<i32>(4, axis, 0);
    fbb.end_table(t)
}

// ── Operator builder with optional builtin options ──

fn build_operator(
    fbb: &mut FlatBufferBuilder,
    inputs: &[u32],
    outputs: &[u32],
    opcode_index: u32,
    builtin_options_type: u8,
    builtin_options: Option<WIPOffset<TableFinishedWIPOffset>>,
) -> WIPOffset<TableFinishedWIPOffset> {
    let inputs_off = fbb.create_vector(&inputs.iter().map(|&i| i as i32).collect::<Vec<_>>());
    let outputs_off = fbb.create_vector(&outputs.iter().map(|&o| o as i32).collect::<Vec<_>>());

    let t = fbb.start_table();
    // Schema: opcode_index=0(uint), inputs=1([int]), outputs=2([int])
    fbb.push_slot_always(4, opcode_index);
    fbb.push_slot_always(6, inputs_off);
    fbb.push_slot_always(8, outputs_off);
    if builtin_options_type != tfl::builtin_options::NONE {
        fbb.push_slot::<u8>(10, builtin_options_type, 0);
        if let Some(off) = builtin_options {
            fbb.push_slot_always(12, off);
        }
    }
    fbb.end_table(t)
}

fn build_subgraph(
    fbb: &mut FlatBufferBuilder,
    tensors: &[WIPOffset<TableFinishedWIPOffset>],
    inputs: &[u32],
    outputs: &[u32],
    operators: &[WIPOffset<TableFinishedWIPOffset>],
    name: &str,
) -> WIPOffset<TableFinishedWIPOffset> {
    let tensors_off = fbb.create_vector(tensors);
    let inputs_off = fbb.create_vector(&inputs.iter().map(|&i| i as i32).collect::<Vec<_>>());
    let outputs_off = fbb.create_vector(&outputs.iter().map(|&o| o as i32).collect::<Vec<_>>());
    let operators_off = fbb.create_vector(operators);
    let name_off = fbb.create_string(name);

    let t = fbb.start_table();
    fbb.push_slot_always(4, tensors_off);
    fbb.push_slot_always(6, inputs_off);
    fbb.push_slot_always(8, outputs_off);
    fbb.push_slot_always(10, operators_off);
    fbb.push_slot_always(12, name_off);
    fbb.end_table(t)
}

fn build_model(
    fbb: &mut FlatBufferBuilder,
    operator_codes: &[WIPOffset<TableFinishedWIPOffset>],
    subgraphs: &[WIPOffset<TableFinishedWIPOffset>],
    buffers: &[WIPOffset<TableFinishedWIPOffset>],
    description: &str,
) -> WIPOffset<TableFinishedWIPOffset> {
    let codes_off = fbb.create_vector(operator_codes);
    let subgraphs_off = fbb.create_vector(subgraphs);
    let buffers_off = fbb.create_vector(buffers);
    let desc_off = fbb.create_string(description);

    let t = fbb.start_table();
    fbb.push_slot::<u32>(4, 3, 0); // version = 3
    fbb.push_slot_always(6, codes_off);
    fbb.push_slot_always(8, subgraphs_off);
    fbb.push_slot_always(10, desc_off);
    fbb.push_slot_always(12, buffers_off);
    fbb.end_table(t)
}

// ── Op name → TFLite builtin code mapping ──

fn webnn_op_to_tflite(op: &str) -> Option<i32> {
    match op {
        "add" => Some(tfl::op::ADD),
        "sub" => Some(tfl::op::SUB),
        "mul" => Some(tfl::op::MUL),
        "div" => Some(tfl::op::DIV),
        "relu" => Some(tfl::op::RELU),
        "relu6" => Some(tfl::op::RELU6),
        "tanh" => Some(tfl::op::TANH),
        "sigmoid" => Some(tfl::op::LOGISTIC),
        "gelu" => Some(tfl::op::GELU),
        "softmax" => Some(tfl::op::SOFTMAX),
        "reshape" => Some(tfl::op::RESHAPE),
        "transpose" => Some(tfl::op::TRANSPOSE),
        "concat" => Some(tfl::op::CONCATENATION),
        "conv2d" | "conv_2d" => Some(tfl::op::CONV_2D),
        "convTranspose2d" | "conv_transpose2d" => Some(tfl::op::TRANSPOSE_CONV),
        "gemm" => Some(tfl::op::BATCH_MATMUL),
        "matmul" => Some(tfl::op::BATCH_MATMUL),
        "averagePool2d" | "average_pool_2d" => Some(tfl::op::AVERAGE_POOL_2D),
        "maxPool2d" | "max_pool_2d" => Some(tfl::op::MAX_POOL_2D),
        "l2Pool2d" | "l2_pool_2d" => Some(tfl::op::L2_POOL_2D),
        "abs" => Some(tfl::op::ABS),
        "neg" => Some(tfl::op::NEG),
        "sqrt" => Some(tfl::op::SQRT),
        "rsqrt" => Some(tfl::op::RSQRT),
        "max" => Some(tfl::op::MAXIMUM),
        "min" => Some(tfl::op::MINIMUM),
        "hardSwish" | "hard_swish" => Some(tfl::op::HARD_SWISH),
        "exp" => Some(tfl::op::EXP),
        "log" => Some(tfl::op::LOG),
        "sin" => Some(tfl::op::SIN),
        "cos" => Some(tfl::op::COS),
        "ceil" => Some(tfl::op::CEIL),
        "floor" => Some(tfl::op::FLOOR),
        "pad" => Some(tfl::op::PAD),
        "prelu" => Some(tfl::op::PRELU),
        "identity" => Some(tfl::op::RESHAPE),
        "hardSigmoid" | "hard_sigmoid" => Some(tfl::op::HARD_SIGMOID),
        "softplus" => Some(tfl::op::SOFTPLUS),
        "softsign" => Some(tfl::op::SOFTSIGN),
        "negative" => Some(tfl::op::NEG),
        "square" => Some(tfl::op::SQUARE),
        "resample2d" | "resample_2d" => Some(tfl::op::RESIZE_BILINEAR),
        "slice" => Some(tfl::op::SLICE),
        "split" => Some(tfl::op::SPLIT_V),
        "equal" => Some(tfl::op::EQUAL),
        "notEqual" => Some(tfl::op::NOT_EQUAL),
        "greater" => Some(tfl::op::GREATER),
        "greaterOrEqual" => Some(tfl::op::GREATER_EQUAL),
        "lesser" => Some(tfl::op::LESS),
        "lesserOrEqual" => Some(tfl::op::LESS_EQUAL),
        "logicalAnd" => Some(tfl::op::LOGICAL_AND),
        "logicalOr" => Some(tfl::op::LOGICAL_OR),
        "logicalNot" => Some(tfl::op::LOGICAL_NOT),
        "leakyRelu" | "leaky_relu" => Some(tfl::op::LEAKY_RELU),
        "tile" => Some(tfl::op::TILE),
        "reduceMean" | "reduce_mean" => Some(tfl::op::MEAN),
        "reduceSum" | "reduce_sum" => Some(tfl::op::SUM),
        "reduceMax" | "reduce_max" => Some(tfl::op::REDUCE_MAX),
        "reduceMin" | "reduce_min" => Some(tfl::op::REDUCE_MIN),
        "reduceProduct" | "reduce_product" => Some(tfl::op::REDUCE_PROD),
        "argMax" | "arg_max" => Some(tfl::op::ARG_MAX),
        "argMin" | "arg_min" => Some(tfl::op::ARG_MIN),
        "gather" => Some(tfl::op::GATHER),
        "gatherNd" | "gather_nd" => Some(tfl::op::GATHER_ND),
        "scatterNd" | "scatter_nd" => Some(tfl::op::SCATTER_ND),
        "elu" => Some(tfl::op::ELU),
        "cast" => Some(tfl::op::CAST),
        "expand" | "expand_dims" => Some(tfl::op::EXPAND_DIMS),
        "squeeze" => Some(tfl::op::SQUEEZE),
        "reverse" | "reverse_v2" => Some(tfl::op::REVERSE_V2),
        "cumulativeSum" | "cumulative_sum" | "cumsum" => Some(tfl::op::CUM_SUM),
        "sign" => Some(tfl::op::SIGN),
        "pow" => Some(tfl::op::POW),
        "where" => Some(tfl::op::SELECT),
        "quantizeLinear" | "quantize_linear" => Some(tfl::op::QUANTIZE),
        "tan" => Some(tfl::op::DIV),
        _ => None,
    }
}

// ── Model builder context ──

struct GraphCompiler {
    tensor_type: HashMap<String, u8>,
    tensor_shape: HashMap<String, Vec<u32>>,
    tensor_idx: HashMap<String, u32>,
    op_codes: Vec<i32>,
    op_code_idx: HashMap<i32, u32>,
    operators: Vec<WIPOffset<TableFinishedWIPOffset>>,
    next_tensor: u32,
}

impl GraphCompiler {
    fn new() -> Self {
        Self {
            tensor_type: HashMap::new(),
            tensor_shape: HashMap::new(),
            tensor_idx: HashMap::new(),
            op_codes: Vec::new(),
            op_code_idx: HashMap::new(),
            operators: Vec::new(),
            next_tensor: 0,
        }
    }

    fn ensure_tensor(&mut self, name: &str, dtype: u8, shape: &[u32]) -> u32 {
        if let Some(&idx) = self.tensor_idx.get(name) {
            return idx;
        }
        let idx = self.next_tensor;
        self.next_tensor += 1;
        self.tensor_idx.insert(name.to_string(), idx);
        self.tensor_type.insert(name.to_string(), dtype);
        self.tensor_shape.insert(name.to_string(), shape.to_vec());
        idx
    }

    fn tensor_id(&self, name: &str) -> u32 {
        self.tensor_idx[name]
    }

    fn reserve_temporary(&mut self, dtype: u8, shape: Vec<u32>) -> (u32, String) {
        let name = format!("_tmp_{}", self.next_tensor);
        let idx = self.next_tensor;
        self.next_tensor += 1;
        self.tensor_idx.insert(name.clone(), idx);
        self.tensor_type.insert(name.clone(), dtype);
        self.tensor_shape.insert(name.clone(), shape);
        (idx, name)
    }

    fn get_or_create_op_code(&mut self, code: i32) -> u32 {
        if let Some(&idx) = self.op_code_idx.get(&code) {
            return idx;
        }
        let idx = self.op_codes.len() as u32;
        self.op_codes.push(code);
        self.op_code_idx.insert(code, idx);
        idx
    }

    fn emit(
        &mut self,
        fbb: &mut FlatBufferBuilder,
        code: i32,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
        opt_type: u8,
        opt: Option<WIPOffset<TableFinishedWIPOffset>>,
    ) {
        let op_idx = self.get_or_create_op_code(code);
        let off = build_operator(fbb, &inputs, &outputs, op_idx, opt_type, opt);
        self.operators.push(off);
    }

    // ── Emulated ops ──

    fn name_for_tensor(&self, idx: u32) -> &str {
        self.tensor_idx
            .iter()
            .find(|(_, v)| **v == idx)
            .map(|(n, _)| n.as_str())
            .unwrap()
    }

    fn emit_batch_normalization(
        &mut self,
        fbb: &mut FlatBufferBuilder,
        input: u32,
        mean: u32,
        variance: u32,
        scale: Option<u32>,
        bias: Option<u32>,
        _epsilon: f32,
        output: u32,
    ) {
        let in_name = self.name_for_tensor(input).to_string();
        let dtype = self.tensor_type[&in_name];
        let shape = self.tensor_shape[&in_name].clone();

        // input - mean
        let (sub_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(fbb, tfl::op::SUB, vec![input, mean], vec![sub_out], 0, None);

        // variance + epsilon
        let (eps_idx, _) = self.reserve_temporary(dtype, vec![]);
        let (var_add_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::ADD,
            vec![variance, eps_idx],
            vec![var_add_out],
            0,
            None,
        );

        // sqrt(variance + epsilon)
        let (sqrt_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::SQRT,
            vec![var_add_out],
            vec![sqrt_out],
            0,
            None,
        );

        // (input - mean) / sqrt(variance + epsilon)
        let mut current = sub_out;
        let (div_out, _) = if scale.is_some() || bias.is_some() {
            self.reserve_temporary(dtype, shape.clone())
        } else {
            (output, String::new())
        };
        self.emit(
            fbb,
            tfl::op::DIV,
            vec![current, sqrt_out],
            vec![div_out],
            0,
            None,
        );
        current = div_out;

        // scale * result
        if let Some(s) = scale {
            let (mul_out, _) = if bias.is_some() {
                self.reserve_temporary(dtype, shape.clone())
            } else {
                (output, String::new())
            };
            self.emit(fbb, tfl::op::MUL, vec![s, current], vec![mul_out], 0, None);
            current = mul_out;
        }

        // result + bias
        if let Some(b) = bias {
            self.emit(fbb, tfl::op::ADD, vec![current, b], vec![output], 0, None);
        }
    }

    fn emit_layer_normalization(
        &mut self,
        fbb: &mut FlatBufferBuilder,
        input: u32,
        scale: Option<u32>,
        bias: Option<u32>,
        axes: &[u32],
        _epsilon: f32,
        output: u32,
    ) {
        let in_name = self.name_for_tensor(input).to_string();
        let dtype = self.tensor_type[&in_name];
        let shape = self.tensor_shape[&in_name].clone();

        // Compute mean via ReduceMean
        let (mean_out, _) = self.reserve_temporary(dtype, shape.clone());
        let (axes_idx, _) = self.reserve_temporary(2, axes.iter().map(|&a| a).collect());
        self.emit(
            fbb,
            tfl::op::MEAN,
            vec![input, axes_idx],
            vec![mean_out],
            0,
            None,
        );

        // input - mean
        let (centered_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::SUB,
            vec![input, mean_out],
            vec![centered_out],
            0,
            None,
        );

        // variance = reduceMean((input - mean)^2)
        let (sq_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::MUL,
            vec![centered_out, centered_out],
            vec![sq_out],
            0,
            None,
        );
        let (var_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::MEAN,
            vec![sq_out, axes_idx],
            vec![var_out],
            0,
            None,
        );

        // variance + epsilon
        let (eps_idx, _) = self.reserve_temporary(dtype, vec![]);
        let (var_add_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::ADD,
            vec![var_out, eps_idx],
            vec![var_add_out],
            0,
            None,
        );

        // sqrt(variance + epsilon)
        let (std_out, _) = self.reserve_temporary(dtype, shape.clone());
        self.emit(
            fbb,
            tfl::op::SQRT,
            vec![var_add_out],
            vec![std_out],
            0,
            None,
        );

        // (input - mean) / sqrt(variance + epsilon)
        let mut current = centered_out;
        let (norm_out, _) = if scale.is_some() || bias.is_some() {
            self.reserve_temporary(dtype, shape.clone())
        } else {
            (output, String::new())
        };
        self.emit(
            fbb,
            tfl::op::DIV,
            vec![current, std_out],
            vec![norm_out],
            0,
            None,
        );
        current = norm_out;

        // scale * result
        if let Some(s) = scale {
            let (mul_out, _) = if bias.is_some() {
                self.reserve_temporary(dtype, shape.clone())
            } else {
                (output, String::new())
            };
            self.emit(fbb, tfl::op::MUL, vec![s, current], vec![mul_out], 0, None);
            current = mul_out;
        }

        // result + bias
        if let Some(b) = bias {
            self.emit(fbb, tfl::op::ADD, vec![current, b], vec![output], 0, None);
        }
    }
}

// ── NHWC transpose insertion (Chromium-style pre-processing) ──

const NHWC_SENSITIVE_OPS: &[&str] = &[
    "conv2d", "conv_2d", "convTranspose2d", "conv_transpose2d",
    "maxPool2d", "max_pool_2d", "averagePool2d", "average_pool_2d",
    "l2Pool2d", "l2_pool_2d", "resample2d", "resample_2d",
];

const LAYOUT_BOUNDARY_OPS: &[&str] = &[
    "transpose", "reshape", "concat",
];

/// Insert NCHW↔NHWC transpose ops around NHWC-sensitive operations.
///
/// For each conv2d/pool2d/resample2d/convTranspose2d with a 4-D input:
///  1. Insert TRANSPOSE([0,2,3,1]) before the op (NCHW → NHWC)
///  2. Insert TRANSPOSE([0,3,1,2]) after  the op (NHWC → NCHW)
///  3. For conv2d: transpose filter data OIHW → OHWI
///
/// Returns (transformed_nodes, extra_filter_data).
fn insert_nhwc_transposes(
    nodes: &[GraphNode],
    constant_data: &std::collections::HashMap<&str, &[u8]>,
    input_infos: &[(String, Vec<u32>, DataType)],
) -> (Vec<GraphNode>, std::collections::HashMap<String, Vec<u8>>) {
    let mut result: Vec<GraphNode> = Vec::with_capacity(nodes.len() * 2);
    let mut extra_data: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();

    // Build forward shape map: output_name → (shape, DataType), using owned keys.
    let mut shape_map: std::collections::HashMap<String, (Vec<u32>, DataType)> =
        std::collections::HashMap::new();
    for node in nodes {
        shape_map.insert(
            node.output.clone(),
            (node.desc.shape.clone(), node.desc.data_type),
        );
    }
    // Seed with graph input shapes.
    for (name, shape, dt) in input_infos {
        shape_map.insert(name.clone(), (shape.clone(), *dt));
    }

    // Set of all input names (which are output names of some node).
    let _nhwc_set: std::collections::HashSet<String> = std::collections::HashSet::from_iter(
        nodes
            .iter()
            .flat_map(|n| n.inputs.iter())
            .cloned(),
    );
    let mut is_nhwc: std::collections::HashSet<String> = std::collections::HashSet::new();

    for node in nodes {
        let is_nhwc_op = NHWC_SENSITIVE_OPS.contains(&node.op.as_str());
        let rank = node.desc.shape.len();

        if !is_nhwc_op || rank != 4 {
            result.push(node.clone());
            continue;
        }

        let in_name = &node.inputs[0];
        let in_shape = shape_map.get(in_name).cloned();
        let Some((nchw_in, in_dtype)) = in_shape else {
            result.push(node.clone());
            continue;
        };
        if nchw_in.len() != 4 {
            result.push(node.clone());
            continue;
        }

        let nhwc_in: Vec<u32> = vec![nchw_in[0], nchw_in[2], nchw_in[3], nchw_in[1]];
        let nchw_out = node.desc.shape.clone();
        let nhwc_out: Vec<u32> = vec![nchw_out[0], nchw_out[2], nchw_out[3], nchw_out[1]];
        let out_dtype = node.desc.data_type;

        log::error!("NHWC preproc: op={} in_name={:?} nchw_in={:?} nhwc_in={:?} nchw_out={:?} nhwc_out={:?}", node.op, in_name, nchw_in, nhwc_in, nchw_out, nhwc_out);

        // ── Input TRANSPOSE ──
        let nhwc_in_name: String;
        let producer_op = nodes.iter().find(|n| n.output == *in_name);

        log::error!("NHWC preproc: producer_op={} for in_name={:?}", producer_op.map_or("(none)", |p| p.op.as_str()), in_name);

        if producer_op.map_or(false, |p| LAYOUT_BOUNDARY_OPS.contains(&p.op.as_str())) {
            nhwc_in_name = format!("{in_name}_nhwc");
            let perm = vec![0u32, 2, 3, 1];
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("perm_len".into(), perm.len() as f64);
            for (i, &v) in perm.iter().enumerate() {
                attrs.insert(format!("perm_{i}"), v as f64);
            }
            result.push(GraphNode {
                op: "transpose".into(),
                inputs: vec![in_name.clone()],
                output: nhwc_in_name.clone(),
                desc: TensorDesc {
                    data_type: in_dtype,
                    shape: nhwc_in.clone(),
                },
                attrs,
                data: None,
            });
            is_nhwc.insert(nhwc_in_name.clone());
        } else if is_nhwc.contains(in_name) {
            nhwc_in_name = in_name.clone();
            log::error!("NHWC preproc: input {} already nhwc, reusing directly", in_name);
        } else {
            // Input is NCHW (from graph input, elementwise op like relu, etc.)
            // Insert explicit TRANSPOSE to convert to NHWC.
            nhwc_in_name = format!("{in_name}_nhwc");
            let perm = vec![0u32, 2, 3, 1];
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("perm_len".into(), perm.len() as f64);
            for (i, &v) in perm.iter().enumerate() {
                attrs.insert(format!("perm_{i}"), v as f64);
            }
            result.push(GraphNode {
                op: "transpose".into(),
                inputs: vec![in_name.clone()],
                output: nhwc_in_name.clone(),
                desc: TensorDesc {
                    data_type: in_dtype,
                    shape: nhwc_in.clone(),
                },
                attrs,
                data: None,
            });
            is_nhwc.insert(nhwc_in_name.clone());
            log::error!("NHWC preproc: TRANSPOSE {} -> {} perm={:?}", in_name, nhwc_in_name, perm);
        }

        // ── Modified NHWC op ──
        let nhwc_tmp_out = format!("{}_nhwc_tmp", node.output);
        let mut nhwc_node = node.clone();
        nhwc_node.inputs[0] = nhwc_in_name.clone();
        nhwc_node.output = nhwc_tmp_out.clone();
        nhwc_node.desc = TensorDesc {
            data_type: out_dtype,
            shape: nhwc_out.clone(),
        };
        is_nhwc.insert(nhwc_tmp_out.clone());
        shape_map.insert(nhwc_tmp_out.clone(), (nhwc_out.clone(), out_dtype));

        // For conv2d: rename filter + transpose data (OIHW → OHWI)
        // For convTranspose2d: rename filter + transpose data (IOHW → OHWI)
        let is_conv_transpose = node.op == "convTranspose2d" || node.op == "conv_transpose2d";
        if node.op == "conv2d" || node.op == "conv_2d" || is_conv_transpose {
            let filt_name = &node.inputs[1];
            if let Some(filt_shape) = shape_map.get(filt_name).cloned() {
                if filt_shape.0.len() == 4 {
                    let (f_shape, _) = &filt_shape;
                    let nhwc_filt_name = format!("{filt_name}_nhwc");
                    nhwc_node.inputs[1] = nhwc_filt_name.clone();
                    let groups = node.attrs.get("groups").copied().unwrap_or(1.0) as usize;
                    let is_depthwise = !is_conv_transpose && groups > 1 && f_shape[1] == 1;
                    let (nhwc_filt_shape, filter_perm, _filter_trans_fn) = if is_conv_transpose {
                        // IOHW [C_in, C_out, H, W] → OHWI [C_out, H, W, C_in]
                        let perm = vec![1u32, 2, 3, 0];
                        let shape = vec![f_shape[1], f_shape[2], f_shape[3], f_shape[0]];
                        (shape, perm, "iohw_to_ohwi" as &str)
                    } else if is_depthwise {
                        let perm = vec![0u32, 2, 3, 1];
                        let shape = vec![1, f_shape[2], f_shape[3], f_shape[0]];
                        (shape, perm, "depthwise" as &str)
                    } else {
                        let perm = vec![0u32, 2, 3, 1];
                        let shape = vec![f_shape[0], f_shape[2], f_shape[3], f_shape[1]];
                        (shape, perm, "ohwi" as &str)
                    };
                    let mut has_transposed = false;
                    if let Some(fdata) = constant_data.get(filt_name.as_str()) {
                        let o = f_shape[0] as usize;
                        let i = f_shape[1] as usize;
                        let h = f_shape[2] as usize;
                        let w = f_shape[3] as usize;
                        let trans = if is_conv_transpose {
                            transpose_iohw_to_ohwi(fdata, i, o, h, w)
                        } else if is_depthwise {
                            transpose_oihw_to_depthwise(fdata, o, h, w)
                        } else {
                            transpose_oihw_to_ohwi(fdata, o, i, h, w)
                        };
                        extra_data.insert(nhwc_filt_name.clone(), trans);
                        has_transposed = true;
                    }
                    shape_map
                        .insert(nhwc_filt_name.clone(), (nhwc_filt_shape.clone(), out_dtype));
                    if has_transposed {
                        let transposed_data =
                            extra_data.get(&nhwc_filt_name).cloned().unwrap_or_default();
                        result.push(GraphNode {
                            op: "constant".into(),
                            inputs: vec![],
                            output: nhwc_filt_name.clone(),
                            desc: TensorDesc {
                                data_type: out_dtype,
                                shape: nhwc_filt_shape,
                            },
                            attrs: std::collections::HashMap::new(),
                            data: Some(transposed_data),
                        });
                    } else {
                        // Filter is a graph input (no embedded data).
                        // Insert a TRANSPOSE op to convert filter layout at runtime.
                        let mut attrs = std::collections::HashMap::new();
                        attrs.insert("perm_len".into(), filter_perm.len() as f64);
                        for (i, &v) in filter_perm.iter().enumerate() {
                            attrs.insert(format!("perm_{i}"), v as f64);
                        }
                        result.push(GraphNode {
                            op: "transpose".into(),
                            inputs: vec![filt_name.clone()],
                            output: nhwc_filt_name.clone(),
                            desc: TensorDesc {
                                data_type: out_dtype,
                                shape: nhwc_filt_shape,
                            },
                            attrs,
                            data: None,
                        });
                    }
                }
            }
        }

        result.push(nhwc_node);

        // ── Output TRANSPOSE (NHWC → NCHW) ──
        let perm = vec![0u32, 3, 1, 2];
        let mut out_attrs = std::collections::HashMap::new();
        out_attrs.insert("perm_len".into(), perm.len() as f64);
        for (i, &v) in perm.iter().enumerate() {
            out_attrs.insert(format!("perm_{i}"), v as f64);
        }
        result.push(GraphNode {
            op: "transpose".into(),
            inputs: vec![nhwc_tmp_out],
            output: node.output.clone(),
            desc: TensorDesc {
                data_type: out_dtype,
                shape: nchw_out,
            },
            attrs: out_attrs,
            data: None,
        });
    }

    (result, extra_data)
}

// ── Transpose elimination (Chromium Phase 2) ──

const LAYOUT_AGNOSTIC_UNARY_OPS: &[&str] = &[
    "relu", "relu6", "sigmoid", "tanh", "elu", "gelu", "hardSwish",
    "hardSigmoid", "softplus", "softsign", "leakyRelu", "clamp",
    "abs", "ceil", "floor", "negative", "identity", "exp", "log",
    "cos", "sin", "sqrt", "square", "cast",
];

fn is_nhwc_to_nchw(node: &GraphNode) -> bool {
    if node.op != "transpose" {
        return false;
    }
    let n = node.attrs.get("perm_len").copied().unwrap_or(0.0) as usize;
    n == 4
        && node.attrs.get("perm_0").copied().unwrap_or(-1.0) as i32 == 0
        && node.attrs.get("perm_1").copied().unwrap_or(-1.0) as i32 == 3
        && node.attrs.get("perm_2").copied().unwrap_or(-1.0) as i32 == 1
        && node.attrs.get("perm_3").copied().unwrap_or(-1.0) as i32 == 2
}

fn is_nchw_to_nhwc(node: &GraphNode) -> bool {
    if node.op != "transpose" {
        return false;
    }
    let n = node.attrs.get("perm_len").copied().unwrap_or(0.0) as usize;
    n == 4
        && node.attrs.get("perm_0").copied().unwrap_or(-1.0) as i32 == 0
        && node.attrs.get("perm_1").copied().unwrap_or(-1.0) as i32 == 2
        && node.attrs.get("perm_2").copied().unwrap_or(-1.0) as i32 == 3
        && node.attrs.get("perm_3").copied().unwrap_or(-1.0) as i32 == 1
}

fn nchw_shape_to_nhwc(shape: &[u32]) -> Vec<u32> {
    if shape.len() != 4 {
        return shape.to_vec();
    }
    vec![shape[0], shape[2], shape[3], shape[1]]
}

/// Eliminate redundant NCHW↔NHWC transpose pairs around layout-agnostic ops.
///
/// Pattern: T_out(NHWC→NCHW) → [unary ops] → T_in(NCHW→NHWC)
/// Becomes: [unary ops in NHWC] (transposes removed)
///
/// This is the Chromium TransposeEliminationTransformer logic: when a
/// NHWC→NCHW transpose's NCHW output flows only through layout-agnostic
/// unary ops and then into a NCHW→NHWC transpose, both transposes can
/// be removed and the ops between them stay in NHWC.
fn eliminate_transpose_pairs(mut nodes: Vec<GraphNode>, graph_outputs: &[String]) -> Vec<GraphNode> {
    let graph_output_set: std::collections::HashSet<String> = graph_outputs.iter().cloned().collect();

    loop {
        let mut output_to_idx: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            output_to_idx.insert(node.output.clone(), i);
        }

        let mut consumers: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            for input in &node.inputs {
                consumers.entry(input.clone()).or_default().push(i);
            }
        }

        let mut found = false;
        let mut t1_idx: usize = 0;
        let mut t2_idx: usize = 0;
        let mut chain: Vec<usize> = Vec::new();
        let mut t1_input_name: String = String::new();
        let mut t2_output_name: String = String::new();
        let mut replacement_name: String = String::new();

        for i in 0..nodes.len() {
            if !is_nhwc_to_nchw(&nodes[i]) {
                continue;
            }

            let t1_out = nodes[i].output.clone();
            let t1_in = nodes[i].inputs[0].clone();

            let t1_cons = consumers.get(&t1_out).map(|v| v.len()).unwrap_or(0);
            if t1_cons != 1 {
                continue;
            }

            let first_idx = consumers.get(&t1_out).unwrap()[0];

            let mut cur_idx = first_idx;
            let mut cur_chain: Vec<usize> = Vec::new();
            let mut reached_t2 = false;
            let mut t2_candidate: usize = 0;

            loop {
                if is_nchw_to_nhwc(&nodes[cur_idx]) {
                    reached_t2 = true;
                    t2_candidate = cur_idx;
                    break;
                }
                if LAYOUT_AGNOSTIC_UNARY_OPS.contains(&nodes[cur_idx].op.as_str())
                    && nodes[cur_idx].inputs.len() == 1
                {
                    cur_chain.push(cur_idx);
                    let op_out = nodes[cur_idx].output.clone();
                    let op_cons = consumers.get(&op_out).map(|v| v.len()).unwrap_or(0);
                    if op_cons != 1 {
                        break;
                    }
                    cur_idx = consumers.get(&op_out).unwrap()[0];
                } else {
                    break;
                }
            }

            if reached_t2 {
                // Don't eliminate if T_in produces a graph output —
                // the user explicitly requested that tensor.
                if graph_output_set.contains(&nodes[t2_candidate].output) {
                    continue;
                }
                // Don't eliminate if any chain node is a graph output.
                let chain_is_output = cur_chain.iter().any(|&ci| {
                    graph_output_set.contains(&nodes[ci].output)
                });
                if chain_is_output {
                    continue;
                }
                found = true;
                t1_idx = i;
                t2_idx = t2_candidate;
                chain = cur_chain;
                t1_input_name = t1_in;
                t2_output_name = nodes[t2_candidate].output.clone();
                if chain.is_empty() {
                    replacement_name = t1_input_name.clone();
                } else {
                    replacement_name = nodes[t2_idx].inputs[0].clone();
                }
                break;
            }
        }

        if !found {
            break;
        }

        let chain_ops: Vec<&str> = chain.iter().map(|&i| nodes[i].op.as_str()).collect();
        log::error!(
            "Transpose elimination: removing T_out({}) → [{}] → T_in({}), replacing {} with {}",
            nodes[t1_idx].output,
            chain_ops.join(" → "),
            nodes[t2_idx].output,
            t2_output_name,
            replacement_name,
        );

        if !chain.is_empty() {
            let t1_out_name = nodes[t1_idx].output.clone();
            for input in nodes[chain[0]].inputs.iter_mut() {
                if *input == t1_out_name {
                    *input = t1_input_name.clone();
                }
            }
        }

        for &ci in &chain {
            let nchw = nodes[ci].desc.shape.clone();
            if nchw.len() == 4 {
                nodes[ci].desc.shape = nchw_shape_to_nhwc(&nchw);
            }
        }

        for node in nodes.iter_mut() {
            for input in node.inputs.iter_mut() {
                if *input == t2_output_name {
                    *input = replacement_name.clone();
                }
            }
        }

        let mut kept = Vec::new();
        for (i, node) in nodes.into_iter().enumerate() {
            if i != t1_idx && i != t2_idx {
                kept.push(node);
            }
        }
        nodes = kept;
    }

    nodes
}

/// Compile a WebNN graph into a TFLite flatbuffer model.
///
/// Returns an owned byte buffer suitable for `litert::Model::from_bytes()`.
pub fn compile(nodes: &[GraphNode]) -> Result<Vec<u8>, String> {
    let result = compile_with_input_infos(nodes, &[], &[])?;
    Ok(result.flatbuf)
}

/// Compile a WebNN graph into a TFLite flatbuffer model.
///
/// `input_infos` provides authoritative (name, shape, dtype) for graph input tensors,
/// overriding the node output shape that `GraphNode.desc.shape` incorrectly carries.
pub struct CompileResult {
    pub flatbuf: Vec<u8>,
    pub nhwc_inputs: Vec<String>,
    pub nhwc_outputs: Vec<String>,
}

pub fn compile_with_input_infos(
    nodes: &[GraphNode],
    input_infos: &[(String, Vec<u32>, DataType)],
    output_names: &[String],
) -> Result<CompileResult, String> {
    if nodes.is_empty() {
        return Err("Cannot compile empty graph".to_string());
    }

    let mut fbb = FlatBufferBuilder::new();
    let mut g = GraphCompiler::new();

    // Pre-scan constant nodes to collect buffer data.
    let mut constant_data: std::collections::HashMap<&str, &[u8]> =
        std::collections::HashMap::new();
    for node in nodes {
        if node.op == "constant" {
            if let Some(ref data) = node.data {
                constant_data.insert(&node.output, data.as_slice());
            }
        }
    }

    // Chromium-style NHWC transpose insertion.
    let (transformed_nodes, nhwc_extra_data) = insert_nhwc_transposes(nodes, &constant_data, input_infos);

    // Chromium-style transpose elimination: remove redundant NCHW↔NHWC
    // transpose pairs around layout-agnostic unary ops (relu, sigmoid, etc.)
    let transformed_nodes = eliminate_transpose_pairs(transformed_nodes, output_names);

    let nodes = &transformed_nodes;
    let mut extra_constant_data: std::collections::HashMap<String, Vec<u8>> = nhwc_extra_data;

    let input_shape_map: std::collections::HashMap<&str, (Vec<u32>, DataType)> = input_infos
        .iter()
        .map(|(name, shape, dt)| (name.as_str(), (shape.clone(), *dt)))
        .collect();

    let ensure_input =
        |g: &mut GraphCompiler, name: &str, fallback_dtype: u8, fallback_shape: &[u32]| {
            if let Some((s, dt)) = input_shape_map.get(name) {
                g.ensure_tensor(name, webnn_type_to_tflite(*dt), s)
            } else {
                g.ensure_tensor(name, fallback_dtype, fallback_shape)
            }
        };

    let mut split_groups: std::collections::HashMap<u64, Vec<usize>> =
        std::collections::HashMap::new();
    let mut split_skip: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, node) in nodes.iter().enumerate() {
        if node.op == "split" {
            if let Some(&gid) = node.attrs.get("split_group") {
                split_groups.entry(gid as u64).or_default().push(i);
            }
        }
    }
    for (_, indices) in &split_groups {
        for &idx in &indices[1..] {
            split_skip.insert(idx);
        }
    }

    for (node_idx, node) in nodes.iter().enumerate() {
        let dtype = webnn_type_to_tflite(node.desc.data_type);
        let shape = node.desc.shape.clone();

        match node.op.as_str() {
            "dequantizeLinear" | "dequantize_linear" => {
                if node.inputs.len() < 3 {
                    return Err("dequantizeLinear needs input, scale, zeroPoint".to_string());
                }
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                let scale_idx = ensure_input(
                    &mut g,
                    &node.inputs[1],
                    webnn_type_to_tflite(DataType::Float32),
                    &shape,
                );
                let zp_idx = ensure_input(&mut g, &node.inputs[2], dtype, &shape);
                let (cast_input, _) =
                    g.reserve_temporary(webnn_type_to_tflite(DataType::Float32), shape.clone());
                let cast_in_opt =
                    build_cast_options(&mut fbb, dtype, webnn_type_to_tflite(DataType::Float32));
                g.emit(
                    &mut fbb,
                    tfl::op::CAST,
                    vec![input_idx],
                    vec![cast_input],
                    tfl::builtin_options::CAST,
                    Some(cast_in_opt),
                );
                let (cast_zp, _) =
                    g.reserve_temporary(webnn_type_to_tflite(DataType::Float32), shape.clone());
                let cast_zp_opt =
                    build_cast_options(&mut fbb, dtype, webnn_type_to_tflite(DataType::Float32));
                g.emit(
                    &mut fbb,
                    tfl::op::CAST,
                    vec![zp_idx],
                    vec![cast_zp],
                    tfl::builtin_options::CAST,
                    Some(cast_zp_opt),
                );
                let (sub_out, _) =
                    g.reserve_temporary(webnn_type_to_tflite(DataType::Float32), shape.clone());
                g.emit(
                    &mut fbb,
                    tfl::op::SUB,
                    vec![cast_input, cast_zp],
                    vec![sub_out],
                    0,
                    None,
                );
                let out_idx = g.ensure_tensor(
                    &node.output,
                    webnn_type_to_tflite(DataType::Float32),
                    &shape,
                );
                g.emit(
                    &mut fbb,
                    tfl::op::MUL,
                    vec![sub_out, scale_idx],
                    vec![out_idx],
                    0,
                    None,
                );
                continue;
            },
            "batchNormalization" | "batch_normalization" => {
                if node.inputs.len() < 3 {
                    return Err("batchNormalization needs at least 3 inputs".to_string());
                }
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                let mean_idx = ensure_input(&mut g, &node.inputs[1], dtype, &shape);
                let var_idx = ensure_input(&mut g, &node.inputs[2], dtype, &shape);
                let scale_idx = if node.inputs.len() > 3 {
                    Some(ensure_input(&mut g, &node.inputs[3], dtype, &shape))
                } else {
                    None
                };
                let bias_idx = if node.inputs.len() > 4 {
                    Some(ensure_input(&mut g, &node.inputs[4], dtype, &shape))
                } else {
                    None
                };
                let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                g.emit_batch_normalization(
                    &mut fbb, input_idx, mean_idx, var_idx, scale_idx, bias_idx, 1e-5, out_idx,
                );
                continue;
            },
            "tan" => {
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                let (sin_out, _) = g.reserve_temporary(dtype, shape.clone());
                g.emit(
                    &mut fbb,
                    tfl::op::SIN,
                    vec![input_idx],
                    vec![sin_out],
                    0,
                    None,
                );
                let (cos_out, _) = g.reserve_temporary(dtype, shape.clone());
                g.emit(
                    &mut fbb,
                    tfl::op::COS,
                    vec![input_idx],
                    vec![cos_out],
                    0,
                    None,
                );
                let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                g.emit(
                    &mut fbb,
                    tfl::op::DIV,
                    vec![sin_out, cos_out],
                    vec![out_idx],
                    0,
                    None,
                );
                continue;
            },
            "clamp" => {
                let min_val = node.attrs.get("minValue").copied();
                let max_val = node.attrs.get("maxValue").copied();
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                if min_val.is_none() && max_val.is_none() {
                    let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                    g.emit(
                        &mut fbb,
                        tfl::op::RELU6,
                        vec![input_idx],
                        vec![out_idx],
                        0,
                        None,
                    );
                    continue;
                }
                let has_both = min_val.is_some() && max_val.is_some();
                let mut current = input_idx;
                if let Some(max) = max_val {
                    let (max_idx, max_name) = g.reserve_temporary(dtype, vec![]);
                    extra_constant_data.insert(max_name, (max as f32).to_le_bytes().to_vec());
                    if has_both {
                        let (mid_out, _) = g.reserve_temporary(dtype, shape.clone());
                        g.emit(
                            &mut fbb,
                            tfl::op::MINIMUM,
                            vec![current, max_idx],
                            vec![mid_out],
                            0,
                            None,
                        );
                        current = mid_out;
                    } else {
                        let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                        g.emit(
                            &mut fbb,
                            tfl::op::MINIMUM,
                            vec![current, max_idx],
                            vec![out_idx],
                            0,
                            None,
                        );
                        continue;
                    }
                }
                if let Some(min) = min_val {
                    let (min_idx, min_name) = g.reserve_temporary(dtype, vec![]);
                    extra_constant_data.insert(min_name, (min as f32).to_le_bytes().to_vec());
                    let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                    g.emit(
                        &mut fbb,
                        tfl::op::MAXIMUM,
                        vec![current, min_idx],
                        vec![out_idx],
                        0,
                        None,
                    );
                }
                continue;
            },
            "linear" => {
                let alpha = node.attrs.get("alpha").copied().unwrap_or(1.0) as f32;
                let beta = node.attrs.get("beta").copied().unwrap_or(0.0) as f32;
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                let (alpha_idx, alpha_name) = g.reserve_temporary(dtype, vec![]);
                let alpha_bytes: Vec<u8> = alpha.to_le_bytes().to_vec();
                extra_constant_data.insert(alpha_name, alpha_bytes);
                if beta != 0.0 {
                    let (mul_out, _) = g.reserve_temporary(dtype, shape.clone());
                    g.emit(
                        &mut fbb,
                        tfl::op::MUL,
                        vec![input_idx, alpha_idx],
                        vec![mul_out],
                        0,
                        None,
                    );
                    let (beta_idx, beta_name) = g.reserve_temporary(dtype, vec![]);
                    let beta_bytes: Vec<u8> = beta.to_le_bytes().to_vec();
                    extra_constant_data.insert(beta_name, beta_bytes);
                    g.emit(
                        &mut fbb,
                        tfl::op::ADD,
                        vec![mul_out, beta_idx],
                        vec![out_idx],
                        0,
                        None,
                    );
                } else {
                    g.emit(
                        &mut fbb,
                        tfl::op::MUL,
                        vec![input_idx, alpha_idx],
                        vec![out_idx],
                        0,
                        None,
                    );
                }
                continue;
            },
            "where" => {
                if node.inputs.len() < 3 {
                    return Err("where needs condition, true_value, false_value".to_string());
                }
                let input_dtype = |nodes: &[GraphNode], name: &str| -> u8 {
                    nodes
                        .iter()
                        .find(|n| n.output == name)
                        .map(|n| webnn_type_to_tflite(n.desc.data_type))
                        .unwrap_or(dtype)
                };
                let cond_dtype = input_dtype(&nodes, &node.inputs[0]);
                let cond_idx = ensure_input(&mut g, &node.inputs[0], cond_dtype, &shape);
                let true_idx = ensure_input(&mut g, &node.inputs[1], dtype, &shape);
                let false_idx = ensure_input(&mut g, &node.inputs[2], dtype, &shape);
                let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                g.emit(
                    &mut fbb,
                    tfl::op::SELECT,
                    vec![cond_idx, true_idx, false_idx],
                    vec![out_idx],
                    tfl::builtin_options::SELECT,
                    None,
                );
                continue;
            },
            "layerNormalization" | "layer_normalization" => {
                if node.inputs.is_empty() {
                    return Err("layerNormalization needs at least 1 input".to_string());
                }
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                let scale_idx = if node.inputs.len() > 1 {
                    Some(ensure_input(&mut g, &node.inputs[1], dtype, &shape))
                } else {
                    None
                };
                let bias_idx = if node.inputs.len() > 2 {
                    Some(ensure_input(&mut g, &node.inputs[2], dtype, &shape))
                } else {
                    None
                };
                let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                let axes = vec![shape.len().saturating_sub(1) as u32];
                g.emit_layer_normalization(
                    &mut fbb, input_idx, scale_idx, bias_idx, &axes, 1e-5, out_idx,
                );
                continue;
            },
            "instanceNormalization" | "instance_normalization" => {
                if node.inputs.is_empty() {
                    return Err("instanceNormalization needs at least 1 input".to_string());
                }
                let input_idx = ensure_input(&mut g, &node.inputs[0], dtype, &shape);
                let scale_idx = if node.inputs.len() > 1 {
                    Some(ensure_input(&mut g, &node.inputs[1], dtype, &shape))
                } else {
                    None
                };
                let bias_idx = if node.inputs.len() > 2 {
                    Some(ensure_input(&mut g, &node.inputs[2], dtype, &shape))
                } else {
                    None
                };
                let out_idx = g.ensure_tensor(&node.output, dtype, &shape);
                let axes: Vec<u32> = if shape.len() >= 4 {
                    vec![2, 3]
                } else {
                    vec![shape.len().saturating_sub(2) as u32, shape.len().saturating_sub(1) as u32]
                };
                g.emit_layer_normalization(
                    &mut fbb, input_idx, scale_idx, bias_idx, &axes, 1e-5, out_idx,
                );
                continue;
            },
            _ => {},
        }

        let input_dtype_of = |name: &str| -> u8 {
            nodes
                .iter()
                .find(|n| n.output == name)
                .map(|n| webnn_type_to_tflite(n.desc.data_type))
                .unwrap_or(dtype)
        };
        for inp in &node.inputs {
            ensure_input(&mut g, inp, input_dtype_of(inp), &shape);
        }
        g.ensure_tensor(&node.output, dtype, &shape);

        if node.op == "split" {
            if let Some(&gid) = node.attrs.get("split_group") {
                let group = &split_groups[&(gid as u64)];
                if group[0] == node_idx {
                    let axis = node.attrs.get("axis").copied().unwrap_or(0.0) as i32;
                    let axis_name = format!("_split_axis_g{}", gid as u64);
                    g.ensure_tensor(&axis_name, 2, &[1]);
                    extra_constant_data.insert(axis_name, axis.to_le_bytes().to_vec());
                    let sizes_name = format!("_split_sizes_g{}", gid as u64);
                    let num_splits = node.attrs.get("splits").copied().unwrap_or(2.0) as usize;
                    g.ensure_tensor(&sizes_name, 2, &[num_splits as u32]);
                    let mut sizes: Vec<i32> = Vec::new();
                    for &idx in group.iter() {
                        let node_shape = &nodes[idx].desc.shape;
                        let ax = node.attrs.get("axis").copied().unwrap_or(0.0) as usize;
                        if ax < node_shape.len() {
                            sizes.push(node_shape[ax] as i32);
                        }
                    }
                    extra_constant_data.insert(
                        sizes_name,
                        sizes.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    );
                }
            } else {
                let axis = node.attrs.get("axis").copied().unwrap_or(0.0) as i32;
                let axis_name = format!("_split_axis_{}", node.output);
                g.ensure_tensor(&axis_name, 2, &[1]);
                extra_constant_data.insert(axis_name, axis.to_le_bytes().to_vec());
                let sizes_name = format!("_split_sizes_{}", node.output);
                let num_splits = node.attrs.get("splits").copied().unwrap_or(2.0) as usize;
                g.ensure_tensor(&sizes_name, 2, &[num_splits as u32]);
                let total: u32 = node.desc.shape.iter().product();
                let per_split = total / num_splits as u32;
                let sizes: Vec<i32> = vec![per_split as i32; num_splits];
                extra_constant_data.insert(
                    sizes_name,
                    sizes.iter().flat_map(|v| v.to_le_bytes()).collect(),
                );
            }
        }
    }

    for (node_idx, node) in nodes.iter().enumerate() {
        log::error!(
            "TFL compile op #{}: {} (inputs: {:?}, output: {})",
            node_idx,
            node.op,
            node.inputs,
            node.output
        );
        // Skip constant ops – their data is already embedded in the buffer table.
        if node.op == "constant" {
            continue;
        }
        // Skip secondary split nodes – they are handled by the primary one.
        if node.op == "split" && split_skip.contains(&node_idx) {
            continue;
        }
    // Skip decomposed ops – already handled in the first pass.
        if matches!(
            node.op.as_str(),
            "batchNormalization" |
                "batch_normalization" |
                "tan" |
                "linear" |
                "where" |
                "layerNormalization" |
                "layer_normalization" |
                "instanceNormalization" |
                "instance_normalization" |
                "dequantizeLinear" |
                "dequantize_linear" |
                "clamp"
        ) {
            continue;
        }
        let tfl_code = webnn_op_to_tflite(&node.op)
            .ok_or_else(|| format!("Unsupported WebNN op '{}'", node.op))?;

        let input_indices: Vec<u32> =
            node.inputs.iter().map(|name| g.tensor_id(name)).collect();
        let output_idx = g.tensor_id(&node.output);
        let attrs = &node.attrs;

        match tfl_code {
            tfl::op::CONV_2D => {
                if input_indices.len() < 2 {
                    return Err("conv2d needs input and filter".to_string());
                }
                let filt_name = &node.inputs[1];
                let filt_shape = g.tensor_shape.get(filt_name).cloned().unwrap_or_default();
                let groups = attrs.get("groups").copied().unwrap_or(1.0) as usize;
                let is_depthwise = groups > 1 && filt_shape.first() == Some(&1);
                let out_channels = if is_depthwise {
                    filt_shape.get(3).copied().unwrap_or(0) as usize
                } else {
                    filt_shape.get(0).copied().unwrap_or(0) as usize
                };
                let out_dtype = webnn_type_to_tflite(node.desc.data_type);
                let all_pads_zero = (0..4)
                    .all(|i| attrs.get(&format!("pad{}", i)).copied().unwrap_or(0.0) == 0.0);
                let (pad, padded_input_idx) = if all_pads_zero {
                    (tfl::padding::VALID, input_indices[0])
                } else {
                    // Insert a PAD op before conv2d and use VALID padding.
                    let pad_top = attrs.get("pad0").copied().unwrap_or(0.0) as i32;
                    let pad_bottom = attrs.get("pad1").copied().unwrap_or(0.0) as i32;
                    let pad_left = attrs.get("pad2").copied().unwrap_or(0.0) as i32;
                    let pad_right = attrs.get("pad3").copied().unwrap_or(0.0) as i32;
                    // After NHWC preproc, input is in NHWC format.
                    // Padding attrs from WebNN are [top, bottom, left, right] in NCHW spatial order.
                    // In NHWC, pad_dims = [0, top, bottom, left, right, 0] (batch, H_top, H_bot, W_left, W_right, channels).
                    let default_shape = node.desc.shape.clone();
                    let input_shape = g
                        .tensor_shape
                        .get(&node.inputs[0])
                        .cloned()
                        .unwrap_or(default_shape);
                    let n = *input_shape.get(0).unwrap_or(&1);
                    let h = *input_shape.get(1).unwrap_or(&1);
                    let w = *input_shape.get(2).unwrap_or(&1);
                    let c = *input_shape.get(3).unwrap_or(&1);
                    let padded_h = h + (pad_top + pad_bottom) as u32;
                    let padded_w = w + (pad_left + pad_right) as u32;
                    let padded_shape = vec![n, padded_h, padded_w, c];
                    let padded_name = format!("{}_padded", node.output);
                    let padded_idx = g.ensure_tensor(&padded_name, out_dtype, &padded_shape);
                    // PAD op: TFLite expects paddings shape [rank, 2].
                    // NHWC layout: [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]
                    let pad_dims = vec![0i32, 0, pad_top, pad_bottom, pad_left, pad_right, 0, 0];
                    let pd_shape: Vec<u32> = vec![4, 2];
                    let (pd_idx, pd_name) = g.reserve_temporary(2, pd_shape);
                    let pd_bytes: Vec<u8> = pad_dims.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(pd_name, pd_bytes);
                    g.emit(
                        &mut fbb,
                        tfl::op::PAD,
                        vec![input_indices[0], pd_idx],
                        vec![padded_idx],
                        0,
                        None,
                    );
                    (tfl::padding::VALID, padded_idx)
                };
                let s_h = attrs.get("stride_h").copied().unwrap_or(1.0) as i32;
                let s_w = attrs.get("stride_w").copied().unwrap_or(1.0) as i32;
                let d_h = attrs.get("dilation_h").copied().unwrap_or(1.0) as i32;
                let d_w = attrs.get("dilation_w").copied().unwrap_or(1.0) as i32;
                let bias_idx = if node.inputs.len() > 2 {
                    g.tensor_id(&node.inputs[2])
                } else {
                    let (idx, name) = g.reserve_temporary(out_dtype, vec![out_channels as u32]);
                    extra_constant_data.insert(name, vec![0u8; out_channels * 4]);
                    idx
                };
                let mut conv_inputs = vec![padded_input_idx];
                for i in 1..node.inputs.len().min(input_indices.len()) {
                    conv_inputs.push(input_indices[i]);
                }
                if node.inputs.len() <= 2 {
                    conv_inputs.push(bias_idx);
                }
                if is_depthwise {
                    let opt = build_depthwise_conv2d_options(
                        &mut fbb, pad as u8, s_w, s_h, d_w, d_h, 0, 1,
                    );
                    g.emit(
                        &mut fbb,
                        tfl::op::DEPTHWISE_CONV_2D,
                        conv_inputs,
                        vec![output_idx],
                        tfl::builtin_options::DEPTHWISE_CONV_2D,
                        Some(opt),
                    );
                } else {
                    let opt =
                        build_conv2d_options(&mut fbb, pad as u8, s_w, s_h, d_w, d_h, 0);
                    g.emit(
                        &mut fbb,
                        tfl_code,
                        conv_inputs,
                        vec![output_idx],
                        tfl::builtin_options::CONV_2D,
                        Some(opt),
                    );
                }
            },
            tfl::op::TRANSPOSE_CONV => {
                if input_indices.len() < 2 {
                    return Err("convTranspose2d needs input and filter".to_string());
                }
                // TFLite TRANSPOSE_CONV input order: [output_shape, filter, input, bias?]
                // node.desc.shape is already NHWC from the pre-processing pass.
                let out_shape_nhwc: Vec<i32> = node.desc.shape.iter().map(|&d| d as i32).collect();
                let out_shape_shape: Vec<u32> = vec![4];
                let (out_shape_idx, out_shape_name) = g.reserve_temporary(2, out_shape_shape);
                let out_shape_bytes: Vec<u8> = out_shape_nhwc.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(out_shape_name, out_shape_bytes);
                let s_h = attrs.get("stride_h").copied().unwrap_or(1.0) as i32;
                let s_w = attrs.get("stride_w").copied().unwrap_or(1.0) as i32;
                let all_pads_zero = (0..4)
                    .all(|i| attrs.get(&format!("pad{}", i)).copied().unwrap_or(0.0) == 0.0);
                let pad = if all_pads_zero {
                    tfl::padding::VALID
                } else {
                    tfl::padding::SAME
                };
                let filter_idx = input_indices[1];
                let opt = build_transpose_conv_options(&mut fbb, pad, s_w, s_h, 0);
                let mut tc_inputs = vec![out_shape_idx, filter_idx, input_indices[0]];
                if node.inputs.len() > 2 && input_indices.len() > 2 {
                    tc_inputs.push(input_indices[2]);
                }
                g.emit(
                    &mut fbb,
                    tfl_code,
                    tc_inputs,
                    vec![output_idx],
                    tfl::builtin_options::TRANSPOSE_CONV,
                    Some(opt),
                );
            },
            tfl::op::AVERAGE_POOL_2D | tfl::op::MAX_POOL_2D | tfl::op::L2_POOL_2D => {
                let in_name = &node.inputs[0];
                let in_shape = g.tensor_shape.get(in_name).cloned().unwrap_or_default();
                let w_h = attrs.get("window_h").copied().unwrap_or(in_shape[1] as f64) as i32;
                let w_w = attrs.get("window_w").copied().unwrap_or(in_shape[2] as f64) as i32;
                let s_h = attrs.get("stride_h").copied().unwrap_or(1.0) as i32;
                let s_w = attrs.get("stride_w").copied().unwrap_or(1.0) as i32;
                let all_pads_zero =
                    (0..4).all(|i| attrs.get(&format!("pad{}", i)).copied().unwrap_or(0.0) == 0.0);
                let (pad, pool_input_idx) = if all_pads_zero {
                    (tfl::padding::VALID, input_indices[0])
                } else {
                    let pad_top = attrs.get("pad0").copied().unwrap_or(0.0) as i32;
                    let pad_bottom = attrs.get("pad1").copied().unwrap_or(0.0) as i32;
                    let pad_left = attrs.get("pad2").copied().unwrap_or(0.0) as i32;
                    let pad_right = attrs.get("pad3").copied().unwrap_or(0.0) as i32;
                    let n = *in_shape.get(0).unwrap_or(&1);
                    let h = *in_shape.get(1).unwrap_or(&1);
                    let w = *in_shape.get(2).unwrap_or(&1);
                    let c = *in_shape.get(3).unwrap_or(&1);
                    let padded_h = h + (pad_top + pad_bottom) as u32;
                    let padded_w = w + (pad_left + pad_right) as u32;
                    let padded_shape = vec![n, padded_h, padded_w, c];
                    let out_dtype = webnn_type_to_tflite(node.desc.data_type);
                    let padded_name = format!("{}_padded", node.output);
                    let padded_idx = g.ensure_tensor(&padded_name, out_dtype, &padded_shape);
                    let pad_dims = vec![0i32, 0, pad_top, pad_bottom, pad_left, pad_right, 0, 0];
                    let pd_shape: Vec<u32> = vec![4, 2];
                    let (pd_idx, pd_name) = g.reserve_temporary(2, pd_shape);
                    let pd_bytes: Vec<u8> = pad_dims.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(pd_name, pd_bytes);
                    g.emit(
                        &mut fbb,
                        tfl::op::PAD,
                        vec![input_indices[0], pd_idx],
                        vec![padded_idx],
                        0,
                        None,
                    );
                    (tfl::padding::VALID, padded_idx)
                };
                let opt = build_pool2d_options(&mut fbb, pad, s_w, s_h, w_w, w_h, 0);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    vec![pool_input_idx],
                    vec![output_idx],
                    tfl::builtin_options::POOL_2D,
                    Some(opt),
                );
            },
            tfl::op::SOFTMAX => {
                let rank = node.desc.shape.len() as i32;
                let axis = attrs.get("axis").copied().unwrap_or((rank - 1) as f64) as i32;
                log::error!(
                    "TFL softmax: rank={}, axis={}, shape={:?}",
                    rank,
                    axis,
                    node.desc.shape
                );
                let out_dtype = webnn_type_to_tflite(node.desc.data_type);
                if rank > 0 && axis != rank - 1 {
                    let orig_shape = &node.desc.shape;
                    let mut perm: Vec<i32> = (0..rank).collect();
                    perm.swap(axis as usize, (rank - 1) as usize);
                    let mut inv_perm: Vec<i32> = (0..rank).collect();
                    inv_perm.swap(axis as usize, (rank - 1) as usize);
                    let transposed_shape: Vec<u32> =
                        perm.iter().map(|&i| orig_shape[i as usize]).collect();
                    let perm_shape: Vec<u32> = vec![rank as u32];
                    let (perm_idx, perm_name) = g.reserve_temporary(2, perm_shape.clone());
                    let perm_bytes: Vec<u8> = perm.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(perm_name, perm_bytes);
                    let (inv_perm_idx, inv_perm_name) = g.reserve_temporary(2, perm_shape);
                    let inv_perm_bytes: Vec<u8> =
                        inv_perm.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(inv_perm_name, inv_perm_bytes);
                    let (trans_out, _) = g.reserve_temporary(out_dtype, transposed_shape.clone());
                    g.emit(
                        &mut fbb,
                        tfl::op::TRANSPOSE,
                        vec![input_indices[0], perm_idx],
                        vec![trans_out],
                        0,
                        None,
                    );
                    let (soft_out, _) = g.reserve_temporary(out_dtype, transposed_shape);
                    let softmax_opt = build_softmax_options(&mut fbb, 1.0);
                    g.emit(
                        &mut fbb,
                        tfl::op::SOFTMAX,
                        vec![trans_out],
                        vec![soft_out],
                        tfl::builtin_options::SOFTMAX,
                        Some(softmax_opt),
                    );
                    g.emit(
                        &mut fbb,
                        tfl::op::TRANSPOSE,
                        vec![soft_out, inv_perm_idx],
                        vec![output_idx],
                        0,
                        None,
                    );
                } else {
                    let softmax_opt = build_softmax_options(&mut fbb, 1.0);
                    g.emit(
                        &mut fbb,
                        tfl_code,
                        input_indices.clone(),
                        vec![output_idx],
                        tfl::builtin_options::SOFTMAX,
                        Some(softmax_opt),
                    );
                }
            },
            tfl::op::CONCATENATION => {
                let axis = node.attrs.get("axis").copied().unwrap_or(0.0) as i32;
                let opt = build_concatenation_options(&mut fbb, axis);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::CONCATENATION,
                    Some(opt),
                );
            },
            tfl::op::BATCH_MATMUL => {
                let a_transpose = node.attrs.get("aTranspose").copied().unwrap_or(0.0) != 0.0;
                let b_transpose = node.attrs.get("bTranspose").copied().unwrap_or(0.0) != 0.0;
                let adj_x = a_transpose;
                let adj_y = b_transpose;
                let opt = build_batch_matmul_options(&mut fbb, adj_x, adj_y);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::BATCH_MATMUL,
                    Some(opt),
                );
            },
            tfl::op::PAD => {
                let rank = node.attrs.get("pad_rank").copied().unwrap_or(0.0) as usize;
                let mut padding_data: Vec<i32> = Vec::with_capacity(rank * 2);
                for i in 0..rank {
                    let begin = node
                        .attrs
                        .get(&format!("pad_begin_{}", i))
                        .copied()
                        .unwrap_or(0.0) as i32;
                    let end = node
                        .attrs
                        .get(&format!("pad_end_{}", i))
                        .copied()
                        .unwrap_or(0.0) as i32;
                    padding_data.push(begin);
                    padding_data.push(end);
                }
if padding_data.is_empty() {
                    return Err("pad needs padding data".to_string());
                }
                let pad_shape: Vec<u32> = vec![rank as u32, 2];
                let (pad_idx, pad_name) = g.reserve_temporary(2, pad_shape);
                let pad_bytes: Vec<u8> =
                    padding_data.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(pad_name, pad_bytes);
                let pad_inputs = vec![input_indices[0], pad_idx];
                let opt = build_pad_options(&mut fbb);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    pad_inputs,
                    vec![output_idx],
                    tfl::builtin_options::PAD,
                    Some(opt),
                );
            },
            tfl::op::RESIZE_BILINEAR | tfl::op::RESIZE_NEAREST_NEIGHBOR => {
                let actual_code = if node.attrs.get("mode").map(|&v| v as i32) == Some(1) {
                    tfl::op::RESIZE_NEAREST_NEIGHBOR
                } else {
                    tfl::op::RESIZE_BILINEAR
                };
                let sizes: Vec<i32> = {
                    let num_sizes = node.attrs.get("sizes_len").copied().unwrap_or(0.0) as usize;
                    if num_sizes >= 2 {
                        (0..num_sizes)
                            .filter_map(|i| node.attrs.get(&format!("sizes_{}", i)))
                            .map(|&v| v as i32)
                            .collect()
                    } else {
                        let default_shape = node.desc.shape.clone();
                        let input_shape = g
                            .tensor_shape
                            .get(&node.inputs[0])
                            .cloned()
                            .unwrap_or(default_shape);
                        let scale_h = node.attrs.get("scale_h").copied().unwrap_or(1.0);
                        let scale_w = node.attrs.get("scale_w").copied().unwrap_or(1.0);
                        // After NHWC preprocessing, 4D spatial inputs are always NHWC [N,H,W,C].
                        let (h, w) = if input_shape.len() >= 4 {
                            (input_shape[1] as f64 * scale_h as f64, input_shape[2] as f64 * scale_w as f64)
                        } else {
                            let h = if input_shape.len() >= 2 { input_shape[input_shape.len() - 2] as f64 } else { 1.0 };
                            let w = if input_shape.len() >= 1 { input_shape[input_shape.len() - 1] as f64 } else { 1.0 };
                            (h * scale_h as f64, w * scale_w as f64)
                        };
                        vec![h as i32, w as i32]
                    }
                };
                log::error!("RESIZE sizes={:?} input={:?} desc_shape={:?}", sizes, g.tensor_shape.get(&node.inputs[0]), node.desc.shape);
                let default_shape = node.desc.shape.clone();
                let input_shape = g
                    .tensor_shape
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or(default_shape);
                let rank = input_shape.len() as i32;
                if rank == 4 {
                    let _out_dtype = webnn_type_to_tflite(node.desc.data_type);
                    let size_shape: Vec<u32> = vec![sizes.len() as u32];
                    let (size_idx, size_name) = g.reserve_temporary(2, size_shape);
                    let size_bytes: Vec<u8> = sizes.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(size_name, size_bytes);
                    let opts = build_resize_bilinear_options(
                        &mut fbb,
                        false,
                        actual_code == tfl::op::RESIZE_BILINEAR,
                    );
                    let builtin_opt = if actual_code == tfl::op::RESIZE_NEAREST_NEIGHBOR {
                        tfl::builtin_options::RESIZE_NEAREST_NEIGHBOR
                    } else {
                        tfl::builtin_options::RESIZE_BILINEAR
                    };
                    g.emit(
                        &mut fbb,
                        actual_code,
                        vec![input_indices[0], size_idx],
                        vec![output_idx],
                        builtin_opt,
                        Some(opts),
                    );
                } else if rank >= 3 {
                    let out_dtype = webnn_type_to_tflite(node.desc.data_type);
                    let perm: Vec<i32> = if rank == 4 {
                        vec![0, 2, 3, 1]
                    } else {
                        let mut p: Vec<i32> = (1..rank).collect();
                        p.push(0);
                        p
                    };
                    let inv_perm: Vec<i32> = {
                        let mut ip = vec![0i32; rank as usize];
                        for (i, &p) in perm.iter().enumerate() {
                            ip[p as usize] = i as i32;
                        }
                        ip
                    };
                    let transposed_shape: Vec<u32> =
                        perm.iter().map(|&i| input_shape[i as usize]).collect();
                    let perm_shape: Vec<u32> = vec![rank as u32];
                    let (perm_idx, perm_name) = g.reserve_temporary(2, perm_shape.clone());
                    let perm_bytes: Vec<u8> = perm.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(perm_name, perm_bytes);
                    let (inv_perm_idx, inv_perm_name) = g.reserve_temporary(2, perm_shape);
                    let inv_perm_bytes: Vec<u8> =
                        inv_perm.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(inv_perm_name, inv_perm_bytes);
                    let (trans_in, _) = g.reserve_temporary(out_dtype, transposed_shape);
                    g.emit(
                        &mut fbb,
                        tfl::op::TRANSPOSE,
                        vec![input_indices[0], perm_idx],
                        vec![trans_in],
                        0,
                        None,
                    );
                    let size_shape: Vec<u32> = vec![sizes.len() as u32];
                    let (size_idx, size_name) = g.reserve_temporary(2, size_shape);
                    let size_bytes: Vec<u8> = sizes.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(size_name, size_bytes);
                    let resize_nhwc_shape: Vec<u32> = if rank == 4 {
                        vec![
                            input_shape[0],
                            sizes[0] as u32,
                            sizes[1] as u32,
                            input_shape[1],
                        ]
                    } else {
                        vec![sizes[0] as u32, sizes[1] as u32, input_shape[0]]
                    };
                    let (trans_out, _) = g.reserve_temporary(out_dtype, resize_nhwc_shape);
                    let opts = build_resize_bilinear_options(
                        &mut fbb,
                        false,
                        actual_code == tfl::op::RESIZE_BILINEAR,
                    );
                    let builtin_opt = if actual_code == tfl::op::RESIZE_NEAREST_NEIGHBOR {
                        tfl::builtin_options::RESIZE_NEAREST_NEIGHBOR
                    } else {
                        tfl::builtin_options::RESIZE_BILINEAR
                    };
                    g.emit(
                        &mut fbb,
                        actual_code,
                        vec![trans_in, size_idx],
                        vec![trans_out],
                        builtin_opt,
                        Some(opts),
                    );
                    g.emit(
                        &mut fbb,
                        tfl::op::TRANSPOSE,
                        vec![trans_out, inv_perm_idx],
                        vec![output_idx],
                        0,
                        None,
                    );
                } else {
                    let size_shape: Vec<u32> = vec![sizes.len() as u32];
                    let (size_idx, size_name) = g.reserve_temporary(2, size_shape);
                    let size_bytes: Vec<u8> = sizes.iter().flat_map(|v| v.to_le_bytes()).collect();
                    extra_constant_data.insert(size_name, size_bytes);
                    let opts = build_resize_bilinear_options(
                        &mut fbb,
                        false,
                        actual_code == tfl::op::RESIZE_BILINEAR,
                    );
                    let builtin_opt = if actual_code == tfl::op::RESIZE_NEAREST_NEIGHBOR {
                        tfl::builtin_options::RESIZE_NEAREST_NEIGHBOR
                    } else {
                        tfl::builtin_options::RESIZE_BILINEAR
                    };
                    g.emit(
                        &mut fbb,
                        actual_code,
                        vec![input_indices[0], size_idx],
                        vec![output_idx],
                        builtin_opt,
                        Some(opts),
                    );
                }
            },
            tfl::op::FULLY_CONNECTED => {
                let opt = build_fully_connected_options(&mut fbb, tfl::activation::NONE);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::FULLY_CONNECTED,
                    Some(opt),
                );
            },
            tfl::op::RESHAPE => {
                let out_shape: Vec<i32> = node.desc.shape.iter().map(|&s| s as i32).collect();
                let shape_tensor_shape: Vec<u32> = vec![out_shape.len() as u32];
                let (shape_idx, shape_name) = g.reserve_temporary(2, shape_tensor_shape);
                let shape_bytes: Vec<u8> = out_shape.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(shape_name, shape_bytes);
                let opt = build_reshape_options(&mut fbb, &out_shape);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    vec![input_indices[0], shape_idx],
                    vec![output_idx],
                    tfl::builtin_options::RESHAPE,
                    Some(opt),
                );
            },
            tfl::op::SQUEEZE => {
                let dims: Vec<i32> = node
                    .attrs
                    .get("axes")
                    .map(|_| node.desc.shape.iter().map(|&s| s as i32).collect())
                    .unwrap_or_default();
                let opt = build_squeeze_options(&mut fbb, &dims);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::SQUEEZE,
                    Some(opt),
                );
            },
            tfl::op::SPLIT_V => {
                let num_splits = node.attrs.get("splits").copied().unwrap_or(2.0) as i32;
                let opt = build_split_options(&mut fbb, num_splits);
                let (axis_name, sizes_name, split_outputs) =
                    if let Some(&gid) = node.attrs.get("split_group") {
                        let an = format!("_split_axis_g{}", gid as u64);
                        let sn = format!("_split_sizes_g{}", gid as u64);
                        let mut outs = vec![output_idx];
                        let group = &split_groups[&(gid as u64)];
                        for &idx in &group[1..] {
                            outs.push(g.tensor_id(&nodes[idx].output));
                        }
                        (an, sn, outs)
                    } else {
                        let an = format!("_split_axis_{}", node.output);
                        let sn = format!("_split_sizes_{}", node.output);
                        (an, sn, vec![output_idx])
                    };
                let axis_idx = g.tensor_id(&axis_name);
                let sizes_idx = g.tensor_id(&sizes_name);
                // SPLIT_V inputs: [input, split_sizes, axis]
                let mut split_inputs = input_indices.clone();
                split_inputs.push(sizes_idx);
                split_inputs.push(axis_idx);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    split_inputs,
                    split_outputs,
                    tfl::builtin_options::SPLIT_V,
                    Some(opt),
                );
            },
            tfl::op::LEAKY_RELU => {
                let alpha = node.attrs.get("alpha").copied().unwrap_or(0.01) as f32;
                let opt = build_leaky_relu_options(&mut fbb, alpha);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::LEAKY_RELU,
                    Some(opt),
                );
            },
            tfl::op::CAST => {
                let out_dt = webnn_type_to_tflite(node.desc.data_type);
                let in_dt = g
                    .tensor_type
                    .get(&node.inputs[0])
                    .copied()
                    .unwrap_or(out_dt);
                let opt = build_cast_options(&mut fbb, in_dt, out_dt);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::CAST,
                    Some(opt),
                );
            },
            tfl::op::CUM_SUM => {
                let exclusive = node.attrs.get("exclusive").copied().unwrap_or(0.0) != 0.0;
                let reverse = node.attrs.get("reverse").copied().unwrap_or(0.0) != 0.0;
                let opt = build_cumsum_options(&mut fbb, exclusive, reverse);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::CUM_SUM,
                    Some(opt),
                );
            },
            tfl::op::ARG_MAX => {
                let opt = build_arg_max_options(&mut fbb, 0);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::ARG_MAX,
                    Some(opt),
                );
            },
            tfl::op::ARG_MIN => {
                let opt = build_arg_min_options(&mut fbb, 0);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::ARG_MIN,
                    Some(opt),
                );
            },
            tfl::op::GATHER => {
                let axis = node.attrs.get("axis").copied().unwrap_or(0.0) as i32;
                let opt = build_gather_options(&mut fbb, axis);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    input_indices.clone(),
                    vec![output_idx],
                    tfl::builtin_options::GATHER,
                    Some(opt),
                );
            },
            tfl::op::MEAN |
            tfl::op::SUM |
            tfl::op::REDUCE_MAX |
            tfl::op::REDUCE_MIN |
            tfl::op::REDUCE_PROD => {
                let num_axes = node.attrs.get("axes_len").copied().unwrap_or(0.0) as usize;
                let axes: Vec<i32> = if num_axes > 0 {
                    (0..num_axes)
                        .filter_map(|i| node.attrs.get(&format!("axis_{}", i)))
                        .map(|&v| v as i32)
                        .collect()
                } else {
                    let default_shape = node.desc.shape.clone();
                    let input_shape = g
                        .tensor_shape
                        .get(&node.inputs[0])
                        .cloned()
                        .unwrap_or(default_shape);
                    (0..input_shape.len() as i32).collect()
                };
                let axes_shape: Vec<u32> = vec![axes.len() as u32];
                let (axes_idx, axes_name) = g.reserve_temporary(2, axes_shape);
                let axes_bytes: Vec<u8> = axes.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(axes_name, axes_bytes);
                let keep_dims = node.attrs.get("keepDimensions").copied().map(|v| v != 0.0).unwrap_or(true);
                let opt = build_reducer_options(&mut fbb, keep_dims);
                g.emit(
                    &mut fbb,
                    tfl_code,
                    vec![input_indices[0], axes_idx],
                    vec![output_idx],
                    tfl::builtin_options::REDUCER,
                    Some(opt),
                );
            },
            tfl::op::REVERSE_V2 => {
                let axes: Vec<i32> = node
                    .attrs
                    .keys()
                    .filter(|k| k.starts_with("axis_"))
                    .filter_map(|k| k.strip_prefix("axis_"))
                    .filter_map(|idx| idx.parse::<usize>().ok())
                    .filter_map(|idx| node.attrs.get(&format!("axis_{}", idx)).copied())
                    .map(|v| v as i32)
                    .collect();
                let axes_data: Vec<i32> = if axes.is_empty() { vec![0i32] } else { axes };
                let axes_shape: Vec<u32> = vec![axes_data.len() as u32];
                let (axes_idx, axes_name) = g.reserve_temporary(2, axes_shape);
                let axes_bytes: Vec<u8> = axes_data.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(axes_name, axes_bytes);
                let rev_inputs = vec![input_indices[0], axes_idx];
                g.emit(&mut fbb, tfl_code, rev_inputs, vec![output_idx], 0, None);
            },
            tfl::op::SLICE => {
                let mut starts: Vec<i32> = Vec::new();
                let mut sizes: Vec<i32> = Vec::new();
                let mut i = 0;
                while let Some(&v) = node.attrs.get(&format!("start_{}", i)) {
                    starts.push(v as i32);
                    i += 1;
                }
                let mut j = 0;
                while let Some(&v) = node.attrs.get(&format!("size_{}", j)) {
                    sizes.push(v as i32);
                    j += 1;
                }
                let start_shape: Vec<u32> = vec![starts.len() as u32];
                let size_shape: Vec<u32> = vec![sizes.len() as u32];
                let (start_idx, start_name) = g.reserve_temporary(2, start_shape);
                let start_bytes: Vec<u8> = starts.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(start_name, start_bytes);
                let (size_idx, size_name) = g.reserve_temporary(2, size_shape);
                let size_bytes: Vec<u8> = sizes.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(size_name, size_bytes);
                let slice_inputs = vec![input_indices[0], start_idx, size_idx];
                g.emit(&mut fbb, tfl_code, slice_inputs, vec![output_idx], 0, None);
            },
            tfl::op::TRANSPOSE => {
                let perm: Vec<i32> = {
                    let perm_len = node.attrs.get("perm_len").copied().unwrap_or(0.0) as usize;
                    if perm_len > 0 {
                        (0..perm_len)
                            .filter_map(|i| node.attrs.get(&format!("perm_{}", i)))
                            .map(|&v| v as i32)
                            .collect()
                    } else {
                        let rank = g
                            .tensor_shape
                            .get(&node.inputs[0])
                            .map(|s| s.len())
                            .unwrap_or(0);
                        (0..rank).rev().map(|i| i as i32).collect()
                    }
                };
                let perm_shape: Vec<u32> = vec![perm.len() as u32];
                let (perm_idx, perm_name) = g.reserve_temporary(2, perm_shape);
                let perm_bytes: Vec<u8> = perm.iter().flat_map(|v| v.to_le_bytes()).collect();
                extra_constant_data.insert(perm_name, perm_bytes);
                let trans_inputs = vec![input_indices[0], perm_idx];
                g.emit(&mut fbb, tfl_code, trans_inputs, vec![output_idx], 0, None);
            },
            tfl::op::QUANTIZE => {
                g.emit(
                    &mut fbb,
                    tfl_code,
                    vec![input_indices[0]],
                    vec![output_idx],
                    tfl::builtin_options::QUANTIZE,
                    None,
                );
            },
            _ => {
                g.emit(&mut fbb, tfl_code, input_indices, vec![output_idx], 0, None);
            },
        }
    }

    // ── NHWC shape propagation is now handled by insert_nhwc_transposes() ──
    // Pre-processing inserts explicit TRANSPOSE ops with correct NHWC shapes.
    // No post-hoc shape conversion needed here.

    // ── Finalize ──

    let tensor_names: Vec<String> = {
        let mut v: Vec<(u32, String)> = g.tensor_idx.iter().map(|(n, &i)| (i, n.clone())).collect();
        v.sort_by_key(|(i, _)| *i);
        v.into_iter().map(|(_, n)| n).collect()
    };

    // Build buffers: index 0 is always empty. Subsequent indices hold constant data.
    let mut buffer_data: Vec<Vec<u8>> = vec![vec![]]; // buffer 0 = empty
    let mut tensor_buffer_map: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    for name in &tensor_names {
        if let Some(data) = constant_data.get(name.as_str()) {
            let buf_idx = buffer_data.len() as u32;
            buffer_data.push(data.to_vec());
            tensor_buffer_map.insert(name.clone(), buf_idx);
        } else if let Some(data) = extra_constant_data.get(name) {
            let buf_idx = buffer_data.len() as u32;
            buffer_data.push(data.clone());
            tensor_buffer_map.insert(name.clone(), buf_idx);
        } else {
            tensor_buffer_map.insert(name.clone(), 0);
        }
    }

    let mut tensor_offsets: Vec<WIPOffset<TableFinishedWIPOffset>> = Vec::new();
    for name in &tensor_names {
        let dtype = g.tensor_type[name];
        let shape = &g.tensor_shape[name];
        let buffer = tensor_buffer_map[name];
        tensor_offsets.push(build_tensor(&mut fbb, shape, dtype, buffer, name));
    }

    let buf_offsets: Vec<WIPOffset<TableFinishedWIPOffset>> = buffer_data
        .iter()
        .map(|d| build_buffer(&mut fbb, d.as_slice()))
        .collect();

    let op_code_offsets: Vec<WIPOffset<TableFinishedWIPOffset>> = g
        .op_codes
        .iter()
        .map(|&code| build_operator_code(&mut fbb, code))
        .collect();

    let input_set: HashSet<&str> = nodes
        .iter()
        .flat_map(|n| n.inputs.iter().map(|s| s.as_str()))
        .collect();
    let output_set: HashSet<&str> = nodes.iter().map(|n| n.output.as_str()).collect();
    let mut subgraph_input_names: Vec<&str> = input_set
        .iter()
        .filter(|n| !output_set.contains(*n))
        .copied()
        .collect();
    subgraph_input_names.sort();
    let subgraph_inputs: Vec<u32> = subgraph_input_names
        .iter()
        .map(|n| g.tensor_id(n))
        .collect();
    let subgraph_outputs: Vec<u32> = {
        let output_set: std::collections::HashSet<&str> =
            output_names.iter().map(|s| s.as_str()).collect();
        if !output_set.is_empty() {
            // Only include the requested output nodes.
            let mut outs = Vec::new();
            for n in nodes {
                if output_set.contains(n.output.as_str()) {
                    outs.push(g.tensor_id(&n.output));
                }
            }
            outs
        } else {
            // Legacy fallback: include all non-constant node outputs.
            let mut outs = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for n in nodes {
                if n.op == "constant" {
                    continue;
                }
                if seen.insert(n.output.as_str()) {
                    outs.push(g.tensor_id(&n.output));
                }
            }
            outs
        }
    };

    log::error!(
        "TFL compile: {} ops, {} tensors, {} inputs, {} outputs",
        g.operators.len(),
        tensor_names.len(),
        subgraph_inputs.len(),
        subgraph_outputs.len()
    );
    log::error!("TFL input names: {:?}", subgraph_input_names);
    log::error!("TFL output tensor IDs: {:?}", subgraph_outputs);
    for (name, &idx) in &g.tensor_idx {
        log::error!(
            "  tensor '{}' -> idx={}, shape={:?}, dtype={}",
            name,
            idx,
            g.tensor_shape
                .get(name)
                .map(|s| s.clone())
                .unwrap_or_default(),
            g.tensor_type.get(name).copied().unwrap_or(0)
        );
    }

    let subgraph = build_subgraph(
        &mut fbb,
        &tensor_offsets,
        &subgraph_inputs,
        &subgraph_outputs,
        &g.operators,
        "main",
    );

    let model = build_model(
        &mut fbb,
        &op_code_offsets,
        &[subgraph],
        &buf_offsets,
        "Servo WebNN",
    );

    fbb.finish(model, Some("TFL3"));
    let result = fbb.finished_data().to_vec();

    // Diagnostic: dump first 64 bytes
    let n = result.len().min(64);
    log::error!("Flatbuffer size: {} bytes", result.len());
    log::error!("First {} bytes: {:02x?}", n, &result[..n]);
    if result.len() >= 8 {
        let root_off = u32::from_le_bytes(result[0..4].try_into().unwrap());
        let file_id = std::str::from_utf8(&result[4..8]);
        log::error!(
            "Root offset at bytes 0-3: {} (0x{:08x})",
            root_off,
            root_off
        );
        log::error!("File identifier at bytes 4-7: {:?}", file_id);
    }
    // Write to file for external analysis
    if let Err(e) = std::fs::write("/tmp/webnn_model.tflite", &result) {
        log::error!("Failed to write diagnostic file: {}", e);
    }

    Ok(CompileResult {
        flatbuf: result,
        nhwc_inputs: Vec::new(),
        nhwc_outputs: Vec::new(),
    })
}
