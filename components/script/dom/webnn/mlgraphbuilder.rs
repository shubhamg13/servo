/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use dom_struct::dom_struct;
use js::context::JSContext;
use js::gc::HandleObject;
use script_bindings::cell::DomRefCell;
use script_bindings::cformat;
use script_bindings::codegen::GenericBindings::WindowBinding::WindowMethods;
use script_bindings::record::Record;
use script_bindings::reflector::{Reflector, reflect_weak_referenceable_dom_object_with_proto};
use script_bindings::root::DomRoot;

use crate::dom::bindings::buffer_source::get_buffer_source_copy;
use crate::dom::bindings::codegen::Bindings::PermissionStatusBinding::PermissionName;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLConv2dFilterOperandLayout, MLConv2dOptions, MLConvTranspose2dFilterOperandLayout,
    MLConvTranspose2dOptions, MLGemmOptions, MLGraphBuilderMethods, MLInputOperandLayout,
    MLInterpolationMode, MLLeakyReluOptions, MLOperandDataType, MLOperandDescriptor,
    MLOperatorOptions, MLPadOptions, MLPaddingMode, MLPool2dOptions, MLReduceOptions,
    MLResample2dOptions, MLRoundingType, MLSliceOptions, MLSplitOptions, MLTransposeOptions,
};
use crate::dom::bindings::codegen::UnionTypes::{
    MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer,
    RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence,
};
use crate::dom::bindings::error::Error;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::str::USVString;
use crate::dom::bindings::weakref::WeakRef;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::{MLContext, check_dimensions, validate_buffer_with_descriptor};
use crate::dom::webnn::mlgraph::MLGraph;
use crate::dom::webnn::mloperand::MLOperand;
use crate::routed_promise::callback_promise;

/// <https://www.w3.org/TR/webnn/#mlgraphbuilder-validate-operand>
fn validate_operand(builder: &MLGraphBuilder, operand: &MLOperand) -> bool {
    // > To validate operand given MLGraphBuilder builder and MLOperand operand,
    // return true if operand.[[builder]] is builder, and false otherwise.
    operand.builder() == builder
}

/// <https://www.w3.org/TR/webnn/#bidirectionally-broadcasting>
fn bidirectionally_broadcast(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, ()> {
    // Step 1. Let sizeA be shapeA's size.
    let size_a = shape_a.len();
    // Step 2. Let sizeB be shapeB's size.
    let size_b = shape_b.len();
    // Step 3. Let outputSize be the maximum of sizeA and sizeB.
    let output_size = std::cmp::max(size_a, size_b);
    // Step 4. Let paddedA be a clone of shapeA.
    let mut padded_a = shape_a.to_vec();
    // Step 5. While paddedA's size is less than outputSize, prepend 1 to paddedA.
    while padded_a.len() < output_size {
        padded_a.insert(0, 1);
    }
    // Step 6. Let paddedB be a clone of shapeB.
    let mut padded_b = shape_b.to_vec();
    // Step 7. While paddedB's size is less than outputSize, prepend 1 to paddedB.
    while padded_b.len() < output_size {
        padded_b.insert(0, 1);
    }
    // Step 8. Let outputShape be a new list.
    let mut output_shape = Vec::new();
    // Step 9. For each index in the range 0 to outputSize, exclusive:
    for index in 0..output_size {
        // Step 9.1. Let dimA be paddedA[index].
        let dim_a = padded_a[index];
        // Step 9.2. Let dimB be paddedB[index].
        let dim_b = padded_b[index];
        // Step 9.3. If dimA is not equal to dimB, and dimA is not equal to 1, and dimB is not equal to 1, then return failure.
        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return Err(());
        }
        // Step 9.4. Append the maximum of dimA and dimB to outputShape.
        output_shape.push(std::cmp::max(dim_a, dim_b));
    }
    // Step 10. Return outputShape.
    Ok(output_shape)
}

/// Returns the string form of an `MLInputOperandLayout`.
fn input_layout_str(layout: MLInputOperandLayout) -> &'static str {
    match layout {
        MLInputOperandLayout::Nchw => "nchw",
        MLInputOperandLayout::Nhwc => "nhwc",
    }
}

/// Returns the string form of an `MLConv2dFilterOperandLayout`.
fn filter_layout_str(layout: MLConv2dFilterOperandLayout) -> &'static str {
    match layout {
        MLConv2dFilterOperandLayout::Oihw => "oihw",
        MLConv2dFilterOperandLayout::Hwio => "hwio",
        MLConv2dFilterOperandLayout::Ohwi => "ohwi",
        MLConv2dFilterOperandLayout::Ihwo => "ihwo",
    }
}

/// Returns the string form of an `MLConvTranspose2dFilterOperandLayout`.
fn conv_transpose_filter_layout_str(layout: MLConvTranspose2dFilterOperandLayout) -> &'static str {
    match layout {
        MLConvTranspose2dFilterOperandLayout::Iohw => "iohw",
        MLConvTranspose2dFilterOperandLayout::Hwoi => "hwoi",
        MLConvTranspose2dFilterOperandLayout::Ohwi => "ohwi",
    }
}

/// Returns the string form of an `MLRoundingType`.
fn rounding_type_str(rounding: MLRoundingType) -> &'static str {
    match rounding {
        MLRoundingType::Floor => "floor",
        MLRoundingType::Ceil => "ceil",
    }
}

/// Returns the string form of an `MLInterpolationMode`.
fn interpolation_mode_str(mode: MLInterpolationMode) -> &'static str {
    match mode {
        MLInterpolationMode::Nearest_neighbor => "nearest-neighbor",
        MLInterpolationMode::Linear => "linear",
    }
}

/// Returns the string form of an `MLPaddingMode`.
fn padding_mode_str(mode: MLPaddingMode) -> &'static str {
    match mode {
        MLPaddingMode::Constant => "constant",
        MLPaddingMode::Edge => "edge",
        MLPaddingMode::Reflection => "reflection",
        MLPaddingMode::Symmetric => "symmetric",
    }
}

/// <https://www.w3.org/TR/webnn/#mlgraphbuilder-pad>
fn pad_output_shape(
    input_shape: &[u32],
    beginning_padding: &[u32],
    ending_padding: &[u32],
) -> Result<Vec<u32>, Error> {
    // Step 1. If beginningPadding's size is not equal to input's rank, then
    // throw a TypeError.
    // Step 2. If endingPadding's size is not equal to input's rank, then throw
    // a TypeError.
    if beginning_padding.len() != input_shape.len() || ending_padding.len() != input_shape.len() {
        return Err(Error::Type(
            c"pad padding length does not match input rank.".to_owned(),
        ));
    }
    // Step 3. The output shape is input's shape with each dimension increased
    // by the corresponding beginning and ending padding.
    Ok(input_shape
        .iter()
        .zip(beginning_padding.iter().zip(ending_padding.iter()))
        .map(|(&dim, (&begin, &end))| dim + begin + end)
        .collect())
}

/// Computes a single conv/pool output dimension.
fn conv_dim(
    input: u32,
    kernel: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
) -> Result<u32, Error> {
    if stride == 0 || dilation == 0 {
        return Err(Error::Type(
            c"Stride and dilation must be greater than 0.".to_owned(),
        ));
    }
    let kernel_extent = (dilation as i64)
        .checked_mul(kernel as i64 - 1)
        .ok_or_else(|| Error::Type(c"Kernel extent is too large.".to_owned()))?;
    let numerator = input as i64 + padding as i64 - kernel_extent - 1;
    if numerator < 0 {
        return Err(Error::Type(
            c"Output dimension is invalid for the given input, kernel, and padding.".to_owned(),
        ));
    }
    u32::try_from(numerator / stride as i64 + 1)
        .map_err(|_| Error::Type(c"Output dimension is too large.".to_owned()))
}

/// Computes the output shape of a 2D convolution for the supported input and
/// filter operand layouts.
fn conv2d_output_shape(
    input: &[u32],
    filter: &[u32],
    options: &MLConv2dOptions,
) -> Result<Vec<u32>, Error> {
    if input.len() != 4 || filter.len() != 4 {
        return Err(Error::Type(
            c"conv2d requires 4-D input and filter.".to_owned(),
        ));
    }
    let strides = options.strides.clone().unwrap_or_else(|| vec![1, 1]);
    let dilations = options.dilations.clone().unwrap_or_else(|| vec![1, 1]);
    let padding = options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || dilations.len() != 2 || padding.len() != 4 {
        return Err(Error::Type(c"Invalid conv2d options.".to_owned()));
    }

    // The input spatial dimensions depend on the input layout; NHWC stores
    // channels last.
    let (input_h, input_w) = match options.inputLayout {
        MLInputOperandLayout::Nchw => (input[2], input[3]),
        MLInputOperandLayout::Nhwc => (input[1], input[2]),
    };

    // The output channel count and filter spatial dimensions depend on the
    // filter layout.
    let (output_channels, filter_h, filter_w) = match options.filterLayout {
        MLConv2dFilterOperandLayout::Oihw => (filter[0], filter[2], filter[3]),
        MLConv2dFilterOperandLayout::Hwio => (filter[3], filter[0], filter[1]),
        MLConv2dFilterOperandLayout::Ohwi => (filter[0], filter[1], filter[2]),
        MLConv2dFilterOperandLayout::Ihwo => (filter[3], filter[1], filter[2]),
    };

    let output_height = conv_dim(
        input_h,
        filter_h,
        padding[0] + padding[1],
        strides[0],
        dilations[0],
    )?;
    let output_width = conv_dim(
        input_w,
        filter_w,
        padding[2] + padding[3],
        strides[1],
        dilations[1],
    )?;

    match options.inputLayout {
        MLInputOperandLayout::Nchw => {
            Ok(vec![input[0], output_channels, output_height, output_width])
        },
        MLInputOperandLayout::Nhwc => {
            Ok(vec![input[0], output_height, output_width, output_channels])
        },
    }
}

/// Computes a single pooling output dimension.
fn pool_dim(
    input: u32,
    window: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
    ceil: bool,
) -> Result<u32, Error> {
    if stride == 0 || dilation == 0 {
        return Err(Error::Type(
            c"Stride and dilation must be greater than 0.".to_owned(),
        ));
    }
    let window_extent = (dilation as i64)
        .checked_mul(window as i64 - 1)
        .ok_or_else(|| Error::Type(c"Window extent is too large.".to_owned()))?;
    let numerator = input as i64 + padding as i64 - window_extent - 1;
    if numerator < 0 {
        return Err(Error::Type(
            c"Output dimension is invalid for the given input, window, and padding.".to_owned(),
        ));
    }
    let dim = if ceil {
        (numerator + stride as i64 - 1) / stride as i64 + 1
    } else {
        numerator / stride as i64 + 1
    };
    u32::try_from(dim).map_err(|_| Error::Type(c"Output dimension is too large.".to_owned()))
}

/// Computes the output shape of a 2D pooling operation for the supported
/// layouts.
fn pool2d_output_shape(input: &[u32], options: &MLPool2dOptions) -> Result<Vec<u32>, Error> {
    if input.len() != 4 {
        return Err(Error::Type(c"pool2d requires a 4-D input.".to_owned()));
    }
    let window = options.windowDimensions.clone().unwrap_or_default();
    if window.len() != 2 {
        return Err(Error::Type(
            c"pool2d requires 2 window dimensions.".to_owned(),
        ));
    }
    let strides = options.strides.clone().unwrap_or_else(|| vec![1, 1]);
    let dilations = options.dilations.clone().unwrap_or_else(|| vec![1, 1]);
    let padding = options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || dilations.len() != 2 || padding.len() != 4 {
        return Err(Error::Type(c"Invalid pool2d options.".to_owned()));
    }

    // The spatial dimensions depend on the layout; NHWC stores channels last.
    let (input_h, input_w) = match options.layout {
        MLInputOperandLayout::Nchw => (input[2], input[3]),
        MLInputOperandLayout::Nhwc => (input[1], input[2]),
    };

    let (output_height, output_width) = match &options.outputSizes {
        Some(sizes) if sizes.len() == 2 => (sizes[0], sizes[1]),
        _ => {
            let ceil = matches!(options.outputShapeRounding, MLRoundingType::Ceil);
            (
                pool_dim(
                    input_h,
                    window[0],
                    padding[0]
                        .checked_add(padding[1])
                        .ok_or_else(|| Error::Type(c"Padding is too large.".to_owned()))?,
                    strides[0],
                    dilations[0],
                    ceil,
                )?,
                pool_dim(
                    input_w,
                    window[1],
                    padding[2]
                        .checked_add(padding[3])
                        .ok_or_else(|| Error::Type(c"Padding is too large.".to_owned()))?,
                    strides[1],
                    dilations[1],
                    ceil,
                )?,
            )
        },
    };

    match options.layout {
        MLInputOperandLayout::Nchw => Ok(vec![input[0], input[1], output_height, output_width]),
        MLInputOperandLayout::Nhwc => Ok(vec![input[0], output_height, output_width, input[3]]),
    }
}

/// Computes the output shape of `resample2d` (NCHW).
fn resample2d_output_shape(
    input: &[u32],
    options: &MLResample2dOptions,
) -> Result<Vec<u32>, Error> {
    if input.len() != 4 {
        return Err(Error::Type(c"resample2d requires a 4-D input.".to_owned()));
    }
    // The spatial axes to resize default to the last two axes (NCHW).
    let axes = options.axes.clone().unwrap_or_else(|| vec![2, 3]);
    if axes.len() != 2 {
        return Err(Error::Type(c"resample2d requires 2 axes.".to_owned()));
    }
    let (axis_h, axis_w) = (axes[0] as usize, axes[1] as usize);
    if axis_h >= input.len() || axis_w >= input.len() {
        return Err(Error::Type(c"resample2d axis is out of bounds.".to_owned()));
    }
    let mut output = input.to_vec();
    if let Some(sizes) = &options.sizes {
        if sizes.len() != 2 {
            return Err(Error::Type(
                c"resample2d requires 2 output sizes.".to_owned(),
            ));
        }
        output[axis_h] = sizes[0];
        output[axis_w] = sizes[1];
        return Ok(output);
    }
    if let Some(scales) = &options.scales {
        if scales.len() != 2 {
            return Err(Error::Type(c"resample2d requires 2 scales.".to_owned()));
        }
        output[axis_h] = (output[axis_h] as f32 * (*scales[0])).floor() as u32;
        output[axis_w] = (output[axis_w] as f32 * (*scales[1])).floor() as u32;
        return Ok(output);
    }
    Err(Error::Type(
        c"resample2d requires either sizes or scales.".to_owned(),
    ))
}

/// Computes the output shape of `concat`.
fn concat_shape(inputs: &[&[u32]], axis: u32) -> Result<Vec<u32>, Error> {
    let Some(first) = inputs.first() else {
        return Err(Error::Type(
            c"concat requires at least one input.".to_owned(),
        ));
    };
    let rank = first.len();
    if axis as usize >= rank {
        return Err(Error::Type(c"concat axis is out of bounds.".to_owned()));
    }
    let mut output = first.to_vec();
    let mut axis_total = 0u32;
    for shape in inputs {
        if shape.len() != rank {
            return Err(Error::Type(
                c"concat inputs must have the same rank.".to_owned(),
            ));
        }
        for dim in 0..rank {
            if dim == axis as usize {
                axis_total += shape[dim];
            } else if shape[dim] != first[dim] {
                return Err(Error::Type(
                    c"concat inputs must match on non-axis dimensions.".to_owned(),
                ));
            }
        }
    }
    output[axis as usize] = axis_total;
    Ok(output)
}

/// Computes the output shape of `reshape`, resolving the `-1` inference
/// dimension and `0` copy-dimension sentinels.
fn reshape_shape(input_shape: &[u32], new_shape: &[i32]) -> Result<Vec<u32>, Error> {
    let input_elements: usize = input_shape
        .iter()
        .map(|&dimension| dimension as usize)
        .product();
    let mut output = Vec::with_capacity(new_shape.len());
    let mut inferred_index = None;
    let mut known_elements: usize = 1;
    for (index, &dimension) in new_shape.iter().enumerate() {
        if dimension == -1 {
            if inferred_index.is_some() {
                return Err(Error::Type(
                    c"reshape newShape may contain at most one -1.".to_owned(),
                ));
            }
            inferred_index = Some(index);
            output.push(1);
        } else if dimension == 0 {
            let dim = *input_shape
                .get(index)
                .ok_or_else(|| Error::Type(c"reshape 0 dimension is out of bounds.".to_owned()))?;
            output.push(dim);
            known_elements *= dim as usize;
        } else if dimension > 0 {
            output.push(dimension as u32);
            known_elements *= dimension as usize;
        } else {
            return Err(Error::Type(
                c"reshape newShape contains an invalid dimension.".to_owned(),
            ));
        }
    }
    if let Some(index) = inferred_index {
        if known_elements == 0 || input_elements % known_elements != 0 {
            return Err(Error::Type(
                c"reshape newShape is incompatible with the input.".to_owned(),
            ));
        }
        output[index] = (input_elements / known_elements) as u32;
    } else if known_elements != input_elements {
        return Err(Error::Type(
            c"reshape newShape does not match the input element count.".to_owned(),
        ));
    }
    Ok(output)
}

/// Computes the output shape of `transpose`.
fn transpose_shape(input_shape: &[u32], permutation: &[u32]) -> Result<Vec<u32>, Error> {
    let rank = input_shape.len();
    if permutation.len() != rank {
        return Err(Error::Type(
            c"transpose permutation must match the input rank.".to_owned(),
        ));
    }
    let mut seen = vec![false; rank];
    let mut output = vec![0u32; rank];
    for (index, &axis) in permutation.iter().enumerate() {
        let axis = axis as usize;
        if axis >= rank || seen[axis] {
            return Err(Error::Type(
                c"transpose permutation must be a permutation of the axes.".to_owned(),
            ));
        }
        seen[axis] = true;
        output[index] = input_shape[axis];
    }
    Ok(output)
}

/// Computes the output shape of `slice`, resolving `0` sizes to "the rest of
/// the input dimension from the corresponding start".
fn slice_shape(input_shape: &[u32], starts: &[u32], sizes: &[u32]) -> Result<Vec<u32>, Error> {
    if sizes.len() != input_shape.len() || starts.len() != input_shape.len() {
        return Err(Error::Type(
            c"slice starts and sizes must match the input rank.".to_owned(),
        ));
    }
    let mut output = Vec::with_capacity(sizes.len());
    for index in 0..sizes.len() {
        let size = if sizes[index] == 0 {
            input_shape[index]
                .checked_sub(starts[index])
                .ok_or_else(|| Error::Type(c"slice start is out of bounds.".to_owned()))?
        } else {
            sizes[index]
        };
        output.push(size);
    }
    Ok(output)
}

/// Computes the per-output shapes of `split`.
fn split_shapes(input_shape: &[u32], splits: &[u32], axis: u32) -> Result<Vec<Vec<u32>>, Error> {
    if axis as usize >= input_shape.len() {
        return Err(Error::Type(c"split axis is out of bounds.".to_owned()));
    }
    if splits.iter().sum::<u32>() != input_shape[axis as usize] {
        return Err(Error::Type(
            c"split sizes do not sum to the input dimension.".to_owned(),
        ));
    }
    let mut outputs = Vec::with_capacity(splits.len());
    for &size in splits {
        let mut shape = input_shape.to_vec();
        shape[axis as usize] = size;
        outputs.push(shape);
    }
    Ok(outputs)
}

/// Computes the output shape of `reduceSum`. An empty `axes` reduces over all
/// dimensions.
fn reduce_shape(
    input_shape: &[u32],
    axes: &[u32],
    keep_dimensions: bool,
) -> Result<Vec<u32>, Error> {
    let rank = input_shape.len();
    let mut reduced = vec![false; rank];
    if axes.is_empty() {
        for flag in &mut reduced {
            *flag = true;
        }
    } else {
        for &axis in axes {
            if axis as usize >= rank {
                return Err(Error::Type(c"reduceSum axis is out of bounds.".to_owned()));
            }
            reduced[axis as usize] = true;
        }
    }
    let mut output = Vec::new();
    for (dimension, &is_reduced) in input_shape.iter().copied().zip(reduced.iter()) {
        if is_reduced {
            if keep_dimensions {
                output.push(1);
            }
        } else {
            output.push(dimension);
        }
    }
    Ok(output)
}

/// Computes the output shape of `matmul`, broadcasting batch dimensions.
fn matmul_output_shape(a: &[u32], b: &[u32]) -> Result<Vec<u32>, Error> {
    if a.len() < 2 || b.len() < 2 {
        return Err(Error::Type(
            c"matmul requires at least 2-D inputs.".to_owned(),
        ));
    }
    let (m, k_a) = (a[a.len() - 2], a[a.len() - 1]);
    let (k_b, n) = (b[b.len() - 2], b[b.len() - 1]);
    if k_a != k_b {
        return Err(Error::Type(
            c"matmul inner dimensions do not match.".to_owned(),
        ));
    }
    let mut output = bidirectionally_broadcast(&a[..a.len() - 2], &b[..b.len() - 2])
        .map_err(|_| Error::Type(c"matmul batch dimensions are not broadcastable.".to_owned()))?;
    output.push(m);
    output.push(n);
    Ok(output)
}

/// Computes the output shape of `gemm` (2-D `[M, N]`).
fn gemm_output_shape(a: &[u32], b: &[u32], options: &MLGemmOptions) -> Result<Vec<u32>, Error> {
    if a.len() != 2 || b.len() != 2 {
        return Err(Error::Type(c"gemm requires 2-D inputs.".to_owned()));
    }
    let (m, k_a) = if options.aTranspose {
        (a[1], a[0])
    } else {
        (a[0], a[1])
    };
    let (k_b, n) = if options.bTranspose {
        (b[1], b[0])
    } else {
        (b[0], b[1])
    };
    if k_a != k_b {
        return Err(Error::Type(
            c"gemm inner dimensions do not match.".to_owned(),
        ));
    }
    Ok(vec![m, n])
}

/// Computes the output shape of a 2D transposed convolution.
fn conv_transpose2d_output_shape(
    input: &[u32],
    filter: &[u32],
    options: &MLConvTranspose2dOptions,
) -> Result<Vec<u32>, Error> {
    if input.len() != 4 || filter.len() != 4 {
        return Err(Error::Type(
            c"convTranspose2d requires 4-D input and filter.".to_owned(),
        ));
    }
    let strides = options.strides.clone().unwrap_or_else(|| vec![1, 1]);
    let dilations = options.dilations.clone().unwrap_or_else(|| vec![1, 1]);
    let padding = options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
    let output_padding = options.outputPadding.clone().unwrap_or_default();
    if strides.len() != 2 || dilations.len() != 2 || padding.len() != 4 {
        return Err(Error::Type(c"Invalid convTranspose2d options.".to_owned()));
    }
    let output_padding = if output_padding.is_empty() {
        vec![0, 0]
    } else if output_padding.len() == 2 {
        output_padding
    } else {
        return Err(Error::Type(
            c"convTranspose2d outputPadding must have 2 values.".to_owned(),
        ));
    };

    let (batch, input_h, input_w) = match options.inputLayout {
        MLInputOperandLayout::Nchw => (input[0], input[2], input[3]),
        MLInputOperandLayout::Nhwc => (input[0], input[1], input[2]),
    };

    // Output channels (per group) and kernel dims depend on the filter layout.
    let (out_channels_per_group, kernel_h, kernel_w) = match options.filterLayout {
        MLConvTranspose2dFilterOperandLayout::Iohw => (filter[1], filter[2], filter[3]),
        MLConvTranspose2dFilterOperandLayout::Ohwi => (filter[0], filter[1], filter[2]),
        MLConvTranspose2dFilterOperandLayout::Hwoi => (filter[2], filter[0], filter[1]),
    };
    let groups = if options.groups == 0 {
        1
    } else {
        options.groups
    };
    let output_channels = out_channels_per_group
        .checked_mul(groups)
        .ok_or_else(|| Error::Type(c"Output channels are too large.".to_owned()))?;

    if strides[0] == 0 || strides[1] == 0 || dilations[0] == 0 || dilations[1] == 0 {
        return Err(Error::Type(
            c"Stride and dilation must be greater than 0.".to_owned(),
        ));
    }
    let effective_kernel_h = (dilations[0] as i64) * (kernel_h as i64 - 1) + 1;
    let effective_kernel_w = (dilations[1] as i64) * (kernel_w as i64 - 1) + 1;

    let output_extent = |input: u32,
                         effective_kernel: i64,
                         stride: u32,
                         pad_begin: u32,
                         pad_end: u32,
                         output_padding: u32|
     -> Result<u32, Error> {
        let extent = (input as i64 - 1) * stride as i64 + effective_kernel - pad_begin as i64
            - pad_end as i64
            + output_padding as i64;
        u32::try_from(extent).map_err(|_| {
            Error::Type(c"Output dimension is invalid for the given options.".to_owned())
        })
    };

    let output_h = match &options.outputSizes {
        Some(sizes) if sizes.len() == 2 => sizes[0],
        _ => output_extent(
            input_h,
            effective_kernel_h,
            strides[0],
            padding[0],
            padding[1],
            output_padding[0],
        )?,
    };
    let output_w = match &options.outputSizes {
        Some(sizes) if sizes.len() == 2 => sizes[1],
        _ => output_extent(
            input_w,
            effective_kernel_w,
            strides[1],
            padding[2],
            padding[3],
            output_padding[1],
        )?,
    };

    match options.inputLayout {
        MLInputOperandLayout::Nchw => Ok(vec![batch, output_channels, output_h, output_w]),
        MLInputOperandLayout::Nhwc => Ok(vec![batch, output_h, output_w, output_channels]),
    }
}

/// <https://www.w3.org/TR/webnn/#mlgraphbuilder>
#[dom_struct]
pub(crate) struct MLGraphBuilder {
    reflector_: Reflector,
    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-context-slot
    context: WeakRef<MLContext>,
    #[no_trace]
    #[ignore_malloc_size_of = "GenericSender"]
    channel: webnn::WebNN,
    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-context-slot>
    has_built: Cell<bool>,
    builder_id: Cell<webnn::BuilderId>,
    /// Identifier of this builder's context on the shared WebNN backend thread.
    #[no_trace]
    context_id: webnn::ContextId,
    next_operand_id: Cell<webnn::OperandId>,
    /// Names of the graph's input operands, used to reject duplicate input names.
    input_names: DomRefCell<HashMap<String, webnn::OperandId>>,
}

impl MLGraphBuilder {
    pub(crate) fn new_inherited(context: &MLContext) -> MLGraphBuilder {
        let channel = context.channel().clone();
        let context_id = context.context_id();
        let id = channel.create_builder(context_id);
        MLGraphBuilder {
            reflector_: Reflector::new(),
            context: WeakRef::new(context),
            channel,
            has_built: Cell::new(false),
            builder_id: Cell::new(id),
            context_id,
            next_operand_id: Cell::new(1),
            input_names: DomRefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn new(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        context: &MLContext,
        cx: &mut JSContext,
    ) -> DomRoot<MLGraphBuilder> {
        reflect_weak_referenceable_dom_object_with_proto(
            cx,
            Rc::new(Self::new_inherited(context)),
            global,
            proto,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn context(&self) -> &WeakRef<MLContext> {
        &self.context
    }

    fn next_operand_id(&self) -> webnn::OperandId {
        let id = self.next_operand_id.get();
        self.next_operand_id.set(id + 1);
        id
    }

    /// <https://www.w3.org/TR/webnn/#mlgraphbuilder-can-build>
    pub(crate) fn can_build(&self) -> bool {
        // > An MLGraphBuilder can build if its [[hasBuilt]] is false and its [[context]] is not lost.
        if self.has_built.get() {
            return false;
        }
        if let Some(context) = self.context.root() {
            if context.is_lost() {
                return false;
            }
        }
        true
    }

    /// <https://www.w3.org/TR/webnn/#mlgraphbuilder-element-wise-binary-op>
    fn create_element_wise_binary(
        &self,
        cx: &mut JSContext,
        operator: webnn::Operator,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 2. If this can not build, then throw an "InvalidStateError"
        // DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 3. If validating operand with this and any of a and b returns
        // false, then throw a TypeError.
        if !validate_operand(self, a) || !validate_operand(self, b) {
            return Err(Error::Type(cformat!(
                "Input is from another builder. [{}]",
                options.label.0
            )));
        }
        // Step 4. If a's dataType is not equal to b's dataType, then throw a
        // TypeError.
        if a.data_type() != b.data_type() {
            return Err(Error::Type(cformat!(
                "Inputs must have the same data type. [{}]",
                options.label.0
            )));
        }
        // Step 5. Let outputShape be the result of bidirectionally broadcasting
        // a's shape and b's shape.
        let shape = bidirectionally_broadcast(a.shape(), b.shape()).map_err(|_| {
            Error::Type(cformat!(
                "Input shapes are not broadcastable. [{}]",
                options.label.0
            ))
        })?;
        // Steps 6-7. Record the operator on the backend and create the output
        // operand.
        let input_ids = [a.operand_id(), b.operand_id()];
        Ok(self.add_single_output_operator(
            cx,
            operator,
            &input_ids,
            a.data_type(),
            shape,
            options.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#mlgraphbuilder-element-wise-unary-op>
    fn create_element_wise_unary(
        &self,
        cx: &mut JSContext,
        operator: webnn::Operator,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this can not build, then throw an "InvalidStateError"
        // DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. If validating operand with this and input returns false,
        // then throw a TypeError.
        if !validate_operand(self, input) {
            return Err(Error::Type(cformat!(
                "Input is from another builder. [{}]",
                options.label.0
            )));
        }
        // Step 3. Make graph connections: the output has the same shape and
        // data type as the input.
        let shape = input.shape().to_vec();
        Ok(self.add_single_output_operator(
            cx,
            operator,
            &[input.operand_id()],
            input.data_type(),
            shape,
            options.label.0.as_str(),
        ))
    }

    /// Records a single-output operator on the backend and returns the output
    /// operand.
    fn add_single_output_operator(
        &self,
        cx: &mut JSContext,
        operator: webnn::Operator,
        inputs: &[webnn::OperandId],
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        label: &str,
    ) -> DomRoot<MLOperand> {
        let operand_id = self.next_operand_id();
        self.channel.add_operator(
            self.context_id,
            self.builder_id.get(),
            &operator,
            inputs,
            &[(operand_id, data_type as u32, shape.clone())],
            label,
        );
        MLOperand::new(
            &self.global(),
            operand_id,
            data_type,
            shape,
            self,
            false,
            false,
            cx,
        )
    }

    /// Records a multi-output `split` operator on the backend and returns the
    /// output operands.
    fn add_split_operator(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        splits: Vec<u32>,
        axis: u32,
        label: &str,
    ) -> Result<Vec<DomRoot<MLOperand>>, Error> {
        let data_type = input.data_type();
        let shapes = split_shapes(input.shape(), &splits, axis)?;
        let mut output_specs = Vec::with_capacity(shapes.len());
        let mut outputs = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let operand_id = self.next_operand_id();
            output_specs.push((operand_id, data_type as u32, shape.clone()));
            outputs.push(MLOperand::new(
                &self.global(),
                operand_id,
                data_type,
                shape,
                self,
                false,
                false,
                cx,
            ));
        }
        self.channel.add_operator(
            self.context_id,
            self.builder_id.get(),
            &webnn::Operator::Split { splits, axis },
            &[input.operand_id()],
            &output_specs,
            label,
        );
        Ok(outputs)
    }
}

impl MLGraphBuilderMethods<crate::DomTypeHolder> for MLGraphBuilder {
    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-mlgraphbuilder>
    fn Constructor(
        cx: &mut JSContext,
        global: &GlobalScope,
        proto: Option<HandleObject>,
        context: &MLContext,
    ) -> Result<DomRoot<MLGraphBuilder>, Error> {
        // Step 1. If this's relevant global object's associated Document is
        // not allowed to use the webnn feature, then throw a "SecurityError"
        // DOMException.
        let window = global.as_window();
        let document = window.Document();
        if !document.allowed_to_use_feature(PermissionName::WebNN) {
            return Err(Error::Security(Some("WebNN not allowed to use".into())));
        }
        // Step 2. If context is lost, then throw an "InvalidStateError"
        // DOMException.
        if context.is_lost() {
            return Err(Error::InvalidState(Some(
                "Cannot construct MLGraphBuilder: context is lost.".to_owned(),
            )));
        }
        // Step 3. Set this.[[context]] to context.
        // Step 4. Set this.[[hasBuilt]] to false.
        Ok(MLGraphBuilder::new(global, proto, context, cx))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-input>
    fn Input(
        &self,
        cx: &mut JSContext,
        name: USVString,
        descriptor: &MLOperandDescriptor,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        };
        // Step 2. If name is empty, throw a TypeError.
        if name.0.is_empty() {
            return Err(Error::Type(c"The name is empty.".to_owned()));
        }
        // Step 3. If any MLOperands in this's graph's inputs have a [[name]]
        // equal to name, then throw a TypeError.
        if self.input_names.borrow().contains_key(name.0.as_str()) {
            return Err(Error::Type(cformat!(
                "Input name '{}' is already used.",
                name.0
            )));
        }
        // Step 4. If checking dimensions given descriptor returns false, throw a TypeError.
        if !check_dimensions(descriptor) {
            return Err(Error::Type(c"A dimension size cannot be 0.".to_owned()));
        }

        // Step 5. Make graph connections:
        let operand_id = self.next_operand_id();
        self.channel.add_input(
            self.context_id,
            self.builder_id.get(),
            operand_id,
            name.0.as_str(),
            descriptor.dataType as u32,
            &descriptor.shape,
        );
        self.input_names
            .borrow_mut()
            .insert(name.0.clone(), operand_id);
        let operand = MLOperand::new(
            &self.global(),
            operand_id,
            descriptor.dataType,
            descriptor.shape.clone(),
            self,
            true,
            false,
            cx,
        );
        // Step 6. Return operand.
        Ok(operand)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-constant>
    fn Constant(
        &self,
        cx: &mut JSContext,
        descriptor: &MLOperandDescriptor,
        buffer: MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        };
        // Step 2. If checking dimensions given descriptor returns false, throw a TypeError.
        if !check_dimensions(descriptor) {
            return Err(Error::Type(c"A dimension size cannot be 0.".to_owned()));
        }
        // Step 3. If validating buffer with descriptor returns false, throw a TypeError.
        if !validate_buffer_with_descriptor(descriptor, &buffer) {
            return Err(Error::Type(
                c"Buffer size does not match the expected size for the operand descriptor."
                    .to_owned(),
            ));
        }
        // Step 4. Make graph connections:
        let bytes = match &buffer {
            MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer::ArrayBufferView(view) => {
                get_buffer_source_copy(view.into())
            },
            MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer::ArrayBuffer(buffer) => {
                get_buffer_source_copy(buffer.into())
            },
        };
        let operand_id = self.next_operand_id();
        self.channel.add_constant(
            self.context_id,
            self.builder_id.get(),
            operand_id,
            descriptor.dataType as u32,
            &descriptor.shape,
            &bytes,
        );
        let operand = MLOperand::new(
            &self.global(),
            operand_id,
            descriptor.dataType,
            descriptor.shape.clone(),
            self,
            false,
            true,
            cx,
        );
        // Step 5. Return operand.
        Ok(operand)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-build>
    fn Build(
        &self,
        cx: &mut JSContext,
        outputs: Record<USVString, DomRoot<MLOperand>>,
    ) -> Result<Rc<Promise>, Error> {
        let global = &self.global();
        // Step 2. If this can not build, then return a new promise rejected
        // with an "InvalidStateError" DOMException.
        if !self.can_build() {
            let promise = Promise::new(cx, global);
            promise.reject_error(cx, Error::InvalidState(Some("Cannot build.".into())));
            return Ok(promise);
        }
        // Step 3. If outputs is empty, then return a new promise rejected
        // with a TypeError.
        if outputs.iter().next().is_none() {
            let promise = Promise::new(cx, global);
            promise.reject_error(cx, Error::Type(c"outputs is empty.".to_owned()));
            return Ok(promise);
        }
        // Step 4. For each name -> operand of outputs, validate.
        for (name, operand) in outputs.iter() {
            if name.0.is_empty() {
                let promise = Promise::new(cx, global);
                promise.reject_error(cx, Error::Type(c"output name is empty.".to_owned()));
                return Ok(promise);
            }
            if !validate_operand(self, &operand) {
                let promise = Promise::new(cx, global);
                promise.reject_error(
                    cx,
                    Error::Type(cformat!(
                        "Output operand is from another builder. [{}]",
                        name.0
                    )),
                );
                return Ok(promise);
            }
            if operand.is_input() || operand.is_constant() {
                let promise = Promise::new(cx, global);
                promise.reject_error(
                    cx,
                    Error::Type(c"Output operand cannot be an input or constant.".to_owned()),
                );
                return Ok(promise);
            }
        }
        // Step 11. Let graph be a new MLGraph in realm.
        let graph = MLGraph::new(global, &*self.context.root().unwrap(), cx);
        // Step 15. Record the output descriptors on the graph.
        let mut output_descriptors: HashMap<String, (MLOperandDataType, Vec<u32>)> = HashMap::new();
        for (name, operand) in outputs.iter() {
            output_descriptors.insert(
                name.0.clone(),
                (operand.data_type(), operand.shape().to_vec()),
            );
        }
        graph.set_output_descriptors(output_descriptors);
        let output_pairs: Vec<(String, webnn::OperandId)> = outputs
            .iter()
            .map(|(label, operand)| (label.0.clone(), operand.operand_id()))
            .collect();
        // Step 16. Set this.[[hasBuilt]] to true.
        self.has_built.set(true);
        // Step 17. Let promise be a new promise in realm.
        let promise = Promise::new(cx, &self.global());
        // Step 18. Build runs asynchronously on the backend thread; the
        // callback resolves/rejects the promise.
        let callback = callback_promise(
            &promise,
            &*graph,
            self.global().task_manager().ml_task_source(),
        );
        self.channel.build(
            self.context_id,
            self.builder_id.get(),
            &output_pairs,
            callback,
        );
        // Step 19. Return promise.
        Ok(promise)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-add>
    fn Add(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Add, a, b, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sub>
    fn Sub(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Sub, a, b, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-mul>
    fn Mul(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Mul, a, b, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-div>
    fn Div(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Div, a, b, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-min>
    fn Min(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Min, a, b, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-max>
    fn Max(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Max, a, b, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-matmul>
    fn Matmul(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operands.
        if !validate_operand(self, a) || !validate_operand(self, b) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        // Step 3. If a's dataType is not equal to b's dataType, throw a TypeError.
        if a.data_type() != b.data_type() {
            return Err(Error::Type(
                c"Inputs must have the same data type.".to_owned(),
            ));
        }
        let output_shape = matmul_output_shape(a.shape(), b.shape())?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Matmul,
            &[a.operand_id(), b.operand_id()],
            a.data_type(),
            output_shape,
            options.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gemm>
    fn Gemm(
        &self,
        cx: &mut JSContext,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLGemmOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operands.
        if !validate_operand(self, a) || !validate_operand(self, b) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        if a.data_type() != b.data_type() {
            return Err(Error::Type(
                c"Inputs must have the same data type.".to_owned(),
            ));
        }
        if let Some(c) = &options.c {
            if !validate_operand(self, c) {
                return Err(Error::Type(c"gemm c is from another builder.".to_owned()));
            }
        }
        let output_shape = gemm_output_shape(a.shape(), b.shape(), options)?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Gemm(webnn::GemmOptions {
                c: options.c.as_ref().map(|operand| operand.operand_id()),
                alpha: *options.alpha,
                beta: *options.beta,
                a_transpose: options.aTranspose,
                b_transpose: options.bTranspose,
            }),
            &[a.operand_id(), b.operand_id()],
            a.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-prelu>
    fn Prelu(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        slope: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_binary(cx, webnn::Operator::Prelu, input, slope, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sigmoid>
    fn Sigmoid(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_unary(cx, webnn::Operator::Sigmoid, input, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-relu>
    fn Relu(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_unary(cx, webnn::Operator::Relu, input, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-sqrt>
    fn Sqrt(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_unary(cx, webnn::Operator::Sqrt, input, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reciprocal>
    fn Reciprocal(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_unary(cx, webnn::Operator::Reciprocal, input, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-gelu>
    fn Gelu(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        self.create_element_wise_unary(cx, webnn::Operator::Gelu, input, options)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-leakyrelu>
    fn LeakyRelu(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLLeakyReluOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        // Step 3. The output has the same shape and data type as the input.
        let shape = input.shape().to_vec();
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::LeakyRelu {
                alpha: *options.alpha,
            },
            &[input.operand_id()],
            input.data_type(),
            shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-cast>
    fn Cast(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        output_data_type: MLOperandDataType,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. If validating operand with this and input returns false,
        // then throw a TypeError.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        // Step 3. Make graph connections: the output has the same shape as the
        // input but the requested output data type.
        let shape = input.shape().to_vec();
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Cast {
                output_data_type: output_data_type as u32,
            },
            &[input.operand_id()],
            output_data_type,
            shape,
            options.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-concat>
    fn Concat(
        &self,
        cx: &mut JSContext,
        inputs: Vec<DomRoot<MLOperand>>,
        axis: u32,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. If inputs is empty, throw a TypeError.
        if inputs.is_empty() {
            return Err(Error::Type(
                c"concat requires at least one input.".to_owned(),
            ));
        }
        // Step 3. Validate the operands.
        let first = &inputs[0];
        for input in &inputs {
            if !validate_operand(self, input) {
                return Err(Error::Type(c"Input is from another builder.".to_owned()));
            }
            if input.data_type() != first.data_type() {
                return Err(Error::Type(
                    c"Inputs must have the same data type.".to_owned(),
                ));
            }
        }
        let shapes: Vec<&[u32]> = inputs.iter().map(|operand| operand.shape()).collect();
        let output_shape = concat_shape(&shapes, axis)?;
        let input_ids: Vec<webnn::OperandId> =
            inputs.iter().map(|operand| operand.operand_id()).collect();
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Concat { axis },
            &input_ids,
            first.data_type(),
            output_shape,
            options.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-conv2d>
    fn Conv2d(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        filter: &MLOperand,
        options: &MLConv2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operands.
        if !validate_operand(self, input) || !validate_operand(self, filter) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        // Step 3. If input's dataType is not equal to filter's dataType, throw
        // a TypeError.
        if input.data_type() != filter.data_type() {
            return Err(Error::Type(
                c"conv2d input and filter must have the same data type.".to_owned(),
            ));
        }
        if let Some(bias) = &options.bias {
            if !validate_operand(self, bias) {
                return Err(Error::Type(
                    c"conv2d bias is from another builder.".to_owned(),
                ));
            }
            if bias.data_type() != input.data_type() {
                return Err(Error::Type(
                    c"conv2d bias must have the same data type as the input.".to_owned(),
                ));
            }
        }
        let output_shape = conv2d_output_shape(input.shape(), filter.shape(), options)?;
        let backend_options = webnn::Conv2dOptions {
            padding: options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]),
            strides: options.strides.clone().unwrap_or_else(|| vec![1, 1]),
            dilations: options.dilations.clone().unwrap_or_else(|| vec![1, 1]),
            groups: options.groups,
            input_layout: input_layout_str(options.inputLayout).to_string(),
            filter_layout: filter_layout_str(options.filterLayout).to_string(),
            bias: options.bias.as_ref().map(|operand| operand.operand_id()),
        };
        let input_ids = [input.operand_id(), filter.operand_id()];
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Conv2d(backend_options),
            &input_ids,
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-convtranspose2d>
    fn ConvTranspose2d(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        filter: &MLOperand,
        options: &MLConvTranspose2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operands.
        if !validate_operand(self, input) || !validate_operand(self, filter) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        if input.data_type() != filter.data_type() {
            return Err(Error::Type(
                c"convTranspose2d input and filter must have the same data type.".to_owned(),
            ));
        }
        if let Some(bias) = &options.bias {
            if !validate_operand(self, bias) {
                return Err(Error::Type(
                    c"convTranspose2d bias is from another builder.".to_owned(),
                ));
            }
        }
        let output_shape = conv_transpose2d_output_shape(input.shape(), filter.shape(), options)?;
        let backend_options = webnn::ConvTranspose2dOptions {
            padding: options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]),
            strides: options.strides.clone().unwrap_or_else(|| vec![1, 1]),
            dilations: options.dilations.clone().unwrap_or_else(|| vec![1, 1]),
            output_padding: options.outputPadding.clone().unwrap_or_default(),
            output_sizes: options.outputSizes.clone(),
            groups: options.groups,
            input_layout: input_layout_str(options.inputLayout).to_string(),
            filter_layout: conv_transpose_filter_layout_str(options.filterLayout).to_string(),
            bias: options.bias.as_ref().map(|operand| operand.operand_id()),
        };
        let input_ids = [input.operand_id(), filter.operand_id()];
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::ConvTranspose2d(backend_options),
            &input_ids,
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-maxpool2d>
    fn MaxPool2d(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLPool2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let output_shape = pool2d_output_shape(input.shape(), options)?;
        let backend_options = webnn::Pool2dOptions {
            window_dimensions: options.windowDimensions.clone().unwrap_or_default(),
            padding: options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]),
            strides: options.strides.clone().unwrap_or_else(|| vec![1, 1]),
            dilations: options.dilations.clone().unwrap_or_else(|| vec![1, 1]),
            layout: input_layout_str(options.layout).to_string(),
            output_shape_rounding: rounding_type_str(options.outputShapeRounding).to_string(),
            output_sizes: options.outputSizes.clone(),
        };
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::MaxPool2d(backend_options),
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-averagepool2d>
    fn AveragePool2d(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLPool2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let output_shape = pool2d_output_shape(input.shape(), options)?;
        let backend_options = webnn::Pool2dOptions {
            window_dimensions: options.windowDimensions.clone().unwrap_or_default(),
            padding: options.padding.clone().unwrap_or_else(|| vec![0, 0, 0, 0]),
            strides: options.strides.clone().unwrap_or_else(|| vec![1, 1]),
            dilations: options.dilations.clone().unwrap_or_else(|| vec![1, 1]),
            layout: input_layout_str(options.layout).to_string(),
            output_shape_rounding: rounding_type_str(options.outputShapeRounding).to_string(),
            output_sizes: options.outputSizes.clone(),
        };
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::AveragePool2d(backend_options),
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-pad>
    fn Pad(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        beginning_padding: Vec<u32>,
        ending_padding: Vec<u32>,
        options: &MLPadOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        // Step 3. Compute the output shape from the padding.
        let output_shape = pad_output_shape(input.shape(), &beginning_padding, &ending_padding)?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Pad {
                beginning_padding,
                ending_padding,
                mode: padding_mode_str(options.mode).to_string(),
                value: *options.value,
            },
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reshape>
    fn Reshape(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        new_shape: Vec<i32>,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let output_shape = reshape_shape(input.shape(), &new_shape)?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Reshape {
                new_shape: output_shape.clone(),
            },
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-resample2d>
    fn Resample2d(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLResample2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let output_shape = resample2d_output_shape(input.shape(), options)?;
        let backend_options = webnn::Resample2dOptions {
            mode: interpolation_mode_str(options.mode).to_string(),
            scales: options
                .scales
                .as_ref()
                .map(|scales| scales.iter().map(|scale| **scale).collect())
                .unwrap_or_default(),
            sizes: options.sizes.clone(),
            axes: options.axes.clone().unwrap_or_default(),
        };
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Resample2d(backend_options),
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-slice>
    fn Slice(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        starts: Vec<u32>,
        sizes: Vec<u32>,
        options: &MLSliceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let output_shape = slice_shape(input.shape(), &starts, &sizes)?;
        let strides = options
            .strides
            .clone()
            .unwrap_or_else(|| vec![1u32; input.shape().len()]);
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Slice {
                starts,
                sizes: output_shape.clone(),
                strides,
            },
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-softmax>
    fn Softmax(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        axis: u32,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        if axis as usize >= input.shape().len() {
            return Err(Error::Type(c"softmax axis is out of bounds.".to_owned()));
        }
        let output_shape = input.shape().to_vec();
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Softmax { axis },
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-split>
    fn Split(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        splits: RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence,
        options: &MLSplitOptions,
    ) -> Result<Vec<DomRoot<MLOperand>>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let axis = options.axis;
        let axis_dim = *input
            .shape()
            .get(axis as usize)
            .ok_or_else(|| Error::Type(c"split axis is out of bounds.".to_owned()))?;
        let splits = match splits {
            RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence::RangeEnforcedUnsignedLong(
                count,
            ) => {
                if count == 0 || axis_dim % count != 0 {
                    return Err(Error::Type(
                        c"split count does not divide the input dimension.".to_owned(),
                    ));
                }
                let size = axis_dim / count;
                vec![size; count as usize]
            },
            RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence::RangeEnforcedUnsignedLongSequence(
                sizes,
            ) => sizes,
        };
        self.add_split_operator(cx, input, splits, axis, options.parent.label.0.as_str())
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-transpose>
    fn Transpose(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLTransposeOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let rank = input.shape().len();
        let permutation = options
            .permutation
            .clone()
            .unwrap_or_else(|| (0..rank as u32).rev().collect());
        let output_shape = transpose_shape(input.shape(), &permutation)?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::Transpose { permutation },
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducesum>
    fn ReduceSum(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let axes = options.axes.clone().unwrap_or_default();
        let output_shape = reduce_shape(input.shape(), &axes, options.keepDimensions)?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::ReduceSum(webnn::ReduceOptions {
                axes,
                keep_dimensions: options.keepDimensions,
            }),
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }

    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-reducemean>
    fn ReduceMean(
        &self,
        cx: &mut JSContext,
        input: &MLOperand,
        options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        // Step 1. If this cannot build, throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(Some("Cannot build.".to_owned())));
        }
        // Step 2. Validate the operand.
        if !validate_operand(self, input) {
            return Err(Error::Type(c"Input is from another builder.".to_owned()));
        }
        let axes = options.axes.clone().unwrap_or_default();
        let output_shape = reduce_shape(input.shape(), &axes, options.keepDimensions)?;
        Ok(self.add_single_output_operator(
            cx,
            webnn::Operator::ReduceMean(webnn::ReduceOptions {
                axes,
                keep_dimensions: options.keepDimensions,
            }),
            &[input.operand_id()],
            input.data_type(),
            output_shape,
            options.parent.label.0.as_str(),
        ))
    }
}
