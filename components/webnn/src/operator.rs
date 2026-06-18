/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use serde::{Deserialize, Serialize};

use crate::OperandId;

/// A typed description of a WebNN operator, including its per-operator
/// parameters and options.
///
/// This is the single source of truth for what the DOM layer requests and what
/// each backend consumes via [`crate::Backend::add_operator`]. Adding a new
/// operator means adding a variant here (plus any option struct) and handling it
/// in every backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Sigmoid,
    Prelu,
    Cast {
        /// Target data type as the WebIDL `MLOperandDataType` discriminant.
        output_data_type: u32,
    },
    Concat {
        axis: u32,
    },
    Softmax {
        axis: u32,
    },
    Reshape {
        /// Target shape with any `-1` inference dimension already resolved.
        new_shape: Vec<u32>,
    },
    Transpose {
        permutation: Vec<u32>,
    },
    Slice {
        starts: Vec<u32>,
        sizes: Vec<u32>,
        strides: Vec<u32>,
    },
    Split {
        /// Size of each output along `axis`. The DOM layer resolves the spec's
        /// "split into N equal parts" form into explicit sizes.
        splits: Vec<u32>,
        axis: u32,
    },
    Conv2d(Conv2dOptions),
    MaxPool2d(Pool2dOptions),
    Resample2d(Resample2dOptions),
    ReduceSum(ReduceOptions),
}

/// <https://www.w3.org/TR/webnn/#dictdef-mlreduceoptions>
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReduceOptions {
    /// Axes to reduce over. An empty list means "reduce over all axes".
    pub axes: Vec<u32>,
    pub keep_dimensions: bool,
}

/// <https://www.w3.org/TR/webnn/#dictdef-mlconv2doptions>
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv2dOptions {
    pub padding: Vec<u32>,
    pub strides: Vec<u32>,
    pub dilations: Vec<u32>,
    pub groups: u32,
    pub input_layout: String,
    pub filter_layout: String,
    pub bias: Option<OperandId>,
}

/// <https://www.w3.org/TR/webnn/#dictdef-mlpool2doptions>
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pool2dOptions {
    pub window_dimensions: Vec<u32>,
    pub padding: Vec<u32>,
    pub strides: Vec<u32>,
    pub dilations: Vec<u32>,
    pub layout: String,
    pub output_shape_rounding: String,
    pub output_sizes: Option<Vec<u32>>,
}

/// <https://www.w3.org/TR/webnn/#dictdef-mlresample2doptions>
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resample2dOptions {
    pub mode: String,
    pub scales: Vec<f32>,
    pub sizes: Option<Vec<u32>>,
    pub axes: Vec<u32>,
}
