/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Backend trait for the WebNN compute engine.

use std::collections::HashMap;

/// Operator attributes: param name → f64 value.
pub type OpAttrs = HashMap<String, f64>;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Int8 = 6,
    Uint8 = 7,
}

impl DataType {
    pub fn from_u32(v: u32) -> Self {
        match v {
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

    pub fn element_byte_size(&self) -> usize {
        match self {
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => 4,
            DataType::Float16 => 2,
            DataType::Int64 | DataType::Uint64 => 8,
            DataType::Int8 | DataType::Uint8 => 1,
        }
    }
}

/// Descriptor for a tensor in the compute graph.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub data_type: DataType,
    pub shape: Vec<u32>,
}

impl TensorDesc {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn byte_length(&self) -> usize {
        self.num_elements() * self.data_type.element_byte_size()
    }
}

/// A single node in the compute graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub desc: TensorDesc,
    pub attrs: OpAttrs,
    pub data: Option<Vec<u8>>,
}

/// A compiled model ready for inference.
pub enum CompiledModel {
    LiteRt {
        compiled: Box<dyn std::any::Any>,
        input_shapes: Vec<(String, Vec<u32>, DataType)>,
        output_shapes: Vec<(String, Vec<u32>, DataType)>,
        nhwc_inputs: std::collections::HashSet<String>,
        nhwc_outputs: std::collections::HashSet<String>,
    },
}

// SAFETY: LiteRtState contains CompiledModel + Environment which are Send.
unsafe impl Send for CompiledModel {}

/// Inference result.
pub struct RunResult {
    pub outputs: Vec<Vec<u8>>,
}

/// The backend trait — each backend implements this.
pub trait Backend: Send {
    fn name(&self) -> &'static str;
    fn compile_with_input_infos(
        &self,
        nodes: &[GraphNode],
        input_infos: &[(String, Vec<u32>, DataType)],
        output_names: &[String],
    ) -> Result<CompiledModel, String>;
    fn run(&self, model: &CompiledModel, inputs: &[(&str, &[u8])]) -> Result<RunResult, String>;
}
