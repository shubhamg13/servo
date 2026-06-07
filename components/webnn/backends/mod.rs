/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Backend registry for WebNN compute.

#[cfg(feature = "litert")]
pub(crate) mod litert;

#[cfg(feature = "litert")]
use crate::backend::Backend;
use crate::backend::{CompiledModel, DataType, GraphNode, RunResult};

/// Compile and run inference using the LiteRT backend.
pub fn infer(
    nodes: &[GraphNode],
    inputs: &[(&str, &[u8])],
    input_infos: &[(String, Vec<u32>, DataType)],
    output_names: &[String],
) -> Result<RunResult, String> {
    let model = compile(nodes, input_infos, output_names)?;
    run(&model, inputs)
}

/// Compile a WebNN graph into a LiteRT model (without running).
pub fn compile(
    nodes: &[GraphNode],
    input_infos: &[(String, Vec<u32>, DataType)],
    output_names: &[String],
) -> Result<CompiledModel, String> {
    let backend = litert::LiteRtBackend;
    log::error!("WebNN using backend: {}", backend.name());
    let model = backend
        .compile_with_input_infos(nodes, input_infos, output_names)
        .map_err(|e| {
            log::error!("WebNN compile FAILED: {}", e);
            e
        })?;
    Ok(model)
}

/// Run inference with a pre-compiled model.
pub fn run(model: &CompiledModel, inputs: &[(&str, &[u8])]) -> Result<RunResult, String> {
    let backend = litert::LiteRtBackend;
    backend.run(model, inputs)
}

#[cfg(not(feature = "litert"))]
pub fn compile(
    _nodes: &[GraphNode],
    _input_infos: &[(String, Vec<u32>, DataType)],
    _output_names: &[String],
) -> Result<CompiledModel, String> {
    Err("WebNN backend not available (enable the litert feature)".to_string())
}

#[cfg(not(feature = "litert"))]
pub fn infer(
    _nodes: &[GraphNode],
    _inputs: &[(&str, &[u8])],
    _input_infos: &[(String, Vec<u32>, DataType)],
    _output_names: &[String],
) -> Result<RunResult, String> {
    Err("WebNN backend not available (enable the litert feature)".to_string())
}

#[cfg(not(feature = "litert"))]
pub fn run(
    _model: &CompiledModel,
    _inputs: &[(&str, &[u8])],
) -> Result<RunResult, String> {
    Err("WebNN backend not available (enable the litert feature)".to_string())
}
