/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Backend registry for WebNN compute.

#[cfg(feature = "litert")]
pub(crate) mod litert;

#[cfg(feature = "litert")]
use crate::backend::Backend;
use crate::backend::{DataType, GraphNode, RunResult};

/// Compile and run inference using the LiteRT backend.
pub fn infer(
    nodes: &[GraphNode],
    inputs: &[(&str, &[u8])],
    input_infos: &[(String, Vec<u32>, DataType)],
) -> Result<RunResult, String> {
    let backend = litert::LiteRtBackend;
    log::error!("WebNN using backend: {}", backend.name());
    let model = backend
        .compile_with_input_infos(nodes, input_infos)
        .map_err(|e| {
            log::error!("WebNN compile FAILED: {}", e);
            e
        })?;
    let result = backend.run(&model, inputs).map_err(|e| {
        log::error!("WebNN run FAILED: {}", e);
        e
    })?;
    log::error!("WebNN inference OK: {} outputs", result.outputs.len());
    Ok(result)
}

#[cfg(not(feature = "litert"))]
pub fn infer(
    _nodes: &[GraphNode],
    _inputs: &[(&str, &[u8])],
    _input_infos: &[(String, Vec<u32>, DataType)],
) -> Result<RunResult, String> {
    Err("WebNN backend not available (enable the litert feature)".to_string())
}
