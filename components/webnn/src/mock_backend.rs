/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{Backend, BuilderId, GraphId, OperandId, Operator};

#[allow(dead_code)]
struct Node {
    /// `None` for `input` / `constant` pseudo-nodes.
    operator: Option<Operator>,
    inputs: Vec<OperandId>,
    data_type: u32,
    shape: Vec<u32>,
    data: Option<Vec<u8>>,
    label: String,
}

struct BuilderState {
    nodes: HashMap<OperandId, Node>,
    input_names: HashMap<String, OperandId>,
}

struct GraphState {
    #[allow(dead_code)]
    outputs: HashMap<String, OperandId>,
}

pub struct MockBackend {
    builders: Mutex<HashMap<BuilderId, BuilderState>>,
    graphs: Mutex<HashMap<GraphId, GraphState>>,
    tensors: Mutex<HashMap<u32, Vec<u8>>>,
    next_builder_id: AtomicUsize,
    next_graph_id: AtomicUsize,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            builders: Mutex::new(HashMap::new()),
            graphs: Mutex::new(HashMap::new()),
            tensors: Mutex::new(HashMap::new()),
            next_builder_id: AtomicUsize::new(1),
            next_graph_id: AtomicUsize::new(1),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }

    fn create_builder(&self) -> BuilderId {
        let id = self.next_builder_id.fetch_add(1, Ordering::Relaxed);
        self.builders.lock().unwrap().insert(
            id,
            BuilderState {
                nodes: HashMap::new(),
                input_names: HashMap::new(),
            },
        );
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
            state.nodes.insert(
                operand_id,
                Node {
                    operator: None,
                    inputs: Vec::new(),
                    data_type,
                    shape: shape.to_vec(),
                    data: None,
                    label: String::new(),
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
            state.nodes.insert(
                operand_id,
                Node {
                    operator: None,
                    inputs: Vec::new(),
                    data_type,
                    shape: shape.to_vec(),
                    data: Some(data.to_vec()),
                    label: String::new(),
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
            for (operand_id, data_type, shape) in outputs {
                state.nodes.insert(
                    *operand_id,
                    Node {
                        operator: Some(operator.clone()),
                        inputs: inputs.to_vec(),
                        data_type: *data_type,
                        shape: shape.clone(),
                        data: None,
                        label: label.to_string(),
                    },
                );
            }
        }
    }

    fn build(
        &self,
        builder_id: BuilderId,
        outputs: &[(String, OperandId)],
    ) -> Result<GraphId, String> {
        let id = self.next_graph_id.fetch_add(1, Ordering::Relaxed);
        let output_map: HashMap<String, OperandId> = outputs.iter().cloned().collect();
        self.graphs.lock().unwrap().insert(
            id,
            GraphState {
                outputs: output_map,
            },
        );
        self.builders.lock().unwrap().remove(&builder_id);
        Ok(id)
    }

    fn run(
        &self,
        _graph_id: GraphId,
        inputs: &[(String, u32)],
        outputs: &[(String, u32)],
    ) -> Result<(), String> {
        // Mock behavior: echo the first input tensor into every output tensor.
        let data = self
            .tensors
            .lock()
            .unwrap()
            .get(&inputs.first().map(|(_, id)| *id).unwrap_or(0))
            .cloned()
            .unwrap_or_default();
        let mut tensors = self.tensors.lock().unwrap();
        for (_, id) in outputs {
            tensors.insert(*id, data.clone());
        }
        Ok(())
    }

    fn create_tensor(&self, tensor_id: u32, _data_type: u32, _shape: &[u32]) -> Result<(), String> {
        self.tensors.lock().unwrap().insert(tensor_id, Vec::new());
        Ok(())
    }

    fn write_tensor(&self, tensor_id: u32, bytes: &[u8]) -> Result<(), String> {
        self.tensors
            .lock()
            .unwrap()
            .insert(tensor_id, bytes.to_vec());
        Ok(())
    }

    fn read_tensor(&self, tensor_id: u32) -> Result<Vec<u8>, String> {
        self.tensors
            .lock()
            .unwrap()
            .get(&tensor_id)
            .cloned()
            .ok_or_else(|| format!("unknown tensor {tensor_id}"))
    }

    fn destroy_tensor(&self, tensor_id: u32) {
        self.tensors.lock().unwrap().remove(&tensor_id);
    }

    fn destroy_graph(&self, graph_id: GraphId) {
        self.graphs.lock().unwrap().remove(&graph_id);
    }
}
