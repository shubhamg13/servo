/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! <https://webmachinelearning.github.io/webnn/#api-mlgraph>

use std::collections::HashMap;

use dom_struct::dom_struct;
use jstraceable_derive::JSTraceable;
use malloc_size_of_derive::MallocSizeOf;
use script_bindings::cell::DomRefCell;
use script_bindings::reflector::{Reflector, reflect_dom_object};

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLGraphMethods, MLOperandDataType};
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::script_runtime::CanGc;

/// Operator attributes: param name → f64 value.
pub(crate) type OpAttrs = HashMap<String, f64>;

/// A single node in the compute graph.
#[derive(Clone, JSTraceable, MallocSizeOf)]
pub(crate) struct ComputeNode {
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub data_type: MLOperandDataType,
    pub shape: Vec<u32>,
    pub attrs: OpAttrs,
    #[ignore_malloc_size_of = "optional vec"]
    pub data: Option<Vec<u8>>,
}

#[derive(Clone, JSTraceable, MallocSizeOf)]
pub(crate) struct InputOperandInfo {
    pub data_type: MLOperandDataType,
    #[ignore_malloc_size_of = "Vec"]
    pub shape: Vec<u32>,
}

#[dom_struct]
pub(crate) struct MLGraph {
    reflector_: Reflector,
    nodes: DomRefCell<Vec<ComputeNode>>,
    input_names: DomRefCell<Vec<String>>,
    output_names: DomRefCell<Vec<String>>,
    output_internal_names: DomRefCell<Vec<String>>,
    input_operand_info: DomRefCell<HashMap<String, InputOperandInfo>>,
    graph_id: std::cell::Cell<usize>,
}

impl MLGraph {
    pub(crate) fn new_inherited() -> MLGraph {
        MLGraph {
            reflector_: Reflector::new(),
            nodes: DomRefCell::new(Vec::new()),
            input_names: DomRefCell::new(Vec::new()),
            output_names: DomRefCell::new(Vec::new()),
            output_internal_names: DomRefCell::new(Vec::new()),
            input_operand_info: DomRefCell::new(HashMap::new()),
            graph_id: std::cell::Cell::new(0),
        }
    }

    pub(crate) fn set_nodes(&self, nodes: Vec<ComputeNode>) {
        *self.nodes.borrow_mut() = nodes;
    }
    pub(crate) fn nodes(&self) -> std::cell::Ref<'_, Vec<ComputeNode>> {
        self.nodes.borrow()
    }
    pub(crate) fn set_input_names(&self, names: Vec<String>) {
        *self.input_names.borrow_mut() = names;
    }
    pub(crate) fn set_output_names(&self, names: Vec<String>) {
        *self.output_names.borrow_mut() = names;
    }
    pub(crate) fn output_names(&self) -> std::cell::Ref<'_, Vec<String>> {
        self.output_names.borrow()
    }
    pub(crate) fn set_output_internal_names(&self, names: Vec<String>) {
        *self.output_internal_names.borrow_mut() = names;
    }
    pub(crate) fn output_internal_names(&self) -> std::cell::Ref<'_, Vec<String>> {
        self.output_internal_names.borrow()
    }
    pub(crate) fn set_input_operand_info(&self, info: HashMap<String, InputOperandInfo>) {
        *self.input_operand_info.borrow_mut() = info;
    }
    pub(crate) fn input_operand_info(
        &self,
    ) -> std::cell::Ref<'_, HashMap<String, InputOperandInfo>> {
        self.input_operand_info.borrow()
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<MLGraph> {
        reflect_dom_object(Box::new(MLGraph::new_inherited()), global, can_gc)
    }

    pub(crate) fn graph_id(&self) -> usize { self.graph_id.get() }
    pub(crate) fn set_graph_id(&self, id: usize) { self.graph_id.set(id); }
}

impl MLGraphMethods<crate::DomTypeHolder> for MLGraph {
    fn Destroy(&self) {
        self.nodes.borrow_mut().clear();
    }
}
