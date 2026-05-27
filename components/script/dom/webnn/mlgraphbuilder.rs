/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder>

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::rc::Rc;

use dom_struct::dom_struct;
use js::jsapi::JSObject;
use js::rust::Handle;
use script_bindings::cell::DomRefCell;
use script_bindings::record::Record;
use script_bindings::reflector::{Reflector, reflect_dom_object};
use script_bindings::root::DomRoot;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConv2dOptions,
    MLConvTranspose2dOptions, MLCumulativeSumOptions, MLEluOptions, MLGatherOptions, MLGemmOptions,
    MLGraphBuilderMethods, MLHardSigmoidOptions, MLInstanceNormalizationOptions,
    MLInterpolationMode, MLLayerNormalizationOptions, MLLeakyReluOptions, MLLinearOptions,
    MLOperandDataType, MLOperandDescriptor, MLOperatorOptions, MLPadOptions, MLPool2dOptions,
    MLReduceOptions, MLResample2dOptions, MLReverseOptions, MLScatterOptions, MLSliceOptions,
    MLSplitOptions, MLTransposeOptions, MLTriangularOptions,
};
use crate::dom::bindings::codegen::GenericUnionTypes;
use crate::dom::bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;
use crate::dom::bindings::error::Error;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::str::USVString;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::MLContext;
use crate::dom::webnn::mlgraph::{ComputeNode, InputOperandInfo, MLGraph, OpAttrs};
use crate::dom::webnn::mloperand::MLOperand;
use crate::realms::InRealm;
use crate::script_runtime::CanGc;

fn err_type(msg: &str, label: &str) -> Error {
    let full = if label.is_empty() {
        msg.to_string()
    } else {
        format!("{} [{}]", msg, label)
    };
    Error::Type(CString::new(full).unwrap())
}

fn make_node_op(
    builder: &MLGraphBuilder,
    global: &GlobalScope,
    op: &str,
    inputs: &[&MLOperand],
    data_type: MLOperandDataType,
    shape: Vec<u32>,
) -> DomRoot<MLOperand> {
    make_node_op_with_attrs(
        builder,
        global,
        op,
        inputs,
        data_type,
        shape,
        OpAttrs::new(),
        None,
    )
}

fn make_node_op_with_data(
    builder: &MLGraphBuilder,
    global: &GlobalScope,
    op: &str,
    inputs: &[&MLOperand],
    data_type: MLOperandDataType,
    shape: Vec<u32>,
    data: Vec<u8>,
) -> DomRoot<MLOperand> {
    make_node_op_with_attrs(
        builder,
        global,
        op,
        inputs,
        data_type,
        shape,
        OpAttrs::new(),
        Some(data),
    )
}

fn make_node_op_with_attrs(
    builder: &MLGraphBuilder,
    global: &GlobalScope,
    op: &str,
    inputs: &[&MLOperand],
    data_type: MLOperandDataType,
    shape: Vec<u32>,
    attrs: OpAttrs,
    data: Option<Vec<u8>>,
) -> DomRoot<MLOperand> {
    let id = builder.next_id.get();
    builder.next_id.set(id + 1);
    let name = format!("_op_{}", id);

    let input_names: Vec<String> = inputs.iter().map(|op| op.name().to_string()).collect();

    builder.nodes.borrow_mut().push(ComputeNode {
        op: op.to_string(),
        inputs: input_names,
        output: name.clone(),
        data_type,
        shape: shape.clone(),
        attrs,
        data,
    });

    MLOperand::new(
        global,
        name,
        data_type,
        shape,
        builder,
        CanGc::deprecated_note(),
    )
}

fn check_same_builder<'a>(
    builder: &MLGraphBuilder,
    inputs: impl IntoIterator<Item = &'a MLOperand>,
    label: &str,
) -> Result<(), Error> {
    for input in inputs {
        if !input.is_from_builder(builder) {
            return Err(err_type("Input is from another builder.", label));
        }
    }
    Ok(())
}

fn check_broadcastable(a: &[u32], b: &[u32]) -> bool {
    let mut ai = a.len() as i32 - 1;
    let mut bi = b.len() as i32 - 1;
    while ai >= 0 || bi >= 0 {
        let da = if ai >= 0 { a[ai as usize] } else { 1 };
        let db = if bi >= 0 { b[bi as usize] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return false;
        }
        ai -= 1;
        bi -= 1;
    }
    true
}

fn broadcast_shape(a: &[u32], b: &[u32], output: &mut Vec<u32>) {
    output.clear();
    let mut ai = a.len() as i32 - 1;
    let mut bi = b.len() as i32 - 1;
    while ai >= 0 || bi >= 0 {
        let da = if ai >= 0 { a[ai as usize] } else { 1 };
        let db = if bi >= 0 { b[bi as usize] } else { 1 };
        output.push(std::cmp::max(da, db));
        ai -= 1;
        bi -= 1;
    }
    output.reverse();
}

fn check_same_type(a: &MLOperand, b: &MLOperand, label: &str) -> Result<(), Error> {
    if a.data_type() != b.data_type() {
        return Err(err_type("Inputs must have the same data type.", label));
    }
    Ok(())
}

fn check_binary_inputs(
    builder: &MLGraphBuilder,
    a: &MLOperand,
    b: &MLOperand,
    label: &str,
) -> Result<Vec<u32>, Error> {
    check_same_builder(builder, [a, b], label)?;
    check_same_type(a, b, label)?;
    if !check_broadcastable(a.shape(), b.shape()) {
        return Err(err_type("Input shapes are not broadcastable.", label));
    }
    let mut shape = Vec::new();
    broadcast_shape(a.shape(), b.shape(), &mut shape);
    Ok(shape)
}

#[dom_struct]
pub(crate) struct MLGraphBuilder {
    reflector_: Reflector,
    named_operands: DomRefCell<HashMap<String, (MLOperandDataType, Vec<u32>)>>,
    nodes: DomRefCell<Vec<ComputeNode>>,
    next_id: Cell<u64>,
}

impl MLGraphBuilder {
    pub(crate) fn new_inherited() -> MLGraphBuilder {
        MLGraphBuilder {
            reflector_: Reflector::new(),
            named_operands: DomRefCell::new(HashMap::new()),
            nodes: DomRefCell::new(Vec::new()),
            next_id: Cell::new(0),
        }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<MLGraphBuilder> {
        reflect_dom_object(Box::new(MLGraphBuilder::new_inherited()), global, can_gc)
    }
}

impl MLGraphBuilderMethods<crate::DomTypeHolder> for MLGraphBuilder {
    fn Constructor(
        global: &GlobalScope,
        _cx: Option<Handle<*mut JSObject>>,
        can_gc: CanGc,
        _context: &MLContext,
    ) -> DomRoot<MLGraphBuilder> {
        MLGraphBuilder::new(global, can_gc)
    }

    fn Input(
        &self,
        name: USVString,
        descriptor: &MLOperandDescriptor,
    ) -> Result<DomRoot<MLOperand>, Error> {
        if name.0.is_empty() {
            return Err(Error::Type(CString::new("The name is empty.").unwrap()));
        }
        if descriptor.shape.iter().any(|&d| d == 0) {
            return Err(Error::Type(
                CString::new("A dimension size cannot be 0.").unwrap(),
            ));
        }
        self.named_operands.borrow_mut().insert(
            name.0.clone(),
            (descriptor.dataType, descriptor.shape.clone()),
        );
        Ok(MLOperand::new(
            &self.global(),
            name.0,
            descriptor.dataType,
            descriptor.shape.clone(),
            self,
            CanGc::deprecated_note(),
        ))
    }

    #[allow(unsafe_code)]
    fn Constant(
        &self,
        descriptor: &MLOperandDescriptor,
        buffer: ArrayBufferViewOrArrayBuffer,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let global = &self.global();
        let data: Vec<u8> = match buffer {
            ArrayBufferViewOrArrayBuffer::ArrayBuffer(ref buf) => unsafe {
                buf.as_slice().to_vec()
            },
            ArrayBufferViewOrArrayBuffer::ArrayBufferView(ref view) => unsafe {
                view.as_slice().to_vec()
            },
        };
        Ok(make_node_op_with_data(
            self,
            global,
            "constant",
            &[],
            descriptor.dataType,
            descriptor.shape.clone(),
            data,
        ))
    }

    fn Build(
        &self,
        outputs: Record<USVString, DomRoot<MLOperand>>,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Result<Rc<Promise>, Error> {
        let global = &self.global();
        let graph = MLGraph::new(global, can_gc);

        let nodes = self.nodes.borrow().clone();
        let produced: HashSet<&str> = nodes.iter().map(|n| n.output.as_str()).collect();

        // Inputs are named operands not produced by any node.
        let named = self.named_operands.borrow();
        let input_names: Vec<String> = named
            .keys()
            .filter(|n| !produced.contains(n.as_str()))
            .cloned()
            .collect();

        let output_names: Vec<String> = outputs.iter().map(|(k, _)| k.0.clone()).collect();

        let input_operand_info: HashMap<String, InputOperandInfo> = named
            .iter()
            .filter(|(n, _)| !produced.contains(n.as_str()))
            .map(|(n, (dt, shape))| {
                (
                    n.clone(),
                    InputOperandInfo {
                        data_type: *dt,
                        shape: shape.clone(),
                    },
                )
            })
            .collect();

        graph.set_nodes(nodes);
        graph.set_input_names(input_names);
        graph.set_output_names(output_names);
        graph.set_input_operand_info(input_operand_info);

        let promise = Promise::new_in_current_realm(comp, can_gc);
        promise.resolve_native(&graph, can_gc);
        Ok(promise)
    }

    // ── Binary ──

    fn Add(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "add",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    fn Sub(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "sub",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    fn Mul(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "mul",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    fn Div(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "div",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    fn Max(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "max",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    fn Min(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "min",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    fn Pow(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "pow",
            &[a, b],
            a.data_type(),
            shape,
        ))
    }

    // ── Logical ──

    fn Equal(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "equal",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn NotEqual(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "notEqual",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn Greater(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "greater",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn GreaterOrEqual(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "greaterOrEqual",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn Lesser(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "lesser",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn LesserOrEqual(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "lesserOrEqual",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn LogicalNot(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "logicalNot",
            &[input],
            MLOperandDataType::Uint8,
            input.shape().to_vec(),
        ))
    }

    fn LogicalAnd(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "logicalAnd",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn LogicalOr(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "logicalOr",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    fn LogicalXor(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        let shape = check_binary_inputs(self, a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "logicalXor",
            &[a, b],
            MLOperandDataType::Uint8,
            shape,
        ))
    }

    // ── Unary ──

    fn Abs(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "abs",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Ceil(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "ceil",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Cos(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "cos",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Erf(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "erf",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Exp(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "exp",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Floor(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "floor",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Identity(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "identity",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Log(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "log",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Neg(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "neg",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Reciprocal(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reciprocal",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Sin(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "sin",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Sqrt(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "sqrt",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Tan(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "tan",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    // ── Activation ──

    fn Relu(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "relu",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Sigmoid(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "sigmoid",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Tanh(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "tanh",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Softmax(
        &self,
        input: &MLOperand,
        axis: u32,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        let rank = input.shape().len() as u32;
        if rank == 0 || axis >= rank {
            return Err(err_type("Axis is out of range.", &options.label.0));
        }
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "softmax",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
            {
                let mut attrs = OpAttrs::new();
                attrs.insert("axis".to_string(), axis as f64);
                attrs
            },
            None,
        ))
    }

    fn Gelu(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "gelu",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn HardSigmoid(
        &self,
        input: &MLOperand,
        _options: &MLHardSigmoidOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "hardSigmoid",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn HardSwish(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "hardSwish",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Elu(&self, input: &MLOperand, _options: &MLEluOptions) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "elu",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn LeakyRelu(
        &self,
        input: &MLOperand,
        _options: &MLLeakyReluOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "leakyRelu",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Linear(
        &self,
        input: &MLOperand,
        options: &MLLinearOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let mut attrs = OpAttrs::new();
        attrs.insert("alpha".to_string(), *options.alpha);
        attrs.insert("beta".to_string(), *options.beta);
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "linear",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
            attrs,
            None,
        ))
    }

    fn Softplus(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "softplus",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Softsign(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "softsign",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Clamp(
        &self,
        input: &MLOperand,
        options: &MLClampOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let mut attrs = OpAttrs::new();
        if let Some(v) = options.minValue {
            attrs.insert("minValue".to_string(), *v);
        }
        if let Some(v) = options.maxValue {
            attrs.insert("maxValue".to_string(), *v);
        }
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "clamp",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
            attrs,
            None,
        ))
    }

    // ── Layout ──

    fn Reshape(
        &self,
        input: &MLOperand,
        new_shape: Vec<u32>,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        if new_shape.iter().any(|&d| d == 0) {
            return Err(Error::Type(
                CString::new("A dimension size cannot be 0.").unwrap(),
            ));
        }
        Ok(make_node_op(
            self,
            &self.global(),
            "reshape",
            &[input],
            input.data_type(),
            new_shape,
        ))
    }

    fn Transpose(
        &self,
        input: &MLOperand,
        options: &MLTransposeOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let input_shape = input.shape();
        let perm: Vec<u32> = options.permutation.clone().unwrap_or_else(|| {
            let rank = input_shape.len() as u32;
            (0..rank).rev().collect()
        });
        let output_shape: Vec<u32> = perm.iter().map(|&i| input_shape[i as usize]).collect();
        let mut attrs = OpAttrs::new();
        attrs.insert("perm_len".to_string(), perm.len() as f64);
        for (i, &p) in perm.iter().enumerate() {
            attrs.insert(format!("perm_{}", i), p as f64);
        }
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "transpose",
            &[input],
            input.data_type(),
            output_shape,
            attrs,
            None,
        ))
    }

    fn Concat(
        &self,
        inputs: Vec<DomRoot<MLOperand>>,
        axis: u32,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        if inputs.is_empty() {
            return Err(err_type("The inputs list is empty.", &options.label.0));
        }
        for input in &inputs {
            check_same_builder(self, [&**input], &options.label.0)?;
        }
        let data_type = inputs[0].data_type();
        for input in &inputs {
            if input.data_type() != data_type {
                return Err(err_type(
                    "Inputs must have the same data type.",
                    &options.label.0,
                ));
            }
        }
        let rank = inputs[0].shape().len() as u32;
        if rank == 0 || axis >= rank {
            return Err(err_type("Axis is out of range.", &options.label.0));
        }
        let ref_shape = inputs[0].shape();
        let mut out_shape = ref_shape.to_vec();
        let mut total = 0u32;
        for input in &inputs {
            let s = input.shape();
            if s.len() != ref_shape.len() {
                return Err(err_type(
                    "Inputs must have the same rank.",
                    &options.label.0,
                ));
            }
            for i in 0..s.len() {
                if i != axis as usize && s[i] != ref_shape[i] {
                    return Err(err_type(
                        "Input shapes must match except on the concat axis.",
                        &options.label.0,
                    ));
                }
            }
            total += s[axis as usize];
        }
        out_shape[axis as usize] = total;
        let refs: Vec<&MLOperand> = inputs.iter().map(|op| &**op).collect();
        Ok(make_node_op(
            self,
            &self.global(),
            "concat",
            &refs,
            data_type,
            out_shape,
        ))
    }

    fn Slice(
        &self,
        input: &MLOperand,
        starts: Vec<u32>,
        sizes: Vec<u32>,
        options: &MLSliceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let mut attrs = OpAttrs::new();
        for (i, &s) in starts.iter().enumerate() {
            attrs.insert(format!("start_{}", i), s as f64);
        }
        for (i, &s) in sizes.iter().enumerate() {
            attrs.insert(format!("size_{}", i), s as f64);
        }
        if let Some(ref strides) = options.strides {
            for (i, &s) in strides.iter().enumerate() {
                attrs.insert(format!("stride_{}", i), s as f64);
            }
        }
        let mut out_shape = input.shape().to_vec();
        for (i, &s) in sizes.iter().enumerate() {
            if i < out_shape.len() {
                out_shape[i] = s;
            }
        }
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "slice",
            &[input],
            input.data_type(),
            out_shape,
            attrs,
            None,
        ))
    }

    fn Split(
        &self,
        input: &MLOperand,
        splits: GenericUnionTypes::RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence,
        options: &MLSplitOptions,
    ) -> Result<Vec<DomRoot<MLOperand>>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;

        let axis = options.axis as usize;
        let input_shape = input.shape();

        let split_sizes: Vec<u32> = match splits {
            GenericUnionTypes::RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence::RangeEnforcedUnsignedLong(n) => {
                let n = n as usize;
                if n == 0 {
                    return Err(err_type("splits must be > 0", &options.parent.label.0));
                }
                let dim = input_shape[axis];
                if dim % n as u32 != 0 {
                    return Err(err_type("dimension not evenly divisible by splits", &options.parent.label.0));
                }
                let size = dim / n as u32;
                vec![size; n]
            },
            GenericUnionTypes::RangeEnforcedUnsignedLongOrRangeEnforcedUnsignedLongSequence::RangeEnforcedUnsignedLongSequence(sizes) => {
                sizes
            },
        };

        let num_splits = split_sizes.len();
        let group_id = self.next_id.get();

        let mut results = Vec::with_capacity(num_splits);
        for split_size in &split_sizes {
            let mut out_shape = input_shape.to_vec();
            out_shape[axis] = *split_size;

            let mut attrs = OpAttrs::new();
            attrs.insert("splits".to_string(), num_splits as f64);
            attrs.insert("axis".to_string(), axis as f64);
            attrs.insert("split_group".to_string(), group_id as f64);

            results.push(make_node_op_with_attrs(
                self,
                &self.global(),
                "split",
                &[input],
                input.data_type(),
                out_shape,
                attrs,
                None,
            ));
        }

        Ok(results)
    }

    fn Pad(
        &self,
        input: &MLOperand,
        beginning: Vec<u32>,
        ending: Vec<u32>,
        _options: &MLPadOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        let input_shape = input.shape();
        if beginning.len() != input_shape.len() || ending.len() != input_shape.len() {
            return Err(Error::Type(
                CString::new("Pad: beginning/ending length must match input rank").unwrap(),
            ));
        }
        let output_shape: Vec<u32> = input_shape
            .iter()
            .zip(beginning.iter())
            .zip(ending.iter())
            .map(|((&d, &b), &e)| d + b + e)
            .collect();
        let mut attrs = OpAttrs::new();
        attrs.insert("pad_rank".to_string(), input_shape.len() as f64);
        for (i, &b) in beginning.iter().enumerate() {
            attrs.insert(format!("pad_begin_{}", i), b as f64);
        }
        for (i, &e) in ending.iter().enumerate() {
            attrs.insert(format!("pad_end_{}", i), e as f64);
        }
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "pad",
            &[input],
            input.data_type(),
            output_shape,
            attrs,
            None,
        ))
    }

    fn Tile(
        &self,
        input: &MLOperand,
        _repetitions: Vec<u32>,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "tile",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Reverse(
        &self,
        input: &MLOperand,
        options: &MLReverseOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let mut attrs = OpAttrs::new();
        if let Some(ref axes) = options.axes {
            for (i, &a) in axes.iter().enumerate() {
                attrs.insert(format!("axis_{}", i), a as f64);
            }
        }
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "reverse",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
            attrs,
            None,
        ))
    }

    fn Expand(
        &self,
        input: &MLOperand,
        _new_shape: Vec<u32>,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "expand",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Gather(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        _options: &MLGatherOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, indices], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "gather",
            &[input, indices],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn GatherElements(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        _options: &MLGatherOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, indices], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "gatherElements",
            &[input, indices],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn GatherND(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, indices], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "gatherND",
            &[input, indices],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ScatterElements(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        updates: &MLOperand,
        _options: &MLScatterOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, indices, updates], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "scatterElements",
            &[input, indices, updates],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ScatterND(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        updates: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, indices, updates], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "scatterND",
            &[input, indices, updates],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    // ── Matrix ──

    fn Matmul(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [a, b], &options.label.0)?;
        check_same_type(a, b, &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "matmul",
            &[a, b],
            a.data_type(),
            a.shape().to_vec(),
        ))
    }

    fn Gemm(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLGemmOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [a, b], &options.parent.label.0)?;
        check_same_type(a, b, &options.parent.label.0)?;
        let mut inputs: Vec<&MLOperand> = vec![a, b];
        if let Some(c) = &options.c {
            inputs.push(c);
        }
        let mut attrs = OpAttrs::new();
        attrs.insert("alpha".to_string(), *options.alpha);
        attrs.insert("beta".to_string(), *options.beta);
        if options.aTranspose {
            attrs.insert("aTranspose".to_string(), 1.0);
        }
        if options.bTranspose {
            attrs.insert("bTranspose".to_string(), 1.0);
        }
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if options.aTranspose && a_shape.len() >= 2 {
            a_shape[a_shape.len() - 1]
        } else if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let n = if options.bTranspose && b_shape.len() >= 2 {
            b_shape[b_shape.len() - 2]
        } else if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 1]
        } else {
            1
        };
        let output_shape = vec![m, n];
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "gemm",
            &inputs,
            a.data_type(),
            output_shape,
            attrs,
            None,
        ))
    }

    // ── Pooling ──

    fn AveragePool2d(
        &self,
        input: &MLOperand,
        options: &MLPool2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "averagePool2d",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn MaxPool2d(
        &self,
        input: &MLOperand,
        options: &MLPool2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "maxPool2d",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn L2Pool2d(
        &self,
        input: &MLOperand,
        options: &MLPool2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "l2Pool2d",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    // ── Convolution ──

    fn Conv2d(
        &self,
        input: &MLOperand,
        filter: &MLOperand,
        options: &MLConv2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, filter], &options.parent.label.0)?;
        let mut attrs = OpAttrs::new();
        if let Some(ref p) = options.padding {
            for (i, &v) in p.iter().enumerate() {
                attrs.insert(format!("pad{}", i), v as f64);
            }
        }
        if let Some(ref s) = options.strides {
            attrs.insert("stride_h".to_string(), s[0] as f64);
            if s.len() > 1 {
                attrs.insert("stride_w".to_string(), s[1] as f64);
            }
        }
        if let Some(ref d) = options.dilations {
            attrs.insert("dilation_h".to_string(), d[0] as f64);
            if d.len() > 1 {
                attrs.insert("dilation_w".to_string(), d[1] as f64);
            }
        }
        attrs.insert("groups".to_string(), options.groups as f64);
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "conv2d",
            &[input, filter],
            input.data_type(),
            input.shape().to_vec(),
            attrs,
            None,
        ))
    }

    fn ConvTranspose2d(
        &self,
        input: &MLOperand,
        filter: &MLOperand,
        _options: &MLConvTranspose2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, filter], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "convTranspose2d",
            &[input, filter],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    // ── Reduction ──

    fn ReduceL1(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceL1",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceL2(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceL2",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceLogSum(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceLogSum",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceLogSumExp(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceLogSumExp",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceMax(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceMax",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceMean(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceMean",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceMin(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceMin",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceProduct(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceProduct",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceSum(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceSum",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ReduceSumSquare(
        &self,
        input: &MLOperand,
        _options: &MLReduceOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "reduceSumSquare",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn ArgMin(
        &self,
        input: &MLOperand,
        axis: u32,
        options: &MLArgMinMaxOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let rank = input.shape().len() as u32;
        if rank == 0 || axis >= rank {
            return Err(err_type("Axis is out of range.", &options.parent.label.0));
        }
        Ok(make_node_op(
            self,
            &self.global(),
            "argMin",
            &[input],
            options.outputDataType,
            input.shape().to_vec(),
        ))
    }

    fn ArgMax(
        &self,
        input: &MLOperand,
        axis: u32,
        options: &MLArgMinMaxOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let rank = input.shape().len() as u32;
        if rank == 0 || axis >= rank {
            return Err(err_type("Axis is out of range.", &options.parent.label.0));
        }
        Ok(make_node_op(
            self,
            &self.global(),
            "argMax",
            &[input],
            options.outputDataType,
            input.shape().to_vec(),
        ))
    }

    // ── Normalization ──

    fn BatchNormalization(
        &self,
        input: &MLOperand,
        mean: &MLOperand,
        variance: &MLOperand,
        _options: &MLBatchNormalizationOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, mean, variance], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "batchNormalization",
            &[input, mean, variance],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn LayerNormalization(
        &self,
        input: &MLOperand,
        _options: &MLLayerNormalizationOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "layerNormalization",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn InstanceNormalization(
        &self,
        input: &MLOperand,
        _options: &MLInstanceNormalizationOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "instanceNormalization",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    // ── Quantization ──

    fn Cast(
        &self,
        input: &MLOperand,
        data_type: MLOperandDataType,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "cast",
            &[input],
            data_type,
            input.shape().to_vec(),
        ))
    }

    fn DequantizeLinear(
        &self,
        input: &MLOperand,
        scale: &MLOperand,
        zero_point: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, scale, zero_point], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "dequantizeLinear",
            &[input, scale, zero_point],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn QuantizeLinear(
        &self,
        input: &MLOperand,
        scale: &MLOperand,
        zero_point: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, scale, zero_point], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "quantizeLinear",
            &[input, scale, zero_point],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    // ── Misc ──

    fn Prelu(
        &self,
        input: &MLOperand,
        slope: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input, slope], &options.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "prelu",
            &[input, slope],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Where(
        &self,
        condition: &MLOperand,
        true_value: &MLOperand,
        false_value: &MLOperand,
        options: &MLOperatorOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [condition, true_value, false_value], &options.label.0)?;
        let mut output_shape = Vec::new();
        broadcast_shape(true_value.shape(), false_value.shape(), &mut output_shape);
        let mut final_shape = Vec::new();
        broadcast_shape(condition.shape(), &output_shape, &mut final_shape);
        Ok(make_node_op(
            self,
            &self.global(),
            "where",
            &[condition, true_value, false_value],
            true_value.data_type(),
            final_shape,
        ))
    }

    fn Triangular(
        &self,
        input: &MLOperand,
        _options: &MLTriangularOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "triangular",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn CumulativeSum(
        &self,
        input: &MLOperand,
        _axis: u32,
        _options: &MLCumulativeSumOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &_options.parent.label.0)?;
        Ok(make_node_op(
            self,
            &self.global(),
            "cumulativeSum",
            &[input],
            input.data_type(),
            input.shape().to_vec(),
        ))
    }

    fn Resample2d(
        &self,
        input: &MLOperand,
        options: &MLResample2dOptions,
    ) -> Result<DomRoot<MLOperand>, Error> {
        check_same_builder(self, [input], &options.parent.label.0)?;
        let input_shape = input.shape();
        let mut attrs = OpAttrs::new();
        match options.mode {
            MLInterpolationMode::Nearest_neighbor => {
                attrs.insert("mode".to_string(), 1.0);
            },
            MLInterpolationMode::Linear => {
                attrs.insert("mode".to_string(), 0.0);
            },
        }
        let output_shape: Vec<u32> = if let Some(ref sizes) = options.sizes {
            attrs.insert("sizes_len".to_string(), sizes.len() as f64);
            for (i, &s) in sizes.iter().enumerate() {
                attrs.insert(format!("sizes_{}", i), s as f64);
            }
            let mut shape = input_shape.to_vec();
            let rank = shape.len();
            if rank >= 2 {
                shape[rank - 2] = sizes[0];
                shape[rank - 1] = sizes[1];
            }
            shape
        } else if let Some(ref scales) = options.scales {
            let h = scales.get(0).map(|f| **f).unwrap_or(1.0_f32);
            let w = scales.get(1).map(|f| **f).unwrap_or(1.0_f32);
            attrs.insert("scale_h".to_string(), h as f64);
            attrs.insert("scale_w".to_string(), w as f64);
            let mut shape = input_shape.to_vec();
            let rank = shape.len();
            if rank >= 2 {
                shape[rank - 2] = (shape[rank - 2] as f32 * h) as u32;
                shape[rank - 1] = (shape[rank - 1] as f32 * w) as u32;
            }
            shape
        } else {
            input_shape.to_vec()
        };
        Ok(make_node_op_with_attrs(
            self,
            &self.global(),
            "resample2d",
            &[input],
            input.data_type(),
            output_shape,
            attrs,
            None,
        ))
    }
}
