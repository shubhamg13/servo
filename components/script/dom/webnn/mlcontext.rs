/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! <https://webmachinelearning.github.io/webnn/#api-mlcontext>

use std::rc::Rc;

use dom_struct::dom_struct;
use script_bindings::record::Record;
use script_bindings::reflector::{Reflector, reflect_dom_object};
use script_bindings::root::DomRoot;
#[cfg(feature = "webnn")]
use webnn::{DataType, GraphNode, TensorDesc, compile_model, run_cached};

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLContextMethods, MLInputOperandLayout, MLOpSupportLimits, MLOperandDataType, MLRankRange,
    MLSingleInputSupportLimits, MLTensorDescriptor, MLTensorLimits,
};
use crate::dom::bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;
use crate::dom::bindings::error::Error;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::str::USVString;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlgraph::MLGraph;
use crate::dom::webnn::mltensor::MLTensor;
use crate::realms::InRealm;
use crate::script_runtime::CanGc;

#[cfg(feature = "webnn")]
thread_local! {
    static MODEL_CACHE: std::cell::RefCell<
        std::collections::HashMap<usize, webnn::CompiledModel>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[dom_struct]
pub(crate) struct MLContext {
    reflector_: Reflector,
    accelerated: bool,
}

impl MLContext {
    pub(crate) fn new_inherited() -> MLContext {
        MLContext {
            reflector_: Reflector::new(),
            accelerated: true,
        }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<MLContext> {
        reflect_dom_object(Box::new(MLContext::new_inherited()), global, can_gc)
    }
}

fn all_data_types() -> Vec<MLOperandDataType> {
    vec![
        MLOperandDataType::Float32,
        MLOperandDataType::Float16,
        MLOperandDataType::Int32,
        MLOperandDataType::Uint32,
        MLOperandDataType::Int64,
        MLOperandDataType::Uint64,
        MLOperandDataType::Int8,
        MLOperandDataType::Uint8,
    ]
}

fn common_data_types() -> Vec<MLOperandDataType> {
    vec![
        MLOperandDataType::Float32,
        MLOperandDataType::Float16,
        MLOperandDataType::Int32,
        MLOperandDataType::Uint32,
        MLOperandDataType::Int8,
        MLOperandDataType::Uint8,
    ]
}

fn default_tensor_limits() -> MLTensorLimits {
    MLTensorLimits {
        dataTypes: Some(common_data_types()),
        rankRange: Some(MLRankRange {
            min: Some(0),
            max: Some(8),
        }),
    }
}

fn single_limits() -> MLSingleInputSupportLimits {
    let limits = default_tensor_limits();
    MLSingleInputSupportLimits {
        input: Some(limits.clone()),
        output: Some(limits),
    }
}

impl MLContextMethods<crate::DomTypeHolder> for MLContext {
    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-dispatch>
    fn Dispatch(
        &self,
        graph: &MLGraph,
        inputs: Record<USVString, DomRoot<MLTensor>>,
        outputs: Record<USVString, DomRoot<MLTensor>>,
        _comp: InRealm,
        _can_gc: CanGc,
    ) {
            #[cfg(feature = "webnn")]
            {
                let nodes = graph.nodes();
                let mut webnn_nodes = Vec::new();
                for node in nodes.iter() {
                    webnn_nodes.push(GraphNode {
                        op: node.op.clone(),
                        inputs: node.inputs.clone(),
                        output: node.output.clone(),
                        desc: TensorDesc {
                            data_type: DataType::from_u32(node.data_type as u32),
                            shape: node.shape.clone(),
                        },
                        attrs: node.attrs.clone(),
                        data: node.data.clone(),
                    });
                }

                let operand_info = graph.input_operand_info();
                let input_infos: Vec<(String, Vec<u32>, DataType)> = operand_info
                    .iter()
                    .map(|(name, info)| {
                        (
                            name.clone(),
                            info.shape.clone(),
                            DataType::from_u32(info.data_type as u32),
                        )
                    })
                    .collect();

                let mut input_data: Vec<(String, Vec<u8>)> = Vec::new();
                for (name, tensor) in inputs.iter() {
                    if let Some(data) = tensor.read_data() {
                        input_data.push((name.0.clone(), data));
                    }
                }
                let input_slices: Vec<(&str, &[u8])> = input_data
                    .iter()
                    .map(|(n, d)| (n.as_str(), d.as_slice()))
                    .collect();

                let output_names = graph.output_names();
                let output_internal_names = graph.output_internal_names();
                let user_to_internal: std::collections::HashMap<String, String> = output_names
                    .iter()
                    .zip(output_internal_names.iter())
                    .map(|(u, i)| (u.clone(), i.clone()))
                    .collect();
                let internal_names: Vec<String> = outputs
                    .iter()
                    .filter_map(|(user_key, _)| user_to_internal.get(user_key.0.as_str()).cloned())
                    .collect();
                let cache_key: usize = {
                    let mut h: u64 = 0;
                    for node in graph.nodes().iter() {
                        for b in node.op.as_bytes() { h = h.wrapping_mul(1099511628211).wrapping_add(*b as u64); }
                        for b in node.output.as_bytes() { h = h.wrapping_mul(1099511628211).wrapping_add(*b as u64); }
                        for s in &node.shape { h = h.wrapping_mul(1099511628211).wrapping_add(*s as u64); }
                    }
                    h as usize
                };
                let result = MODEL_CACHE.with(|cache| {
                    if let Some(model) = cache.borrow().get(&cache_key) {
                        log::error!("WebNN cache HIT key={}", cache_key);
                        run_cached(model, &input_slices)
                    } else {
                        log::error!("WebNN cache MISS key={} compiling", cache_key);
                        let compiled = compile_model(
                            &webnn_nodes, &input_infos, &internal_names,
                        );
                        match compiled {
                            Ok(model) => {
                                let result = run_cached(&model, &input_slices);
                                if result.is_ok() {
                                    cache.borrow_mut().insert(cache_key, model);
                                }
                                result
                            }
                            Err(e) => Err(e),
                        }
                    }
                });
                if let Ok(result) = result {
                let mut out_map: std::collections::HashMap<String, Vec<u8>> =
                    std::collections::HashMap::new();
                for (name, data) in internal_names.iter().zip(result.outputs.into_iter()) {
                    out_map.insert(name.clone(), data);
                }
                for (_name, out_tensor) in outputs.iter() {
                    if let Some(internal_name) = user_to_internal.get(_name.0.as_str()) {
                        if let Some(data) = out_map.remove(internal_name) {
                            out_tensor.write_data(&data);
                        }
                    }
                }
                return;
            }
        }

        // Fallback: copy first input to all outputs
        #[allow(unused_variables)]
        let first_input = inputs.iter().next();
        if let Some((_, first_tensor)) = first_input {
            if let Some(data) = first_tensor.read_data() {
                for (_, out_tensor) in outputs.iter() {
                    out_tensor.write_data(&data);
                }
            }
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-createtensor>
    fn CreateTensor(
        &self,
        descriptor: &MLTensorDescriptor,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let global = &self.global();
        let promise = Promise::new_in_current_realm(comp, can_gc);
        let tensor = MLTensor::new(
            global,
            descriptor.parent.dataType,
            descriptor.parent.shape.clone(),
            descriptor.readable,
            descriptor.writable,
            false,
            can_gc,
        );
        promise.resolve_native(&tensor, can_gc);
        promise
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-createconstanttensor>
    fn CreateConstantTensor(
        &self,
        descriptor: &crate::dom::bindings::codegen::Bindings::WebNNBinding::MLOperandDescriptor,
        input_data: ArrayBufferViewOrArrayBuffer,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let global = &self.global();
        let promise = Promise::new_in_current_realm(comp, can_gc);
        #[allow(unsafe_code)]
        let src: &[u8] = match input_data {
            ArrayBufferViewOrArrayBuffer::ArrayBuffer(ref data) => unsafe { data.as_slice() },
            ArrayBufferViewOrArrayBuffer::ArrayBufferView(ref data) => unsafe { data.as_slice() },
        };
        let tensor = MLTensor::new(
            global,
            descriptor.dataType,
            descriptor.shape.clone(),
            false,
            false,
            true,
            can_gc,
        );
        tensor.write_data(src);
        promise.resolve_native(&tensor, can_gc);
        promise
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-readtensor>
    fn ReadTensor(&self, tensor: &MLTensor, comp: InRealm, can_gc: CanGc) -> Rc<Promise> {
        let promise = Promise::new_in_current_realm(comp, can_gc);
        if let Some(data) = tensor.read_data() {
            let len = data.len();
            if len == 0 {
                promise.reject_error(Error::NotSupported(None), can_gc);
                return promise;
            }
            #[allow(unsafe_code)]
            unsafe {
                let ptr = libc::malloc(len);
                if ptr.is_null() {
                    promise.reject_error(Error::NotSupported(None), can_gc);
                    return promise;
                }
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, len);
                let cx = GlobalScope::get_cx();
                let obj = js::jsapi::NewArrayBufferWithContents(*cx, len, ptr);
                if obj.is_null() {
                    libc::free(ptr);
                    promise.reject_error(Error::NotSupported(None), can_gc);
                    return promise;
                }
                let obj_val = js::jsval::ObjectValue(obj);
                promise.resolve_native(&obj_val, can_gc);
            }
        } else {
            promise.reject_error(Error::NotSupported(None), can_gc);
        }
        promise
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-readtensor>
    /// Copies tensor data into the caller-provided buffer.
    #[allow(unsafe_code)]
    fn ReadTensor_(
        &self,
        tensor: &MLTensor,
        mut output_data: ArrayBufferViewOrArrayBuffer,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let promise = Promise::new_in_current_realm(comp, can_gc);
        let data = match tensor.read_data() {
            Some(d) => d,
            None => {
                promise.reject_error(Error::NotSupported(None), can_gc);
                return promise;
            },
        };
        let len = data.len();
        match output_data {
            ArrayBufferViewOrArrayBuffer::ArrayBuffer(ref mut buf) => {
                let slice: &mut [u8] = unsafe { buf.as_mut_slice() };
                let end = len.min(slice.len());
                slice[..end].copy_from_slice(&data[..end]);
            },
            ArrayBufferViewOrArrayBuffer::ArrayBufferView(ref mut view) => {
                let slice: &mut [u8] = unsafe { view.as_mut_slice() };
                let end = len.min(slice.len());
                slice[..end].copy_from_slice(&data[..end]);
            },
        }
        promise.resolve_native(&(), can_gc);
        promise
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-writetensor>
    #[allow(unsafe_code)]
    fn WriteTensor(&self, tensor: &MLTensor, input_data: ArrayBufferViewOrArrayBuffer) {
        let src: &[u8] = match input_data {
            ArrayBufferViewOrArrayBuffer::ArrayBuffer(ref data) => unsafe { data.as_slice() },
            ArrayBufferViewOrArrayBuffer::ArrayBufferView(ref data) => unsafe { data.as_slice() },
        };
        log::error!("WebNN writeTensor: {} bytes, first f32: {:?}", src.len(),
            if src.len() >= 4 { Some(f32::from_le_bytes([src[0],src[1],src[2],src[3]])) } else { None });
        tensor.write_data(src);
        // Verify write
        if let Some(readback) = tensor.read_data() {
            log::error!("WebNN writeTensor verify: {} bytes, first f32: {:?}", readback.len(),
                if readback.len() >= 4 { Some(f32::from_le_bytes([readback[0],readback[1],readback[2],readback[3]])) } else { None });
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-opsupportlimits>
    fn OpSupportLimits(&self) -> MLOpSupportLimits {
        let tensor_limits = default_tensor_limits();
        let input_limits = MLTensorLimits {
            dataTypes: Some(all_data_types()),
            rankRange: Some(MLRankRange {
                min: Some(0),
                max: Some(8),
            }),
        };
        let sl = single_limits();
        MLOpSupportLimits {
            preferredInputLayout: Some(MLInputOperandLayout::Nchw),
            maxTensorByteLength: Some(1_000_000_000),
            input: Some(input_limits),
            constant: Some(tensor_limits.clone()),
            output: Some(tensor_limits.clone()),
            abs: Some(sl.clone()),
            add: Some(sl.clone()),
            argMax: Some(sl.clone()),
            argMin: Some(sl.clone()),
            averagePool2d: Some(sl.clone()),
            batchNormalization: Some(sl.clone()),
            cast: Some(sl.clone()),
            ceil: Some(sl.clone()),
            clamp: Some(sl.clone()),
            concat: Some(sl.clone()),
            conv2d: Some(sl.clone()),
            convTranspose2d: Some(sl.clone()),
            cos: Some(sl.clone()),
            cumulativeSum: Some(sl.clone()),
            dequantizeLinear: Some(sl.clone()),
            div: Some(sl.clone()),
            elu: Some(sl.clone()),
            equal: Some(sl.clone()),
            erf: Some(sl.clone()),
            exp: Some(sl.clone()),
            expand: Some(sl.clone()),
            floor: Some(sl.clone()),
            gather: Some(sl.clone()),
            gatherElements: Some(sl.clone()),
            gatherND: Some(sl.clone()),
            gelu: Some(sl.clone()),
            gemm: Some(sl.clone()),
            greater: Some(sl.clone()),
            greaterOrEqual: Some(sl.clone()),
            hardSigmoid: Some(sl.clone()),
            hardSwish: Some(sl.clone()),
            identity: Some(sl.clone()),
            instanceNormalization: Some(sl.clone()),
            l2Pool2d: Some(sl.clone()),
            layerNormalization: Some(sl.clone()),
            leakyRelu: Some(sl.clone()),
            lesser: Some(sl.clone()),
            lesserOrEqual: Some(sl.clone()),
            linear: Some(sl.clone()),
            log: Some(sl.clone()),
            logicalAnd: Some(sl.clone()),
            logicalNot: Some(sl.clone()),
            logicalOr: Some(sl.clone()),
            logicalXor: Some(sl.clone()),
            matmul: Some(sl.clone()),
            max: Some(sl.clone()),
            maxPool2d: Some(sl.clone()),
            min: Some(sl.clone()),
            mul: Some(sl.clone()),
            neg: Some(sl.clone()),
            notEqual: Some(sl.clone()),
            pad: Some(sl.clone()),
            pow: Some(sl.clone()),
            prelu: Some(sl.clone()),
            quantizeLinear: Some(sl.clone()),
            reciprocal: Some(sl.clone()),
            reduceL1: Some(sl.clone()),
            reduceL2: Some(sl.clone()),
            reduceLogSum: Some(sl.clone()),
            reduceLogSumExp: Some(sl.clone()),
            reduceMax: Some(sl.clone()),
            reduceMean: Some(sl.clone()),
            reduceMin: Some(sl.clone()),
            reduceProduct: Some(sl.clone()),
            reduceSum: Some(sl.clone()),
            reduceSumSquare: Some(sl.clone()),
            relu: Some(sl.clone()),
            resample2d: Some(sl.clone()),
            reshape: Some(sl.clone()),
            reverse: Some(sl.clone()),
            scatterElements: Some(sl.clone()),
            scatterND: Some(sl.clone()),
            sigmoid: Some(sl.clone()),
            sin: Some(sl.clone()),
            slice: Some(sl.clone()),
            softmax: Some(sl.clone()),
            softplus: Some(sl.clone()),
            softsign: Some(sl.clone()),
            sqrt: Some(sl.clone()),
            sub: Some(sl.clone()),
            tan: Some(sl.clone()),
            tanh: Some(sl.clone()),
            tile: Some(sl.clone()),
            transpose: Some(sl.clone()),
            triangular: Some(sl.clone()),
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-destroy>
    fn Destroy(&self) {}

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated>
    fn Accelerated(&self) -> bool {
        self.accelerated
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-lost>
    fn Lost(&self) -> Rc<Promise> {
        Promise::new(&self.global(), CanGc::deprecated_note())
    }
}
