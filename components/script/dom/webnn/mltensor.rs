/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! <https://webmachinelearning.github.io/webnn/#api-mltensor>

use dom_struct::dom_struct;
use js::conversions::ToJSValConvertible;
use js::rust::MutableHandleValue;
use script_bindings::cell::DomRefCell;
use script_bindings::reflector::{Reflector, reflect_dom_object};
use script_bindings::root::DomRoot;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLOperandDataType, MLTensorMethods};
use crate::dom::bindings::import::base::SafeJSContext;
use crate::dom::globalscope::GlobalScope;
use crate::script_runtime::CanGc;

/// Byte size per element for each MLOperandDataType.
fn element_byte_size(data_type: MLOperandDataType) -> usize {
    match data_type {
        MLOperandDataType::Float32 | MLOperandDataType::Int32 | MLOperandDataType::Uint32 => 4,
        MLOperandDataType::Float16 => 2,
        MLOperandDataType::Int64 | MLOperandDataType::Uint64 => 8,
        MLOperandDataType::Int8 | MLOperandDataType::Uint8 => 1,
    }
}

/// Total byte length of a tensor given its data type and shape.
pub(crate) fn tensor_byte_length(data_type: MLOperandDataType, shape: &[u32]) -> usize {
    let num_elements: usize = shape.iter().map(|&d| d as usize).product();
    num_elements * element_byte_size(data_type)
}

#[dom_struct]
pub(crate) struct MLTensor {
    reflector_: Reflector,
    data_type: MLOperandDataType,
    shape: Vec<u32>,
    readable: bool,
    writable: bool,
    constant: bool,
    data: DomRefCell<Option<Vec<u8>>>,
}

impl MLTensor {
    pub(crate) fn new_inherited(
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        readable: bool,
        writable: bool,
        constant: bool,
    ) -> MLTensor {
        let byte_len = tensor_byte_length(data_type, &shape);
        let data = if byte_len > 0 {
            Some(vec![0u8; byte_len])
        } else {
            None
        };
        MLTensor {
            reflector_: Reflector::new(),
            data_type,
            shape,
            readable,
            writable,
            constant,
            data: DomRefCell::new(data),
        }
    }

    pub(crate) fn new(
        global: &GlobalScope,
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        readable: bool,
        writable: bool,
        constant: bool,
        can_gc: CanGc,
    ) -> DomRoot<MLTensor> {
        reflect_dom_object(
            Box::new(MLTensor::new_inherited(
                data_type, shape, readable, writable, constant,
            )),
            global,
            can_gc,
        )
    }

    /// Write data from a byte slice into the tensor's buffer.
    pub(crate) fn write_data(&self, src: &[u8]) {
        let mut data = self.data.borrow_mut();
        if let Some(ref mut buf) = *data {
            let len = src.len().min(buf.len());
            buf[..len].copy_from_slice(&src[..len]);
        }
    }

    /// Read a copy of the tensor's data buffer.
    pub(crate) fn read_data(&self) -> Option<Vec<u8>> {
        self.data.borrow().clone()
    }
}

impl MLTensorMethods<crate::DomTypeHolder> for MLTensor {
    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-datatype>
    fn DataType(&self) -> MLOperandDataType {
        self.data_type
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-shape>
    #[allow(unsafe_code)]
    fn Shape(&self, cx: SafeJSContext, retval: MutableHandleValue) {
        let js_shape: Vec<f64> = self.shape.iter().map(|&d| d as f64).collect();
        unsafe { js_shape.to_jsval(*cx, retval) }
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-readable>
    fn Readable(&self) -> bool {
        self.readable
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-writable>
    fn Writable(&self) -> bool {
        self.writable
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-constant>
    fn Constant(&self) -> bool {
        self.constant
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-destroy>
    fn Destroy(&self) {
        *self.data.borrow_mut() = None;
    }
}
