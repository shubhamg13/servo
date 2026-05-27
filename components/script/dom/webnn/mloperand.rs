/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! <https://webmachinelearning.github.io/webnn/#api-mloperand>

use dom_struct::dom_struct;
use js::conversions::ToJSValConvertible;
use js::rust::MutableHandleValue;
use script_bindings::reflector::{Reflector, reflect_dom_object};
use script_bindings::root::DomRoot;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLOperandDataType, MLOperandMethods};
use crate::dom::bindings::import::base::SafeJSContext;
use crate::dom::bindings::weakref::WeakRef;
use crate::dom::globalscope::GlobalScope;
use crate::dom::webnn::mlgraphbuilder::MLGraphBuilder;
use crate::script_runtime::CanGc;

#[dom_struct]
pub(crate) struct MLOperand {
    reflector_: Reflector,
    name: String,
    data_type: MLOperandDataType,
    shape: Vec<u32>,
    builder: WeakRef<MLGraphBuilder>,
}

impl MLOperand {
    pub(crate) fn new_inherited(
        name: String,
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        builder: &MLGraphBuilder,
    ) -> MLOperand {
        MLOperand {
            reflector_: Reflector::new(),
            name,
            data_type,
            shape,
            builder: WeakRef::new(builder),
        }
    }

    pub(crate) fn new(
        global: &GlobalScope,
        name: String,
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        builder: &MLGraphBuilder,
        can_gc: CanGc,
    ) -> DomRoot<MLOperand> {
        reflect_dom_object(
            Box::new(MLOperand::new_inherited(name, data_type, shape, builder)),
            global,
            can_gc,
        )
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn data_type(&self) -> MLOperandDataType {
        self.data_type
    }

    pub(crate) fn shape(&self) -> &[u32] {
        &self.shape
    }

    pub(crate) fn is_from_builder(&self, builder: &MLGraphBuilder) -> bool {
        self.builder.root().map_or(false, |b| {
            let b_ptr: *const MLGraphBuilder = &*b;
            let builder_ptr: *const MLGraphBuilder = builder;
            std::ptr::eq(b_ptr, builder_ptr)
        })
    }
}

impl MLOperandMethods<crate::DomTypeHolder> for MLOperand {
    /// <https://webmachinelearning.github.io/webnn/#dom-mloperand-datatype>
    fn DataType(&self) -> MLOperandDataType {
        self.data_type
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mloperand-shape>
    #[allow(unsafe_code)]
    fn Shape(&self, cx: SafeJSContext, retval: MutableHandleValue) {
        let js_shape: Vec<f64> = self.shape.iter().map(|&d| d as f64).collect();
        unsafe { js_shape.to_jsval(*cx, retval) }
    }
}
