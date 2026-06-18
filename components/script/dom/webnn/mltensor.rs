/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;
use std::rc::Rc;

use dom_struct::dom_struct;
use js::context::JSContext;
use js::conversions::ToJSValConvertible;
use js::rust::MutableHandleValue;
use script_bindings::cell::DomRefCell;
use script_bindings::reflector::{Reflector, reflect_dom_object_with_cx};
use script_bindings::root::DomRoot;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLOperandDataType, MLTensorMethods};
use crate::dom::bindings::codegen::UnionTypes::MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer;
use crate::dom::bindings::weakref::WeakRef;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::MLContext;

/// A pending read request queued on a tensor.
#[derive(JSTraceable)]
pub(crate) enum PendingRead {
    /// A normal `readTensor()` request; resolve the promise with the returned
    /// bytes.
    Read(Rc<Promise>),
    /// A BYOB overload; write the returned bytes into `output` and resolve
    /// `promise` with `undefined`.
    ReadByob {
        promise: Rc<Promise>,
        output: MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer,
    },
}

/// <https://www.w3.org/TR/webnn/#mltensor>
#[dom_struct]
pub(crate) struct MLTensor {
    reflector_: Reflector,
    /// <https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-context-slot>
    context: WeakRef<MLContext>,
    /// <https://www.w3.org/TR/webnn/#dom-mloperanddescriptor-datatype>
    data_type: MLOperandDataType,
    /// <https://www.w3.org/TR/webnn/#dom-mloperanddescriptor-shape>
    shape: Vec<u32>,
    /// <https://www.w3.org/TR/webnn/#dom-mltensordescriptor-readable>
    readable: bool,
    /// <https://www.w3.org/TR/webnn/#dom-mltensordescriptor-writable>
    writable: bool,
    /// <https://www.w3.org/TR/webnn/#dom-mltensor-isconstant-slot>
    is_constant: bool,
    /// <https://www.w3.org/TR/webnn/#dom-mltensor-isdestroyed-slot>
    is_destroyed: Cell<bool>,
    /// Script-visible tensor id assigned by the context (0 means no backend id).
    tensor_id: Cell<u32>,
    /// Number of in-flight dispatches writing to this tensor.
    pending_dispatches: Cell<u32>,
    /// Read requests waiting for in-flight dispatches to complete.
    #[ignore_malloc_size_of = "Rc"]
    pending_reads: DomRefCell<Vec<PendingRead>>,
    /// Whether a backend read is already in flight for this tensor.
    read_in_flight: Cell<bool>,
}

impl MLTensor {
    pub(crate) fn new_inherited(
        context: &MLContext,
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        readable: bool,
        writable: bool,
        is_constant: bool,
        tensor_id: u32,
    ) -> MLTensor {
        MLTensor {
            reflector_: Reflector::new(),
            context: WeakRef::new(context),
            data_type,
            shape,
            readable,
            writable,
            is_constant,
            is_destroyed: Cell::new(false),
            tensor_id: Cell::new(tensor_id),
            pending_dispatches: Cell::new(0),
            pending_reads: DomRefCell::new(Vec::new()),
            read_in_flight: Cell::new(false),
        }
    }

    pub(crate) fn new(
        global: &GlobalScope,
        context: &MLContext,
        data_type: MLOperandDataType,
        shape: Vec<u32>,
        readable: bool,
        writable: bool,
        is_constant: bool,
        tensor_id: u32,
        cx: &mut JSContext,
    ) -> DomRoot<MLTensor> {
        reflect_dom_object_with_cx(
            Box::new(MLTensor::new_inherited(
                context,
                data_type,
                shape,
                readable,
                writable,
                is_constant,
                tensor_id,
            )),
            global,
            cx,
        )
    }

    /// Script-visible tensor id assigned by the context (0 means no backend id).
    pub(crate) fn tensor_id(&self) -> u32 {
        self.tensor_id.get()
    }

    /// Number of dispatches currently writing to this tensor.
    pub(crate) fn pending_dispatches(&self) -> u32 {
        self.pending_dispatches.get()
    }

    /// Increments the count of in-flight dispatches writing to this tensor.
    pub(crate) fn incr_pending_dispatches(&self) {
        self.pending_dispatches
            .set(self.pending_dispatches.get() + 1);
    }

    /// Decrements the count of in-flight dispatches writing to this tensor and
    /// returns the new count.
    pub(crate) fn decr_pending_dispatches(&self) -> u32 {
        let count = self.pending_dispatches.get().saturating_sub(1);
        self.pending_dispatches.set(count);
        count
    }

    /// Appends a normal (non-BYOB) read request to the queue.
    pub(crate) fn append_read(&self, promise: Rc<Promise>) {
        self.pending_reads.borrow_mut().push(PendingRead::Read(promise));
    }

    /// Appends a BYOB read request along with its output buffer.
    pub(crate) fn append_read_byob(
        &self,
        promise: Rc<Promise>,
        output: MaybeSharedArrayBufferViewOrMaybeSharedArrayBuffer,
    ) {
        self.pending_reads
            .borrow_mut()
            .push(PendingRead::ReadByob { promise, output });
    }

    /// Removes and returns all queued read requests.
    pub(crate) fn take_pending_reads(&self) -> Vec<PendingRead> {
        self.pending_reads.borrow_mut().drain(..).collect()
    }

    /// Whether there are any queued read requests.
    pub(crate) fn has_pending_reads(&self) -> bool {
        !self.pending_reads.borrow().is_empty()
    }

    /// Whether a backend read is already in flight for this tensor.
    pub(crate) fn read_in_flight(&self) -> bool {
        self.read_in_flight.get()
    }

    /// Marks whether a backend read is in flight for this tensor.
    pub(crate) fn set_read_in_flight(&self, in_flight: bool) {
        self.read_in_flight.set(in_flight);
    }

    pub(crate) fn data_type(&self) -> MLOperandDataType {
        self.data_type
    }

    pub(crate) fn shape(&self) -> &[u32] {
        &self.shape
    }

    pub(crate) fn context(&self) -> &WeakRef<MLContext> {
        &self.context
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-isdestroyed-slot>
    pub(crate) fn is_destroyed(&self) -> bool {
        self.is_destroyed.get()
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-isconstant-slot>
    pub(crate) fn is_constant(&self) -> bool {
        self.is_constant
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensordescriptor-readable>
    pub(crate) fn readable(&self) -> bool {
        self.readable
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensordescriptor-writable>
    pub(crate) fn writable(&self) -> bool {
        self.writable
    }
}

impl MLTensorMethods<crate::DomTypeHolder> for MLTensor {
    /// <https://www.w3.org/TR/webnn/#dom-mltensor-datatype>
    fn DataType(&self) -> MLOperandDataType {
        // > The dataType getter steps are to return this’s dataType.
        self.data_type
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-destroy>
    fn Destroy(&self) {
        // Step 1. Set this.[[isDestroyed]] to true.
        self.is_destroyed.set(true);
        // TODO Step 2. For each promise in this.[[pendingPromises]]:
        // TODO Step 2.1. Remove promise from this.[[pendingPromises]].
        // TODO Step 2.2. Reject promise with an "InvalidStateError" DOMException.
        // Step 3. Enqueue the following steps to this.[[context]].[[timeline]]:
        // Step 3.1. Release this.[[data]].
        // Note: The tensor bytes are owned by the backend and released when the
        // context is destroyed, so there is no DOM-side data to release here.
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-shape>
    fn Shape(&self, cx: &mut JSContext, retval: MutableHandleValue) {
        // > The shape getter steps are to return this’s shape.
        let js_shape: Vec<f64> = self
            .shape
            .iter()
            .map(|&dimension| dimension as f64)
            .collect();
        js_shape.safe_to_jsval(cx, retval)
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-readable>
    fn Readable(&self) -> bool {
        // > The readable getter steps are to return this.[[descriptor]].readable.
        self.readable
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-writable>
    fn Writable(&self) -> bool {
        // > The writable getter steps are to return this.[[descriptor]].writable.
        self.writable
    }

    /// <https://www.w3.org/TR/webnn/#dom-mltensor-constant>
    fn Constant(&self) -> bool {
        // > The constant getter steps are to return this’s [[isConstant]].
        self.is_constant
    }
}
