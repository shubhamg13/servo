/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use servo_base::generic_channel::{
    self, GenericCallback, GenericOneshotSender, GenericReceiver, GenericSender,
};
use servo_base::id::MLContextId;

mod mock_backend;
mod operator;
#[cfg(any(feature = "rustnn", feature = "cann"))]
mod rustnn_backend;

pub use mock_backend::MockBackend;
pub use operator::{Conv2dOptions, Operator, Pool2dOptions, ReduceOptions, Resample2dOptions};

pub type GraphId = usize;
pub type BuilderId = usize;
pub type OperandId = usize;
/// Identifier of an `MLContext` on the shared WebNN backend thread.
pub type ContextId = MLContextId;

#[derive(Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub outputs: Vec<Vec<u8>>,
}

// ── Backend options ──

/// <https://www.w3.org/TR/webnn/#enumdef-mlpowerpreference>
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BackendPowerPreference {
    Default,
    HighPerformance,
    LowPower,
}

/// Backend-agnostic subset of `MLContextOptions`, used to select a backend.
/// <https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions>
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendOptions {
    pub power_preference: BackendPowerPreference,
    pub accelerated: bool,
}

// ── Backend trait ──

pub trait Backend: Send + 'static {
    fn name(&self) -> &str;
    fn create_builder(&self) -> BuilderId;
    fn add_input(
        &self,
        builder_id: BuilderId,
        operand_id: OperandId,
        name: &str,
        data_type: u32,
        shape: &[u32],
    );
    fn add_constant(
        &self,
        builder_id: BuilderId,
        operand_id: OperandId,
        data_type: u32,
        shape: &[u32],
        data: &[u8],
    );
    fn add_operator(
        &self,
        builder_id: BuilderId,
        operator: &Operator,
        inputs: &[OperandId],
        outputs: &[(OperandId, u32, Vec<u32>)],
        label: &str,
    );
    fn build(
        &self,
        builder_id: BuilderId,
        outputs: &[(String, OperandId)],
    ) -> Result<GraphId, String>;
    fn run(
        &self,
        graph_id: GraphId,
        inputs: &[(String, &[u8])],
        output_labels: &[String],
    ) -> Result<RunResult, String>;
    fn destroy_graph(&self, graph_id: GraphId);
}

// ── Backend selection ──

/// Select a backend for a new context.
///
/// This is the seam each backend plugs into. With the `cann` feature enabled
/// this returns a `RustnnBackend` targeting the CANN (HiAI) NPU; with the
/// `rustnn` feature it targets ONNX Runtime. In both cases it falls back to the
/// mock backend if the context cannot be created (e.g. no runtime available);
/// otherwise it always returns the mock backend.
#[cfg(feature = "cann")]
pub fn create_backend(options: &BackendOptions) -> Box<dyn Backend> {
    match rustnn_backend::RustnnBackend::new_cann(options) {
        Ok(backend) => Box::new(backend),
        Err(err) => {
            log::error!("cann backend creation failed ({err}); falling back to mock backend");
            Box::new(MockBackend::new())
        },
    }
}

/// Select a backend for a new context (ONNX Runtime via rustnn).
#[cfg(all(feature = "rustnn", not(feature = "cann")))]
pub fn create_backend(options: &BackendOptions) -> Box<dyn Backend> {
    match rustnn_backend::RustnnBackend::new(options) {
        Ok(backend) => Box::new(backend),
        Err(err) => {
            log::error!("rustnn backend creation failed ({err}); falling back to mock backend");
            Box::new(MockBackend::new())
        },
    }
}

/// Select a backend for a new context (mock-only build).
#[cfg(not(any(feature = "rustnn", feature = "cann")))]
pub fn create_backend(_options: &BackendOptions) -> Box<dyn Backend> {
    Box::new(MockBackend::new())
}

// ── Async responses (backend -> script thread via GenericCallback) ──

#[derive(Serialize, Deserialize)]
pub struct BuildResponse {
    pub graph_id: Result<GraphId, String>,
}

#[derive(Serialize, Deserialize)]
pub struct RunResponse {
    pub result: Result<RunResult, String>,
}

/// Response to a `ReadTensor` request, carrying the tensor's raw bytes.
#[derive(Serialize, Deserialize)]
pub struct ReadTensorResponse {
    pub bytes: Result<Vec<u8>, ()>,
}

// ── Thread requests ──

#[derive(Serialize, Deserialize)]
enum WebNNRequest {
    NewContext {
        context_id: ContextId,
        options: BackendOptions,
    },
    DestroyContext {
        context_id: ContextId,
    },
    CreateBuilder {
        context_id: ContextId,
        reply: GenericOneshotSender<BuilderId>,
    },
    AddInput {
        context_id: ContextId,
        builder_id: BuilderId,
        operand_id: OperandId,
        name: String,
        data_type: u32,
        shape: Vec<u32>,
    },
    AddConstant {
        context_id: ContextId,
        builder_id: BuilderId,
        operand_id: OperandId,
        data_type: u32,
        shape: Vec<u32>,
        data: Vec<u8>,
    },
    AddOperator {
        context_id: ContextId,
        builder_id: BuilderId,
        operator: Operator,
        inputs: Vec<OperandId>,
        outputs: Vec<(OperandId, u32, Vec<u32>)>,
        label: String,
    },
    Build {
        context_id: ContextId,
        builder_id: BuilderId,
        outputs: Vec<(String, OperandId)>,
        callback: GenericCallback<BuildResponse>,
    },
    CreateTensor {
        context_id: ContextId,
        tensor_id: u32,
        byte_length: usize,
    },
    CreateConstantTensor {
        context_id: ContextId,
        tensor_id: u32,
        bytes: Vec<u8>,
    },
    WriteTensor {
        context_id: ContextId,
        tensor_id: u32,
        bytes: Vec<u8>,
    },
    ReadTensor {
        context_id: ContextId,
        tensor_id: u32,
        callback: GenericCallback<ReadTensorResponse>,
    },
    Run {
        context_id: ContextId,
        graph_id: GraphId,
        inputs: Vec<(String, u32)>,
        outputs: Vec<(String, u32)>,
        callback: GenericCallback<RunResponse>,
    },
    DestroyGraph {
        context_id: ContextId,
        graph_id: GraphId,
    },
    #[allow(dead_code)]
    Shutdown,
}

// ── WebNN channel ──

/// Channel from script thread to the shared WebNN backend thread.
/// A single backend thread hosts one backend per MLContext.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WebNN(pub(crate) GenericSender<WebNNRequest>);

static WEBNN_CHANNEL: std::sync::OnceLock<WebNN> = std::sync::OnceLock::new();

impl WebNN {
    /// Get the shared WebNN thread, spawning it on first call.
    pub fn shared() -> &'static WebNN {
        WEBNN_CHANNEL.get_or_init(|| {
            let (sender, receiver) =
                generic_channel::channel::<WebNNRequest>().expect("WebNN channel creation");
            std::thread::Builder::new()
                .name("WebNN".into())
                .spawn(move || {
                    run_webnn_thread(receiver);
                })
                .expect("WebNN thread spawn");
            WebNN(sender)
        })
    }
}

/// Run the WebNN backend thread's main loop, routing requests to the backend
/// owned by each context.
fn run_webnn_thread(receiver: GenericReceiver<WebNNRequest>) {
    let mut backends: HashMap<ContextId, Box<dyn Backend>> = HashMap::new();
    // Authoritative tensor byte storage, keyed by (context id, tensor id).
    // Dispatch reads its inputs from here at execution time and writes its
    // outputs back here, so chained dispatches observe each other's results.
    let mut tensor_store: HashMap<(ContextId, u32), Arc<Vec<u8>>> = HashMap::new();
    while let Ok(request) = receiver.recv() {
        match request {
            WebNNRequest::NewContext {
                context_id,
                options,
            } => {
                backends.insert(context_id, create_backend(&options));
            },
            WebNNRequest::DestroyContext { context_id } => {
                backends.remove(&context_id);
            },
            WebNNRequest::CreateBuilder { context_id, reply } => {
                let id = backends
                    .get(&context_id)
                    .map(|backend| backend.create_builder())
                    .unwrap_or(0);
                reply.send_or_warn(id);
            },
            WebNNRequest::AddInput {
                context_id,
                builder_id,
                operand_id,
                name,
                data_type,
                shape,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    backend.add_input(builder_id, operand_id, &name, data_type, &shape);
                }
            },
            WebNNRequest::AddConstant {
                context_id,
                builder_id,
                operand_id,
                data_type,
                shape,
                data,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    backend.add_constant(builder_id, operand_id, data_type, &shape, &data);
                }
            },
            WebNNRequest::AddOperator {
                context_id,
                builder_id,
                operator,
                inputs,
                outputs,
                label,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    backend.add_operator(builder_id, &operator, &inputs, &outputs, &label);
                }
            },
            WebNNRequest::Build {
                context_id,
                builder_id,
                outputs,
                callback,
            } => {
                let result = backends
                    .get(&context_id)
                    .map(|backend| backend.build(builder_id, &outputs))
                    .unwrap_or_else(|| Err("unknown context".to_string()));
                let _ = callback.send(BuildResponse { graph_id: result });
            },
            WebNNRequest::CreateTensor {
                context_id,
                tensor_id,
                byte_length,
            } => {
                tensor_store.insert((context_id, tensor_id), Arc::new(vec![0u8; byte_length]));
            },
            WebNNRequest::CreateConstantTensor {
                context_id,
                tensor_id,
                bytes,
            } => {
                tensor_store.insert((context_id, tensor_id), Arc::new(bytes));
            },
            WebNNRequest::WriteTensor {
                context_id,
                tensor_id,
                bytes,
            } => {
                tensor_store.insert((context_id, tensor_id), Arc::new(bytes));
            },
            WebNNRequest::ReadTensor {
                context_id,
                tensor_id,
                callback,
            } => {
                let bytes = tensor_store
                    .get(&(context_id, tensor_id))
                    .map(|buffer| buffer.as_ref().clone())
                    .ok_or(());
                let _ = callback.send(ReadTensorResponse { bytes });
            },
            WebNNRequest::Run {
                context_id,
                graph_id,
                inputs,
                outputs,
                callback,
            } => {
                let input_refs: Vec<(String, &[u8])> = inputs
                    .iter()
                    .map(|(name, tensor_id)| {
                        let bytes = tensor_store
                            .get(&(context_id, *tensor_id))
                            .map(|buffer| buffer.as_slice())
                            .unwrap_or(&[]);
                        (name.clone(), bytes)
                    })
                    .collect();
                let output_labels: Vec<String> =
                    outputs.iter().map(|(name, _)| name.clone()).collect();
                let result = backends
                    .get(&context_id)
                    .map(|backend| backend.run(graph_id, &input_refs, &output_labels))
                    .unwrap_or_else(|| Err("unknown context".to_string()));
                if let Ok(run_result) = &result {
                    for ((_, tensor_id), bytes) in outputs.iter().zip(run_result.outputs.iter()) {
                        tensor_store.insert((context_id, *tensor_id), Arc::new(bytes.clone()));
                    }
                }
                let _ = callback.send(RunResponse { result });
            },
            WebNNRequest::DestroyGraph {
                context_id,
                graph_id,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    backend.destroy_graph(graph_id);
                }
            },
            WebNNRequest::Shutdown => break,
        }
    }
}

impl WebNN {
    /// Register a new context and its backend on the shared thread.
    pub fn new_context(&self, context_id: ContextId, options: &BackendOptions) {
        self.0.send_or_warn(WebNNRequest::NewContext {
            context_id,
            options: options.clone(),
        });
    }

    /// Remove a context and its backend from the shared thread.
    pub fn destroy_context(&self, context_id: ContextId) {
        self.0
            .send_or_warn(WebNNRequest::DestroyContext { context_id });
    }

    pub fn create_builder(&self, context_id: ContextId) -> BuilderId {
        let (tx, rx) = generic_channel::oneshot().expect("WebNN oneshot");
        self.0.send_or_warn(WebNNRequest::CreateBuilder {
            context_id,
            reply: tx,
        });
        rx.recv().unwrap_or(0)
    }

    pub fn add_input(
        &self,
        context_id: ContextId,
        builder_id: BuilderId,
        operand_id: OperandId,
        name: &str,
        data_type: u32,
        shape: &[u32],
    ) {
        self.0.send_or_warn(WebNNRequest::AddInput {
            context_id,
            builder_id,
            operand_id,
            name: name.to_string(),
            data_type,
            shape: shape.to_vec(),
        });
    }

    pub fn add_constant(
        &self,
        context_id: ContextId,
        builder_id: BuilderId,
        operand_id: OperandId,
        data_type: u32,
        shape: &[u32],
        data: &[u8],
    ) {
        self.0.send_or_warn(WebNNRequest::AddConstant {
            context_id,
            builder_id,
            operand_id,
            data_type,
            shape: shape.to_vec(),
            data: data.to_vec(),
        });
    }

    pub fn add_operator(
        &self,
        context_id: ContextId,
        builder_id: BuilderId,
        operator: &Operator,
        inputs: &[OperandId],
        outputs: &[(OperandId, u32, Vec<u32>)],
        label: &str,
    ) {
        self.0.send_or_warn(WebNNRequest::AddOperator {
            context_id,
            builder_id,
            operator: operator.clone(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            label: label.to_string(),
        });
    }

    pub fn build(
        &self,
        context_id: ContextId,
        builder_id: BuilderId,
        outputs: &[(String, OperandId)],
        callback: GenericCallback<BuildResponse>,
    ) {
        self.0.send_or_warn(WebNNRequest::Build {
            context_id,
            builder_id,
            outputs: outputs.to_vec(),
            callback,
        });
    }

    pub fn create_tensor(&self, context_id: ContextId, tensor_id: u32, byte_length: usize) {
        self.0.send_or_warn(WebNNRequest::CreateTensor {
            context_id,
            tensor_id,
            byte_length,
        });
    }

    pub fn create_constant_tensor(&self, context_id: ContextId, tensor_id: u32, bytes: Vec<u8>) {
        self.0.send_or_warn(WebNNRequest::CreateConstantTensor {
            context_id,
            tensor_id,
            bytes,
        });
    }

    pub fn write_tensor(&self, context_id: ContextId, tensor_id: u32, bytes: Vec<u8>) {
        self.0.send_or_warn(WebNNRequest::WriteTensor {
            context_id,
            tensor_id,
            bytes,
        });
    }

    pub fn read_tensor(
        &self,
        context_id: ContextId,
        tensor_id: u32,
        callback: GenericCallback<ReadTensorResponse>,
    ) {
        self.0.send_or_warn(WebNNRequest::ReadTensor {
            context_id,
            tensor_id,
            callback,
        });
    }

    pub fn run(
        &self,
        context_id: ContextId,
        graph_id: GraphId,
        inputs: &[(String, u32)],
        outputs: &[(String, u32)],
        callback: GenericCallback<RunResponse>,
    ) {
        self.0.send_or_warn(WebNNRequest::Run {
            context_id,
            graph_id,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            callback,
        });
    }

    pub fn destroy_graph(&self, context_id: ContextId, graph_id: GraphId) {
        self.0.send_or_warn(WebNNRequest::DestroyGraph {
            context_id,
            graph_id,
        });
    }
}
