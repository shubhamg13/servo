/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use servo_base::generic_channel::{
    self, GenericCallback, GenericOneshotSender, GenericReceiver, GenericSender,
};
use servo_base::id::MLContextId;

mod mock_backend;
mod operator;
#[cfg(feature = "rustnn")]
mod rustnn_backend;

pub use mock_backend::MockBackend;
pub use operator::{
    Conv2dOptions, ConvTranspose2dOptions, GemmOptions, Operator, Pool2dOptions, ReduceOptions,
    Resample2dOptions,
};

pub type GraphId = usize;
pub type BuilderId = usize;
pub type OperandId = usize;
/// Identifier of an `MLContext` on the shared WebNN backend thread.
pub type ContextId = MLContextId;

// ── Backend options ──

/// <https://www.w3.org/TR/webnn/#enumdef-mlpowerpreference>
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BackendPowerPreference {
    Default,
    HighPerformance,
    LowPower,
}

/// <https://www.w3.org/TR/webnn/#enumdef-mldevicetype>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendDeviceType {
    Cpu,
    Gpu,
    Npu,
}

/// Backend-agnostic subset of `MLContextOptions`, used to select a backend.
/// <https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions>
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendOptions {
    pub power_preference: BackendPowerPreference,
    pub accelerated: bool,
    /// Requested WebNN device type; used to select the concrete backend.
    pub device_type: BackendDeviceType,
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
    /// Runs a compiled graph with device-resident tensors referenced by Servo
    /// tensor id. Inputs and outputs are `(name, tensor_id)` pairs; the backend
    /// resolves ids to its own device tensors and never round-trips host bytes,
    /// so a tensor produced by one graph can be consumed by another on device.
    fn run(
        &self,
        graph_id: GraphId,
        inputs: &[(String, u32)],
        outputs: &[(String, u32)],
    ) -> Result<(), String>;
    /// Allocates a device tensor for `tensor_id` with the given descriptor.
    fn create_tensor(&self, tensor_id: u32, data_type: u32, shape: &[u32]) -> Result<(), String>;
    /// Copies host bytes into the device tensor `tensor_id`.
    fn write_tensor(&self, tensor_id: u32, bytes: &[u8]) -> Result<(), String>;
    /// Materializes the device tensor `tensor_id` back to host bytes.
    fn read_tensor(&self, tensor_id: u32) -> Result<Vec<u8>, String>;
    /// Releases the device tensor `tensor_id`.
    fn destroy_tensor(&self, tensor_id: u32);
    fn destroy_graph(&self, graph_id: GraphId);
}

// ── Backend selection ──

/// Select a backend for a new context.
///
/// This is the seam each backend plugs into. With the `rustnn` feature the
/// choice is rustnn's — it resolves the requested options against the backends
/// compiled in for this target — and it falls back to the mock backend if no
/// context can be created (e.g. no runtime available). Without the feature it
/// always returns the mock backend.
#[cfg(feature = "rustnn")]
pub fn create_backend(options: &BackendOptions) -> Box<dyn Backend> {
    match rustnn_backend::RustnnBackend::new(options) {
        Ok(backend) => Box::new(backend),
        Err(err) => {
            log::error!("webnn backend creation failed ({err}); falling back to mock backend");
            Box::new(MockBackend::new())
        },
    }
}

/// Select a backend for a new context (mock-only build).
#[cfg(not(feature = "rustnn"))]
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
    pub result: Result<(), String>,
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
        data_type: u32,
        shape: Vec<u32>,
    },
    CreateConstantTensor {
        context_id: ContextId,
        tensor_id: u32,
        data_type: u32,
        shape: Vec<u32>,
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
    DestroyTensor {
        context_id: ContextId,
        tensor_id: u32,
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
    while let Ok(request) = receiver.recv() {
        match request {
            WebNNRequest::NewContext {
                context_id,
                options,
            } => {
                backends.insert(context_id, create_backend(&options));
                log::error!(
                    "[webnn-leak] new_ctx {:?} backends={}",
                    context_id,
                    backends.len()
                );
            },
            WebNNRequest::DestroyContext { context_id } => {
                backends.remove(&context_id);
                log::error!(
                    "[webnn-leak] destroy_ctx {:?} backends={}",
                    context_id,
                    backends.len()
                );
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
                data_type,
                shape,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    if let Err(e) = backend.create_tensor(tensor_id, data_type, &shape) {
                        log::error!("create_tensor({tensor_id}) failed: {e}");
                    }
                }
            },
            WebNNRequest::CreateConstantTensor {
                context_id,
                tensor_id,
                data_type,
                shape,
                bytes,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    if let Err(e) = backend.create_tensor(tensor_id, data_type, &shape) {
                        log::error!("create_constant_tensor({tensor_id}) failed: {e}");
                    } else if let Err(e) = backend.write_tensor(tensor_id, &bytes) {
                        log::error!("create_constant_tensor({tensor_id}) write failed: {e}");
                    }
                }
            },
            WebNNRequest::WriteTensor {
                context_id,
                tensor_id,
                bytes,
            } => {
                if let Some(backend) = backends.get(&context_id) {
                    if let Err(e) = backend.write_tensor(tensor_id, &bytes) {
                        log::error!("write_tensor({tensor_id}) failed: {e}");
                    }
                }
            },
            WebNNRequest::ReadTensor {
                context_id,
                tensor_id,
                callback,
            } => {
                let t0 = std::time::Instant::now();
                let bytes = backends
                    .get(&context_id)
                    .map(|backend| backend.read_tensor(tensor_id))
                    .unwrap_or_else(|| Err("unknown context".to_string()))
                    .map_err(|_| ());
                let t1 = std::time::Instant::now();
                log::error!(
                    "[webnn-timing] read_handler = {:.2}ms",
                    (t1 - t0).as_secs_f64() * 1e3
                );
                let _ = callback.send(ReadTensorResponse { bytes });
            },
            WebNNRequest::DestroyTensor {
                context_id,
                tensor_id,
            } => {
                log::error!("[webnn-leak] destroy_tensor {:?} {tensor_id}", context_id);
                if let Some(backend) = backends.get(&context_id) {
                    backend.destroy_tensor(tensor_id);
                }
            },
            WebNNRequest::Run {
                context_id,
                graph_id,
                inputs,
                outputs,
                callback,
            } => {
                let t0 = std::time::Instant::now();
                let result = backends
                    .get(&context_id)
                    .map(|backend| backend.run(graph_id, &inputs, &outputs))
                    .unwrap_or_else(|| Err("unknown context".to_string()));
                let t1 = std::time::Instant::now();
                log::error!(
                    "[webnn-timing] run = {:.2}ms",
                    (t1 - t0).as_secs_f64() * 1e3
                );
                let _ = callback.send(RunResponse { result });
            },
            WebNNRequest::DestroyGraph {
                context_id,
                graph_id,
            } => {
                log::error!("[webnn-leak] destroy_graph {:?} {graph_id}", context_id);
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

    pub fn create_tensor(
        &self,
        context_id: ContextId,
        tensor_id: u32,
        data_type: u32,
        shape: &[u32],
    ) {
        self.0.send_or_warn(WebNNRequest::CreateTensor {
            context_id,
            tensor_id,
            data_type,
            shape: shape.to_vec(),
        });
    }

    pub fn create_constant_tensor(
        &self,
        context_id: ContextId,
        tensor_id: u32,
        data_type: u32,
        shape: &[u32],
        bytes: Vec<u8>,
    ) {
        self.0.send_or_warn(WebNNRequest::CreateConstantTensor {
            context_id,
            tensor_id,
            data_type,
            shape: shape.to_vec(),
            bytes,
        });
    }

    pub fn destroy_tensor(&self, context_id: ContextId, tensor_id: u32) {
        self.0.send_or_warn(WebNNRequest::DestroyTensor {
            context_id,
            tensor_id,
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
