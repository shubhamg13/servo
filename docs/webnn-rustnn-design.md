# WebNN with rustnn Backend — Design Document

## 1. Architecture

```
JS (WebNN API)
  │ navigator.ml.createContext({ accelerated, powerPreference })
  │ new MLGraphBuilder(context)
  │ builder.input("a", desc), builder.add(a, b), builder.build({c})
  │ context.dispatch(graph, inputs, outputs)
  ▼
┌──────────────────────────────────────────────────────┐
│ DOM Bindings (components/script/dom/webnn/)          │
│  ~3,000 lines — unchanged, no rustnn types inside    │
│                                                      │
│  Build()  → GraphNode[]  →  webnn::compile()         │
│  Dispatch → inputs+names →  webnn::run()             │
└───────────────────┬──────────────────────────────────┘
                    │ GraphNode[]
                    ▼
┌──────────────────────────────────────────────────────┐
│ servo-webnn crate (components/webnn/src/lib.rs)      │
│                                          ~230 lines   │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ Backend trait (swappable)                     │    │
│  │  fn compile(nodes) → Result<usize, String>    │    │
│  │  fn run(graph_id, inputs, outputs) → RunResult│    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  Implementations:                                     │
│  ┌──────────────────┐  ┌──────────────────────┐      │
│  │ RustnnBackend     │  │ MockBackend           │      │
│  │ → ORT/CoreML/TRT  │  │ → copies first input  │      │
│  │ → auto-selects    │  │ → zero deps           │      │
│  └──────────────────┘  └──────────────────────┘      │
│                                                      │
│  Shared converter:                                    │
│  nodes_to_graph_info() — GraphNode[] → GraphInfo     │
│  make_op() — string → Operation variant (46 ops)     │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ rustnn crate (local clone, /home/.../servo/rustnn)   │
│                                                      │
│  MLContext::create()  → select_backend()             │
│  MLGraphBuilder       → build_graph_info()           │
│  MLTensor             → create_tensor, write, read   │
│                                                      │
│  Send+Sync added to MLBackendContext (our patch)     │
└──────────────────────────────────────────────────────┘
```

## 2. Backend Selection

```
rustnn::MLContext::create(&options)
  │
  ▼
select_backend(options)
  │
  ├── macOS  + coreml-runtime → CoreML GPU/Neural Engine
  ├── macOS  + onnx-runtime   → ORT CPU (XNNPACK)
  ├── Windows + onnx-runtime  → ORT DirectML (GPU) or CPU
  ├── Linux   + onnx-runtime  → ORT CPU (XNNPACK)
  ├── Linux   + trtx-runtime  → TensorRT (NVIDIA GPU)
  ├── Android / OHOS           → NoBackendAvialable
  └── WASM                     → Not yet implemented
```

## 3. Swappable Backend

```rust
// Trait — any backend implements these two methods:
pub trait Backend: Send + Sync {
    fn compile(&self, nodes: &[GraphNode], output_names: &[String]) -> Result<usize, String>;
    fn run(&self, graph_id: usize, inputs: &[(&str, &[u8])], output_names: &[String]) -> Result<RunResult, String>;
}

// Swap at any time:
webnn::set_backend(Box::new(MockBackend));      // for tests/no-ORT envs
webnn::set_backend(Box::new(TfliteBackend));    // Android/OHOS
webnn::set_backend(Box::new(RustnnBackend));    // desktop (default)
```

**To add a new backend (3 steps):**
```rust
// 1. Implement the trait
pub struct TfliteBackend;
impl Backend for TfliteBackend {
    fn compile(&self, nodes, output_names) -> Result<usize, String> {
        // convert GraphNode[] → TFLite flatbuffer, store in registry
    }
    fn run(&self, gid, inputs, output_names) -> Result<RunResult, String> {
        // load flatbuffer, run LiteRT, return outputs
    }
}

// 2. Swap at startup
webnn::set_backend(Box::new(TfliteBackend));

// 3. Zero changes to DOM code, converter, or op mappings
```

## 4. Code Line Counts

| Component | Lines | Notes |
|-----------|-------|-------|
| `webnn/src/lib.rs` | 230 | Backend trait, RustnnBackend, MockBackend, converter, 46 ops |
| DOM `mlgraphbuilder.rs` | 2,291 | All operator methods, shape computation, validation |
| DOM `mlcontext.rs` | 400 | dispatch, tensor I/O, createContext options |
| DOM `mlgraph.rs` | 110 | MLGraph storage (ComputeNode[] + graph_id) |
| DOM `mloperand.rs` | 93 | MLOperand (name, dtype, shape) |
| DOM `mltensor.rs` | 136 | MLTensor (raw byte buffer) |
| DOM `ml.rs` | 52 | navigator.ml getter |
| **Total production code** | **3,312** | |
| **Deletable if JSTraceable fixed** | **~2,700** | DOM builder + converter |

## 5. Integration Blockers

### Blocker 1: JSTraceable (PRIMARY)

`#[dom_struct]` requires all fields: `JSTraceable` + `MallocSizeOf`. Rustnn types don't implement these. `#[no_trace]` (used in WebGPU) panics with lifetime params in types.

**Workaround:** Global `HashMap<usize, MLGraph>` registry. DOM `MLGraph` stores `Cell<usize>` (graph_id). No rustnn types in DOM structs.

**Fix in rustnn:** Remove phantom lifetimes from `MLGraph`, `MLContext`, `MLGraphBuilder`.

### Blocker 2: Send + Sync on MLBackendContext

`MLContext<'static>` couldn't be stored in `Mutex` because `Box<dyn MLBackendContext>` lacked `Send`.

**Fix applied (local clone):** Added to `rustnn/src/mlcontext.rs:32,63`:
```rust
pub(crate) trait MLBackendContext<'context>: Debug + Send + Sync
pub(crate) trait MLBackendBuilder<'context, 'builder>: Debug + Send
```

### Blocker 3: Private `descriptor()` on MLTensor

`pub(crate)` — can't compute tensor element count from outside rustnn.

**Workaround:** Use `shape()` × element size manually.

### Blocker 4: No pre-compiled graph dispatch

`dispatch()` takes `&mut MLGraph` — can't cache compiled graphs easily.

**Workaround:** Global `HashMap<usize, MLGraph>` — extract, dispatch, re-insert.

## 6. Platform Coverage

| Platform | Backend | Status |
|----------|---------|--------|
| Linux x86_64 | ORT CPU (XNNPACK) | Working |
| macOS | CoreML GPU/Neural Engine | Working (via rustnn) |
| Windows | DirectML GPU / ORT CPU | Working (via rustnn) |
| Android | TFLite/LiteRT | Needs TfliteBackend impl |
| OpenHarmony | TFLite/LiteRT | Needs TfliteBackend impl |

## 7. Supported Operations (46)

**Binary (8):** add, sub, mul, div, pow, max, min, matmul
**Unary (18):** relu, sigmoid, tanh, abs, neg, exp, log, sqrt, sin, cos, ceil, floor, tan, erf, reciprocal, identity, softplus, softsign
**Activation (4):** gelu, hardSigmoid, hardSwish, leakyRelu
**Comparison (6):** equal, greater, greaterOrEqual, lesser, lesserOrEqual, notEqual
**Logical (4):** logicalNot, logicalAnd, logicalOr, logicalXor
**Layout (2):** transpose, reshape
**Norm (2):** elu, prelu
**Other (2):** concat, softmax, clamp

Adding an op: 1 line in `make_op()`.

## 8. Runtime Requirements

```
# One-time setup (Linux)
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so

# Build
PATH=/tmp/protoc/bin:$PATH cargo build --features webnn

# Run
RUST_LOG=info ./mach run --features webnn

# Tests
ORT_DYLIB_PATH=... cargo test -p servo-webnn -- --nocapture
```

## 9. What We Use From rustnn

| API | Purpose |
|-----|---------|
| `MLContext::create()` | Backend selection + runtime |
| `MLGraphBuilder::build_graph_info()` | Graph compilation |
| `MLContext::dispatch()` | Multi-backend execution |
| `MLTensor`, `MLTensorDescriptor` | Tensor I/O |
| `Operation::Add, Relu, ...` (46 types) | Typed op construction |
| `GraphInfo, Operand, OperandDescriptor` | Graph IR |
| `OnnxConverter` | (not used directly — `build_graph_info` handles it) |

## 10. Our rustnn Patch

**File:** `/home/shubham/space/servo/rustnn/src/mlcontext.rs`

```diff
-pub(crate) trait MLBackendContext<'context>: std::fmt::Debug {
+pub(crate) trait MLBackendContext<'context>: std::fmt::Debug + Send + Sync {

-pub(crate) trait MLBackendBuilder<'context, 'builder>: std::fmt::Debug {
+pub(crate) trait MLBackendBuilder<'context, 'builder>: std::fmt::Debug + Send {
```

2 words changed. Enables `Mutex<MLContext>` for thread-safe browser usage. PR to upstream recommended.
