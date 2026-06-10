# WebNN with rustnn Backend — Design Document

## 1. Architecture Overview

```
   JavaScript (WebNN API)
          │
          ▼
   ┌─────────────────────────────────────────────┐
   │  DOM Bindings (components/script/dom/webnn/) │
   │                                              │
   │  MLGraphBuilder  — 30+ operator methods      │
   │  MLContext       — dispatch(), tensor I/O    │
   │  MLGraph         — ComputeNode[] storage     │
   │  MLTensor        — data buffer               │
   │  MLOperand       — name, dtype, shape        │
   └────────────────────┬────────────────────────┘
                        │ ComputeNode[] + input_tensors
                        ▼
   ┌─────────────────────────────────────────────┐
   │  servo-webnn crate (components/webnn/)       │
   │                                              │
   │  ┌──────────────────────────────────────┐    │
   │  │  converter.rs (~260 lines)            │    │
   │  │                                       │    │
   │  │  String-based GraphNode[]             │    │
   │  │       │                               │    │
   │  │       │ name → u32 index mapping      │    │
   │  │       │ OperandKind detection          │    │
   │  │       │ Operation enum construction    │    │
   │  │       ▼                               │    │
   │  │  rustnn::GraphInfo                    │    │
   │  │  (backend-agnostic IR)                │    │
   │  └──────────────┬───────────────────────┘    │
   │                 │                            │
   │  ┌──────────────▼───────────────────────┐    │
   │  │  backends.rs (~140 lines)             │    │
   │  │                                       │    │
   │  │  compile():                           │    │
   │  │    GraphInfo → OnnxConverter           │    │
   │  │    → ONNX protobuf bytes              │    │
   │  │                                       │    │
   │  │  run():                               │    │
   │  │    ONNX bytes → ort::Session           │    │
   │  │    → Session::run() → f32 outputs     │    │
   │  │    → RunResult { outputs: Vec<u8> }   │    │
   │  └──────────────────────────────────────┘    │
   └────────────────────┬────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   ┌──────────────┐          ┌────────────────────┐
   │  rustnn crate │          │  ort crate          │
   │  (v0.5.12)    │          │  (v2.0.0-rc.11)     │
   │               │          │                     │
   │  GraphInfo    │          │  load-dynamic:      │
   │  Operation    │          │  ORT_DYLIB_PATH     │
   │  OnnxConverter│          │     │               │
   │  Dimension    │          │     ▼               │
   │  DataType     │          │  libonnxruntime.so  │
   └──────────────┘          │     │               │
                             │     ▼               │
                             │  XNNPACK (CPU)      │
                             └────────────────────┘
```

## 2. End-to-End Code Flow

### Step 0 — Startup

`components/webnn/src/backends.rs`

1. `ORT_DYLIB_PATH` env var checked (lazy, once)
2. If not set → `ort_available()` returns false → all WebNN ops return `Err("ORT not available")`
3. If set → `libonnxruntime.so` loads dynamically via `ort` crate's `load-dynamic` feature
4. No global init — each `run()` creates a fresh `ort::Session`

### Step 1 — Build Graph (JS → ComputeNode[])

`components/script/dom/webnn/mlgraphbuilder.rs`

1. JS creates `new MLGraphBuilder(context)`
2. JS calls `builder.input("x", desc)` → creates MLOperand { name: "_input_x", ... }
3. JS calls `builder.add(a, b)` → validates operands, computes output shape, creates ComputeNode:
   ```rust
   ComputeNode {
       op: "add",
       inputs: ["_input_a", "_input_b"],
       output: "_op_1",
       data_type: Float32,
       shape: [4],
       attrs: {},
       data: None,
   }
   ```
4. JS calls `builder.build({y: outputOperand})` → collects all ComputeNode[] + I/O info → MLGraph

### Step 2 — Dispatch (ComputeNode[] → GraphInfo → ONNX → Run)

`components/script/dom/webnn/mlcontext.rs::dispatch()`

```rust
// 1. Convert ComputeNode[] → GraphNode[] (string-based IR)
let webnn_nodes: Vec<GraphNode> = nodes.iter().map(|n| GraphNode {
    op: n.op.clone(),
    inputs: n.inputs.clone(),
    output: n.output.clone(),
    desc: TensorDesc { data_type: ..., shape: n.shape.clone() },
    attrs: n.attrs.clone(),
    data: n.data.clone(),
}).collect();

// 2. Compile (or use cache)
let model_bytes = webnn::compile_cached(&webnn_nodes, &output_names, cache_key)?;

// 3. Run inference
let result = webnn::run(&model_bytes, &input_slices)?;

// 4. Write outputs to MLTensors
for (name, out_tensor) in outputs.iter() {
    out_tensor.write_data(&result.outputs[i]);
}
```

### Step 3 — GraphNode → GraphInfo Conversion

`components/webnn/src/converter.rs::to_rustnn_graph_info()`

```
Input:  GraphNode[]  (String-based, flat list)
Output: rustnn::GraphInfo  (u32-index-based, structured)

Algorithm:
1. First pass: register all producer outputs as operands
2. Second pass: register graph inputs (not produced by any node)
3. Third pass: build rustnn::Operation enum variants
4. Fix operand kinds (Input/Output/Constant)

Key transformation:
   GraphNode { op: "add", inputs: ["a","b"], output: "c" }
       │
       │  name_to_idx: {"a"→0, "b"→1, "c"→2}
       ▼
   Operation::Add { a: 0, b: 1, options: None, outputs: [2] }
```

### Step 4 — ONNX Conversion

`rustnn::converters::onnx::OnnxConverter::convert()`

1. Takes `GraphInfo` (operands + operations)
2. Maps each `Operation` variant → ONNX `NodeProto`
3. Maps `DataType` → ONNX tensor element types
4. Creates `ModelProto` with graph inputs, outputs, initializers
5. Serializes to ONNX protobuf bytes

### Step 5 — ORT Execution

`components/webnn/src/backends.rs::run()`

```rust
// 1. Convert raw bytes → f32 slices
let float_data: Vec<f32> = data.chunks(4)
    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    .collect();

// 2. Create session from ONNX bytes (guarded by catch_unwind)
let mut session = ort::Session::builder()?
    .commit_from_memory(model_bytes)?;

// 3. Create input tensors
let tensor = ort::Value::from_array((shape, float_data))?;

// 4. Run inference
let outputs = session.run(inputs.as_slice())?;

// 5. Extract f32 data → Vec<u8>
let bytes: Vec<u8> = f32_data.iter()
    .flat_map(|f| f.to_le_bytes())
    .collect();
```

### Step 6 — Model Caching

```
MODEL_CACHE: OnceLock<Mutex<HashMap<usize, Vec<u8>>>>

compile_cached(nodes, output_names, cache_key):
  1. Lock cache
  2. If cache_key exists → return cached ONNX bytes
  3. Else → compile → store → return
  
Key: hash of op types + output names + shapes
Cache is global (not thread-local) → shared across all script threads
```

## 3. Differences from Conventional TFLite Implementation

| Aspect | TFLite (conventional) | rustnn (new) |
|--------|----------------------|--------------|
| **Compiler** | Custom ~2,800 line TFLite flatbuffer compiler in `compiler.rs` | rustnn `OnnxConverter` (~4,400 lines in rustnn crate) |
| **Layout handling** | Chromium-style 2-phase TRANSPOSE insertion (NCHW→NHWC) for TFLite compatibility | None needed — ONNX is NCHW-native |
| **Runtime** | LiteRT via `litert-sys` FFI (XNNPACK delegate) | ONNX Runtime via `ort` crate (`libonnxruntime.so` → XNNPACK) |
| **Graph IR** | `GraphNode` with String-based inputs/outputs | `GraphInfo` with u32-index-based operands |
| **Operation encoding** | String op name + `HashMap<String, f64>` attrs (loose) | Typed `Operation` enum variants with named fields (strict) |
| **Shape inference** | Inline in `mlgraphbuilder.rs` DOM bindings | Dedicated `shape_inference.rs` module in rustnn |
| **Model format** | TFLite flatbuffer (`.tflite`) | ONNX protobuf |
| **Device support** | CPU only (XNNPACK) | CPU (XNNPACK), GPU (CUDA), NPU (CoreML) via ONNX Runtime providers |
| **Model cache** | `thread_local! RefCell<HashMap<usize, CompiledModel>>` | `OnceLock<Mutex<HashMap<usize, Vec<u8>>>>` (global, thread-safe) |
| **Error handling** | `Result<CompiledModel, String>` | `Result<Vec<u8>, String>` — simpler, model bytes are the compiled form |
| **Lines of code (webnn crate)** | ~3,300 lines (compiler + litert + backend) | ~460 lines (converter + backends + types) |
| **Lines deleted** | ~2,800 (compiler.rs) + ~515 (litert.rs) = ~3,315 | N/A |
| **Lines added** | N/A | ~260 (converter.rs) + ~140 (backends.rs) + ~60 (backend.rs) = ~460 |
| **Net change** | N/A | **~2,855 lines removed** |

### Key Architectural Differences

#### 1. Two-Phase Build vs Single-Phase

**Conventional (TFLite):**
```
GraphNode[] → insert_nhwc_transposes() [Phase 1]
           → eliminate_transpose_pairs() [Phase 2]
           → compile_to_flatbuffer()    [Pass 1: tensors, Pass 2: ops]
           → LiteRtModel::from_bytes()
           → CompiledModel
```
3 phases, 2 passes, ~1,200 lines of NHWC logic.

**rustnn (new):**
```
GraphNode[] → to_rustnn_graph_info() → OnnxConverter::convert() → Vec<u8> (ONNX bytes)
```
Single conversion step, no layout preprocessing.

#### 2. Operation Type Safety

**Conventional:** Loose string matching.
```rust
// compiler.rs
match node.op.as_str() {
    "add" | "ADD" => TflOp::Add,
    // typo-prone, no compile-time checks
}
```

**rustnn:** Typed enum variants.
```rust
// converter.rs
match op_type {
    "add" => Ok(Operation::Add { a: input_ids[0], b: input_ids[1], ... }),
    // compiler verifies field names and types
}
```

#### 3. Backend Abstraction

**Conventional:**
```rust
pub enum CompiledModel {
    LiteRt { compiled: Box<dyn Any>, ... },
}
// To add a new backend: add a new enum variant + modify Backend trait
```

**rustnn:**
```rust
// No CompiledModel enum — model is just Vec<u8> (ONNX bytes)
// Backend trait simplified to:
pub trait Backend {
    fn compile(&self, nodes: &[GraphNode], output_names: &[String]) -> Result<Vec<u8>, String>;
    fn run(&self, model_bytes: &[u8], inputs: &[(&str, &[u8])]) -> Result<RunResult, String>;
}
// New backends just implement the trait — no enum modification needed
```

#### 4. Dependency Stack

**Conventional:**
```
servo-webnn → compiler.rs (custom TFLite) → flatbuffers crate
            → litert.rs → litert-sys → libLiteRT.so
```

**rustnn (new):**
```
servo-webnn → converter.rs → rustnn → OnnxConverter → prost (protobuf)
            → backends.rs → ort → libonnxruntime.so
```

Weight: ONNX Runtime (~50MB .so) vs LiteRT (~15MB .so). Trade: larger binary but far more ops (85 vs 61) and GPU/NPU support.

#### 5. DOM Bindings Integration

Both implementations use identical DOM bindings. The only change in `mlcontext.rs`:

**Conventional:**
```rust
use webnn::{DataType, GraphNode, TensorDesc, compile_model, run_cached};
let model = compile_model(&nodes, &input_infos, &output_names)?;
let result = run_cached(&model, &inputs)?;
```

**rustnn (new):**
```rust
use webnn::GraphNode;
let model_bytes = webnn::compile_cached(&nodes, &output_names, cache_key)?;
let result = webnn::run(&model_bytes, &inputs)?;
```

No `input_infos` needed — ONNX model encodes shapes natively. No `CompiledModel` wrapper — raw bytes cached.

## 4. File Map

| File | Lines | Role |
|------|-------|------|
| `components/webnn/src/backend.rs` | 100 | `GraphNode`, `DataType`, `TensorDesc`, `Backend` trait, `RunResult` |
| `components/webnn/src/converter.rs` | 260 | `GraphNode[]` → `rustnn::GraphInfo` (name→index, 35 op mappings) |
| `components/webnn/src/backends.rs` | 145 | `compile()` via `OnnxConverter`, `run()` via `ort::Session`, `compile_cached()` with global `Mutex` cache |
| `components/webnn/src/lib.rs` | 9 | Public re-exports |
| `components/webnn/Cargo.toml` | 28 | `rustnn` + `ort` deps, `load-dynamic` feature |
| `components/webnn/tests/integration.rs` | 135 | 4 end-to-end tests (add, mul, relu, sigmoid) |
| `components/script/dom/webnn/mlcontext.rs` | 440 | `dispatch()` calling `compile_cached()` + `run()` |
| `components/script/dom/webnn/mlgraphbuilder.rs` | 2300 | All 30+ operator DOM bindings |
| `components/script/dom/webnn/mlgraph.rs` | 95 | `ComputeNode` struct + `MLGraph` storage |
| `components/script/dom/webnn/mltensor.rs` | 136 | Data buffer read/write |
| `components/script/dom/webnn/mloperand.rs` | 93 | Name, dtype, shape |
| `components/script/dom/webnn/ml.rs` | 52 | `navigator.ml` getter |
| `components/script_bindings/webidls/WebNN.webidl` | 292 | Full W3C WebNN IDL |
| `components/script_bindings/codegen/Bindings.conf` | +40 | ML/MLContext/MLGraph/MLGraphBuilder interface config |

## 5. Supported Operations (35 of 85)

| Category | Ops |
|----------|-----|
| **Binary** | add, sub, mul, div, pow, max, min, matmul |
| **Unary** | relu, sigmoid, tanh, abs, neg, exp, log, sqrt, sin, cos, ceil, floor, tan, erf, reciprocal, sign, identity, softplus |
| **Comparison** | equal, greater, greaterOrEqual, lesser, lesserOrEqual, notEqual |
| **Activation** | softsign, gelu, hardSigmoid, hardSwish, prelu |
| **Remaining (50)** | conv2d, convTranspose2d, pool2d, concat, reshape, transpose, slice, pad, batchNorm, layerNorm, softmax, gather, cast, clamp, reduce*, resample2d, etc. |

Adding a new op: add 1-3 lines in `converter.rs::build_operation()` mapping the string name to the typed `Operation::*` variant. The ONNX converter in rustnn already handles all 85 ops.

## 6. Runtime Requirements

```
env:
  ORT_DYLIB_PATH=/path/to/libonnxruntime.so  # required, full file path

build:
  PROTOC=/tmp/protoc/bin/protoc              # needed for prost-build (ONNX protos)
```

## 7. Integration Blockers: Storing rustnn Types in DOM Structs

The goal is to replace Servo's DOM-level `MLGraphBuilder`, `MLContext`, and `MLTensor` with rustnn's native implementations, eliminating ~3,000 lines of bridge code. This requires storing rustnn types (`MLContext<'context>`, `MLGraphBuilder<'context, 'builder>`, `MLGraph<'context>`, `MLTensor`) inside `#[dom_struct]` annotated structs.

### 7.1 Root Cause: JSTraceable Requirement

Every `#[dom_struct]` expands to `#[derive(JSTraceable, MallocSizeOf)]`. Both traits must be implementable for all fields. External crate types don't implement `JSTraceable`, and we can't add impls due to the orphan rule. 

The mechanism to bypass this **exists** in Servo: `#[no_trace]` (field-level attribute in `jstraceable_derive`) skips a field during GC tracing. Combined with `#[ignore_malloc_size_of]` (from `malloc_size_of_derive`), external types can be stored in DOM structs. This pattern is used extensively in WebGPU (`GPUAdapter`, `GPUDevice`, `GPUTexture`, etc.).

### 7.2 Blocker A: Lifetime Parameter in Type Tokens

**File:** `rustnn/src/mlcontext.rs`, `rustnn/src/mlgraphbuilder.rs`

```rust
pub struct MLContext<'context> { ... }
pub struct MLGraphBuilder<'context, 'builder> { ... }
pub struct MLGraph<'context> { ... }
```

**Problem:** The `#[no_trace]` attribute on a field typed `Box<MLGraphBuilder<'static, 'static>>` causes `JSTraceable` and `MallocSizeOf` derive macros to panic. The lifetime tokens `'static` inside `Box<T<'static>>` produce unexpected token streams that the derive macros cannot parse.

**Evidence:** Proc-macro panic at `#[dom_struct]` annotation site when any rustnn type with lifetime parameter is used as a field type. Type aliases (`type RnnB = MLGraphBuilder<'static, 'static>`) do not resolve the issue — the lifetime is still present in the expanded type.

**Impact:** Prevents storing `MLContext`, `MLGraphBuilder`, or `MLGraph` in any DOM struct, even with `#[no_trace]` annotations.

**Proposed Fix (rustnn side):** Remove `'context` and `'builder` lifetime parameters. In a browser embedding, the ML runtime (ORT environment, CUDA context) is process-global and effectively `'static`. The lifetimes only serve CLI use cases where `main()` owns the context.

### 7.3 Blocker B: Missing Send + Sync on MLBackendContext

**File:** `rustnn/src/mlcontext.rs`

```rust
pub(crate) trait MLBackendContext<'context>: std::fmt::Debug {
    fn accelerated(&self) -> bool;
    fn create_builder(&mut self) -> ...;
    fn dispatch(&mut self, ...) -> ...;
    // No Send + Sync bound
}
```

**Problem:** Servo runs multiple script threads, each calling `dispatch()` concurrently. The `MLContext` must be shareable across threads (`Send + Sync`). Placing `MLContext` in a `Mutex` requires `Send`. But `MLBackendContext` (the trait object inside `MLContext`) doesn't require `Send`, so `Box<dyn MLBackendContext>` is not `Send`.

**Evidence:**
```
error[E0277]: `(dyn webnn::mlcontext::MLBackendContext<'static> + 'static)` cannot be sent between threads safely
```

**Impact:** Prevents using `Mutex<MLContext>` for shared thread-safe access. Cannot wrap `MLContext` in any `Send`-requiring container.

**Proposed Fix (rustnn side):** Add `Send + Sync` to `MLBackendContext`:
```rust
pub(crate) trait MLBackendContext<'context>: std::fmt::Debug + Send + Sync { ... }
```
The concrete implementations (`OrtContext`, `TrtxContext`) are already thread-safe — the trait just doesn't declare it.

### 7.4 Blocker C: Private `descriptor()` on MLTensor

**File:** `rustnn/src/mlcontext.rs`

```rust
impl MLTensor {
    pub(crate) fn descriptor(&self) -> &MLTensorDescriptor { ... }
}
```

**Problem:** `pub(crate)` visibility prevents accessing tensor metadata (shape, element count) from outside the `rustnn` crate. After dispatching, we need to know output tensor sizes to allocate read buffers.

**Impact:** Can't compute `read_tensor()` buffer sizes from outside rustnn. Must hardcode element counts or use public `shape()` which returns dimensions but requires manual byte-size calculation.

**Proposed Fix (rustnn side):** Make `descriptor()` or equivalent methods `pub`:
```rust
pub fn descriptor(&self) -> &MLTensorDescriptor { ... }
// or
pub fn element_count(&self) -> usize { ... }
pub fn byte_length(&self) -> usize { ... }
```

### 7.5 Blocker D: No `build_graph_info()` Taking Pre-Built GraphInfo

**File:** `rustnn/src/mlgraphbuilder.rs`

```rust
impl MLGraphBuilder<'context, 'builder> {
    pub fn build_graph_info(&mut self, graph: GraphInfo) -> Result<MLGraph<'context>> { ... }
}
```

**Problem:** This is the ideal API for Servo — the DOM builder already produces a `GraphInfo` via rustnn's typed API. But `build_graph_info` borrows `&mut self` on the builder, and `MLGraph<'context>` borrows the context. With `#[no_trace]` fields on the DOM struct, we'd need `RefCell` for interior mutability. `RefCell<MLGraphBuilder>` with mutable borrow + `MLGraph` holding a reference to the same context creates borrow-checker conflicts that `RefCell` catches at runtime (panic).

**Impact:** The `build()` → `MLGraph` flow works in tests where the builder is stack-allocated, but fails inside `RefCell`-wrapped DOM fields where the graph's implicit lifetime conflicts with subsequent builder use.

**Proposed Fix (rustnn side):** Make `MLGraph` own its backend state rather than borrowing:
```rust
pub struct MLGraph {
    pub(crate) backend: MLBackendGraph,  // owned, no lifetime
    pub input_descriptors: HashMap<String, OperandDescriptor>,
    pub output_descriptors: HashMap<String, OperandDescriptor>,
}
```
This requires `MLBackendGraph` variants to own their sessions/engines.

### 7.6 Blocker E: No Pre-Compiled Graph Dispatch

**File:** `rustnn/src/mlcontext.rs`

```rust
impl MLContext<'context> {
    pub fn dispatch(&mut self, graph: &mut MLGraph, inputs: ..., outputs: ...) -> Result<()> { ... }
}
```

**Problem:** `dispatch()` takes a `&mut MLGraph` built by the same `MLContext`. There is no API to load a pre-compiled graph (e.g., from cached ONNX bytes) and dispatch it. In Servo, graphs are compiled once and dispatched many times. Rebuilding the graph from `GraphInfo` on every dispatch is wasteful.

**Impact:** Model caching is impossible through the rustnn API. Each dispatch recompiles the graph.

**Proposed Fix (rustnn side):** Add a `dispatch_static()` or `dispatch_precompiled()` that takes a serialized engine/ONNX bytes:
```rust
pub fn dispatch_precompiled(&mut self, engine_bytes: &[u8], inputs: ..., outputs: ...) -> Result<()> { ... }
```
Or make `MLGraph` clonable/clone-free for caching.

### 7.7 Summary Table

| # | Blocker | Root Cause | Location | Fix in rustnn |
|---|---------|-----------|----------|---------------|
| A | Lifetime params in types | `'context` + `'builder` expose lifetimes to derive macros | `mlcontext.rs`, `mlgraphbuilder.rs` | Remove lifetimes, use `'static` |
| B | `MLBackendContext` not `Send + Sync` | Trait missing thread-safety bounds | `mlcontext.rs` | Add `Send + Sync` to trait |
| C | Private `descriptor()` on MLTensor | `pub(crate)` visibility | `mlcontext.rs` | Make `pub` or add `element_count()` |
| D | `MLGraph` borrows context | Lifetime tied to builder | `mlgraphbuilder.rs` | Make `MLGraph` own its backend state |
| E | No pre-compiled dispatch | API expects build-then-dispatch | `mlcontext.rs` | Add `dispatch_precompiled()` |
| F | `#[no_trace]` derive panics | Lifetime tokens in type stream confuse derive macros | `jstraceable_derive` (Servo) | Remove lifetimes from rustnn types (same as A) |

Blockers A, B, D, and E are all fixable on the rustnn side with API changes. Blocker C is a one-line visibility change. Blocker F is resolved by fixing A. **Total rustnn-side changes: ~30 lines across 4 files.**

### 7.8 Working Integration: The Demo Test

The `demo_rustnn_api` test (`components/webnn/tests/demo_rustnn_api.rs`) proves that when stack-allocated (no DOM structs involved), rustnn's full API works end-to-end:

```rust
// Stack-allocated — no JSTraceable issues, no lifetime problems
let mut ctx = MLContext::create(&opts)?;
let mut builder = MLGraphBuilder::new(&mut ctx)?;
let c = builder.add(builder.input("a", ...)?, builder.input("b", ...)?)?;
let mut graph = builder.build(&[("c", c)].into())?;
ctx.write_tensor(&a_tensor, &[1., 2., 3., 4.])?;
ctx.dispatch(&mut graph, &inputs, &outputs)?;
ctx.read_tensor(&c_tensor, &mut result)?;
// result = [6.0, 8.0, 10.0, 12.0]  ✓
```

This confirms that all blockers are DOM-storage-specific, not fundamental API issues.

