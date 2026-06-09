# WebNN Implementation with LiteRT Backend — Design Document

## 1. Architecture Overview

```
   JavaScript (WebNN API)
          │
          ▼
   ┌─────────────────────┐
   │   MLGraphBuilder     │  DOM bindings (components/script/dom/webnn/)
   │   MLContext.dispatch  │  ~3,069 lines
   │   MLTensor           │
   └──────────┬──────────┘
              │ GraphNode[] + input_infos + output_names
              ▼
   ┌─────────────────────┐
   │  servo-webnn crate  │  components/webnn/ (~4,200 lines)
   │                      │
   │  ┌────────────────┐ │
   │  │  Pre-processing │ │  insert_nhwc_transposes()
   │  │  (NCHW→NHWC)    │ │  Chromium-style Phase 1
   │  └───────┬────────┘ │
   │          ▼           │
   │  ┌────────────────┐ │
   │  │ Phase 2 Elim.  │ │  eliminate_transpose_pairs()
   │  │ (remove dupes) │ │  Redundant TRANSPOSE removal
   │  └───────┬────────┘ │
   │          ▼           │
   │  ┌────────────────┐ │
   │  │   Compiler      │ │  GraphNode[] → TFLite flatbuffer
   │  │  (compiler.rs)  │ │  ~2,800 lines
   │  └───────┬────────┘ │
   │          ▼           │
   │  ┌────────────────┐ │
   │  │  LiteRT Backend │ │  compile + run via litert-sys FFI
   │  │  (litert.rs)    │ │  ~515 lines
   │  └────────────────┘ │
   └─────────────────────┘
```

### Data Flow

```
JS: builder.conv2d(input, filter, {padding: "same"})
         │
         ▼
MLGraphBuilder produces ComputeNode list
         │
         ▼  dispatch()
Backend::compile_with_input_infos()
         │
         ├── 1. insert_nhwc_transposes(pre_nodes)      [Phase 1]
         │       TRANSPOSE [0,2,3,1] before conv2d/pool2d
         │       TRANSPOSE [0,3,1,2] after  conv2d/pool2d
         │       Filter OIHW→OHWI     (conv2d)
         │       Filter IOHW→OHWI     (convTranspose2d, perm [1,2,3,0])
         │       Depthwise: OIHW→[1,H,W,C_out]
         │
         ├── 2. eliminate_transpose_pairs(nodes)       [Phase 2]
         │       Remove redundant NCHW↔NHWC pairs around
         │       layout-agnostic unary ops (relu, sigmoid, tanh, etc.)
         │
         ├── 3. compile_with_input_infos(transformed_nodes)
         │       Two-pass TFLite flatbuffer generation
         │       First pass: ensure tensors, decompose ops
         │       Second pass: emit TFLite operator codes
         │
         └── 4. LiteRT: Model::from_bytes → CompiledModel
                                                    │
Backend::run()                                      ▼
         ┌── Input buffers (NCHW, as-is) ────→  LiteRT inference
         └── Output buffers (NCHW, from TFLite) ←  XNNPACK
```

**Key invariant**: At the core crate boundary, data is always NCHW. Layout conversion is a backend-specific concern. The TFLite backend handles it via upfront TRANSPOSE insertion — no runtime shuffling, no shape propagation in the compiler.

---

## 2. Design Steps (end-to-end flow)

### Step 0 — Initialize Backend

`components/webnn/litert/mod.rs`, `components/webnn/lib.rs`

1. `navigator.ml.createContext()` → `MLContext::new()` → `start_webnn_backend()`
2. `start_webnn_backend()` calls `litert::initialize()` which creates a `LiteRtEnvironment` (one-time init, lazy, thread-safe)
3. `MLContext` holds no compiled models yet; thread-local cache starts empty

### Step 1 — Build Graph

`components/script/dom/webnn/mlgraphbuilder.rs`

1. JS creates `new MLGraphBuilder()`
2. JS calls operator methods: `builder.conv2d(input, filter, options)`, `builder.convTranspose2d(input, filter, options)`, `builder.relu(x)`, `builder.add(a, b)`, etc.
3. Each call validates operands (same builder, broadcastable shapes), computes output shape, creates a `ComputeNode { op, inputs, output, desc, attrs, data }`
4. `builder.constant(data, descriptor)` creates `ComputeNode { op: "constant", data: Some(bytes), ... }` for filter weights, biases, etc.
5. JS calls `builder.build(outputOperands)` → collects `ComputeNode[]` list with I/O names and `InputOperandInfo[]`
6. Returns an `MLGraph` object wrapping the node list

### Step 2 — Dispatch (Compile + Run)

`components/script/dom/webnn/mlcontext.rs`

1. JS calls `context.dispatch(graph, inputs)` with `MLTensor[]` inputs
2. `MLContext.dispatch()` computes a hash of the graph → checks thread-local model cache
3. **Cache miss**: converts `ComputeNode[]` → `GraphNode[]`, calls `Backend::compile_with_input_infos(nodes, input_infos, output_names)`
4. **Cache hit**: skips compilation, uses cached `CompiledModel`
5. Calls `Backend::run(model, input_data)` → fills input buffers, invokes LiteRT, reads outputs
6. Returns output `MLTensor[]`

### Step 3 — Read Output

`components/script/dom/webnn/mlcontext.rs`

1. JS calls `context.readTensor(outputTensor)` → copies output buffer data to JS ArrayBuffer
2. Result is a `Float32Array`, `Int32Array`, etc. matching the tensor's dataType

### Step 4 — Preprocessing: Phase 1 (NHWC TRANSPOSE insertion)

`components/webnn/compiler.rs: insert_nhwc_transposes()`

1. Receives the original NCHW `GraphNode[]` + `constant_data` + `input_infos`
2. For each NHWC-sensitive op (conv2d, pool2d, resample2d, convTranspose2d) with 4-D input:
   - Insert `TRANSPOSE` with perm `[0,2,3,1]` (NCHW→NHWC) before the op
   - Graph inputs always get explicit TRANSPOSE (never silent reshape)
   - Layout boundary ops (transpose, reshape, concat) also get explicit TRANSPOSE
   - Intermediate NHWC tensors reuse name (no redundant transpose)
   - Permutation stored in `attrs` (`perm_len`, `perm_0`..`perm_N`), NOT as node input
   - **After this pass, `node.desc.shape` for 4-D ops is in NHWC format** (used directly by handlers)
3. For conv2d filter: if constant data exists → transpose OIHW→OHWI at compile time; if graph input → insert TRANSPOSE op
4. For convTranspose2d filter: IOHW `[C_in, C_out, H, W]` → OHWI `[C_out, H, W, C_in]` via perm `[1,2,3,0]` (matching Chromium's `GetConvTranspose2DFilterPermutation`)
5. Depthwise conv2d filter: OIHW `[O, 1, H, W]` → depthwise format `[1, H, W, C_out]`
6. Insert `TRANSPOSE` with perm `[0,3,1,2]` (NHWC→NCHW) after the op
7. Return transformed nodes + extra constant data

**NHWC-sensitive ops** (require NCHW↔NHWC wrapping):
- `conv2d`, `conv_2d`, `convTranspose2d`, `conv_transpose_2d`
- `maxPool2d`, `max_pool_2d`, `averagePool2d`, `average_pool_2d`, `l2Pool2d`, `l2_pool_2d`
- `resample2d`, `resample_2d`

**Layout boundary ops** (produce existing-NHWC tensors that don't need transpose):
- `transpose`, `reshape`, `concat`

### Step 4b — Preprocessing: Phase 2 (Transpose elimination)

`components/webnn/compiler.rs: eliminate_transpose_pairs()`

Removes redundant NCHW↔NHWC transpose pairs around **layout-agnostic unary ops** — operations that produce the same numerical result regardless of CHW vs HWC ordering because they apply element-wise.

**Layout-agnostic unary ops**: relu, sigmoid, tanh, elu, gelu, hardSwish, hardSigmoid, softplus, softsign, leakyRelu, clamp, abs, ceil, floor, negative, identity, exp, log, cos, sin, sqrt, square, cast

**Pattern eliminated**: `T_out(NHWC→NCHW) → [unary ops] → T_in(NCHW→NHWC)` → both transposes removed, chain ops converted to NHWC shapes.

**Protection**: Graph output nodes are never eliminated — T_in nodes producing graph outputs are preserved.

### Step 5 — Compile (TFLite flatbuffer generation)

`components/webnn/compiler.rs: compile_with_input_infos()`

1. Pass 1 (`match node.op.as_str()`): Register tensors via `ensure_tensor()`, decompose complex ops (batchNorm→5 ops, layerNorm→9 ops, etc.), create constant data buffers
2. Pass 2 (`match tfl_code`): Emit TFLite operator codes, create temporary tensors for permutations/constants, build FlatBuffer options structs
3. `webnn_op_to_tflite()` maps 61+ WebNN op names → TFLite op codes
4. Conv2d/pool2d explicit padding: inserts PAD op (`[4,2]` paddings tensor in NHWC `[[0,0],[top,bot],[left,right],[0,0]]` format) + VALID conv/pool (matching Chromium)
5. TransposeConv: emits `output_shape` tensor (NHWC, direct from `node.desc.shape` — already converted by preprocessing), filter in OHWI, padding=VALID for zero-pads or SAME for explicit pads. TransposeConvOptions uses `push_slot_always` for all fields (strides must be explicit; `push_slot` with defaults can omit fields)
6. Reduce ops: axes stored as `axis_0`..`axis_N` attrs; compiler creates Int32 axes tensor as second input
7. FlatBuffer builder: pre-create vectors before tables (avoids 64KB assertion)
8. Return `CompileResult { flatbuf, nhwc_inputs: [], nhwc_outputs: [] }`

### Step 6 — LiteRT Run

`components/webnn/backends/litert.rs`

1. `compile_with_input_infos()`: `LiteRtModel::from_bytes(flatbuf)` → `CompiledModelBuilder` → compile → `LiteRtState`
2. `run()`: Create `TensorBuffer`s for each input, fill with NCHW data, invoke, read outputs
3. `extract_output_region()`: Handle XNNPACK padding — query `LiteRtGetCompiledModelOutputTensorLayouts` for actual stride/size, copy only valid region
4. No runtime NCHW↔NHWC conversion — TRANSPOSE ops in the graph handle layout
5. `nhwc_inputs`/`nhwc_outputs` always empty (dead code — pre-processing handles all layout)

---

## 3. Component Details

### 3.1 Core Types (`backend.rs`, 107 lines)

| Type | Purpose |
|------|---------|
| `DataType` | Enum: Float32/16, Int32/64, Uint8/32/64, Int8 |
| `TensorDesc` | Shape + DataType, with `num_elements()` and `byte_length()` |
| `GraphNode` | Op name, input names, output name, desc, attrs, optional constant data |
| `OpAttrs` | `HashMap<String, f64>` — op parameters (strides, padding, etc.) |
| `CompiledModel` | Extensible newtype (`Box<dyn Any + Send + Sync>`) + I/O shape metadata |
| `RunResult` | Output buffers: `Vec<Vec<u8>>` |
| `Backend` trait | `compile_with_input_infos()` + `run()` — send + sync safe |

### 3.2 DOM Bindings (`components/script/dom/webnn/`, ~3,069 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `mlgraphbuilder.rs` | 2,293 | All operator implementations, validation, shape computation |
| `mlcontext.rs` | 473 | `dispatch()`, `createTensor()`, `readTensor()`, `writeTensor()`, model cache |
| `mlgraph.rs` | 104 | Stores `ComputeNode` list, I/O names, `InputOperandInfo` |
| `mltensor.rs` | 136 | Data buffer, read/write, properties |
| `mloperand.rs` | 93 | Name, data_type, shape, builder weak ref |
| `ml.rs` | 52 | `navigator.ml` getter |
| `mod.rs` | 10 | Module declarations |

### 3.3 WebIDL (`WebNN.webidl`, 292 lines)

Defines: `ML`, `MLContext`, `MLGraphBuilder` (30+ operator methods), `MLOperand`, `MLTensor`, enums (`MLInputOperandLayout`, `MLOperandDataType`, etc.), dictionaries (29 option dicts).

### 3.4 File Map

| File | Lines | Role |
|------|-------|------|
| `webnn/backend.rs` | 107 | DataType, TensorDesc, GraphNode, Backend trait, CompiledModel |
| `webnn/backends/mod.rs` | 73 | Backend registry: compile(), run(), infer() |
| `webnn/lib.rs` | 26 | Re-exports, feature-gated litert module, init |
| `webnn/compiler.rs` | ~2,800 | TFLite FlatBuffer compiler + NHWC pre-processing (Phase 1 + 2) |
| `webnn/backends/litert.rs` | ~515 | LiteRT FFI: compile + run + output extraction |
| `webnn/litert/mod.rs` | 9 | One-time Environment init |
| `webnn/tests/litert.rs` | ~740 | 30 compile tests + 3 inference tests (conv2d, maxPool2d, convTranspose2d) |
| `script/dom/webnn/mlgraphbuilder.rs` | 2,293 | All operator DOM bindings + shape computation |
| `script/dom/webnn/mlcontext.rs` | 473 | dispatch(), tensor I/O, model cache |
| `script/dom/webnn/mlgraph.rs` | 104 | ComputeNode list + I/O names |
| `script/dom/webnn/mltensor.rs` | 136 | Data buffer, read/write |
| `script/dom/webnn/mloperand.rs` | 93 | Name, dtype, shape |
| `script/dom/webnn/ml.rs` | 52 | navigator.ml getter |
| `script/dom/webnn/mod.rs` | 10 | Module declarations |
| `WebNN.webidl` | 292 | Full IDL definition |

---

## 4. Op Coverage

### 4.1 Directly Mapped

| Category | Ops |
|----------|-----|
| Binary arithmetic | add, sub, mul, div, max, min, pow |
| Comparison | equal, notEqual, greater, greaterOrEqual, lesser, lesserOrEqual |
| Logical | logicalAnd, logicalOr, logicalNot |
| Unary math | abs, neg, sqrt, rsqrt, exp, log, sin, cos, ceil, floor, sign |
| Activation | relu, relu6, sigmoid, tanh, gelu, hardSwish, hardSigmoid, elu, leakyRelu, prelu, softplus, softsign |
| Convolution | conv2d, conv_2d, convTranspose2d, conv_transpose2d, depthwise (via groups) |
| Pooling | averagePool2d, maxPool2d, l2Pool2d |
| Layout | reshape, transpose, concat, slice, split, pad, squeeze, tile |
| Resize | resample2d (bilinear + nearest-neighbor) |
| Matrix | matmul, gemm → BATCH_MATMUL |
| Reduction | reduceMean, reduceSum, reduceMax, reduceMin, reduceProduct, reduceL1, reduceL2, reduceLogSum, reduceLogSumExp, reduceSumSquare |
| Indexing | gather, gatherNd, scatterNd |
| Cast/Quantize | cast, quantizeLinear |
| Other | expand, reverse, cumulativeSum, argMax, argMin |

### 4.2 Decomposed

| Op | Decomposition |
|----|---------------|
| batchNormalization | SUB, ADD, SQRT, DIV, MUL, ADD |
| layerNormalization | MEAN, SUB, MUL, MEAN, ADD, SQRT, DIV, MUL, ADD |
| instanceNormalization | Same as layerNorm with axes=[2,3] |
| dequantizeLinear | CAST(f32), CAST(f32), SUB, MUL |
| clamp | MINIMUM + MAXIMUM with constant bounds |
| linear | MUL(alpha) + ADD(beta) |
| tan | SIN + COS + DIV |
| where | SELECT(condition, true, false) |

### 4.3 DOM-Bound (JS API, not yet mapped to TFLite)

| Category | Ops |
|----------|-----|
| Additional unary | reciprocal, erf |
| Additional layout | scatterElements, scatterND, gatherElements, gatherND |
| Misc | triangular |

---

## 5. Key Design Decisions

### 5.1 Chromium-Style TRANSPOSE Insertion

**Problem**: TFLite expects NHWC layout. WebNN spec uses NCHW. Two approaches:
- (a) Runtime data transposition at I/O boundaries — error-prone, requires tracking which tensors are NHWC
- (b) Upfront graph transformation — inserts explicit TRANSPOSE ops, compiler treats shapes as-is

**Chosen**: Approach (b) — two-phase Chromium-style implementation:

**Phase 1** (`insert_nhwc_transposes()`): Naive TRANSPOSE insertion around all NHWC-sensitive ops. Every conv2d/pool2d/convTranspose2d gets wrapped. Graph inputs always get explicit TRANSPOSE.

**Phase 2** (`eliminate_transpose_pairs()`): Remove redundant NCHW↔NHWC pairs around element-wise ops. Pattern `T_out(NHWC→NCHW) → [unary ops] → T_in(NCHW→NHWC)` → both transposes removed, chain runs in NHWC.

Benefits:
- No runtime data shuffling (I/O stays NCHW)
- No shape tracking needed in the compiler
- No post-hoc NHWC propagation
- Matches Chromium WebNN implementation pattern

### 5.2 CompiledModel Extensible Newtype

```rust
pub struct CompiledModel(pub Box<dyn Any + Send + Sync>);
```

New backends only need to implement `Backend` trait and add a variant. No enum modification needed.

### 5.3 In-Process Backend (vs Chromium's Mojo IPC)

Direct function calls via `litert-sys` FFI. Simpler, sufficient for CPU-only inference. No IPC serialization overhead.

### 5.4 Model Caching

Thread-local `HashMap<usize, CompiledModel>` keyed by graph hash. First `dispatch()` compiles, subsequent calls reuse.

### 5.5 FlatBuffer 64KB Table Limit

TFLite flatbuffers hit assertion failure if vectors are created after `start_table()`. Solution: pre-create all vectors before starting the table.

### 5.6 TransposeConv Encoding Details

- **Filter format**: OHWI `[C_out, H, W, C_in]` (matching Chromium). WebNN IOHW → OHWI via perm `[1,2,3,0]`.
- **Output shape**: Direct from `node.desc.shape` (already NHWC from pre-processing — never double-convert).
- **Padding**: VALID when all pads are zero (output_size = stride * (input-1) + filter), SAME for explicit pads.
- **Options**: `push_slot_always` for all TransposeConvOptions fields; strides must be explicit.

### 5.7 Explicit Padding (conv2d/pool2d)

When explicit padding values `[top, bottom, left, right]` are provided and non-zero:
1. Insert a TFLite PAD op before conv2d/pool2d, creating padded input
2. Paddings tensor: shape `[4,2]` = `[[0,0], [top, bottom], [left, right], [0,0]]` for NHWC
3. Use VALID padding in the conv2d/pool2d op
4. DOM computes output shape using explicit padding formula: `(in + pad_begin + pad_end - filter) / stride + 1`

---

## 6. Current Implementation Status

### 6.1 What Works End-to-End

| Feature | Status |
|---------|--------|
| Conv2d (regular + depthwise) | Passing |
| ConvTranspose2d | Passing (IOHW→OHWI, VALID/SAME, strides) |
| MaxPool2d | Passing |
| Explicit padding (conv2d + pool2d) | Passing (PAD + VALID approach) |
| 30+ compile-only ops | Passing |
| NHWC pre-processing (Phase 1) | Complete |
| Transpose elimination (Phase 2) | Complete |
| Filter transposition | OIHW→OHWI, IOHW→OHWI, OIHW→depthwise |
| XNNPACK padded output extraction | Working |
| Model caching | Working |
| DOM bindings | Working (all operator methods) |
| 33 litert tests | Passing (30 compile + 3 inference) |

### 6.2 Known Gaps

| Issue | Impact | Priority |
|-------|--------|----------|
| `log::error!` used for routine diagnostics | Log noise | Low |
| RESIZE handler has legacy inline transpose for 3-D | Dead code path, harmless | Low |
| `nhwc_inputs`/`nhwc_outputs` always empty (dead code) | No functional impact | Low |
| TransposeConv explicit padding: falls back to SAME | Asymmetric padding may produce wrong output | Low (not used in tests) |
| `scatterElements`/`gatherElements`/`triangular` | No TFLite op | Medium |
| `reciprocal`/`erf` | Can decompose (DIV/SQRT etc.) | Low |
| `logicalXor` | No TFLite XOR op, needs decomposition | Low |

---

## 7. GitHub Issue Checkpoints

### Issue 1: WebNN Core Crate — Types, Traits, and Backend Abstraction

- [x] `DataType` enum with all 8 types (Float32/16, Int32, Uint32, Int64, Uint64, Int8, Uint8)
- [x] `TensorDesc` with shape + data_type, `num_elements()`, `byte_length()`
- [x] `GraphNode` with op, inputs, output, desc, attrs, optional data
- [x] `OpAttrs = HashMap<String, f64>`
- [x] `CompiledModel` extensible newtype (`Box<dyn Any + Send + Sync>`) + I/O shape metadata
- [x] `RunResult` with output buffers
- [x] `Backend` trait: `name()`, `compile_with_input_infos()`, `run()`
- [x] `start_webnn_backend()` initialization function
- [x] MPL 2.0 license headers on all files
- [x] Unit tests for DataType conversions and TensorDesc

### Issue 2: TFLite Flatbuffer Compiler

- [x] `webnn_op_to_tflite()` mapping for 61+ op names
- [x] Two-pass compilation: tensor registration → op emission
- [x] `insert_nhwc_transposes()` Chromium-style Phase 1 pass
  - [x] NHWC-sensitive ops: conv2d, pool2d, resample2d, convTranspose2d
  - [x] Layout boundary ops: transpose, reshape, concat
  - [x] Graph inputs always get explicit TRANSPOSE
  - [x] Intermediate NHWC tensors reuse names (no redundant transpose)
  - [x] Filter data transposition: OIHW→OHWI (regular), IOHW→OHWI (convTranspose2d), OIHW→depthwise format
  - [x] Permutation stored in `attrs` (perm_len, perm_0..perm_N), NOT as node input
- [x] `eliminate_transpose_pairs()` Chromium-style Phase 2 pass
- [x] Decomposed ops: batchNorm, layerNorm, instanceNorm, dequantizeLinear, clamp, linear, tan, where
- [x] Custom FlatBuffer builder that pre-creates vectors before tables
- [x] `CompileResult` struct with flatbuf bytes
- [x] No post-hoc NHWC shape propagation in compiler
- [x] Conv2d bias: wire 3rd input as bias tensor
- [x] Explicit padding via PAD + VALID for conv2d/pool2d
- [x] Depthwise conv2d → DEPTHWISE_CONV_2D with [1,H,W,C_out] filter
- [x] Reduce ops: Int32 axes tensor, keepDimensions support
- [x] TransposeConv: output_shape tensor, OHWI filter, VALID/SAME padding, strides
- [x] Compile-only tests for 30+ ops
- [x] End-to-end inference tests for conv2d, maxPool2d, convTranspose2d

### Issue 3: LiteRT Backend

- [x] `LiteRtBackend` implements `Backend` trait
- [x] `initialize()` one-time environment setup
- [x] `compile_with_input_infos()`: flatbuf → Model → CompiledModelBuilder → compile → LiteRtState
- [x] `run()`: TensorBuffer I/O, fill inputs, invoke, extract outputs
- [x] `extract_output_region()` for XNNPACK-padded output handling
- [x] Output layout query via `LiteRtGetCompiledModelOutputTensorLayouts`
- [x] No runtime NCHW↔NHWC conversion (handled by graph transposes)
- [x] Feature-gated behind `litert` in Cargo.toml
- [x] `ports/servoshell/Cargo.toml` has `webnn = []` feature

### Issue 4: DOM Bindings and WebIDL

- [x] `WebNN.webidl` with all interfaces, enums, dictionaries
- [x] `MLGraphBuilder` with 30+ operator methods
- [x] `MLContext.dispatch()` calls Backend via webnn crate
- [x] `MLContext.createTensor()`, `readTensor()`, `writeTensor()`
- [x] `MLTensor` with dataType, shape, readable, writable, constant properties
- [x] Thread-local model cache in MLContext
- [x] `#[cfg(feature = "webnn")]` guards on all webnn code
- [x] `check_same_builder()` validates operands from same builder
- [x] `check_broadcastable()` / `broadcast_shape()` for binary ops

### Issue 5: Integration and Testing

- [x] `cargo test -p servo-webnn --test litert` — all 33 tests pass
- [x] `./mach build -d --use-crown --features webnn` succeeds
- [x] Conv2d inference produces correct numerical output (all-9s for all-ones 3×3 filter)
- [x] MaxPool2d inference produces correct output (`[6, 8, 14, 16]`)
- [x] ConvTranspose2d inference produces correct output (`[1,3,2,4,10,6,3,7,4]`)
- [x] NHWC pre-processing correctly inserts transposes
- [x] No "Custom allocation too small" error from XNNPACK padded outputs
- [x] No flatbuffer 64KB table size assertion
- [ ] WPT conformance test harness (tests/wpt/tests/webnn/)
- [ ] `log::error!` calls replaced with `log::debug!`/`log::trace!` for routine messages

### Issue 6: Remaining Op Support

- [x] Depthwise conv2d: `DEPTHWISE_CONV_2D` with proper filter format
- [x] Conv2d bias: 3rd input as bias tensor
- [x] Explicit padding: PAD + VALID approach for conv2d/pool2d
- [x] TransposeConv: full implementation (DOM options, OHWI filter, output shape)
- [x] Reduce variants: L1, L2, logSum, logSumExp, sumSquare mapped
- [x] hardSigmoid, softplus, softsign directly mapped
- [ ] Decomposed ops: `reciprocal`→div(1,x), `erf`, `logicalXor`
- [ ] `scatterElements`/`gatherElements` (no direct TFLite op)

### Issue 7: Cross-compilation and Platform Support

- [ ] `aarch64-unknown-linux-gnu` builds with litert-sys prebuilts
- [ ] `aarch64-unknown-linux-ohos` target_spec entry for OpenHarmony
- [ ] ARM64 LiteRT bindings generated
- [x] GPU/NPU accelerator registration (graceful fallback to CPU)
- [x] No hardcoded x86_64 assumptions in compiler or backend
