//! LiteRT backend — uses litert v0.2.1 for inference.

use std::collections::HashSet;

use litert::{
    Accelerators, CompilationOptions, CompiledModel, ElementType, Environment, Model, TensorBuffer,
    TensorShape,
};
use litert_sys as sys;

use crate::backend::{Backend, CompiledModel as WebNNModel, DataType, GraphNode, RunResult};
use crate::compiler;

pub(crate) struct LiteRtBackend;

struct LiteRtState {
    compiled: CompiledModel,
    env: Environment,
    /// Actual output layouts after LiteRT delegates (e.g. XNNPACK) have
    /// padded/resized output tensors. Stored as a list of dims-per-output,
    /// in the same order as `CompiledModel::run` expects them.
    output_layouts: Vec<Vec<i32>>,
}

/// Extract the raw `LiteRtCompiledModel` pointer from the high-level
/// `litert::CompiledModel` wrapper.
///
/// This is `unsafe` because it relies on the first field of
/// `litert::CompiledModel` being the `ptr: NonNull<LiteRtCompiledModelT>`.
/// The `litert` 0.2.1 crate does not expose `as_raw()` for `CompiledModel`,
/// so we cannot call `LiteRtGetCompiledModelOutputTensorLayouts` otherwise.
///
/// The 0.2.1 `litert::CompiledModel` struct is `pub struct CompiledModel {
///     ptr: NonNull<sys::LiteRtCompiledModelT>,
///     _env: Environment,
///     _model: Model,
/// }`. The first field is the raw pointer. `NonNull<T>` has the same
/// in-memory representation as `*mut T`, so we read the first 8 bytes of
/// the struct as a raw pointer.
unsafe fn compiled_model_raw(compiled: &CompiledModel) -> sys::LiteRtCompiledModel {
    let base = compiled as *const CompiledModel as *const u8;
    unsafe { std::ptr::read(base as *const sys::LiteRtCompiledModel) }
}

/// Query the actual output tensor layouts from a compiled model.
///
/// Passing `update_allocation=true` causes the underlying runtime (e.g. the
/// XNNPACK delegate) to finalize tensor allocations so the returned layouts
/// reflect the true (potentially padded) output shapes.
fn get_output_layouts(
    raw_compiled: sys::LiteRtCompiledModel,
    signature_index: u32,
    num_outputs: usize,
) -> Result<Vec<Vec<i32>>, String> {
    let mut layouts: Vec<sys::LiteRtLayout> = vec![unsafe { std::mem::zeroed() }; num_outputs];
    let status = unsafe {
        sys::LiteRtGetCompiledModelOutputTensorLayouts(
            raw_compiled,
            signature_index as usize,
            layouts.len(),
            layouts.as_mut_ptr(),
            true, // update_allocation
        )
    };
    if status != sys::kLiteRtStatusOk {
        return Err(format!(
            "LiteRtGetCompiledModelOutputTensorLayouts failed: status {}",
            status
        ));
    }

    Ok(layouts
        .iter()
        .map(|l| {
            let rank = l.rank() as usize;
            l.dimensions[..rank].to_vec()
        })
        .collect())
}

/// Extract elements from a potentially-padded NHWC buffer, keeping only
/// elements within the declared output shape. This handles XNNPACK-style
/// SIMD padding where the actual tensor dimensions (returned by
/// `LiteRtGetCompiledModelOutputTensorLayouts`) may exceed the declared
/// shapes (e.g. width padded from 3 to 4).
fn extract_output_region(
    buf_bytes: &[u8],
    elem_size: usize,
    decl_shape: &[u32],
    actual_layout: &[i32],
) -> Vec<u8> {
    if actual_layout.len() < 4 || decl_shape.len() < 4 {
        return buf_bytes.to_vec();
    }
    let n = decl_shape[0] as usize;
    let h = decl_shape[1] as usize;
    let w = decl_shape[2] as usize;
    let c = decl_shape[3] as usize;
    let ah = actual_layout[1] as usize;
    let aw = actual_layout[2] as usize;
    let ac = actual_layout[3] as usize;

    let total = n * h * w * c;
    let mut out = vec![0u8; total * elem_size];
    for ni in 0..n.min(1) {
        for hi in 0..h.min(ah) {
            for wi in 0..w.min(aw) {
                for ci in 0..c.min(ac) {
                    let src_off = ((ni * ah + hi) * aw + wi) * ac + ci;
                    let dst_off = ((ni * h + hi) * w + wi) * c + ci;
                    let src_byte = src_off * elem_size;
                    let dst_byte = dst_off * elem_size;
                    if src_byte + elem_size <= buf_bytes.len() {
                        out[dst_byte..dst_byte + elem_size]
                            .copy_from_slice(&buf_bytes[src_byte..src_byte + elem_size]);
                    }
                }
            }
        }
    }
    out
}

#[allow(dead_code)]
fn layout_byte_size(dims: &[i32], dt: DataType) -> usize {
    let mut n: usize = 1;
    for &d in dims {
        if d > 0 {
            n *= d as usize;
        }
    }
    n * dt.element_byte_size()
}

fn our_dt_to_litert(dt: DataType) -> ElementType {
    match dt {
        DataType::Float32 => ElementType::Float32,
        DataType::Float16 => ElementType::Float16,
        DataType::Int32 => ElementType::Int32,
        DataType::Uint32 => ElementType::UInt32,
        DataType::Int64 => ElementType::Int64,
        DataType::Uint64 => ElementType::UInt64,
        DataType::Int8 => ElementType::Int8,
        DataType::Uint8 => ElementType::UInt8,
    }
}

fn nchw_to_nhwc(data: &[u8], elem_size: usize, nhwc_shape: &[u32]) -> Vec<u8> {
    let (n, h, w, c) = (
        nhwc_shape[0] as usize,
        nhwc_shape[1] as usize,
        nhwc_shape[2] as usize,
        nhwc_shape[3] as usize,
    );
    let mut out = vec![0u8; data.len()];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ((ni * c + ci) * h + hi) * w + wi;
                    let dst = ((ni * h + hi) * w + wi) * c + ci;
                    out[dst * elem_size..(dst + 1) * elem_size]
                        .copy_from_slice(&data[src * elem_size..(src + 1) * elem_size]);
                }
            }
        }
    }
    out
}

fn nhwc_to_nchw(data: &[u8], elem_size: usize, nhwc: &[u32]) -> Vec<u8> {
    let (n, h, w, c) = (
        nhwc[0] as usize,
        nhwc[1] as usize,
        nhwc[2] as usize,
        nhwc[3] as usize,
    );
    let mut out = vec![0u8; data.len()];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ((ni * h + hi) * w + wi) * c + ci;
                    let dst = ((ni * c + ci) * h + hi) * w + wi;
                    out[dst * elem_size..(dst + 1) * elem_size]
                        .copy_from_slice(&data[src * elem_size..(src + 1) * elem_size]);
                }
            }
        }
    }
    out
}

fn write_typed_input(buf: &mut TensorBuffer, data: &[u8], dt: DataType) -> Result<(), String> {
    macro_rules! write_typed {
        ($rust_ty:ty) => {{
            let mut guard = buf
                .lock_for_write::<$rust_ty>()
                .map_err(|e| format!("lock input: {}", e))?;
            let typed_data: &[$rust_ty] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const $rust_ty,
                    data.len() / std::mem::size_of::<$rust_ty>(),
                )
            };
            guard.copy_from_slice(typed_data);
        }};
    }
    match dt {
        DataType::Float32 => write_typed!(f32),
        DataType::Int32 => write_typed!(i32),
        DataType::Uint32 => write_typed!(u32),
        DataType::Int64 => write_typed!(i64),
        DataType::Uint64 => write_typed!(u64),
        DataType::Int8 => write_typed!(i8),
        DataType::Uint8 => {
            let mut guard = buf
                .lock_for_write::<u8>()
                .map_err(|e| format!("lock input: {}", e))?;
            guard[..data.len()].copy_from_slice(data);
        },
        DataType::Float16 => {
            let mut guard = buf
                .lock_for_write::<u8>()
                .map_err(|e| format!("lock input (f16): {}", e))?;
            guard[..data.len()].copy_from_slice(data);
        },
    }
    Ok(())
}

fn read_typed_output(buf: &TensorBuffer, dt: DataType) -> Result<Vec<u8>, String> {
    macro_rules! read_typed {
        ($rust_ty:ty) => {{
            let guard = buf
                .lock_for_read::<$rust_ty>()
                .map_err(|e| format!("read output: {}", e))?;
            let slice: &[$rust_ty] = &*guard;
            let mut v: Vec<u8> = Vec::with_capacity(slice.len() * std::mem::size_of::<$rust_ty>());
            for val in slice {
                v.extend_from_slice(&val.to_ne_bytes());
            }
            Ok(v)
        }};
    }
    match dt {
        DataType::Float32 => read_typed!(f32),
        DataType::Int32 => read_typed!(i32),
        DataType::Uint32 => read_typed!(u32),
        DataType::Int64 => read_typed!(i64),
        DataType::Uint64 => read_typed!(u64),
        DataType::Int8 => {
            let guard = buf
                .lock_for_read::<i8>()
                .map_err(|e| format!("read output: {}", e))?;
            let slice: &[i8] = &*guard;
            Ok(slice.iter().map(|&v| v as u8).collect())
        },
        DataType::Uint8 => {
            let guard = buf
                .lock_for_read::<u8>()
                .map_err(|e| format!("read output: {}", e))?;
            let slice: &[u8] = &*guard;
            Ok(slice.to_vec())
        },
        DataType::Float16 => {
            let guard = buf
                .lock_for_read::<u8>()
                .map_err(|e| format!("read output (f16): {}", e))?;
            let slice: &[u8] = &*guard;
            Ok(slice.to_vec())
        },
    }
}

impl Backend for LiteRtBackend {
    fn name(&self) -> &'static str {
        "litert"
    }

    fn compile_with_input_infos(
        &self,
        nodes: &[GraphNode],
        input_infos: &[(String, Vec<u32>, DataType)],
    ) -> Result<WebNNModel, String> {
        let compile_result = compiler::compile_with_input_infos(nodes, input_infos)?;
        let flatbuf = compile_result.flatbuf;
        let nhwc_inputs: HashSet<String> = compile_result.nhwc_inputs.into_iter().collect();
        let nhwc_outputs: HashSet<String> = compile_result.nhwc_outputs.into_iter().collect();

        let mut input_shapes: Vec<(String, Vec<u32>, DataType)> = input_infos.to_vec();
        input_shapes.sort_by(|a, b| a.0.cmp(&b.0));
        input_shapes = input_shapes
            .into_iter()
            .map(|(name, shape, dt)| {
                if nhwc_inputs.contains(&name) && shape.len() >= 4 {
                    (name, vec![shape[0], shape[2], shape[3], shape[1]], dt)
                } else {
                    (name, shape, dt)
                }
            })
            .collect();

        let output_shapes: Vec<(String, Vec<u32>, DataType)> = nodes
            .iter()
            .filter(|n| n.op != "constant")
            .map(|n| {
                let shape = if nhwc_outputs.contains(&n.output) && n.desc.shape.len() >= 4 {
                    vec![
                        n.desc.shape[0],
                        n.desc.shape[2],
                        n.desc.shape[3],
                        n.desc.shape[1],
                    ]
                } else {
                    n.desc.shape.clone()
                };
                (n.output.clone(), shape, n.desc.data_type)
            })
            .collect();

        let env = Environment::new().map_err(|e| format!("LiteRT env: {}", e))?;
        let model = Model::from_bytes(flatbuf).map_err(|e| format!("LiteRT load: {}", e))?;
        let mut options =
            CompilationOptions::new().map_err(|e| format!("LiteRT options: {}", e))?;
        options
            .set_accelerators(Accelerators::CPU)
            .map_err(|e| format!("LiteRT options accelerators: {}", e))?;

        let compiled = CompiledModel::new(env, model, &options)
            .map_err(|e| format!("LiteRT compile: {}", e))?;

        // Query actual output layouts after LiteRT delegates (e.g. XNNPACK)
        // have finalized tensor allocations. The returned dims may be larger
        // than the model-declared shapes due to SIMD alignment padding.
        let output_layouts = unsafe {
            let raw = compiled_model_raw(&compiled);
            get_output_layouts(raw, 0, output_shapes.len())
                .map_err(|e| format!("LiteRT output layouts: {}", e))?
        };
        for (i, layout) in output_layouts.iter().enumerate() {
            log::error!(
                "LiteRT output {} actual layout: {:?} (declared: {:?})",
                i,
                layout,
                output_shapes.get(i).map(|(_, s, _)| s.clone())
            );
        }

        let run_env = Environment::new().map_err(|e| format!("LiteRT run env: {}", e))?;
        let state = Box::new(LiteRtState {
            compiled,
            env: run_env,
            output_layouts,
        });

        Ok(WebNNModel::LiteRt {
            compiled: state,
            input_shapes,
            output_shapes,
            nhwc_inputs,
            nhwc_outputs,
        })
    }

    fn run(&self, model: &WebNNModel, inputs: &[(&str, &[u8])]) -> Result<RunResult, String> {
        let (state, input_shapes, output_shapes, nhwc_inputs, nhwc_outputs) = match model {
            WebNNModel::LiteRt {
                compiled,
                input_shapes,
                output_shapes,
                nhwc_inputs,
                nhwc_outputs,
            } => (
                compiled,
                input_shapes,
                output_shapes,
                nhwc_inputs,
                nhwc_outputs,
            ),
        };
        let state = state
            .downcast_ref::<LiteRtState>()
            .ok_or("Bad compiled model")?;
        let env = &state.env;

        let mut input_bufs: Vec<TensorBuffer> = Vec::new();
        for (_name, shape, dt) in input_shapes {
            let ts = TensorShape {
                element_type: our_dt_to_litert(*dt),
                dims: shape.iter().map(|&d| d as i32).collect(),
            };
            let buf =
                TensorBuffer::managed_host(env, &ts).map_err(|e| format!("input buf: {}", e))?;
            input_bufs.push(buf);
        }

        let input_map: std::collections::HashMap<&str, &[u8]> =
            inputs.iter().map(|(n, d)| (*n, *d)).collect();
        for (i, (name, shape, dt)) in input_shapes.iter().enumerate() {
            let data = input_map
                .get(name.as_str())
                .ok_or_else(|| format!("missing input '{}'", name))?;
            let data_to_write: std::borrow::Cow<[u8]> =
                if nhwc_inputs.contains(name.as_str()) && shape.len() >= 4 {
                    let elem_size = dt.element_byte_size();
                    std::borrow::Cow::Owned(nchw_to_nhwc(data, elem_size, shape))
                } else {
                    std::borrow::Cow::Borrowed(data)
                };
            log::error!(
                "LiteRT input '{}' shape={:?} dt={:?} len={}",
                name,
                shape,
                dt,
                data_to_write.len()
            );
            if *dt == DataType::Float32 {
                let fv: Vec<f32> = data_to_write
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                log::error!("LiteRT input '{}' f32 values: {:?}", name, fv);
            }
            write_typed_input(&mut input_bufs[i], &data_to_write, *dt)?;
        }

        // Allocate output buffers using the *actual* (post-delegate) layouts,
        // which may be larger than the model-declared shapes due to padding
        // for SIMD alignment (e.g. XNNPACK pads spatial dims to multiples of
        // a tile size). Allocating to the model-declared size makes LiteRT
        // reject the buffer with "Custom allocation is too small".
        let actual_layouts = &state.output_layouts;
        let mut output_bufs: Vec<TensorBuffer> = Vec::new();
        for (i, (_name, _shape, dt)) in output_shapes.iter().enumerate() {
            let dims = actual_layouts
                .get(i)
                .map(|l| l.clone())
                .unwrap_or_else(|| _shape.iter().map(|&d| d as i32).collect());
            let ts = TensorShape {
                element_type: our_dt_to_litert(*dt),
                dims,
            };
            let buf =
                TensorBuffer::managed_host(env, &ts).map_err(|e| format!("output buf: {}", e))?;
            output_bufs.push(buf);
        }

        log::error!(
            "LiteRT run: {} input bufs, {} output bufs",
            input_bufs.len(),
            output_bufs.len()
        );
        state
            .compiled
            .run(&mut input_bufs, &mut output_bufs)
            .map_err(|e| format!("run: {}", e))?;
        log::error!("LiteRT run completed successfully");

        let mut outputs: Vec<Vec<u8>> = Vec::new();
        for (i, buf) in output_bufs.iter().enumerate() {
            let dt = output_shapes[i].2;
            let shape = &output_shapes[i].1;
            let actual_layout = actual_layouts.get(i).cloned().unwrap_or_default();
            let bytes = read_typed_output(buf, dt)?;
            // The buffer may be larger than the user-requested output shape
            // (padded by the delegate). Extract only the elements within
            // the declared shape range using the actual layout strides.
            let extracted = if actual_layout.len() >= 4 && shape.len() >= 4 {
                let elem_size = dt.element_byte_size();
                extract_output_region(&bytes, elem_size, shape, &actual_layout)
            } else {
                let expected_bytes =
                    shape.iter().map(|&d| d as usize).product::<usize>() * dt.element_byte_size();
                if bytes.len() > expected_bytes {
                    bytes[..expected_bytes].to_vec()
                } else {
                    bytes
                }
            };
            let converted = if nhwc_outputs.contains(&output_shapes[i].0) && shape.len() >= 4 {
                let elem_size = dt.element_byte_size();
                nhwc_to_nchw(&extracted, elem_size, shape)
            } else {
                extracted
            };
            log::error!("LiteRT output {} ({} bytes)", i, converted.len());
            if dt == DataType::Float32 {
                let vals: Vec<f32> = converted
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                log::error!("LiteRT output {} f32 values: {:?}", i, vals);
            }
            outputs.push(converted);
        }

        Ok(RunResult { outputs })
    }
}
