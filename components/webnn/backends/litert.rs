//! LiteRT backend — uses litert v0.2.1 for inference.

use litert::{
    Accelerators, CompilationOptions, CompiledModel, ElementType, Environment, Model, TensorBuffer,
    TensorShape,
};

use crate::backend::{Backend, CompiledModel as WebNNModel, DataType, GraphNode, RunResult};
use crate::compiler;

pub(crate) struct LiteRtBackend;

struct LiteRtState {
    compiled: CompiledModel,
    env: Environment,
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
        let flatbuf = compiler::compile_with_input_infos(nodes, input_infos)?;

        let mut input_shapes: Vec<(String, Vec<u32>, DataType)> = input_infos.to_vec();
        input_shapes.sort_by(|a, b| a.0.cmp(&b.0));

        let output_shapes: Vec<(String, Vec<u32>, DataType)> = nodes
            .iter()
            .filter(|n| n.op != "constant")
            .map(|n| (n.output.clone(), n.desc.shape.clone(), n.desc.data_type))
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

        let run_env = Environment::new().map_err(|e| format!("LiteRT run env: {}", e))?;
        let state = Box::new(LiteRtState {
            compiled,
            env: run_env,
        });

        Ok(WebNNModel::LiteRt {
            compiled: state,
            input_shapes,
            output_shapes,
        })
    }

    fn run(&self, model: &WebNNModel, inputs: &[(&str, &[u8])]) -> Result<RunResult, String> {
        let (state, input_shapes, output_shapes) = match model {
            WebNNModel::LiteRt {
                compiled,
                input_shapes,
                output_shapes,
            } => (compiled, input_shapes, output_shapes),
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
            log::error!(
                "LiteRT input '{}' shape={:?} dt={:?} len={}",
                name,
                shape,
                dt,
                data.len()
            );
            if *dt == DataType::Float32 {
                let fv: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                log::error!("LiteRT input '{}' f32 values: {:?}", name, fv);
            }
            write_typed_input(&mut input_bufs[i], data, *dt)?;
        }

        let mut output_bufs: Vec<TensorBuffer> = Vec::new();
        for (_name, shape, dt) in output_shapes {
            let ts = TensorShape {
                element_type: our_dt_to_litert(*dt),
                dims: shape.iter().map(|&d| d as i32).collect(),
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
            let bytes = read_typed_output(buf, dt)?;
            log::error!("LiteRT output {} ({} bytes)", i, bytes.len());
            if dt == DataType::Float32 {
                let vals: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                log::error!("LiteRT output {} f32 values: {:?}", i, vals);
            }
            outputs.push(bytes);
        }

        Ok(RunResult { outputs })
    }
}
