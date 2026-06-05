//! Integration tests for the LiteRT flatbuffer model path.
//! These use the public crate API (`webnn::compiler::compile`, `webnn::litert::initialize`)
//! plus direct TFLite flatbuffer construction via the `flatbuffers` crate.

#![cfg(feature = "litert")]

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, WIPOffset};
use webnn::backend::{DataType, GraphNode, TensorDesc};

// ── TFLite schema field offsets ──
//
// OperatorCode: deprecated_builtin_code=0(byte), version=4(int),
//               builtin_code=6(int32)
// Tensor:       shape=0([int]), type=2(byte), buffer=4(uint), name=6(string),
//               quantization=8, is_variable=10(bool), sparsity=12,
//               shape_signature=14, has_rank=16(bool)
// Buffer:       data=0([byte])
// Operator:     opcode_index=0(uint), inputs=1([int]), outputs=2([int]),
//               builtin_options_type=3(u8), builtin_options=4(uoffset)
// SubGraph:     tensors=0([Tensor]), inputs=2([int]), outputs=4([int]),
//               operators=6([Operator]), name=8(string)
// Model:        version=0(uint), operator_codes=2([OperatorCode]),
//               subgraphs=4([SubGraph]), description=6(string),
//               buffers=8([Buffer])

fn build_operator_code(
    fbb: &mut FlatBufferBuilder,
    builtin_code: i32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let dep = std::cmp::min(builtin_code, 127);
    let t = fbb.start_table();
    fbb.push_slot::<u8>(4, dep as u8, 0);
    fbb.push_slot::<i32>(8, 1, 1);
    fbb.push_slot::<i32>(10, builtin_code, 0);
    fbb.end_table(t)
}

fn build_tensor(
    fbb: &mut FlatBufferBuilder,
    shape: &[u32],
    dtype: u8,
    buffer: u32,
    name: &str,
) -> WIPOffset<TableFinishedWIPOffset> {
    let shape_off = fbb.create_vector(&shape.iter().map(|&d| d as i32).collect::<Vec<_>>());
    let name_off = fbb.create_string(name);
    let t = fbb.start_table();
    fbb.push_slot_always(4, shape_off);
    fbb.push_slot::<u8>(6, dtype, 0);
    fbb.push_slot::<u32>(8, buffer, 0);
    fbb.push_slot_always(10, name_off);
    fbb.push_slot::<bool>(14, false, false);
    fbb.push_slot_always(20, true);
    fbb.end_table(t)
}

fn build_buffer(fbb: &mut FlatBufferBuilder, data: &[u8]) -> WIPOffset<TableFinishedWIPOffset> {
    let t = fbb.start_table();
    if !data.is_empty() {
        let data_off = fbb.create_vector(data);
        fbb.push_slot_always(4, data_off);
    }
    fbb.end_table(t)
}

fn build_operator(
    fbb: &mut FlatBufferBuilder,
    inputs: &[u32],
    outputs: &[u32],
    opcode_index: u32,
) -> WIPOffset<TableFinishedWIPOffset> {
    let inputs_off = fbb.create_vector(&inputs.iter().map(|&i| i as i32).collect::<Vec<_>>());
    let outputs_off = fbb.create_vector(&outputs.iter().map(|&o| o as i32).collect::<Vec<_>>());
    let t = fbb.start_table();
    fbb.push_slot_always(4, opcode_index);
    fbb.push_slot_always(6, inputs_off);
    fbb.push_slot_always(8, outputs_off);
    fbb.end_table(t)
}

fn build_subgraph(
    fbb: &mut FlatBufferBuilder,
    tensors: &[WIPOffset<TableFinishedWIPOffset>],
    inputs: &[u32],
    outputs: &[u32],
    operators: &[WIPOffset<TableFinishedWIPOffset>],
    name: &str,
) -> WIPOffset<TableFinishedWIPOffset> {
    let tensors_off = fbb.create_vector(tensors);
    let inputs_off = fbb.create_vector(&inputs.iter().map(|&i| i as i32).collect::<Vec<_>>());
    let outputs_off = fbb.create_vector(&outputs.iter().map(|&o| o as i32).collect::<Vec<_>>());
    let operators_off = fbb.create_vector(operators);
    let name_off = fbb.create_string(name);
    let t = fbb.start_table();
    fbb.push_slot_always(4, tensors_off);
    fbb.push_slot_always(6, inputs_off);
    fbb.push_slot_always(8, outputs_off);
    fbb.push_slot_always(10, operators_off);
    fbb.push_slot_always(12, name_off);
    fbb.end_table(t)
}

fn build_model(
    fbb: &mut FlatBufferBuilder,
    operator_codes: &[WIPOffset<TableFinishedWIPOffset>],
    subgraphs: &[WIPOffset<TableFinishedWIPOffset>],
    buffers: &[WIPOffset<TableFinishedWIPOffset>],
    description: &str,
) -> WIPOffset<TableFinishedWIPOffset> {
    let codes_off = fbb.create_vector(operator_codes);
    let subgraphs_off = fbb.create_vector(subgraphs);
    let buffers_off = fbb.create_vector(buffers);
    let desc_off = fbb.create_string(description);
    let t = fbb.start_table();
    fbb.push_slot::<u32>(4, 3, 0);
    fbb.push_slot_always(6, codes_off);
    fbb.push_slot_always(8, subgraphs_off);
    fbb.push_slot_always(10, desc_off);
    fbb.push_slot_always(12, buffers_off);
    fbb.end_table(t)
}

fn create_minimal_model() -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();
    let dtype: u8 = 0;
    let shape: Vec<u32> = vec![5];
    let name = "input";

    let tensor = build_tensor(&mut fbb, &shape, dtype, 0, name);
    let op_code = build_operator_code(&mut fbb, 101);
    let op = build_operator(&mut fbb, &[0], &[0], 0);
    let buf = build_buffer(&mut fbb, &[]);
    let subgraph = build_subgraph(&mut fbb, &[tensor], &[0], &[0], &[op], "main");
    let model = build_model(&mut fbb, &[op_code], &[subgraph], &[buf], "test");
    fbb.finish(model, Some("TFL3"));
    fbb.finished_data().to_vec()
}

// ── Test helpers ──

fn test_model(name: &str, data: Vec<u8>) {
    let out_path = std::env::temp_dir().join(format!("{}.tflite", name));
    std::fs::write(&out_path, &data).unwrap();
    let result = litert::Model::from_bytes(data);
    if let Err(ref e) = result {
        println!("FAIL {}: {}", name, e);
    } else {
        println!("OK   {}", name);
    }
}

fn make_node(
    op: &str,
    inputs: Vec<&str>,
    output: &str,
    shape: Vec<u32>,
    attrs: Vec<(&str, f64)>,
) -> GraphNode {
    GraphNode {
        op: op.to_string(),
        inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
        output: output.to_string(),
        desc: TensorDesc {
            data_type: DataType::Float32,
            shape,
        },
        attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        data: None,
    }
}

fn compile_and_check(name: &str, nodes: Vec<GraphNode>) {
    let result = webnn::compiler::compile(&nodes);
    assert!(
        result.is_ok(),
        "compile({}) failed: {:?}",
        name,
        result.err()
    );
    let data = result.unwrap();
    test_model(name, data);
}

fn build_model_with_fields(add_field6: bool, _add_field7: bool) -> Vec<u8> {
    let mut fbb = flatbuffers::FlatBufferBuilder::new();
    let shape = fbb.create_vector(&[5i32]);
    let name = fbb.create_string("x");
    let tensor = {
        let t = fbb.start_table();
        fbb.push_slot_always(4, shape);
        fbb.push_slot::<u8>(6, 0u8, 0);
        fbb.push_slot::<u32>(8, 0u32, 0);
        fbb.push_slot_always(10, name);
        fbb.push_slot_always(20, true);
        fbb.end_table(t)
    };
    let inputs = fbb.create_vector(&[0i32]);
    let outputs = fbb.create_vector(&[0i32]);
    let op = {
        let t = fbb.start_table();
        fbb.push_slot_always(4, 0u32);
        fbb.push_slot_always(6, inputs);
        fbb.push_slot_always(8, outputs);
        fbb.end_table(t)
    };
    let opcode = {
        let t = fbb.start_table();
        fbb.push_slot_always(4, 101u8.min(127) as u8);
        fbb.push_slot_always(8, 1i32);
        fbb.push_slot_always(10, 101i32);
        fbb.end_table(t)
    };
    let buf = {
        let t = fbb.start_table();
        fbb.end_table(t)
    };
    let sg_name = fbb.create_string("main");
    let sg_inputs = fbb.create_vector(&[0i32]);
    let sg_outputs = fbb.create_vector(&[0i32]);
    let subgraph = {
        let tensors = fbb.create_vector(&[tensor]);
        let ops = fbb.create_vector(&[op]);
        let t = fbb.start_table();
        fbb.push_slot_always(4, tensors);
        fbb.push_slot_always(6, sg_inputs);
        fbb.push_slot_always(8, sg_outputs);
        fbb.push_slot_always(10, ops);
        fbb.push_slot_always(12, sg_name);
        fbb.end_table(t)
    };
    let desc = fbb.create_string("test");
    let oc = fbb.create_vector(&[opcode]);
    let sg = fbb.create_vector(&[subgraph]);
    let bufs = fbb.create_vector(&[buf]);

    let empty_vec = fbb.create_vector::<i32>(&[]);
    let empty_str = fbb.create_string("");

    let model = {
        let t = fbb.start_table();
        fbb.push_slot_always(4, 3u32);
        fbb.push_slot_always(6, oc);
        fbb.push_slot_always(8, sg);
        fbb.push_slot_always(10, desc);
        fbb.push_slot_always(12, bufs);
        if add_field6 || _add_field7 {
            fbb.push_slot_always(14, empty_vec);
        }
        fbb.push_slot_always(16, empty_vec);
        fbb.push_slot_always(18, empty_str);
        fbb.end_table(t)
    };
    fbb.finish(model, Some("TFL3"));
    fbb.finished_data().to_vec()
}

// ── Tests ──

#[test]
fn test_good_model() {
    webnn::litert::initialize().unwrap();
    let alt_path = std::path::Path::new(
        "/home/shubham/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/litert-0.2.1/tests/data/add_10x10.tflite",
    );
    let data = std::fs::read(&alt_path).expect("read test model");
    test_model("good", data);
}

#[test]
fn test_model_fields() {
    webnn::litert::initialize().unwrap();
    test_model("base_0_4", build_model_with_fields(false, false));
    test_model("plus_f6", build_model_with_fields(false, true));
}

#[test]
fn test_production_code() {
    webnn::litert::initialize().unwrap();
    let flatbuf = create_minimal_model();
    let out_path = std::env::temp_dir().join("production_model.tflite");
    std::fs::write(&out_path, &flatbuf).unwrap();
    println!(
        "Production model written to {:?} ({} bytes)",
        out_path,
        flatbuf.len()
    );
    test_model("production", flatbuf);
}

#[test]
fn test_compile_abs() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "abs",
        vec![make_node("abs", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_add() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "add",
        vec![make_node("add", vec!["a", "b"], "c", vec![4], vec![])],
    );
}

#[test]
fn test_compile_mul() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "mul",
        vec![make_node("mul", vec!["a", "b"], "c", vec![4], vec![])],
    );
}

#[test]
fn test_compile_relu() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "relu",
        vec![make_node("relu", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_softmax() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "softmax",
        vec![make_node("softmax", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_reshape() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "reshape",
        vec![make_node("reshape", vec!["x"], "y", vec![2, 2], vec![])],
    );
}

#[test]
fn test_compile_transpose() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "transpose",
        vec![make_node("transpose", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_concat() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "concat",
        vec![make_node("concat", vec!["a", "b"], "c", vec![8], vec![])],
    );
}

#[test]
fn test_compile_exp() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "exp",
        vec![make_node("exp", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_log() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "log",
        vec![make_node("log", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_sqrt() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "sqrt",
        vec![make_node("sqrt", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_neg() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "neg",
        vec![make_node("neg", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_sigmoid() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "sigmoid",
        vec![make_node("sigmoid", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_max() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "max",
        vec![make_node("max", vec!["a", "b"], "c", vec![4], vec![])],
    );
}

#[test]
fn test_compile_pow() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "pow",
        vec![make_node("pow", vec!["a", "b"], "c", vec![4], vec![])],
    );
}

#[test]
fn test_compile_gemm() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "gemm",
        vec![make_node(
            "gemm",
            vec!["a", "b", "c"],
            "d",
            vec![3, 3],
            vec![],
        )],
    );
}

#[test]
fn test_compile_squeeze() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "squeeze",
        vec![make_node("squeeze", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_split() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "split",
        vec![make_node(
            "split",
            vec!["x"],
            "y",
            vec![2],
            vec![("splits", 2.0)],
        )],
    );
}

#[test]
fn test_compile_split_grouped() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "split_grouped",
        vec![
            make_node(
                "split",
                vec!["x"],
                "y0",
                vec![2, 3],
                vec![("splits", 2.0), ("axis", 0.0), ("split_group", 1.0)],
            ),
            make_node(
                "split",
                vec!["x"],
                "y1",
                vec![2, 3],
                vec![("splits", 2.0), ("axis", 0.0), ("split_group", 1.0)],
            ),
        ],
    );
}

#[test]
fn test_compile_leaky_relu() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "leakyRelu",
        vec![make_node(
            "leakyRelu",
            vec!["x"],
            "y",
            vec![4],
            vec![("alpha", 0.01)],
        )],
    );
}

#[test]
fn test_compile_gather() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "gather",
        vec![make_node(
            "gather",
            vec!["x", "indices"],
            "y",
            vec![2],
            vec![("axis", 0.0)],
        )],
    );
}

#[test]
fn test_compile_reduce_mean() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "reduceMean",
        vec![make_node(
            "reduceMean",
            vec!["x", "axes"],
            "y",
            vec![],
            vec![],
        )],
    );
}

#[test]
fn test_compile_arg_max() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "argMax",
        vec![make_node("argMax", vec!["x", "axis"], "y", vec![], vec![])],
    );
}

#[test]
fn test_compile_cumsum() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "cumulativeSum",
        vec![make_node(
            "cumulativeSum",
            vec!["x"],
            "y",
            vec![4],
            vec![("exclusive", 0.0), ("reverse", 0.0)],
        )],
    );
}

#[test]
fn test_compile_cast() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "cast",
        vec![make_node("cast", vec!["x"], "y", vec![4], vec![])],
    );
}

#[test]
fn test_compile_resample2d() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "resample2d",
        vec![make_node(
            "resample2d",
            vec!["x", "sizes"],
            "y",
            vec![1, 3, 3, 1],
            vec![],
        )],
    );
}

#[test]
fn test_compile_average_pool2d() {
    webnn::litert::initialize().unwrap();
    compile_and_check(
        "averagePool2d",
        vec![make_node(
            "averagePool2d",
            vec!["x"],
            "y",
            vec![1, 2, 2, 1],
            vec![],
        )],
    );
}

#[test]
fn test_run_conv2d() {
    use webnn::backends::infer;

    webnn::litert::initialize().unwrap();

    // 1x5x5x1 input convolved with 1x3x3x1 filter (all ones) using VALID
    // padding. The output is 1x3x3x1 of all-9s. This test exercises the
    // path where XNNPACK pads the conv2d output tensor for SIMD alignment,
    // which previously failed with "Custom allocation is too small".
    let mut nodes = vec![
        make_node("constant", vec![], "f", vec![1, 1, 3, 3], vec![]),
        make_node("conv2d", vec!["x", "f"], "y", vec![1, 1, 3, 3], vec![]),
    ];

    // Pre-populate the constant filter with all 1.0s.
    let filter_bytes: Vec<u8> = (0..9).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    nodes[0].data = Some(filter_bytes);

    let input_bytes: Vec<u8> = (0..25).flat_map(|_| 1.0f32.to_le_bytes()).collect();

    let result = infer(
        &nodes,
        &[("x", input_bytes.as_slice())],
        &[("x".to_string(), vec![1, 1, 5, 5], DataType::Float32)],
    )
    .expect("inference should succeed");

    assert_eq!(result.outputs.len(), 1);
    let output = &result.outputs[0];
    assert_eq!(output.len(), 9 * 4, "output should be 9 floats");
    let values: Vec<f32> = output
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for v in values {
        assert!(
            (v - 9.0).abs() < 1e-4,
            "expected each output to be 9.0 (sum of 3x3 ones filter), got {}",
            v
        );
    }
}

#[test]
fn test_run_max_pool2d() {
    use webnn::backends::infer;

    webnn::litert::initialize().unwrap();

    // 1x1x4x4 input, maxPool2d with 2x2 window, stride 2, VALID padding.
    // Input values are [1..16] in NCHW row-major (channel=1, H=4, W=4).
    // With 2x2 max-pooling and stride 2, output is 2x2:
    //   (0,0): max(1, 2,  5,  6)  =  6
    //   (0,1): max(3, 4,  7,  8)  =  8
    //   (1,0): max(9,10, 13, 14)  = 14
    //   (1,1): max(11,12,15,16)  = 16
    let input_f32: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let input_bytes: Vec<u8> = input_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

    let nodes = vec![make_node(
        "maxPool2d",
        vec!["x"],
        "y",
        vec![1, 1, 2, 2],
        vec![
            ("window_h", 2.0),
            ("window_w", 2.0),
            ("stride_h", 2.0),
            ("stride_w", 2.0),
        ],
    )];

    let result = infer(
        &nodes,
        &[("x", input_bytes.as_slice())],
        &[("x".to_string(), vec![1, 1, 4, 4], DataType::Float32)],
    )
    .expect("maxPool2d inference should succeed");

    assert_eq!(result.outputs.len(), 1);
    let output = &result.outputs[0];
    assert_eq!(output.len(), 4 * 4, "output should be 4 floats");
    let values: Vec<f32> = output
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected = [6.0, 8.0, 14.0, 16.0];
    for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "output[{}] expected {}, got {}",
            i,
            exp,
            got
        );
    }
}
