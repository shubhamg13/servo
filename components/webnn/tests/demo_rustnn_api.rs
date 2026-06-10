use rustnn::mlcontext::{MLContext, MLContextOptions, MLPowerPreference, MLOperandDescriptor, MLTensorDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use std::collections::HashMap;

#[test]
fn demo_rustnn_api() {
    let _ = env_logger::builder().is_test(true).try_init();

    let opts = MLContextOptions::new(MLPowerPreference::Default, true);
    let mut ctx = match MLContext::create(&opts) {
        Ok(c) => c,
        Err(_) => { eprintln!("ORT not available, skipping test"); return; }
    };

    let mut builder = MLGraphBuilder::new(&mut ctx).expect("create builder");

    let a = builder.input("a", &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4u64]))
        .expect("input a");
    let b = builder.input("b", &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4u64]))
        .expect("input b");
    let c = builder.add(a, b).expect("add");

    let mut outputs = HashMap::new();
    outputs.insert("c", c);
    let mut graph = builder.build(&outputs).expect("build");

    let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4u64]);
    desc.set_readable(true);
    desc.set_writable(true);

    let a_tensor = ctx.create_tensor(&desc).expect("create a_tensor");
    let b_tensor = ctx.create_tensor(&desc).expect("create b_tensor");
    let c_tensor = ctx.create_tensor(&desc).expect("create c_tensor");

    ctx.write_tensor(&a_tensor, &[1.0f32, 2.0, 3.0, 4.0]).expect("write a");
    ctx.write_tensor(&b_tensor, &[5.0f32, 6.0, 7.0, 8.0]).expect("write b");

    let mut inputs = HashMap::new();
    inputs.insert("a", &a_tensor);
    inputs.insert("b", &b_tensor);
    let mut dispatch_outputs = HashMap::new();
    dispatch_outputs.insert("c", &c_tensor);

    ctx.dispatch(&mut graph, &inputs, &dispatch_outputs).expect("dispatch");

    let mut result = vec![0.0f32; 4];
    ctx.read_tensor(&c_tensor, &mut result).expect("read c");
    println!("result: {:?}", result);
    assert!((result[0] - 6.0).abs() < 0.001);
    assert!((result[1] - 8.0).abs() < 0.001);
    assert!((result[2] - 10.0).abs() < 0.001);
    assert!((result[3] - 12.0).abs() < 0.001);
}
