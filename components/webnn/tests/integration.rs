use webnn::{GraphNode, compile, run};

fn node(op: &str, inputs: Vec<String>, output: &str, shape: Vec<u32>) -> GraphNode {
    GraphNode { op: op.to_string(), inputs, output: output.to_string(),
        data_type: 0, shape, attrs: Default::default(), data: None }
}

#[test]
fn test_add_operator() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a = vec![1.0f32,2.,3.,4.].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let b = vec![5.0f32,6.,7.,8.].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let nodes = vec![node("add", vec!["a".into(), "b".into()], "c", vec![4])];
    let names = vec!["c".to_string()];
    let gid = compile(&nodes, &names).expect("compile");
    let r = run(gid, &[("a", a.as_slice()), ("b", b.as_slice())], &names).expect("run");
    let out: Vec<f32> = r.outputs[0].chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    assert!((out[0]-6.0).abs()<0.001);
}

#[test]
fn test_mul_operator() {
    let _ = env_logger::builder().is_test(true).try_init();
    let a = vec![2.0f32,3.,4.].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let b = vec![5.0f32,6.,7.].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let nodes = vec![node("mul", vec!["a".into(), "b".into()], "c", vec![3])];
    let names = vec!["c".to_string()];
    let gid = compile(&nodes, &names).expect("compile");
    let r = run(gid, &[("a", a.as_slice()), ("b", b.as_slice())], &names).expect("run");
    let out: Vec<f32> = r.outputs[0].chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    assert!((out[0]-10.0).abs()<0.001);
}
