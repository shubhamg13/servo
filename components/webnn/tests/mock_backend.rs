use webnn::{GraphNode, MockBackend, set_backend, compile, run};

#[test]
fn test_mock_backend() {
    set_backend(Box::new(MockBackend));
    let a = vec![1.0f32, 2.0f32].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let nodes = vec![GraphNode {
        op: "add".into(), inputs: vec!["a".into(), "b".into()], output: "c".into(),
        data_type: 0, shape: vec![2], attrs: Default::default(), data: None,
    }];
    let names = vec!["c".to_string()];
    let gid = compile(&nodes, &names).expect("compile");
    assert_eq!(gid, 0);
    let r = run(gid, &[("a", a.as_slice()), ("b", &[])], &names).expect("run");
    assert_eq!(r.outputs.len(), 1);
    assert_eq!(r.outputs[0], a); // mock copies first input
}
