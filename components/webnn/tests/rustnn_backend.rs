/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#![cfg(feature = "rustnn")]

use webnn::{BackendOptions, BackendPowerPreference, Conv2dOptions, Operator};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Builds and runs a simple `sum = add(a, b)` graph through the rustnn
/// backend. Requires ONNX Runtime to be discoverable at runtime (see
/// `ORT_DYLIB_PATH`); when it is not, `create_backend` falls back to the mock
/// backend and this test skips.
#[test]
fn rustnn_backend_runs_add_graph() {
    let backend = webnn::create_backend(&BackendOptions {
        power_preference: BackendPowerPreference::Default,
        accelerated: true,
    });
    if backend.name() != "rustnn" {
        eprintln!(
            "skipping: rustnn backend not available (got '{}'); is libonnxruntime.so discoverable?",
            backend.name()
        );
        return;
    }

    let builder_id = backend.create_builder();
    // float32 (WebIDL `MLOperandDataType` discriminant 0).
    backend.add_input(builder_id, 1, "a", 0, &[2]);
    backend.add_input(builder_id, 2, "b", 0, &[2]);
    backend.add_operator(
        builder_id,
        &Operator::Add,
        &[1, 2],
        &[(3, 0, vec![2])],
        "add",
    );
    let graph_id = backend
        .build(builder_id, &[("sum".to_string(), 3)])
        .expect("build");

    let inputs = [
        ("a".to_string(), f32_bytes(&[1.0, 2.0])),
        ("b".to_string(), f32_bytes(&[3.0, 4.0])),
    ];
    let input_refs: Vec<(String, &[u8])> = inputs
        .iter()
        .map(|(name, data)| (name.clone(), data.as_slice()))
        .collect();

    let result = backend
        .run(graph_id, &input_refs, &["sum".to_string()])
        .expect("run");

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(bytes_to_f32(&result.outputs[0]), vec![4.0, 6.0]);
}

fn backend() -> Option<Box<dyn webnn::Backend>> {
    let backend = webnn::create_backend(&BackendOptions {
        power_preference: BackendPowerPreference::Default,
        accelerated: true,
    });
    if backend.name() != "rustnn" {
        eprintln!(
            "skipping: rustnn backend not available (got '{}')",
            backend.name()
        );
        return None;
    }
    Some(backend)
}

#[test]
fn rustnn_backend_runs_concat() {
    let Some(backend) = backend() else { return };
    let b = backend.create_builder();
    backend.add_input(b, 1, "a", 0, &[2]);
    backend.add_input(b, 2, "b", 0, &[2]);
    backend.add_operator(
        b,
        &Operator::Concat { axis: 0 },
        &[1, 2],
        &[(3, 0, vec![4])],
        "",
    );
    let g = backend.build(b, &[("o".to_string(), 3)]).expect("build");
    let inputs = [
        ("a".to_string(), f32_bytes(&[1.0, 2.0])),
        ("b".to_string(), f32_bytes(&[3.0, 4.0])),
    ];
    let refs: Vec<(String, &[u8])> = inputs
        .iter()
        .map(|(n, d)| (n.clone(), d.as_slice()))
        .collect();
    let r = backend.run(g, &refs, &["o".to_string()]).expect("run");
    assert_eq!(bytes_to_f32(&r.outputs[0]), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rustnn_backend_runs_conv2d() {
    let Some(backend) = backend() else { return };
    let b = backend.create_builder();
    backend.add_input(b, 1, "x", 0, &[1, 1, 3, 3]);
    backend.add_constant(b, 2, 0, &[1, 1, 2, 2], &f32_bytes(&[1.0, 0.0, 0.0, 1.0]));
    backend.add_operator(
        b,
        &Operator::Conv2d(Conv2dOptions {
            padding: vec![0, 0, 0, 0],
            strides: vec![1, 1],
            dilations: vec![1, 1],
            groups: 1,
            input_layout: "nchw".to_string(),
            filter_layout: "oihw".to_string(),
            bias: None,
        }),
        &[1, 2],
        &[(3, 0, vec![1, 1, 2, 2])],
        "",
    );
    let g = backend.build(b, &[("o".to_string(), 3)]).expect("build");
    let inputs = [(
        "x".to_string(),
        f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    )];
    let refs: Vec<(String, &[u8])> = inputs
        .iter()
        .map(|(n, d)| (n.clone(), d.as_slice()))
        .collect();
    let r = backend.run(g, &refs, &["o".to_string()]).expect("run");
    assert_eq!(bytes_to_f32(&r.outputs[0]), vec![6.0, 8.0, 12.0, 14.0]);
}
