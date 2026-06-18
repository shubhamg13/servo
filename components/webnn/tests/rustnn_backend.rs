/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#![cfg(feature = "rustnn")]

use webnn::{
    BackendDeviceType, BackendOptions, BackendPowerPreference, Conv2dOptions, Operator,
    Pool2dOptions,
};

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
        device_type: BackendDeviceType::Cpu,
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

    // Device tensors, keyed by Servo tensor id.
    backend.create_tensor(10, 0, &[2]).expect("create a");
    backend.create_tensor(11, 0, &[2]).expect("create b");
    backend.create_tensor(12, 0, &[2]).expect("create sum");
    backend
        .write_tensor(10, &f32_bytes(&[1.0, 2.0]))
        .expect("write a");
    backend
        .write_tensor(11, &f32_bytes(&[3.0, 4.0]))
        .expect("write b");

    backend
        .run(
            graph_id,
            &[("a".to_string(), 10), ("b".to_string(), 11)],
            &[("sum".to_string(), 12)],
        )
        .expect("run");

    let out = backend.read_tensor(12).expect("read sum");
    assert_eq!(bytes_to_f32(&out), vec![4.0, 6.0]);
}

/// Builds and runs a `y = averagePool2d(x)` graph through the rustnn backend
/// (mirrors the SqueezeNet classification graph's global pooling).
#[test]
fn rustnn_backend_runs_average_pool2d() {
    let Some(backend) = backend() else { return };
    let b = backend.create_builder();
    // NCHW input [1, 1, 4, 4].
    backend.add_input(b, 1, "x", 0, &[1, 1, 4, 4]);
    backend.add_operator(
        b,
        &Operator::AveragePool2d(Pool2dOptions {
            window_dimensions: vec![2, 2],
            padding: vec![0, 0, 0, 0],
            strides: vec![2, 2],
            dilations: vec![1, 1],
            layout: "nchw".to_string(),
            output_shape_rounding: "floor".to_string(),
            output_sizes: None,
        }),
        &[1],
        &[(2, 0, vec![1, 1, 2, 2])],
        "",
    );
    let g = backend.build(b, &[("y".to_string(), 2)]).expect("build");

    backend
        .create_tensor(10, 0, &[1, 1, 4, 4])
        .expect("create x");
    backend
        .create_tensor(11, 0, &[1, 1, 2, 2])
        .expect("create y");
    let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    backend
        .write_tensor(10, &f32_bytes(&input))
        .expect("write x");

    backend
        .run(g, &[("x".to_string(), 10)], &[("y".to_string(), 11)])
        .expect("run");

    let out = bytes_to_f32(&backend.read_tensor(11).expect("read y"));
    // 2x2 average pool with stride 2 over 1..=16 (row major) gives block means.
    assert_eq!(out, vec![3.5, 5.5, 11.5, 13.5]);
}

fn backend() -> Option<Box<dyn webnn::Backend>> {
    let backend = webnn::create_backend(&BackendOptions {
        power_preference: BackendPowerPreference::Default,
        accelerated: true,
        device_type: BackendDeviceType::Cpu,
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

    backend.create_tensor(10, 0, &[2]).expect("create a");
    backend.create_tensor(11, 0, &[2]).expect("create b");
    backend.create_tensor(12, 0, &[4]).expect("create o");
    backend
        .write_tensor(10, &f32_bytes(&[1.0, 2.0]))
        .expect("write a");
    backend
        .write_tensor(11, &f32_bytes(&[3.0, 4.0]))
        .expect("write b");

    backend
        .run(
            g,
            &[("a".to_string(), 10), ("b".to_string(), 11)],
            &[("o".to_string(), 12)],
        )
        .expect("run");

    let out = backend.read_tensor(12).expect("read o");
    assert_eq!(bytes_to_f32(&out), vec![1.0, 2.0, 3.0, 4.0]);
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

    backend
        .create_tensor(10, 0, &[1, 1, 3, 3])
        .expect("create x");
    backend
        .create_tensor(11, 0, &[1, 1, 2, 2])
        .expect("create o");
    backend
        .write_tensor(
            10,
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        )
        .expect("write x");

    backend
        .run(g, &[("x".to_string(), 10)], &[("o".to_string(), 11)])
        .expect("run");

    let out = backend.read_tensor(11).expect("read o");
    assert_eq!(bytes_to_f32(&out), vec![6.0, 8.0, 12.0, 14.0]);
}

/// Exercises the on-device tensor handoff: `y = add(a, b)` is produced by one
/// graph and consumed by another (`z = add(y, c)`) *without* ever reading `y`
/// back to host. This is the Phase 2B device-chain path.
#[test]
fn rustnn_backend_device_handoff() {
    let Some(backend) = backend() else { return };

    // Graph 1: y = add(a, b).
    let b1 = backend.create_builder();
    backend.add_input(b1, 1, "a", 0, &[2]);
    backend.add_input(b1, 2, "b", 0, &[2]);
    backend.add_operator(b1, &Operator::Add, &[1, 2], &[(3, 0, vec![2])], "");
    let g1 = backend
        .build(b1, &[("y".to_string(), 3)])
        .expect("build g1");

    // Graph 2: z = add(y, c) — `y` is a graph-2 input whose tensor id is the
    // same as graph-1's output.
    let b2 = backend.create_builder();
    backend.add_input(b2, 1, "y", 0, &[2]);
    backend.add_input(b2, 2, "c", 0, &[2]);
    backend.add_operator(b2, &Operator::Add, &[1, 2], &[(3, 0, vec![2])], "");
    let g2 = backend
        .build(b2, &[("z".to_string(), 3)])
        .expect("build g2");

    // Device tensors keyed by Servo id. `12` is the shared intermediate.
    backend.create_tensor(10, 0, &[2]).expect("create a");
    backend.create_tensor(11, 0, &[2]).expect("create b");
    backend.create_tensor(12, 0, &[2]).expect("create y");
    backend.create_tensor(13, 0, &[2]).expect("create c");
    backend.create_tensor(14, 0, &[2]).expect("create z");

    backend
        .write_tensor(10, &f32_bytes(&[1.0, 2.0]))
        .expect("write a");
    backend
        .write_tensor(11, &f32_bytes(&[3.0, 4.0]))
        .expect("write b");
    backend
        .write_tensor(13, &f32_bytes(&[10.0, 20.0]))
        .expect("write c");

    // Run graph 1; `y` (12) is written on device, never read.
    backend
        .run(
            g1,
            &[("a".to_string(), 10), ("b".to_string(), 11)],
            &[("y".to_string(), 12)],
        )
        .expect("run g1");

    // Run graph 2 consuming `y` on device.
    backend
        .run(
            g2,
            &[("y".to_string(), 12), ("c".to_string(), 13)],
            &[("z".to_string(), 14)],
        )
        .expect("run g2");

    let z = backend.read_tensor(14).expect("read z");
    // z = (a + b) + c = (1+3, 2+4) + (10, 20) = (14, 26).
    assert_eq!(bytes_to_f32(&z), vec![14.0, 26.0]);
}

/// A tensor created but never dispatched reads back as zeros.
#[test]
fn rustnn_backend_read_never_dispatched_returns_zeros() {
    let Some(backend) = backend() else { return };
    backend.create_tensor(20, 0, &[2]).expect("create");
    let out = backend.read_tensor(20).expect("read");
    assert_eq!(bytes_to_f32(&out), vec![0.0, 0.0]);
}

/// Destroying a tensor makes subsequent reads fail.
#[test]
fn rustnn_backend_destroy_tensor() {
    let Some(backend) = backend() else { return };
    backend.create_tensor(30, 0, &[2]).expect("create");
    backend.destroy_tensor(30);
    assert!(backend.read_tensor(30).is_err());
}
