/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use webnn::{Backend, MockBackend, Operator};

/// Exercises the mock backend's tensor lifecycle (create/write/run/read/destroy)
/// against the id-based `Backend` trait.
#[test]
fn mock_tensor_lifecycle_and_run() {
    let backend = MockBackend::new();

    let b = backend.create_builder();
    backend.add_input(b, 1, "a", 0, &[2]);
    backend.add_operator(b, &Operator::Add, &[1], &[(2, 0, vec![2])], "");
    let g = backend.build(b, &[("o".to_string(), 2)]).expect("build");

    backend.create_tensor(1, 0, &[2]).expect("create a");
    backend.create_tensor(2, 0, &[2]).expect("create o");
    backend.write_tensor(1, &[1, 2, 3, 4]).expect("write a");

    // The mock `run` echoes the first input into every output.
    backend
        .run(g, &[("a".to_string(), 1)], &[("o".to_string(), 2)])
        .expect("run");
    let out = backend.read_tensor(2).expect("read o");
    assert_eq!(out, vec![1, 2, 3, 4]);

    // Destroying a tensor makes subsequent reads fail.
    backend.destroy_tensor(2);
    assert!(backend.read_tensor(2).is_err());
}

/// A tensor that was created but never dispatched reads back empty in the mock
/// (the mock does not pre-zero tensor storage).
#[test]
fn mock_never_dispatched_read_is_empty() {
    let backend = MockBackend::new();
    backend.create_tensor(7, 0, &[2]).expect("create");
    assert_eq!(backend.read_tensor(7).expect("read"), Vec::<u8>::new());
}
