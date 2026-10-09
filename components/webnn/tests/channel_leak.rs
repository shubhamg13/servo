/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Temporary verification for the Phase-2B no-leak instrumentation.
//!
//! Exercises the WebNN channel end-to-end using the exact paths the
//! `DroppableMLTensor`/`DroppableMLGraph`/`DroppableMLContext` GC finalizers
//! call (`destroy_tensor` / `destroy_graph` / `destroy_context`), and asserts
//! the `[webnn-leak]` counters return to baseline.
//!
//! Remove this file together with the `[webnn-leak]` counters in `lib.rs` and
//! `rustnn_backend.rs`.

use std::sync::{Mutex, OnceLock};

use servo_base::id::{PipelineNamespace, TEST_NAMESPACE};
use webnn::{BackendDeviceType, BackendOptions, BackendPowerPreference, WebNN};

static LOG_LINES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

struct CaptureLogger;

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Error
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            LOG_LINES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(format!("{}", record.args()));
        }
    }

    fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger;

fn leak_lines() -> Vec<String> {
    LOG_LINES
        .get()
        .map(|lines| lines.lock().unwrap().clone())
        .unwrap_or_default()
        .into_iter()
        .filter(|line| line.contains("[webnn-leak]"))
        .collect()
}

/// Drive the channel the same way the GC finalizers do and confirm the context
/// backend is fully removed (backends -> 0) once the destroys are processed.
#[test]
fn channel_destroy_paths_return_to_baseline() {
    let _ = LOG_LINES.set(Mutex::new(Vec::new()));
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Trace);

    // `ContextId::new()` reads a per-thread namespace; install the shared test
    // namespace for this thread.
    PipelineNamespace::install(TEST_NAMESPACE);

    let webnn = WebNN::shared();
    let ctx = webnn::ContextId::new();
    let options = BackendOptions {
        power_preference: BackendPowerPreference::Default,
        accelerated: true,
        device_type: BackendDeviceType::Cpu,
    };

    webnn.new_context(ctx, &options);
    // Synchronous barrier: create_builder blocks until the backend thread has
    // processed all prior requests (including `new_context`).
    assert_ne!(
        webnn.create_builder(ctx),
        0,
        "context backend was not created"
    );

    // The exact requests the three `Droppable*` finalizers send.
    webnn.create_tensor(ctx, 1, 0, &[2]);
    webnn.destroy_tensor(ctx, 1);
    webnn.destroy_graph(ctx, 42);
    webnn.destroy_context(ctx);

    // Barrier: the backend thread has now processed every destroy above.
    assert_eq!(
        webnn.create_builder(ctx),
        0,
        "context backend was not removed"
    );

    let lines = leak_lines();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("new_ctx") && l.contains("backends=1")),
        "missing new_ctx backends=1: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("destroy_tensor")),
        "missing destroy_tensor: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("destroy_graph")),
        "missing destroy_graph: {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("destroy_ctx") && l.contains("backends=0")),
        "missing destroy_ctx backends=0: {lines:?}"
    );
}
