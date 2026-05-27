/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

pub mod backend;
pub mod backends;
pub mod compiler;
#[cfg(feature = "litert")]
pub mod litert;

pub use backend::{DataType, GraphNode, TensorDesc};
pub use backends::infer as run_inference;

/// Start the WebNN backend.
pub fn start_webnn_backend() -> bool {
    #[cfg(feature = "litert")]
    {
        if let Err(e) = litert::initialize() {
            log::warn!("Failed to initialize LiteRT backend: {:?}", e);
            return false;
        }
    }
    true
}
