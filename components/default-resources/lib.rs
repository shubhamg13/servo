/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::path::PathBuf;
#[cfg(feature = "snake-game")]
use std::sync::LazyLock;

#[cfg(feature = "snake-game")]
use base64::prelude::*;
use embedder_traits::resources::{Resource, ResourceBytes, ResourceReaderMethods};

/// The network error page with the Snake game. The game wasm is embedded as
/// base64 because the page is shown while the network is unavailable and cannot
/// fetch it. The game source lives in `snake/` as a standalone crate.
#[cfg(feature = "snake-game")]
static NETERROR_HTML: LazyLock<ResourceBytes> = LazyLock::new(|| {
    let template = include_str!("resources/neterror-snake.html");
    let game = BASE64_STANDARD.encode(include_bytes!("resources/snake.wasm"));
    ResourceBytes::from_vec(template.replace("${snake}", &game).into_bytes())
});

#[cfg(feature = "snake-game")]
fn neterror_html() -> ResourceBytes {
    NETERROR_HTML.clone()
}

#[cfg(not(feature = "snake-game"))]
fn neterror_html() -> ResourceBytes {
    ResourceBytes::from_static(&include_bytes!("resources/neterror.html")[..])
}

/// A default resource reader that provides baked in resources.
pub struct DefaultResourceReader;

impl ResourceReaderMethods for DefaultResourceReader {
    fn sandbox_access_files(&self) -> Vec<PathBuf> {
        vec![]
    }
    fn sandbox_access_files_dirs(&self) -> Vec<PathBuf> {
        vec![]
    }
    fn read(&self, file: Resource) -> ResourceBytes {
        let bytes: &'static [u8] = match file {
            Resource::NetErrorHTML => return neterror_html(),
            Resource::BluetoothBlocklist => &include_bytes!("resources/gatt_blocklist.txt")[..],
            Resource::DomainList => &include_bytes!("resources/public_domains.txt")[..],
            Resource::HstsPreloadList => &include_bytes!("resources/hsts_preload.fstmap")[..],
            Resource::BadCertHTML => &include_bytes!("resources/badcert.html")[..],
            Resource::BrokenImageIcon => &include_bytes!("resources/rippy.png")[..],
            Resource::CrashHTML => &include_bytes!("resources/crash.html")[..],
            Resource::DirectoryListingHTML => {
                &include_bytes!("resources/directory-listing.html")[..]
            },
            Resource::AboutMemoryHTML => &include_bytes!("resources/about-memory.html")[..],
            Resource::DebuggerJS => &include_bytes!("resources/debugger.js")[..],
            Resource::JsonViewerHTML => &include_bytes!("resources/json-viewer.html")[..],
        };
        ResourceBytes::from_static(bytes)
    }
}

embedder_traits::submit_resource_reader!(&DefaultResourceReader);
