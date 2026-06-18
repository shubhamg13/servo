/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git_sha() -> Result<String, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        let hash = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
        Ok(hash.trim().to_owned())
    } else {
        let stderr = String::from_utf8(output.stderr).map_err(|e| e.to_string())?;
        Err(stderr)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rustc-check-cfg=cfg(servo_production)");
    println!("cargo::rustc-check-cfg=cfg(servo_do_not_use_in_production)");
    // Cargo does not expose the profile name to crates or their build scripts,
    // but we can extract it from OUT_DIR and set a custom cfg() ourselves.
    let out = std::env::var("OUT_DIR")?;
    let out = Path::new(&out);
    let krate = out.parent().unwrap();
    let build = krate.parent().unwrap();
    let profile = build
        .parent()
        .unwrap()
        .file_name()
        .unwrap()
        .to_string_lossy();
    if profile == "production" || profile.starts_with("production-") {
        println!("cargo:rustc-cfg=servo_production");
    } else {
        println!("cargo:rustc-cfg=servo_do_not_use_in_production");
    }

    // Note: We can't use `#[cfg(windows)]`, since that would check the host platform
    // and not the target platform
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap();

    if target_os == "windows" {
        #[cfg(windows)]
        {
            let mut res = winresource::WindowsResource::new();
            res.set_icon("../../resources/servo.ico");
            res.set_manifest_file("platform/windows/servoshell.exe.manifest");
            res.compile().unwrap();
        }
        #[cfg(not(windows))]
        panic!("Cross-compiling to windows is currently not supported");
    } else if target_os == "android" {
        // FIXME: We need this workaround since jemalloc-sys still links
        // to libgcc instead of libunwind, but Android NDK 23c and above
        // don't have libgcc. We can't disable jemalloc for Android as
        // in 64-bit aarch builds, the system allocator uses tagged
        // pointers by default which causes the assertions in SM & mozjs
        // to fail. See https://github.com/servo/servo/issues/32175.
        let mut libgcc = File::create(out.join("libgcc.a")).unwrap();
        libgcc.write_all(b"INPUT(-lunwind)").unwrap();
        println!("cargo:rustc-link-search=native={}", out.display());
    }

    match git_sha() {
        Ok(hash) => println!("cargo:rustc-env=GIT_SHA={}", hash),
        Err(error) => {
            println!(
                "cargo:warning=Could not generate git version information: {:?}",
                error
            );
            println!("cargo:rustc-env=GIT_SHA=nogit");
        },
    }

    // On MacOS, all dylib dependencies are shipped along with the binary
    // in the "/lib" directory. Setting the rpath here, allows the dynamic
    // linker to locate them. See `man dyld` for more info.
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/lib/");
    }

    // The prebuilt LiteRT runtime is linked by `litert-sys`, which sets an rpath
    // for its own targets. Cargo does not propagate a dependency's
    // `rustc-link-arg` to this binary, so the final artifact needs one itself.
    // `LITERT_LIB_DIR` is litert-sys's own override for the directory holding
    // the prebuilt, so pointing at it cannot disagree with what was linked.
    // Linux is the only target that builds WebNN with LiteRT: OpenHarmony loads
    // the runtime from the application's own library directory, which is already
    // on the loader's search path, and macOS and Windows build it with CoreML
    // and ONNX Runtime instead.
    if target_env != "ohos" &&
        target_os != "macos" &&
        target_os != "windows" &&
        std::env::var("CARGO_FEATURE_WEBNN_RUSTNN").is_ok()
    {
        println!("cargo:rerun-if-env-changed=LITERT_LIB_DIR");
        match std::env::var_os("LITERT_LIB_DIR").map(PathBuf::from) {
            Some(dir) if dir.is_dir() => {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            },
            Some(dir) => println!(
                "cargo:warning=LITERT_LIB_DIR is `{}`, which is not a directory, so the prebuilt \
                 LiteRT runtime that this build links will not be findable at run time.",
                dir.display()
            ),
            None => println!(
                "cargo:warning=LITERT_LIB_DIR is not set, so the prebuilt LiteRT runtime that this \
                 build links will only be found if the platform's library search path provides it \
                 when Servo runs. Set LITERT_LIB_DIR to the directory holding that runtime."
            ),
        }
    }

    // On OpenHarmony, libservoshell.so is loaded by ArkTS as a NAPI module.
    // Passing a version script allows us to inform the linker about required
    // symbol visibility (only one), which improves stripping of unused sections.
    if target_env == "ohos" {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let version_script = Path::new(&manifest_dir)
            .join("platform")
            .join("openharmony")
            .join("libservoshell.ver");
        let version_script_str = version_script.to_str().expect("Expected UTF-8 text");
        assert!(
            version_script.exists(),
            "Expected version script to exist at path `{version_script_str}`"
        );
        println!("cargo:rerun-if-changed={version_script_str}");
        // Using `rustc-link-arg-cdylib` causes a false-positive warning:
        // https://github.com/rust-lang/cargo/issues/16487
        // We work around this by just using the unconditional link-arg, which
        // should be fine, since we always build servo as a cdylib on OpenHarmony.
        println!("cargo:rustc-link-arg=-Wl,--version-script={version_script_str}");
    }
    Ok(())
}
