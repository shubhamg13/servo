//! LiteRT initialization — called once at startup.

/// Initialize the LiteRT runtime. Must be called once before any model ops.
pub fn initialize() -> Result<(), String> {
    let _ =
        ::litert::Environment::new().map_err(|e| format!("LiteRT initialization failed: {}", e))?;
    log::info!("LiteRT backend initialized");
    Ok(())
}
