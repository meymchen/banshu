use std::path::PathBuf;

/// Return the repository root from the fixed `crates/xtask` layout.
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("xtask lives two directories below the workspace root")
        .to_path_buf()
}
