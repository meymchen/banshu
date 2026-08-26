use std::process::Command;

use xtask::workspace_root;

fn package_files() -> Vec<String> {
    let output = Command::new(env!("CARGO"))
        .args(["package", "-p", "banshu-ai", "--list", "--allow-dirty"])
        .current_dir(workspace_root())
        .output()
        .expect("cargo package --list runs");

    assert!(
        output.status.success(),
        "cargo package --list failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("cargo emits UTF-8 paths")
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn package_excludes_non_release_tests_and_credential_files() {
    let files = package_files();
    let unexpected: Vec<&str> = files
        .iter()
        .map(String::as_str)
        .filter(|path| {
            (path.starts_with("tests/")
                && *path != "tests/context_serde.rs"
                && *path != "tests/fixtures/context.json")
                || path.starts_with("proptest-regressions/")
                || path.split('/').any(|component| {
                    matches!(
                        component.to_ascii_lowercase().as_str(),
                        ".scratch"
                            | ".env"
                            | "secrets"
                            | "credentials"
                            | "oauth-token"
                            | "oauth-tokens"
                    ) || component.ends_with(".pem")
                        || component.ends_with(".key")
                })
        })
        .collect();

    assert!(
        unexpected.is_empty(),
        "package contains non-release or credential-like files: {unexpected:#?}"
    );
}

#[test]
fn package_carries_only_the_promised_protocols_and_provider_catalogs() {
    let files = package_files();

    for required in [
        "CHANGELOG.md",
        "README.md",
        "examples/faux_provider.rs",
        "src/api/anthropic_messages.rs",
        "src/api/openai_completions.rs",
        "tests/context_serde.rs",
        "tests/fixtures/context.json",
    ] {
        assert!(
            files.iter().any(|path| path == required),
            "release package is missing {required}"
        );
    }

    let catalogs: Vec<&str> = files
        .iter()
        .map(String::as_str)
        .filter(|path| path.starts_with("src/models/catalog/") && path.ends_with(".json"))
        .collect();
    assert_eq!(
        catalogs,
        [
            "src/models/catalog/deepseek.json",
            "src/models/catalog/kimi.json",
            "src/models/catalog/minimax.json",
            "src/models/catalog/moonshot.json",
            "src/models/catalog/xiaomi.json",
            "src/models/catalog/zai.json",
        ]
    );

    let api_module = std::fs::read_to_string(
        workspace_root()
            .join("crates")
            .join("ai")
            .join("src")
            .join("api")
            .join("mod.rs"),
    )
    .expect("read the packaged public API module");
    let public_protocol_modules: Vec<&str> = api_module
        .lines()
        .filter_map(|line| line.strip_prefix("pub mod "))
        .map(|module| module.trim_end_matches(';'))
        .collect();
    assert_eq!(
        public_protocol_modules,
        ["anthropic_messages", "openai_completions"]
    );

    let provider_module = std::fs::read_to_string(
        workspace_root()
            .join("crates")
            .join("ai")
            .join("src")
            .join("provider")
            .join("mod.rs"),
    )
    .expect("read the packaged provider module");
    for constructor in ["deepseek", "zai", "moonshot", "xiaomi", "kimi", "minimax"] {
        assert!(
            provider_module.contains(&format!("pub fn {constructor}(")),
            "release package is missing Provider::{constructor}"
        );
    }
    assert!(
        !provider_module.contains("pub fn openai()"),
        "OpenAI is a custom-compatible endpoint, not one of the six built-ins"
    );
}
