//! Workspace tooling. Run with `cargo run -p xtask -- <command>`.
//!
//! Commands:
//! - `generate-catalog [--input <api.json>]` — regenerate the bundled model
//!   catalogs for `banshu-ai` from [models.dev](https://models.dev). Fetches
//!   `https://models.dev/api.json` unless `--input` points at a local copy.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use serde_json::Value;

/// (banshu provider id, models.dev provider key).
const PROVIDERS: &[(&str, &str)] = &[
    ("deepseek", "deepseek"),
    ("zai", "zai"),
    ("minimax", "minimax"),
    ("moonshot", "moonshotai"),
    ("kimi", "kimi-for-coding"),
    ("xiaomi", "xiaomi"),
];

const MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// The bundled catalog entry shape consumed by `banshu-ai`.
#[derive(Serialize)]
struct CatalogModel {
    id: String,
    name: String,
    reasoning: bool,
    input: Vec<String>,
    context_window: u32,
    max_tokens: u32,
    cost: CatalogCost,
}

#[derive(Serialize)]
struct CatalogCost {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("generate-catalog") => {
            let mut input = None;
            while let Some(arg) = args.next() {
                if arg == "--input" {
                    input = args.next();
                }
            }
            if let Err(err) = generate_catalog(input.map(PathBuf::from)) {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown command: {other:?}");
            eprintln!("usage: cargo run -p xtask -- generate-catalog [--input <api.json>]");
            std::process::exit(2);
        }
    }
}

fn generate_catalog(input: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let raw = match input {
        Some(path) => std::fs::read_to_string(&path)?,
        None => {
            eprintln!("fetching {MODELS_DEV_URL} …");
            reqwest::blocking::get(MODELS_DEV_URL)?.text()?
        }
    };
    let data: Value = serde_json::from_str(&raw)?;

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("ai")
        .join("src")
        .join("models")
        .join("catalog");
    std::fs::create_dir_all(&out_dir)?;

    for (banshu_id, source_key) in PROVIDERS {
        let models = data
            .get(source_key)
            .and_then(|p| p.get("models"))
            .and_then(Value::as_object)
            .ok_or_else(|| format!("models.dev has no models for `{source_key}`"))?;

        // BTreeMap keeps the output deterministic (sorted by id).
        let mut catalog: BTreeMap<String, CatalogModel> = BTreeMap::new();
        for (id, model) in models {
            catalog.insert(id.clone(), catalog_model(id, model));
        }
        let entries: Vec<&CatalogModel> = catalog.values().collect();

        let path = out_dir.join(format!("{banshu_id}.json"));
        let json = serde_json::to_string_pretty(&entries)?;
        std::fs::write(&path, format!("{json}\n"))?;
        println!("wrote {} ({} models)", path.display(), entries.len());
    }
    Ok(())
}

fn catalog_model(id: &str, model: &Value) -> CatalogModel {
    let cost = &model["cost"];
    CatalogModel {
        id: id.to_string(),
        name: model["name"].as_str().unwrap_or(id).to_string(),
        reasoning: model["reasoning"].as_bool().unwrap_or(false),
        input: model["modalities"]["input"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|m| *m == "text" || *m == "image")
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_else(|| vec!["text".to_string()]),
        context_window: model["limit"]["context"].as_u64().unwrap_or(0) as u32,
        max_tokens: model["limit"]["output"].as_u64().unwrap_or(0) as u32,
        cost: CatalogCost {
            input: cost["input"].as_f64().unwrap_or(0.0),
            output: cost["output"].as_f64().unwrap_or(0.0),
            cache_read: cost["cache_read"].as_f64().unwrap_or(0.0),
            cache_write: cost["cache_write"].as_f64().unwrap_or(0.0),
        },
    }
}
