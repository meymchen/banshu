//! Workspace tooling. Run with `cargo run -p xtask -- <command>`.
//!
//! Commands:
//! - `generate-catalog [--input <api.json>]` — regenerate the bundled model
//!   catalogs for `banshu-ai` from [models.dev](https://models.dev). Fetches
//!   `https://models.dev/api.json` unless `--input` points at a local copy.
//!   Parsing and filtering rules come from `banshu_ai::models_dev`, the same
//!   module the runtime catalog refresh uses.

use std::collections::BTreeMap;
use std::path::PathBuf;

use banshu_ai::models_dev::{ModelsDevModel, models_from_api_json};
use banshu_ai::{Modality, ModelCost};
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

impl From<ModelsDevModel> for CatalogModel {
    fn from(model: ModelsDevModel) -> Self {
        CatalogModel {
            id: model.id,
            name: model.name,
            reasoning: model.reasoning,
            input: model.input.iter().map(modality_str).collect(),
            context_window: model.context_window,
            max_tokens: model.max_tokens,
            cost: CatalogCost::from(model.cost),
        }
    }
}

impl From<ModelCost> for CatalogCost {
    fn from(cost: ModelCost) -> Self {
        CatalogCost {
            input: cost.input,
            output: cost.output,
            cache_read: cost.cache_read,
            cache_write: cost.cache_write,
        }
    }
}

fn modality_str(modality: &Modality) -> String {
    match modality {
        Modality::Text => "text",
        Modality::Image => "image",
        other => unreachable!("catalog filter keeps only text/image modalities: {other:?}"),
    }
    .to_string()
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
        let models = models_from_api_json(&data, source_key)
            .ok_or_else(|| format!("models.dev has no models for `{source_key}`"))?;

        // Only tool-calling text-in/text-out models ship in the catalog.
        // BTreeMap keeps the output deterministic (sorted by id).
        let catalog: BTreeMap<String, CatalogModel> = models
            .into_iter()
            .filter(ModelsDevModel::is_tool_calling_text_model)
            .map(|model| (model.id.clone(), CatalogModel::from(model)))
            .collect();
        let entries: Vec<&CatalogModel> = catalog.values().collect();

        let path = out_dir.join(format!("{banshu_id}.json"));
        let json = serde_json::to_string_pretty(&entries)?;
        std::fs::write(&path, format!("{json}\n"))?;
        println!("wrote {} ({} models)", path.display(), entries.len());
    }
    Ok(())
}
