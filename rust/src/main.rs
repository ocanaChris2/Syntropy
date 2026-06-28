//! Syntropy — streaming lossless neural compressor (skeleton).
//!
//! The DEFAULT build is std-only and verifiable offline: `cargo test` exercises
//! the arithmetic coder (against the Python golden vectors), the quantizer, the
//! classical models, the BPE tokenizer, the mixer, and the full
//! compress/decompress round-trip using a deterministic stub neural model.
//!
//! Wiring in the REAL network (candle + safetensors, behind the `neural`
//! feature) means implementing `model::NeuralModel` for the exported weights;
//! everything else in `pipeline.rs` stays exactly as-is. See
//! docs/rust_integration_blueprint.md.

mod classical;
mod coder;
mod mixer;
mod model;
mod pipeline;
mod quant;
mod tokenizer;

use model::StubModel;
use pipeline::{ClassicalConfig, Pipeline};
use tokenizer::BpeModel;

fn demo_roundtrip() {
    // A tiny end-to-end demonstration with the stub model.
    let tok = BpeModel::new(vec![
        (b"t".to_vec(), b"h".to_vec()),
        (b"th".to_vec(), b"e".to_vec()),
    ]);
    let vocab = tok.vocab_size();
    let mut p = Pipeline::new(tok, StubModel::new(vocab), ClassicalConfig::default());

    let input = b"the the the the lossless the the the";
    let (coded, ntokens) = p.compress(input);
    let out = p.decompress(&coded, ntokens);

    assert_eq!(out, input, "round-trip must be bit-identical");
    println!(
        "demo: {} input bytes -> {} coded bytes ({} tokens), round-trip OK",
        input.len(),
        coded.len(),
        ntokens
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("selftest") | None => {
            demo_roundtrip();
            println!("selftest OK (run `cargo test` for the full suite)");
        }
        Some("compress") | Some("decompress") => {
            eprintln!(
                "compress/decompress on real data require the trained weights and the\n\
                 `neural` feature: build with `cargo build --release --features neural`\n\
                 and implement model::NeuralModel for the exported safetensors.\n\
                 See docs/rust_integration_blueprint.md."
            );
            std::process::exit(2);
        }
        Some(other) => {
            eprintln!("unknown command: {other}\nusage: syntropy [selftest|compress|decompress]");
            std::process::exit(2);
        }
    }
}
