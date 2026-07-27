//! Probe: verify LM tokenizer special-token handling and the audio-code
//! vocab mapping against the real lm_tokenizer.json.
//!
//! Usage: cargo run --release --example lm_probe -- <model_dir>

use maolan_generate::acestep::lm::{self, AudioCodeVocab};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/meka/repos/ace".to_string());
    let lm_tokenizer = Path::new(&model_dir).join("lm_tokenizer.json");

    let vocab = AudioCodeVocab::from_tokenizer_json(&lm_tokenizer)?;
    println!("audio code vocab size: {}", vocab.len());
    for code in [0u32, 1, 2482, 40955, 53754, 63999] {
        println!(
            "code {code} -> token id {:?} -> back {:?}",
            vocab.code_token_id(code),
            vocab
                .code_token_id(code)
                .and_then(|id| vocab.token_id_to_code(id))
        );
    }

    let cot = lm::build_cot_block(
        "Metal guitar with a lot of distortion",
        Some(120.0),
        Some("A minor"),
        Some("4/4"),
        4,
    );
    let prompt = lm::build_codes_prompt("Metal guitar with a lot of distortion", &cot);
    println!("\n===== FULL LM PROMPT =====\n{prompt}");

    let ids = lm::tokenize_prompt(&lm_tokenizer, &prompt)?;
    println!("\n{} token ids", ids.len());
    println!("all: {:?}", ids);
    println!("last 20: {:?}", &ids[ids.len().saturating_sub(20)..]);

    let special = [
        ("<|im_start|>", 151644u32),
        ("<|im_end|>", 151645u32),
        ("<|endoftext|>", 151643u32),
    ];
    for (name, expected) in special {
        let found = ids.contains(&expected);
        println!("{name} id {expected} present in tokenized prompt: {found}");
    }
    Ok(())
}
