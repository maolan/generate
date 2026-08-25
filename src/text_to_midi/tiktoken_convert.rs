//! Convert a Hugging Face `tokenizer.json` to tiktoken `.model` format.
//!
//! The MIDI-LLM checkpoint ships a Hugging Face tokenizer. `maolan_llama`
//! already contains a pure-Rust tiktoken tokenizer (`tokenizer::Tiktoken`), so
//! we convert the HF file once and cache a `tokenizer.model` next to it.

use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Number of base text tokens in the Llama 3.x vocabulary.
const LLAMA_BASE_VOCAB_SIZE: usize = 128_000;

/// Convert `tokenizer.json` to tiktoken `tokenizer.model` format.
///
/// The output file contains one line per base token:
///
/// ```text
/// <base64-encoded token bytes> <token id>
/// ```
///
/// Only the base 128,000 Llama text tokens are emitted. Added tokens (special
/// tokens and the MIDI-LLM extended vocabulary) are handled by the caller/model
/// and are not needed for prompt tokenization.
pub fn convert_hf_tokenizer_to_tiktoken<P, Q>(tokenizer_json: P, output_model: Q) -> Result<()>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let tokenizer_json = tokenizer_json.as_ref();
    let output_model = output_model.as_ref();

    let tokenizer = tokie::Tokenizer::from_json(tokenizer_json)
        .map_err(|err| anyhow::anyhow!("failed to load HF tokenizer with tokie: {err}"))?;

    if tokenizer.vocab_size() < LLAMA_BASE_VOCAB_SIZE {
        bail!(
            "tokenizer vocab size {} is smaller than expected Llama base vocab size {}",
            tokenizer.vocab_size(),
            LLAMA_BASE_VOCAB_SIZE
        );
    }

    if let Some(parent) = output_model.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let file = fs::File::create(output_model)
        .with_context(|| format!("failed to create {}", output_model.display()))?;
    let mut writer = BufWriter::new(file);

    for id in 0..LLAMA_BASE_VOCAB_SIZE {
        let bytes = tokenizer.token_to_bytes(id as u32);
        let line = format!("{} {}\n", STANDARD.encode(bytes), id);
        writer
            .write_all(line.as_bytes())
            .with_context(|| format!("failed to write token {id}"))?;
    }

    writer.flush().context("failed to flush tokenizer.model")?;
    Ok(())
}

/// Return a cached `tokenizer.model` path next to `tokenizer.json`.
///
/// If the cached file is missing or older than `tokenizer.json`, regenerate it.
pub fn ensure_tiktoken_model<P: AsRef<Path>>(tokenizer_json: P) -> Result<PathBuf> {
    let tokenizer_json = tokenizer_json.as_ref();
    let output_model = tokenizer_json.with_file_name("tokenizer.model");

    let should_convert = match fs::metadata(&output_model) {
        Ok(model_meta) => match fs::metadata(tokenizer_json) {
            Ok(json_meta) => {
                let model_mtime = model_meta.modified().unwrap_or(model_meta.created()?);
                let json_mtime = json_meta.modified().unwrap_or(json_meta.created()?);
                json_mtime > model_mtime
            }
            Err(_) => true,
        },
        Err(_) => true,
    };

    if should_convert {
        convert_hf_tokenizer_to_tiktoken(tokenizer_json, &output_model)?;
    }

    Ok(output_model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal synthetic BPE tokenizer fixture in HF tokenizer.json format.
    ///
    /// This is not a real Llama tokenizer; it only exercises the converter's
    /// ability to read a JSON file and emit a tiktoken-format model.
    fn synthetic_tokenizer_json() -> tempfile::TempPath {
        let mut temp = tempfile::NamedTempFile::new().unwrap();
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": { "Regex": "\\p{L}+|\\p{N}+|[^\\s\\p{L}\\p{N}]+" },
                        "behavior": "Isolated",
                        "invert": false
                    }
                ]
            },
            "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": true,
                "vocab": {
                    "<|endoftext|>": 0,
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "ab": 4
                },
                "merges": []
            }
        });
        temp.write_all(json.to_string().as_bytes()).unwrap();
        temp.into_temp_path()
    }

    #[test]
    fn converter_rejects_too_small_vocab() {
        let json_path = synthetic_tokenizer_json();
        let model_path = json_path.with_extension("model");

        // The fixture has only 5 tokens; conversion should fail because it is
        // smaller than the Llama base vocab.
        assert!(convert_hf_tokenizer_to_tiktoken(&json_path, &model_path).is_err());
    }

    /// Download the MIDI-LLM tokenizer and verify that the converted tiktoken
    /// model produces the same token IDs as the original HF tokenizer.
    ///
    /// Ignored by default because it downloads ~18 MB from Hugging Face.
    /// Run with: cargo test -- --ignored
    #[test]
    #[ignore]
    fn midi_llm_tokenizer_conversion_matches() {
        use maolan_llama::tokenizer::{Tiktoken, Tokenizer};

        let repo_id = "slseanwu/MIDI-LLM_Llama-3.2-1B";
        let tokenizer_filename = "tokenizer.json";

        let client = huggingface_hub::HFClientSync::new().expect("HF client");
        let (owner, name) = repo_id.split_once('/').expect("valid repo id");
        let tokenizer_json = client
            .model(owner, name)
            .download_file()
            .filename(tokenizer_filename)
            .send()
            .expect("download tokenizer.json");

        let hf_tokenizer = tokie::Tokenizer::from_json(&tokenizer_json).expect("load HF tokenizer");
        let tiktoken_path = ensure_tiktoken_model(&tokenizer_json).expect("convert tokenizer");
        let tiktoken = Tiktoken::new(tiktoken_path.to_str().unwrap()).expect("load tiktoken model");

        let test_prompts = [
            "Hello, world!",
            "You are a world-class composer. Please compose some music according to the following description: upbeat jazz",
            "C major scale with tempo 120",
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>",
        ];

        for prompt in &test_prompts {
            let hf_ids = hf_tokenizer.encode_ids(prompt, true);
            let tiktoken_ids = tiktoken.encode(prompt, true, false);
            assert_eq!(
                hf_ids, tiktoken_ids,
                "token mismatch for prompt: {prompt:?}\nHF:     {hf_ids:?}\nTiktoken: {tiktoken_ids:?}"
            );
        }
    }
}
