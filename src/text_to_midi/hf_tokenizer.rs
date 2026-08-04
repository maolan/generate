//! Hugging Face tokenizer adapter for `llama-burn`.
//!
//! The official MIDI-LLM checkpoint ships `tokenizer.json` (Hugging Face
//! tokenizers format) rather than the tiktoken `.model` file that
//! `llama-burn::tokenizer::Tiktoken` expects. This module wraps the
//! `tokenizers` crate so the Llama config can be initialized from the HF
//! checkpoint as-is.

use llama_burn::tokenizer::Tokenizer;

/// Tokenizer loaded from a Hugging Face `tokenizer.json` file.
pub struct HfTokenizer {
    tokenizer: tokenizers::Tokenizer,
    bos_id: u32,
    eos_id: u32,
}

impl Tokenizer for HfTokenizer {
    fn new(tokenizer_path: &str) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|err| format!("failed to load Hugging Face tokenizer: {err}"))?;

        let bos_id = tokenizer
            .token_to_id("<|begin_of_text|>")
            .ok_or("missing <|begin_of_text|> special token")?;
        let eos_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .ok_or("missing <|end_of_text|> special token")?;

        Ok(Self {
            tokenizer,
            bos_id,
            eos_id,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| format!("failed to encode text: {err}"))
            .expect("tokenizer encoding should not fail");
        let mut tokens = encoding.get_ids().to_vec();
        if bos {
            tokens.insert(0, self.bos_id);
        }
        if eos {
            tokens.push(self.eos_id);
        }
        tokens
    }

    fn decode(&self, tokens: Vec<u32>) -> String {
        self.tokenizer.decode(&tokens, true).unwrap_or_default()
    }

    fn bos_id(&self) -> u32 {
        self.bos_id
    }

    fn eos_id(&self) -> u32 {
        self.eos_id
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![self.eos_id]
    }
}
