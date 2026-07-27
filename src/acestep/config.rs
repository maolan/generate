//! Serde configuration for the ACE-Step 1.5 DiT checkpoint.
//!
//! Mirrors the keys of `acestep-v15-turbo/config.json`; unknown keys are ignored.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AceStepConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub in_channels: usize,
    pub audio_acoustic_hidden_dim: usize,
    pub patch_size: usize,
    pub text_hidden_dim: usize,
    pub num_lyric_encoder_hidden_layers: usize,
    pub num_timbre_encoder_hidden_layers: usize,
    pub timbre_fix_frame: usize,
    pub pool_window_size: usize,
    pub num_attention_pooler_hidden_layers: usize,
    pub fsq_dim: usize,
    pub fsq_input_levels: Vec<u32>,
    pub vocab_size: u32,
    pub layer_types: Vec<String>,
    pub is_turbo: bool,
}

impl AceStepConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse AceStep config {}", path.display()))
    }

    /// True when `layer_types[layer]` is "sliding_attention".
    pub fn is_sliding_layer(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .is_some_and(|kind| kind == "sliding_attention")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_reference_turbo_config() {
        let config = AceStepConfig::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/acestep/testdata/acestep_v15_turbo_config.json"),
        )
        .expect("reference config must parse");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.in_channels, 192);
        assert_eq!(config.audio_acoustic_hidden_dim, 64);
        assert_eq!(config.patch_size, 2);
        assert_eq!(config.text_hidden_dim, 1024);
        assert_eq!(config.fsq_dim, 2048);
        assert_eq!(config.fsq_input_levels, vec![8, 8, 8, 5, 5, 5]);
        assert_eq!(config.vocab_size, 64003);
        assert_eq!(config.layer_types.len(), 24);
        assert!(config.is_turbo);
        assert!(config.is_sliding_layer(0));
        assert!(!config.is_sliding_layer(1));
        assert!(config.is_sliding_layer(22));
        assert!(!config.is_sliding_layer(23));
        assert!(!config.is_sliding_layer(24));
    }
}
