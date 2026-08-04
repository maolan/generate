//! MIDI-LLM inference for text-to-MIDI generation.
//!
//! This module extends a Llama 3.2 1B model with the AMT music vocabulary and
//! loads the official MIDI-LLM safetensors checkpoint. Generation is performed
//! directly on the Burn backend chosen by the CLI.

use super::amt::tokens_to_events;
use super::hf_tokenizer::HfTokenizer;
use super::midi::{TimeSignature, write_midi};
use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::DType;
use burn::tensor::{
    Bool, DataError, Device, Int, Shape, Tensor, TensorData, activation::softmax, backend::Backend,
};
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    TensorSnapshot,
};
use huggingface_hub::{Repo, RepoType, api::sync::ApiBuilder};
use maolan_llama::llama::{Llama, LlamaConfig, RopeConfig, RopeFrequencyScaling};
use maolan_llama::sampling::Sampler;
use maolan_llama::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Start of the MIDI token range in the extended MIDI-LLM vocabulary.
///
/// MIDI-LLM appends 55,026 AMT tokens after the Llama 3.2 text vocabulary of
/// 128,256 tokens.
pub const MIDI_TOKEN_START: u32 = 128_256;

/// Number of AMT tokens that the model is allowed to emit.
pub const MIDI_TOKEN_COUNT: u32 = 55_026;

/// End (exclusive) of the generated MIDI token range.
pub const MIDI_TOKEN_END: u32 = MIDI_TOKEN_START + MIDI_TOKEN_COUNT; // 183_282

/// Total size of the extended MIDI-LLM vocabulary (Llama 3.2 text tokens +
/// AMT music tokens + four special tokens).
pub const MIDI_VOCAB_SIZE: u32 = 183_286;

/// The special "MIDI-BOS" token appended after the text prompt to cue the
/// model to start generating music tokens.
pub const MIDI_BOS_TOKEN: u32 = MIDI_TOKEN_START + 55_026; // 183_282, AMT AUTOREGRESS

/// System prefix used by the official MIDI-LLM generation scripts.
const SYSTEM_PROMPT: &str = "You are a world-class composer. Please compose some music according to the following description: ";

const MIDI_LLM_REPO_ID: &str = "slseanwu/MIDI-LLM_Llama-3.2-1B";
const TOKENIZER_FILENAME: &str = "tokenizer.json";
const SAFETENSORS_FILENAME: &str = "model.safetensors";
const CONFIG_FILENAME: &str = "config.json";

/// Generation parameters for MIDI-LLM.
#[derive(Clone, Debug)]
pub struct MidiLlmConfig {
    /// Path to the Hugging Face `tokenizer.json` file.
    pub tokenizer_path: PathBuf,
    /// Path to the `model.safetensors` checkpoint.
    pub checkpoint_path: PathBuf,
    /// Maximum number of music tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (1.0 = default).
    pub temperature: f32,
    /// Top-p nucleus-sampling threshold.
    pub top_p: f32,
    /// Seed for the top-p sampler.
    pub seed: u64,
    /// Device to run on.
    pub max_seq_len: usize,
}

impl Default for MidiLlmConfig {
    fn default() -> Self {
        Self {
            tokenizer_path: PathBuf::new(),
            checkpoint_path: PathBuf::new(),
            max_tokens: 1024,
            temperature: 1.0,
            top_p: 0.98,
            seed: 0,
            max_seq_len: 4096,
        }
    }
}

/// Resolve tokenizer and checkpoint paths, downloading from Hugging Face if
/// no local model directory is provided.
pub fn resolve_model_paths(model_dir_override: Option<&Path>) -> Result<(PathBuf, PathBuf)> {
    if let Some(model_dir) = model_dir_override {
        let tokenizer = model_dir.join(TOKENIZER_FILENAME);
        let checkpoint = model_dir.join(SAFETENSORS_FILENAME);
        if !tokenizer.exists() {
            bail!("missing tokenizer file {}", tokenizer.display());
        }
        if !checkpoint.exists() {
            bail!("missing checkpoint file {}", checkpoint.display());
        }
        return Ok((tokenizer, checkpoint));
    }

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("failed to initialize Hugging Face client")?;
    let repo = api.repo(Repo::new(MIDI_LLM_REPO_ID.to_string(), RepoType::Model));

    let tokenizer = repo
        .get(TOKENIZER_FILENAME)
        .with_context(|| format!("failed to fetch {MIDI_LLM_REPO_ID}/{TOKENIZER_FILENAME}"))?;
    let checkpoint = repo
        .get(SAFETENSORS_FILENAME)
        .with_context(|| format!("failed to fetch {MIDI_LLM_REPO_ID}/{SAFETENSORS_FILENAME}"))?;
    // Touch config so the snapshot directory is complete; we don't parse it yet.
    let _ = repo.get(CONFIG_FILENAME);

    Ok((tokenizer, checkpoint))
}

/// Build the Llama 3.2 1B configuration with the extended MIDI vocabulary.
fn midi_llama_config(tokenizer_path: &Path) -> LlamaConfig {
    LlamaConfig::new(
        8192,
        MIDI_VOCAB_SIZE as usize,
        tokenizer_path.to_string_lossy().to_string(),
    )
    .with_d_model(2048)
    .with_num_hidden_layers(16)
    .with_num_attention_heads(32)
    .with_num_key_value_heads(Some(8))
    .with_norm_eps(1e-5)
    .with_rope(
        RopeConfig::new(500_000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.0))),
    )
}

/// Load the MIDI-LLM model on the chosen backend.
pub fn load_model<B: Backend>(
    config: &MidiLlmConfig,
    device: &Device<B>,
) -> Result<Llama<B, HfTokenizer>> {
    let llama_config =
        midi_llama_config(&config.tokenizer_path).with_max_seq_len(config.max_seq_len);
    let mut llama = llama_config
        .init::<B, HfTokenizer>(device)
        .map_err(|err| anyhow!("failed to initialize MIDI-LLM model: {err}"))?;

    load_safetensors_into_model(&mut llama, &config.checkpoint_path)?;
    Ok(llama)
}

/// Adapter that casts BF16 checkpoint weights to F32 so the model can run on
/// backends (such as NdArray CPU) that do not implement BF16 tensor operations.
#[derive(Debug, Clone, Default)]
struct Bf16ToF32Adapter;

impl ModuleAdapter for Bf16ToF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::BF16 {
            return snapshot.clone();
        }
        let original_data_fn = snapshot.clone_data_fn();
        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(DType::F32))
        });
        TensorSnapshot::from_closure(
            cast_data_fn,
            DType::F32,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

fn load_safetensors_into_model<B: Backend>(
    llama: &mut Llama<B, HfTokenizer>,
    checkpoint_path: &Path,
) -> Result<()> {
    let patterns: Vec<(&str, &str)> = vec![
        (r"^model\.embed_tokens\.(.+)$", "tok_embeddings.$1"),
        (r"^lm_head\.(.+)$", "output.$1"),
        // Final RMSNorm: PyTorch `weight` -> Burn `gamma`.
        (r"^model\.norm\.weight$", "norm.gamma"),
        // Per-layer input RMSNorm.
        (
            r"^model\.layers\.([0-9]+)\.input_layernorm\.weight$",
            "layers.$1.attention_norm.gamma",
        ),
        // Per-layer post-attention RMSNorm.
        (
            r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.weight$",
            "layers.$1.ffn_norm.gamma",
        ),
        (
            r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.(.+)$",
            "layers.$1.attention.wq.$2",
        ),
        (
            r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.(.+)$",
            "layers.$1.attention.wk.$2",
        ),
        (
            r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.(.+)$",
            "layers.$1.attention.wv.$2",
        ),
        (
            r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.(.+)$",
            "layers.$1.attention.wo.$2",
        ),
        (
            r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.(.+)$",
            "layers.$1.feed_forward.swiglu.linear_inner.$2",
        ),
        (
            r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.(.+)$",
            "layers.$1.feed_forward.swiglu.linear_outer.$2",
        ),
        (
            r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.(.+)$",
            "layers.$1.feed_forward.w2.$2",
        ),
    ];
    let remapper = KeyRemapper::from_patterns(patterns)
        .map_err(|err| anyhow!("failed to build MIDI-LLM key remapper: {err}"))?;

    if !checkpoint_path.exists() {
        bail!(
            "checkpoint file {} does not exist",
            checkpoint_path.display()
        );
    }
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter.chain(Bf16ToF32Adapter))
        .remap(remapper);

    let result = llama
        .model
        .load_from(&mut store)
        .with_context(|| format!("failed to load weights from {}", checkpoint_path.display()))?;

    if !result.missing.is_empty() {
        let missing: Vec<String> = result.missing.iter().map(|m| m.0.clone()).collect();
        bail!(
            "MIDI-LLM checkpoint is missing required tensors: {}",
            missing.join(", ")
        );
    }

    Ok(())
}

/// Generate AMT-native tokens from a text prompt and write them to a MIDI file.
pub fn generate_midi_file<B: Backend>(
    config: &MidiLlmConfig,
    device: &Device<B>,
    prompt: &str,
    bpm: f32,
    time_signature: Option<TimeSignature>,
    output_path: &Path,
) -> Result<()> {
    let mut llama = load_model::<B>(config, device)?;

    let text_prompt = format!("{SYSTEM_PROMPT}{prompt} ");
    let mut input_tokens = llama.tokenizer.encode(&text_prompt, true, false);
    input_tokens.push(MIDI_BOS_TOKEN);

    let disallowed_mask = build_disallowed_mask(device, MIDI_TOKEN_START, MIDI_VOCAB_SIZE);
    let mut sampler = Sampler::new_top_p(config.top_p as f64, config.seed);
    let mut generated = Vec::with_capacity(config.max_tokens);

    // Prefill with the full prompt + MIDI-BOS.
    let mut input_tensor = tokens_to_tensor::<B>(device, &input_tokens);
    let mut logits = llama
        .model
        .forward(input_tensor, &mut llama.cache, &llama.rope);

    for _ in 0..config.max_tokens {
        let next_token =
            sample_next_token(&logits, &disallowed_mask, config.temperature, &mut sampler)?;
        generated.push(next_token);

        input_tensor = tokens_to_tensor::<B>(device, &[next_token]);
        logits = llama
            .model
            .forward(input_tensor, &mut llama.cache, &llama.rope);
    }

    let amt_tokens: Vec<u32> = generated
        .iter()
        .map(|t| t.saturating_sub(MIDI_TOKEN_START))
        .collect();
    let events = tokens_to_events(&amt_tokens);
    write_midi(output_path, &events, bpm, time_signature)
        .with_context(|| "failed to write generated MIDI file")
}

fn tokens_to_tensor<B: Backend>(device: &Device<B>, tokens: &[u32]) -> Tensor<B, 2, Int> {
    let data = TensorData::new(tokens.to_vec(), Shape::new([1, tokens.len()]));
    Tensor::<B, 2, Int>::from_data(data, device)
}

fn build_disallowed_mask<B: Backend>(
    device: &Device<B>,
    start: u32,
    end: u32,
) -> Tensor<B, 1, Bool> {
    let vocab_size = end as usize;
    let mut mask = vec![true; vocab_size];
    for i in start..end {
        mask[i as usize] = false;
    }
    Tensor::<B, 1, Bool>::from_data(TensorData::new(mask, Shape::new([vocab_size])), device)
}

fn sample_next_token<B: Backend>(
    logits: &Tensor<B, 3>,
    disallowed_mask: &Tensor<B, 1, Bool>,
    temperature: f32,
    sampler: &mut Sampler,
) -> Result<u32> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    if batch_size != 1 {
        bail!("unexpected batch size {batch_size}, expected 1");
    }

    let mut next_logits = logits
        .clone()
        .slice([0..batch_size, (seq_len - 1)..seq_len, 0..vocab_size])
        .squeeze_dim(1); // [1, vocab_size]

    // Zero out disallowed tokens by setting them to negative infinity.
    let mask = disallowed_mask.clone().unsqueeze_dim(0); // [1, vocab_size]
    next_logits = next_logits.mask_fill(mask, f32::NEG_INFINITY);

    if temperature > 0.0 && temperature != 1.0 {
        next_logits = next_logits.div_scalar(temperature);
    }

    let probs = softmax(next_logits, 1);
    let sampled = sampler.sample(probs);
    let value = sampled
        .into_data()
        .as_slice::<i64>()
        .map_err(|e: DataError| anyhow!("failed to read sampled token: {e}"))?
        .first()
        .copied()
        .unwrap_or(0);
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn midi_token_constants_match_discussion() {
        assert_eq!(MIDI_TOKEN_START, 128_256);
        assert_eq!(MIDI_TOKEN_COUNT, 55_026);
        assert_eq!(MIDI_TOKEN_END, 183_282);
        assert_eq!(MIDI_BOS_TOKEN, 183_282);
        assert_eq!(MIDI_VOCAB_SIZE, 183_286);
    }

    #[test]
    fn disallowed_mask_blocks_text_tokens() {
        type B = NdArray<f32>;
        let device = Default::default();
        let mask = build_disallowed_mask::<B>(&device, MIDI_TOKEN_START, MIDI_VOCAB_SIZE);
        let data = mask.into_data().to_vec::<bool>().unwrap();
        assert_eq!(data.len(), MIDI_VOCAB_SIZE as usize);
        assert!(data[MIDI_TOKEN_START as usize - 1]);
        assert!(!data[MIDI_TOKEN_START as usize]);
        assert!(!data[MIDI_VOCAB_SIZE as usize - 1]);
    }

    #[test]
    fn midi_llama_config_has_extended_vocab() {
        let config = midi_llama_config(Path::new("/dummy/tokenizer.model"));
        assert_eq!(config.vocab_size, MIDI_VOCAB_SIZE as usize);
        assert_eq!(config.d_model, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
    }
}
