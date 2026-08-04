//! ACE-Step 1.5 5Hz LM planner: a Qwen3-0.6B causal LM (`tie_word_embeddings =
//! true`) that turns a caption plus musical conditions into audio semantic
//! codes.
//!
//! Prompt contract (ACE-Step 1.5):
//!
//! - The chain-of-thought block is a `<think>...</think>` YAML block with the
//!   keys `bpm`, `caption`, `duration`, `keyscale`, `language`,
//!   `timesignature` in that (sorted) order; only present keys are emitted.
//!   Integral `bpm`/`duration` values print as integers, `language` is always
//!   `unknown` (the training value for instrumental tracks), and
//!   `timesignature` is kept verbatim ("4/4", "6/8", ...).
//! - The full prompt embeds that block in the Qwen chat template with the
//!   `# Lyric` section set to `[Instrumental]` (the official placeholder for
//!   instrumental tracks); see [`build_codes_prompt`].
//! - Generation is autoregressive with the official phase-2 knobs:
//!   temperature 0.85, top-p 0.9, top-k disabled, and CFG 2.0 against an
//!   empty unconditional prompt (see [`build_uncond_codes_prompt`]). Sampling
//!   is restricted to the special-token range (>= `<|im_end|>`) and stops at
//!   `<|im_end|>`. Audio codes are `<|audio_code_N|>` tokens with N in
//!   `0..=63999`, produced at roughly 5 tokens per second of audio.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_store::ModuleStore;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::qwen3::{Qwen3Config, Qwen3Model};

/// `<|im_end|>` — end-of-generation stop token for the LM planner.
pub const IM_END_ID: u32 = 151_645;
/// `<|endoftext|>` — base Qwen3 EOS; not the planner's stop token.
pub const ENDOFTEXT_ID: u32 = 151_643;
/// Highest valid audio code index (`<|audio_code_63999|>`).
pub const MAX_AUDIO_CODE: u32 = 63_999;
/// Audio semantic codes per second of audio (the "5Hz" planner rate).
pub const AUDIO_CODES_PER_SECOND: usize = 5;
/// Default sampling temperature from the ACE-Step 1.5 generation config.
pub const DEFAULT_TEMPERATURE: f32 = 0.85;

/// Number of audio codes the planner should emit for `duration_s` seconds.
pub fn code_count_for_duration(duration_s: usize) -> usize {
    duration_s * AUDIO_CODES_PER_SECOND
}

/// Chain-of-thought `<think>` YAML block with sorted keys.
///
/// `bpm` prints as an integer when integral, `duration_s` always does, and a
/// trailing `/4` is stripped from `time_signature`.
pub fn build_cot_block(
    caption: &str,
    bpm: Option<f32>,
    key_scale: Option<&str>,
    time_signature: Option<&str>,
    duration_s: usize,
) -> String {
    let mut lines = Vec::new();
    if let Some(bpm) = bpm {
        if bpm.fract() == 0.0 {
            lines.push(format!("bpm: {}", bpm as i64));
        } else {
            lines.push(format!("bpm: {bpm}"));
        }
    }
    lines.push(format!("caption: {caption}"));
    lines.push(format!("duration: {duration_s}"));
    if let Some(key_scale) = key_scale {
        lines.push(format!("keyscale: {key_scale}"));
    }
    lines.push("language: unknown".to_string());
    if let Some(time_signature) = time_signature {
        lines.push(format!("timesignature: {time_signature}"));
    }
    format!("<think>\n{}\n</think>", lines.join("\n"))
}

/// Unconditional prompt for CFG during code generation: empty user turn and
/// empty `<think>` block (two inner newlines, matching how Qwen's chat
/// template renders empty reasoning).
pub fn build_uncond_codes_prompt() -> String {
    "<|im_start|>system\n\
     # Instruction\n\
     Generate audio semantic tokens based on the given conditions:\n\
     \n\
     <|im_end|>\n\
     <|im_start|>user\n\
     <|im_end|>\n\
     <|im_start|>assistant\n\
     <think>\n\
     \n\
     </think>\n\
     \n"
    .to_string()
}

/// Full LM-planner prompt: Qwen chat template, `# Lyric` section set to
/// `[Instrumental]` (the official placeholder for instrumental tracks —
/// empty lyrics are out-of-distribution), assistant turn pre-filled with
/// the CoT block.
pub fn build_codes_prompt(caption: &str, cot_block: &str) -> String {
    format!(
        "<|im_start|>system\n\
         # Instruction\n\
         Generate audio semantic tokens based on the given conditions:\n\
         \n\
         <|im_end|>\n\
         <|im_start|>user\n\
         # Caption\n\
         {caption}\n\
         \n\
         # Lyric\n\
         [Instrumental]\n\
         <|im_end|>\n\
         <|im_start|>assistant\n\
         {cot_block}\n\
         \n"
    )
}

/// Bidirectional map between `<|audio_code_N|>` token ids and code indices.
///
/// Parsed from the `added_tokens` array of a HuggingFace `tokenizer.json`
/// (tokie does not enumerate added tokens, so this uses `serde_json`
/// directly — robust against tokie API changes).
#[derive(Clone, Debug, Default)]
pub struct AudioCodeVocab {
    code_to_token: HashMap<u32, u32>,
    token_to_code: HashMap<u32, u32>,
}

impl AudioCodeVocab {
    /// `<|im_end|>` — generation stop token.
    pub const IM_END_ID: u32 = IM_END_ID;
    /// `<|endoftext|>`.
    pub const ENDOFTEXT_ID: u32 = ENDOFTEXT_ID;

    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read tokenizer from {}", path.display()))?;
        let data: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse tokenizer from {}", path.display()))?;
        let mut vocab = Self::default();
        if let Some(added) = data.get("added_tokens").and_then(|v| v.as_array()) {
            for entry in added {
                let Some(content) = entry.get("content").and_then(|c| c.as_str()) else {
                    continue;
                };
                let Some(id) = entry.get("id").and_then(|i| i.as_u64()) else {
                    continue;
                };
                let Some(code) = parse_audio_code_content(content) else {
                    continue;
                };
                let id = u32::try_from(id)
                    .with_context(|| format!("token id out of range for {content}"))?;
                vocab.code_to_token.insert(code, id);
                vocab.token_to_code.insert(id, code);
            }
        }
        Ok(vocab)
    }

    /// Token id of `<|audio_code_N|>`, or `None` if unknown.
    pub fn code_token_id(&self, n: u32) -> Option<u32> {
        self.code_to_token.get(&n).copied()
    }

    /// Code index N for the token id of `<|audio_code_N|>`, or `None` when
    /// `id` is not an audio-code token.
    pub fn token_id_to_code(&self, id: u32) -> Option<u32> {
        self.token_to_code.get(&id).copied()
    }

    pub fn len(&self) -> usize {
        self.code_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code_to_token.is_empty()
    }
}

fn parse_audio_code_content(content: &str) -> Option<u32> {
    content
        .strip_prefix("<|audio_code_")?
        .strip_suffix("|>")?
        .parse()
        .ok()
}

/// Sampling knobs for [`AceStepLm::generate_codes`].
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    /// Nucleus sampling threshold (0.9 officially; 1.0 disables).
    pub top_p: f32,
    /// Top-k limit; 0 disables (official default).
    pub top_k: usize,
    /// Classifier-free guidance scale for code generation (2.0 officially;
    /// 1.0 disables the unconditional branch).
    pub cfg_scale: f32,
    pub seed: u64,
}

impl SamplingConfig {
    /// Official ACE-Step 1.5 phase-2 defaults: temperature 0.85, top-p 0.9,
    /// top-k disabled, CFG 2.0.
    pub fn new(max_new_tokens: usize, seed: u64) -> Self {
        Self {
            max_new_tokens,
            temperature: DEFAULT_TEMPERATURE,
            top_p: 0.9,
            top_k: 0,
            cfg_scale: 2.0,
            seed,
        }
    }
}

/// Qwen3ForCausalLM with a tied lm_head: logits are
/// `hidden @ embed_tokens.weight^T`.
#[derive(Debug)]
pub struct AceStepLm<B: Backend> {
    pub model: Qwen3Model<B>,
}

impl<B: Backend> AceStepLm<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            model: Qwen3Model::new(config, device),
        }
    }

    pub fn from_burnpack(config: &Qwen3Config, path: &Path, device: &B::Device) -> Result<Self> {
        Ok(Self {
            model: Qwen3Model::from_burnpack(config, path, device)?,
        })
    }

    /// Load weights from a burnpack file, casting every tensor to the
    /// backend's native float element (e.g. loading the f32 export into an
    /// f16 planner — halves planner VRAM on wgpu).
    pub fn from_burnpack_cast(
        config: &Qwen3Config,
        path: &Path,
        device: &B::Device,
    ) -> Result<Self> {
        let mut model = Self::new(config, device);
        let snapshots = burn_store::BurnpackStore::from_file(path)
            .zero_copy(true)
            .get_all_snapshots()
            .with_context(|| format!("failed to read snapshots from {}", path.display()))?
            .clone();
        let mut converted = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots.values() {
            let data = snapshot
                .to_data()
                .map_err(|e| anyhow::anyhow!("failed to decode {}: {e:?}", snapshot.full_path()))?
                .convert::<B::FloatElem>();
            converted.push(burn_store::TensorSnapshot::from_data(
                data,
                snapshot.path_stack.clone().unwrap_or_default(),
                snapshot.container_stack.clone().unwrap_or_default(),
                snapshot.tensor_id.unwrap_or_default(),
            ));
        }
        let result =
            burn_store::ModuleSnapshot::apply(&mut model.model, converted, None, None, false);
        if !result.is_success() {
            anyhow::bail!(
                "failed to apply LM weights from {}: {result}",
                path.display()
            );
        }
        Ok(model)
    }

    /// Autoregressively generate audio code indices.
    ///
    /// Prefills `prompt_token_ids`, then samples one token per step. When
    /// `sampling.cfg_scale > 1`, the unconditional prompt (empty user turn,
    /// empty `<think>`) is prefilled alongside and logits combine as
    /// `uncond + cfg * (cond - uncond)`, matching the official phase-2
    /// inference. Sampling is restricted to ids >= `<|im_end|>` (the special
    /// tokens range containing the audio codes) with temperature + top-p
    /// (+ optional top-k), and stops at `<|im_end|>` or after
    /// `sampling.max_new_tokens` steps; sampled tokens that are not
    /// `<|audio_code_N|>` are fed back to both branches but not emitted.
    /// `progress`, when given, is called each step with
    /// `(steps_done, max_new_tokens)`.
    pub fn generate_codes(
        &self,
        prompt_token_ids: &[u32],
        uncond_token_ids: Option<&[u32]>,
        vocab: &AudioCodeVocab,
        sampling: &SamplingConfig,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Vec<u32> {
        let device = self.model.embedding_weight().device();
        let mut rng = SmallRng::seed_from_u64(sampling.seed);
        let mut codes = Vec::new();
        let use_cfg = sampling.cfg_scale > 1.0 && uncond_token_ids.is_some();

        let mut cache = self.model.new_cache();
        let prompt_len = prompt_token_ids.len();
        let hidden =
            self.model
                .forward_cached(ids_tensor::<B>(prompt_token_ids, &device), &mut cache, 0);
        let mut logits = self.last_position_logits(hidden);

        let (mut uncond_cache, mut uncond_logits, uncond_len) = if use_cfg {
            let uncond_ids = uncond_token_ids.expect("cfg requires uncond ids");
            let mut cache = self.model.new_cache();
            let hidden =
                self.model
                    .forward_cached(ids_tensor::<B>(uncond_ids, &device), &mut cache, 0);
            let logits = self.last_position_logits(hidden);
            (Some(cache), Some(logits), uncond_ids.len())
        } else {
            (None, None, 0)
        };

        let mut completed = sampling.max_new_tokens;
        // Official phase-2 constraint: only `<|im_end|>` and audio-code tokens
        // are samplable; everything between them is masked out.
        let code_base = vocab.code_token_id(0);
        for step in 0..sampling.max_new_tokens {
            if let Some(cb) = progress.as_mut() {
                cb(step, sampling.max_new_tokens);
            }
            let combined = match (&logits, &uncond_logits) {
                (cond, Some(uncond)) => cond
                    .iter()
                    .zip(uncond.iter())
                    .map(|(c, u)| u + sampling.cfg_scale * (c - u))
                    .collect(),
                (cond, None) => cond.clone(),
            };
            let token = sample_codes_token(&combined, code_base, sampling, &mut rng);
            if token == IM_END_ID {
                completed = step;
                break;
            }
            if let Some(code) = vocab.token_id_to_code(token) {
                codes.push(code);
            }
            let hidden = self.model.forward_cached(
                ids_tensor::<B>(&[token], &device),
                &mut cache,
                prompt_len + step,
            );
            logits = self.last_position_logits(hidden);
            if let (Some(cache), Some(uncond)) = (&mut uncond_cache, &mut uncond_logits) {
                let hidden = self.model.forward_cached(
                    ids_tensor::<B>(&[token], &device),
                    cache,
                    uncond_len + step,
                );
                *uncond = self.last_position_logits(hidden);
            }
        }
        if let Some(cb) = progress.as_mut() {
            cb(completed, sampling.max_new_tokens);
        }
        codes
    }

    /// Tied lm_head logits at the last sequence position, as a flat f32 vec.
    fn last_position_logits(&self, hidden: Tensor<B, 3>) -> Vec<f32> {
        let [batch, seq_len, hidden_size] = hidden.dims();
        let last = hidden
            .slice([0..batch, seq_len - 1..seq_len, 0..hidden_size])
            .reshape([batch, hidden_size]);
        let weight = self.model.embedding_weight();
        let logits = last.matmul(weight.swap_dims(0, 1));
        logits
            .cast(DType::F32)
            .to_data()
            .to_vec::<f32>()
            .expect("lm_head logits should materialize as f32")
    }
}

fn ids_tensor<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let data: Vec<i64> = ids.iter().map(|&t| i64::from(t)).collect();
    let len = data.len();
    Tensor::<B, 2, Int>::from_data(TensorData::new(data, [1, len]), device)
}

/// Restricted sampling over a logit vector for code generation: only
/// `<|im_end|>` and audio-code tokens (`code_base..`) are eligible, matching
/// the official phase-2 FSM mask (`lc[im_end+1 .. code_base] = -inf`).
/// Temperature scales the logits; `top_p < 1` applies nucleus filtering;
/// `top_k > 0` applies an additional top-k cut. Near-zero temperature or
/// `top_k == 1` is argmax.
fn sample_codes_token(
    logits: &[f32],
    code_base: Option<u32>,
    sampling: &SamplingConfig,
    rng: &mut SmallRng,
) -> u32 {
    let code_base = code_base.unwrap_or(IM_END_ID + 1) as usize;
    let mut indices: Vec<usize> = Vec::with_capacity(logits.len() - code_base + 1);
    if logits.len() > IM_END_ID as usize {
        indices.push(IM_END_ID as usize);
        indices.extend(code_base..logits.len());
    } else {
        // Tiny test vocabs fall back to the full range.
        indices.extend(0..logits.len());
    }
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if sampling.temperature < 1e-3 || sampling.top_k == 1 {
        return indices[0] as u32;
    }
    if sampling.top_k > 0 {
        indices.truncate(sampling.top_k.min(indices.len()).max(1));
    }

    let temperature = sampling.temperature.max(f32::EPSILON);
    let max_scaled = logits[indices[0]] / temperature;
    let mut weights: Vec<f32> = indices
        .iter()
        .map(|&i| (logits[i] / temperature - max_scaled).exp())
        .collect();
    let total: f32 = weights.iter().sum();

    if sampling.top_p < 1.0 {
        let mut keep = weights.len();
        let mut acc = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            acc += w;
            if acc / total >= sampling.top_p {
                keep = i + 1;
                break;
            }
        }
        weights.truncate(keep.max(1));
        indices.truncate(keep.max(1));
    }

    let total: f32 = weights.iter().sum();
    let mut draw = rng.random::<f32>() * total;
    for (pos, &w) in weights.iter().enumerate() {
        draw -= w;
        if draw <= 0.0 {
            return indices[pos] as u32;
        }
    }
    *indices.last().expect("candidate set is non-empty") as u32
}

/// BPE-encode a planner prompt with a HuggingFace `tokenizer.json`.
///
/// `add_special_tokens` is `false`: the template already spells out every
/// `<|...|>` marker as text. tokie matches added tokens (including special
/// ones) against the input before BPE — exactly like HuggingFace — so the
/// `<|im_start|>` / `<|im_end|>` / `<|audio_code_N|>` strings in the prompt
/// map to their special ids directly and no manual pre-pass is needed.
pub fn tokenize_prompt(tokenizer_json_path: &Path, text: &str) -> Result<Vec<u32>> {
    let tokenizer = tokie::Tokenizer::from_json(tokenizer_json_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load tokenizer from {}: {e}",
            tokenizer_json_path.display()
        )
    })?;
    Ok(tokenizer.encode(text, false).ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn testdata(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/acestep/testdata")
            .join(name)
    }

    #[test]
    fn cot_block_all_fields() {
        let block = build_cot_block(
            "epic orchestral score",
            Some(120.0),
            Some("C major"),
            Some("4/4"),
            30,
        );
        assert_eq!(
            block,
            "<think>\n\
             bpm: 120\n\
             caption: epic orchestral score\n\
             duration: 30\n\
             keyscale: C major\n\
             language: unknown\n\
             timesignature: 4/4\n\
             </think>"
        );
    }

    #[test]
    fn cot_block_missing_bpm() {
        let block = build_cot_block("lofi beat", None, Some("A minor"), Some("4/4"), 10);
        assert!(!block.contains("bpm:"));
        assert_eq!(
            block,
            "<think>\n\
             caption: lofi beat\n\
             duration: 10\n\
             keyscale: A minor\n\
             language: unknown\n\
             timesignature: 4/4\n\
             </think>"
        );
    }

    #[test]
    fn cot_block_keeps_time_signature_verbatim() {
        let block = build_cot_block("piano", None, None, Some("4/4"), 5);
        assert!(block.contains("timesignature: 4/4\n"));
        let block = build_cot_block("waltz", None, None, Some("6/8"), 5);
        assert!(block.contains("timesignature: 6/8"));
        let block = build_cot_block("march", Some(128.5), None, Some("3/4"), 5);
        assert!(block.contains("timesignature: 3/4\n"));
        assert!(block.contains("bpm: 128.5"));
    }

    #[test]
    fn codes_prompt_byte_exact() {
        let cot = "<think>\ncaption: calm piano\nlanguage: instrumental\n</think>";
        let prompt = build_codes_prompt("calm piano", cot);
        let expected = "<|im_start|>system\n\
             # Instruction\n\
             Generate audio semantic tokens based on the given conditions:\n\
             \n\
             <|im_end|>\n\
             <|im_start|>user\n\
             # Caption\n\
             calm piano\n\
             \n\
             # Lyric\n\
             [Instrumental]\n\
             <|im_end|>\n\
             <|im_start|>assistant\n\
             <think>\n\
             caption: calm piano\n\
             language: instrumental\n\
             </think>\n\
             \n";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn audio_code_vocab_from_tokenizer_json() {
        let vocab = AudioCodeVocab::from_tokenizer_json(&testdata("lm_tokenizer_min.json"))
            .expect("should parse minimal tokenizer.json");
        assert_eq!(vocab.len(), 3);
        assert!(!vocab.is_empty());
        // The fixture's added-token ids are sequential starting at vocab_size
        // (3) because tokie 0.1+ assigns ids from the vocab end for tokens
        // that are not present in the model vocab.
        assert_eq!(vocab.code_token_id(0), Some(6));
        assert_eq!(vocab.code_token_id(1), Some(7));
        assert_eq!(vocab.code_token_id(42), Some(8));
        assert_eq!(vocab.code_token_id(7), None);
        assert_eq!(vocab.token_id_to_code(7), Some(1));
        assert_eq!(vocab.token_id_to_code(101), None);
        assert_eq!(AudioCodeVocab::IM_END_ID, 151_645);
        assert_eq!(AudioCodeVocab::ENDOFTEXT_ID, 151_643);
    }

    #[test]
    fn tokenize_prompt_maps_special_token_strings() {
        let path = testdata("lm_tokenizer_min.json");
        let ids = tokenize_prompt(&path, "<|im_start|>a<|im_end|>")
            .expect("should tokenize with the minimal fixture");
        // tokie 0.1+ assigns <|im_start|> and <|im_end|> the next sequential
        // ids after the three model vocab entries.
        assert_eq!(ids, vec![4, 0, 5]);
    }

    fn tiny_lm() -> AceStepLm<TestBackend> {
        let config = Qwen3Config {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab_size: 128,
            max_position_embeddings: 512,
            tie_word_embeddings: true,
        };
        let device = Default::default();
        AceStepLm::new(&config, &device)
    }

    fn tiny_vocab() -> AudioCodeVocab {
        // Codes 0..8 live at token ids 1..9 of the 128-token toy vocab.
        let mut vocab = AudioCodeVocab::default();
        for code in 0..8u32 {
            vocab.code_to_token.insert(code, code + 1);
            vocab.token_to_code.insert(code + 1, code);
        }
        vocab
    }

    #[test]
    fn generate_codes_deterministic_and_bounded() {
        let lm = tiny_lm();
        let vocab = tiny_vocab();
        let prompt: Vec<u32> = vec![10, 11, 12, 13];

        let mut calls = Vec::new();
        let sampling = SamplingConfig::new(12, 42);
        let first = {
            let mut cb = |done: usize, total: usize| calls.push((done, total));
            lm.generate_codes(&prompt, Some(&prompt), &vocab, &sampling, Some(&mut cb))
        };
        let second = lm.generate_codes(&prompt, Some(&prompt), &vocab, &sampling, None);

        assert_eq!(first, second, "same seed must reproduce the same codes");
        assert!(first.len() <= 12, "cannot exceed the token budget");
        assert!(first.iter().all(|&c| c < 8), "codes stay in the toy vocab");
        assert!(!calls.is_empty(), "progress callback should fire");
        assert!(calls.iter().all(|&(_, total)| total == 12));
        assert_eq!(calls.last(), Some(&(12, 12)), "budget-bound run completes");

        // A different seed almost surely diverges in which steps emit codes.
        let third = lm.generate_codes(&prompt, None, &vocab, &SamplingConfig::new(12, 7), None);
        assert!(third.len() <= 12);
    }
}
