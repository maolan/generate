//! End-to-end instrumental text2music orchestration for ACE-Step 1.5.
//!
//! Ties the ported components together:
//!
//! 1. Qwen3-Embedding text encoder (bidirectional) → caption hidden states.
//! 2. [`AceStepCondition`] → packed conditioning sequence (lyric dummy +
//!    timbre reference from the silence latent + caption).
//! 3. 5Hz LM planner → audio semantic codes → detokenized 25Hz hint latents
//!    (the `is_covers = 1` path — the only path; there is no fallback).
//! 4. Turbo DiT sampler over [`TURBO_TIMESTEPS`] with seeded Gaussian noise.
//! 5. Oobleck VAE decoder → 48 kHz stereo audio.
//!
//! All callbacks use the shared `(phase, fraction, message)` progress shape;
//! phases are `"loading"`, `"text-encoder"`, `"condition"`, `"lm"`, `"dit"`
//! and `"vae"`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleStore};

use super::condition::AceStepCondition;
use super::config::AceStepConfig;
use super::dit::{AceStepDiT, SFT_TIMESTEPS, TURBO_TIMESTEPS};
use super::lm::{self, AceStepLm, AudioCodeVocab, SamplingConfig};
use super::qwen3::{Qwen3Config, Qwen3Model};
use super::vae::{OobleckDecoder, OobleckVaeConfig};

/// Number of 25Hz latent frames used as the timbre reference
/// (`timbre_fix_frame` of the released checkpoint).
const TIMBRE_REFERENCE_FRAMES: usize = 750;

/// Which ACE-Step model variant to load from a model directory.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AceStepVariant {
    /// Turbo DiT (8 steps) + the 0.6B LM planner.
    #[default]
    Turbo,
    /// SFT DiT (50 steps) + the 4B LM planner.
    Sft,
}

impl AceStepVariant {
    fn prefix(self) -> &'static str {
        match self {
            Self::Turbo => "",
            Self::Sft => "sft-",
        }
    }
}

/// Model-file layout of one ACE-Step model directory.
///
/// Shared files (both variants): `qwen3-encoder.bpk`, `qwen3_config.json`,
/// `tokenizer.json`, `acestep-vae.bpk`, `vae_config.json`,
/// `silence_latent.bpk`. Per-variant files live under the variant prefix
/// (`""` for turbo, `"sft-"` for SFT): `<prefix>acestep-lm.bpk`,
/// `<prefix>lm_config.json`, `<prefix>lm_tokenizer.json`,
/// `<prefix>acestep-dit.bpk`, `<prefix>dit_config.json`,
/// `<prefix>acestep-condition.bpk`.
#[derive(Clone, Debug)]
pub struct AceStepModelPaths {
    /// `qwen3-encoder.bpk` — Qwen3-Embedding text encoder weights.
    pub text_encoder_bpk: PathBuf,
    /// `qwen3_config.json` — text encoder config.
    pub text_encoder_config: PathBuf,
    /// `<prefix>acestep-lm.bpk` — 5Hz LM planner weights.
    pub lm_bpk: PathBuf,
    /// `<prefix>lm_config.json` — LM planner config.
    pub lm_config: PathBuf,
    /// `<prefix>lm_tokenizer.json` — LM planner tokenizer (holds the audio codes).
    pub lm_tokenizer: PathBuf,
    /// `<prefix>acestep-dit.bpk` — DiT weights.
    pub dit_bpk: PathBuf,
    /// `<prefix>dit_config.json` — [`AceStepConfig`] (shared by DiT and condition).
    pub dit_config: PathBuf,
    /// `<prefix>acestep-condition.bpk` — condition stack weights.
    pub condition_bpk: PathBuf,
    /// `acestep-vae.bpk` — Oobleck VAE decoder weights.
    pub vae_bpk: PathBuf,
    /// `vae_config.json` — [`OobleckVaeConfig`].
    pub vae_config: PathBuf,
    /// `silence_latent.bpk` — single tensor `silence_latent`, f32, [1, S, 64].
    pub silence_latent_bpk: PathBuf,
    /// `tokenizer.json` — Qwen3-Embedding HF tokenizer.
    pub tokenizer_json: PathBuf,
}

impl AceStepModelPaths {
    /// Model files required inside the model directory, relative to its root.
    /// Used by the downloader to fetch every artifact up front.
    pub fn required_relative_files(variant: AceStepVariant) -> &'static [&'static str] {
        const TURBO: [&str; 12] = [
            "qwen3-encoder.bpk",
            "qwen3_config.json",
            "tokenizer.json",
            "acestep-vae.bpk",
            "vae_config.json",
            "silence_latent.bpk",
            "acestep-lm.bpk",
            "lm_config.json",
            "lm_tokenizer.json",
            "acestep-dit.bpk",
            "dit_config.json",
            "acestep-condition.bpk",
        ];
        const SFT: [&str; 12] = [
            "qwen3-encoder.bpk",
            "qwen3_config.json",
            "tokenizer.json",
            "acestep-vae.bpk",
            "vae_config.json",
            "silence_latent.bpk",
            "sft-acestep-lm.bpk",
            "sft-lm_config.json",
            "sft-lm_tokenizer.json",
            "sft-acestep-dit.bpk",
            "sft-dit_config.json",
            "sft-acestep-condition.bpk",
        ];
        match variant {
            AceStepVariant::Turbo => &TURBO,
            AceStepVariant::Sft => &SFT,
        }
    }

    /// Resolve every required file inside `model_dir` for `variant`, failing
    /// with a clear error that lists the missing relative paths.
    pub fn resolve(model_dir: &Path, variant: AceStepVariant) -> Result<Self> {
        let prefix = variant.prefix();
        let paths = Self {
            text_encoder_bpk: model_dir.join("qwen3-encoder.bpk"),
            text_encoder_config: model_dir.join("qwen3_config.json"),
            lm_bpk: model_dir.join(format!("{prefix}acestep-lm.bpk")),
            lm_config: model_dir.join(format!("{prefix}lm_config.json")),
            lm_tokenizer: model_dir.join(format!("{prefix}lm_tokenizer.json")),
            dit_bpk: model_dir.join(format!("{prefix}acestep-dit.bpk")),
            dit_config: model_dir.join(format!("{prefix}dit_config.json")),
            condition_bpk: model_dir.join(format!("{prefix}acestep-condition.bpk")),
            vae_bpk: model_dir.join("acestep-vae.bpk"),
            vae_config: model_dir.join("vae_config.json"),
            silence_latent_bpk: model_dir.join("silence_latent.bpk"),
            tokenizer_json: model_dir.join("tokenizer.json"),
        };
        let relative = Self::required_relative_files(variant);
        let missing: Vec<&'static str> = relative
            .iter()
            .copied()
            .zip(paths.all_absolute())
            .filter(|(_, absolute)| !absolute.exists())
            .map(|(relative, _)| relative)
            .collect();
        if !missing.is_empty() {
            bail!(
                "ACE-Step model directory {} is missing required files: {}",
                model_dir.display(),
                missing.join(", ")
            );
        }
        Ok(paths)
    }

    fn all_absolute(&self) -> [&PathBuf; 12] {
        [
            &self.text_encoder_bpk,
            &self.text_encoder_config,
            &self.lm_bpk,
            &self.lm_config,
            &self.lm_tokenizer,
            &self.dit_bpk,
            &self.dit_config,
            &self.condition_bpk,
            &self.vae_bpk,
            &self.vae_config,
            &self.silence_latent_bpk,
            &self.tokenizer_json,
        ]
    }
}

/// Loadable wrapper around the single `silence_latent` tensor
/// (`silence_latent.bpk`, f32, shape [1, S, 64]): VAE-encoded silence used as
/// the timbre reference for text2music.
#[derive(Module, Debug)]
pub struct SilenceLatent<B: Backend> {
    pub silence_latent: Param<Tensor<B, 3>>,
}

impl<B: Backend> SilenceLatent<B> {
    pub fn from_burnpack(path: &Path, device: &B::Device) -> Result<Self> {
        // Load the snapshot directly: the frame count S is only known from the
        // file, so shape-validating module load (which needs a pre-built
        // placeholder of the right shape) cannot work here.
        let data = BurnpackStore::from_file(path)
            .zero_copy(true)
            .get_all_snapshots()
            .with_context(|| format!("failed to read snapshots from {}", path.display()))?
            .iter()
            .find_map(|(_, snap)| {
                (snap.full_path() == "silence_latent").then(|| snap.to_data().ok())
            })
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("missing silence_latent tensor in {}", path.display()))?
            .convert::<f32>();
        let tensor = Tensor::<B, 3>::from_data(data, device);
        Ok(Self {
            silence_latent: Param::from_tensor(tensor),
        })
    }

    /// Latent frames tiled/repeated along dim 1 and cropped to exactly
    /// `frames` frames.
    pub fn slice(&self, frames: usize) -> Tensor<B, 3> {
        tile_frames(self.silence_latent.val(), frames)
    }

    /// The timbre reference: the first [`TIMBRE_REFERENCE_FRAMES`] frames,
    /// tiled if the stored latent is shorter.
    pub fn timbre_reference(&self) -> Tensor<B, 3> {
        self.slice(TIMBRE_REFERENCE_FRAMES)
    }
}

/// Metadata describing one generated audio clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerateAudioMeta {
    /// Audio channel count (2 = stereo).
    pub channels: usize,
    /// Total samples per channel.
    pub frames: usize,
    /// Sample rate of the decoded audio (48000 for the released VAE).
    pub sample_rate_hz: u32,
    /// Number of DiT sampler steps that were run.
    pub steps: usize,
    /// Number of caption tokens fed to the text encoder.
    pub prompt_tokens: usize,
}

/// Optional musical metadata conditioning for one generation request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerateMetadata<'a> {
    /// Tempo in beats per minute.
    pub bpm: Option<f32>,
    /// Musical key, e.g. "A minor".
    pub key_scale: Option<&'a str>,
    /// Time signature formatted as "N/D", e.g. "4/4".
    pub time_signature: Option<&'a str>,
}

/// Per-DiT-step statistics collected during a traced run.
#[derive(Clone, Copy, Debug)]
pub struct DitStepStat {
    /// Timestep of this sampler step.
    pub t: f32,
    /// RMS of the latent going into the step.
    pub xt_rms: f32,
    /// RMS of the predicted velocity.
    pub v_rms: f32,
}

/// Phase artifacts collected by [`AceStepPipeline::generate_traced`].
#[derive(Clone, Debug, Default)]
pub struct AceStepTrace {
    /// The exact caption-branch prompt fed to the text encoder.
    pub text_prompt: String,
    /// The lyric-branch prompt (instrumental template).
    pub lyric_prompt: String,
    /// The constructed CoT `<think>` metadata block for the LM.
    pub cot_block: String,
    /// The full LM prompt (chat template + CoT).
    pub lm_prompt: String,
    /// Raw FSQ codes emitted by the LM (before pad/truncate).
    pub codes: Vec<u32>,
    /// Mean of the packed conditioning sequence.
    pub enc_mean: f32,
    /// Standard deviation of the packed conditioning sequence.
    pub enc_std: f32,
    /// RMS of the 25Hz hint latents (LM plan through FSQ + detokenizer).
    pub hints_latent_rms: f32,
    /// Hint latents decoded by the VAE, bypassing the DiT:
    /// `(interleaved_samples, channels, frames)`.
    pub hints_audio: Option<(Vec<f32>, usize, usize)>,
    /// Per-step DiT sampler statistics.
    pub dit_steps: Vec<DitStepStat>,
    /// RMS of the final DiT output latents.
    pub final_latent_rms: f32,
}

/// Fully loaded ACE-Step 1.5 text2music pipeline (instrumental).
///
/// Memory staging: the LM planner is NOT kept resident. Large configurations
/// (e.g. 4B LM in f32 ≈ 17 GB) would not fit in VRAM alongside everything
/// else, so the LM is loaded on demand in [`Self::plan_codes`] and dropped
/// right after use; peak residency stays at max(LM, everything-else).
pub struct AceStepPipeline<B: Backend> {
    /// Resolved model paths (kept for the LM tokenizer path).
    pub paths: AceStepModelPaths,
    /// Device all components live on (also used for the on-demand LM stage).
    pub device: B::Device,
    /// Qwen3-Embedding text encoder (bidirectional).
    pub text_encoder: Qwen3Model<B>,
    /// Condition stack (text projector, lyric/timbre encoders, detokenizer).
    pub condition: AceStepCondition<B>,
    /// Turbo DiT decoder.
    pub dit: AceStepDiT<B>,
    /// Oobleck VAE decoder.
    pub vae: OobleckDecoder<B>,
    /// VAE-encoded silence used as the timbre reference.
    pub silence: SilenceLatent<B>,
    /// DiT/condition config.
    pub dit_config: AceStepConfig,
    /// VAE config.
    pub vae_config: OobleckVaeConfig,
    /// `<|audio_code_N|>` token map parsed from `lm_tokenizer.json`.
    pub audio_code_vocab: AudioCodeVocab,
    /// Load the LM planner in f16 instead of the pipeline's float type.
    /// EXPERIMENTAL: currently broken — cubecl's f16 kernels do not upcast
    /// intermediates, and the large activations of these models overflow
    /// f16 (max ±65504), producing all-NaN logits and a degenerate
    /// constant-code plan. Kept for future use; stays off by default.
    pub lm_f16: bool,
    text_tokenizer: tokie::Tokenizer,
}

impl<B: Backend> AceStepPipeline<B> {
    /// Load every component except the LM planner (see the struct docs for
    /// the staging rationale), reporting `(loading, fraction, component)`
    /// along the way.
    pub fn load(
        paths: &AceStepModelPaths,
        device: &B::Device,
        progress: &mut dyn FnMut(&str, f32, &str),
    ) -> Result<Self> {
        progress("loading", 0.0, "text encoder config");
        let text_config = Qwen3Config::load(&paths.text_encoder_config)?;
        progress("loading", 0.1, "text encoder weights");
        let text_encoder =
            Qwen3Model::from_burnpack(&text_config, &paths.text_encoder_bpk, device)?;

        progress("loading", 0.4, "dit config");
        let dit_config = AceStepConfig::load(&paths.dit_config)?;
        progress("loading", 0.5, "condition stack weights");
        let condition = AceStepCondition::from_burnpack(&dit_config, &paths.condition_bpk, device)?;
        progress("loading", 0.7, "dit weights");
        let dit = AceStepDiT::from_burnpack(&dit_config, &paths.dit_bpk, device)?;

        progress("loading", 0.85, "vae config");
        let vae_config = OobleckVaeConfig::load(&paths.vae_config)?;
        progress("loading", 0.9, "vae decoder weights");
        let vae = OobleckDecoder::from_burnpack(&vae_config, &paths.vae_bpk, device)?;

        progress("loading", 0.95, "silence latent");
        let silence = SilenceLatent::from_burnpack(&paths.silence_latent_bpk, device)?;

        progress("loading", 0.98, "tokenizers");
        let text_tokenizer = tokie::Tokenizer::from_json(&paths.tokenizer_json).map_err(|e| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {e}",
                paths.tokenizer_json.display()
            )
        })?;
        let audio_code_vocab = AudioCodeVocab::from_tokenizer_json(&paths.lm_tokenizer)?;

        progress("loading", 1.0, "ready");
        Ok(Self {
            paths: paths.clone(),
            device: device.clone(),
            text_encoder,
            condition,
            dit,
            vae,
            silence,
            dit_config,
            vae_config,
            audio_code_vocab,
            lm_f16: false,
            text_tokenizer,
        })
    }

    /// Run the LM planner stage: load the 5Hz LM, generate FSQ audio codes
    /// for `caption` + `metadata`, then drop the LM and release its memory.
    ///
    /// Returns `(codes, cot_block, lm_prompt)` — the latter two feed the
    /// trace. Budget and count follow the official `duration * 5` codes.
    pub fn plan_codes(
        &self,
        caption: &str,
        metadata: &GenerateMetadata<'_>,
        duration_ms: usize,
        seed: u64,
        progress: &mut dyn FnMut(&str, f32, &str),
    ) -> Result<(Vec<u32>, String, String)> {
        let duration_s = duration_ms / 1000;
        let cot = lm::build_cot_block(
            caption,
            metadata.bpm,
            metadata.key_scale,
            metadata.time_signature,
            duration_s,
        );
        let prompt = lm::build_codes_prompt(caption, &cot);
        let prompt_ids = lm::tokenize_prompt(&self.paths.lm_tokenizer, &prompt)?;
        let sampling = SamplingConfig::new(lm::code_count_for_duration(duration_s) * 6, seed);
        let uncond_prompt_ids = if sampling.cfg_scale > 1.0 {
            Some(lm::tokenize_prompt(
                &self.paths.lm_tokenizer,
                &lm::build_uncond_codes_prompt(),
            )?)
        } else {
            None
        };

        progress("loading", 0.0, "lm planner config");
        let lm_config = Qwen3Config::load(&self.paths.lm_config)?;

        let use_f16 = self.lm_f16
            && B::name(&self.device).contains("wgpu")
            && std::env::var_os("MAOLAN_ACESTEP_LM_F32").is_none();
        let codes = if use_f16 {
            progress("loading", 0.05, "lm planner weights (f16)");
            let device = Default::default();
            let lm =
                AceStepLm::<burn::backend::Wgpu<burn::tensor::f16, i64, u32>>::from_burnpack_cast(
                    &lm_config,
                    &self.paths.lm_bpk,
                    &device,
                )?;
            let codes = {
                let mut lm_progress = |done: usize, total: usize| {
                    let fraction = if total == 0 {
                        1.0
                    } else {
                        done as f32 / total as f32
                    };
                    progress("lm", fraction, "planning audio codes");
                };
                lm.generate_codes(
                    &prompt_ids,
                    uncond_prompt_ids.as_deref(),
                    &self.audio_code_vocab,
                    &sampling,
                    Some(&mut lm_progress),
                )
            };
            drop(lm);
            burn::backend::Wgpu::<burn::tensor::f16, i64, u32>::memory_cleanup(&device);
            codes
        } else {
            progress("loading", 0.05, "lm planner weights");
            let lm = AceStepLm::<B>::from_burnpack(&lm_config, &self.paths.lm_bpk, &self.device)?;
            let codes = {
                let mut lm_progress = |done: usize, total: usize| {
                    let fraction = if total == 0 {
                        1.0
                    } else {
                        done as f32 / total as f32
                    };
                    progress("lm", fraction, "planning audio codes");
                };
                lm.generate_codes(
                    &prompt_ids,
                    uncond_prompt_ids.as_deref(),
                    &self.audio_code_vocab,
                    &sampling,
                    Some(&mut lm_progress),
                )
            };
            drop(lm);
            B::memory_cleanup(&self.device);
            codes
        };
        Ok((codes, cot, prompt))
    }

    /// Generate an instrumental clip from a caption.
    ///
    /// - `metadata` carries optional BPM / key-scale / time-signature values,
    ///   appended to the conditioning caption and fed to the LM planner.
    /// - `duration_ms` is the target clip length in milliseconds.
    /// - `seed` drives both the LM sampling and the initial DiT noise, so a
    ///   (caption, seed) pair is deterministic.
    ///
    /// Returns `[1, channels, samples]` audio plus its [`GenerateAudioMeta`].
    pub fn generate(
        &self,
        caption: &str,
        metadata: &GenerateMetadata<'_>,
        duration_ms: usize,
        seed: u64,
        progress: &mut dyn FnMut(&str, f32, &str),
    ) -> Result<(Tensor<B, 3>, GenerateAudioMeta)> {
        self.generate_impl(caption, metadata, duration_ms, seed, progress, None)
    }

    /// Like [`Self::generate`], but collects per-phase artifacts into `trace`
    /// for debugging and verification (see [`AceStepTrace`]).
    pub fn generate_traced(
        &self,
        caption: &str,
        metadata: &GenerateMetadata<'_>,
        duration_ms: usize,
        seed: u64,
        progress: &mut dyn FnMut(&str, f32, &str),
        trace: &mut AceStepTrace,
    ) -> Result<(Tensor<B, 3>, GenerateAudioMeta)> {
        self.generate_impl(caption, metadata, duration_ms, seed, progress, Some(trace))
    }

    fn generate_impl(
        &self,
        caption: &str,
        metadata: &GenerateMetadata<'_>,
        duration_ms: usize,
        seed: u64,
        progress: &mut dyn FnMut(&str, f32, &str),
        mut trace: Option<&mut AceStepTrace>,
    ) -> Result<(Tensor<B, 3>, GenerateAudioMeta)> {
        let acoustic_dim = self.dit_config.audio_acoustic_hidden_dim;
        let latent_frames = latent_frames_for_duration(duration_ms);
        let duration_s = duration_ms / 1000;
        let device = self.silence.silence_latent.device();

        // Text encoding, matching the official training/inference format:
        // the DiT caption branch consumes SFT_GEN_PROMPT (instruction +
        // caption + metadata block) embedded by Qwen3-Embedding (max 256
        // tokens, causal forward, explicit EOS appended); the lyric branch is
        // a RAW embed_tokens lookup of the "# Languages / # Lyric" template
        // with [Instrumental] lyrics (max 2048 tokens + EOS).
        let metas_block = build_metas_block(
            metadata.bpm,
            metadata.key_scale,
            metadata.time_signature,
            duration_s,
        );
        progress("text-encoder", 0.0, "tokenizing caption");
        let text_prompt = build_dit_text_prompt(caption, &metas_block);
        if let Some(trace) = trace.as_deref_mut() {
            trace.text_prompt = text_prompt.clone();
            trace.lyric_prompt = INSTRUMENTAL_LYRIC_PROMPT.to_string();
        }
        let mut ids = self.text_tokenizer.encode(&text_prompt, false).ids;
        ids.truncate(MAX_TEXT_TOKENS);
        // The official appends an explicit EOS (their pipeline encodes with
        // add_eos=true) — the DiT cross-attention sees it as a normal token.
        ids.push(lm::ENDOFTEXT_ID);
        let prompt_tokens = ids.len();
        let ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
        let ids_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, prompt_tokens]), &device);
        progress("text-encoder", 0.3, "encoding caption");
        let text_hidden = self.text_encoder.forward(ids_tensor, true);

        let mut lyric_ids = self
            .text_tokenizer
            .encode(INSTRUMENTAL_LYRIC_PROMPT, false)
            .ids;
        lyric_ids.truncate(MAX_LYRIC_TOKENS);
        // The official appends an explicit EOS after the lyric template
        // (their pipeline encodes with add_eos=true).
        lyric_ids.push(lm::ENDOFTEXT_ID);
        let lyric_len = lyric_ids.len();
        let lyric_ids: Vec<i64> = lyric_ids.into_iter().map(i64::from).collect();
        let lyric_ids_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(lyric_ids, [1, lyric_len]), &device);
        // Lyric branch: RAW embedding-table lookup of the lyric tokens — the
        // official `infer_lyric_embeddings` is `text_encoder.embed_tokens(ids)`
        // with NO transformer forward.
        let lyric_hidden = self.text_encoder.embed_tokens.forward(lyric_ids_tensor);
        let lyric_mask = Tensor::<B, 2, Int>::ones([1, lyric_len], &device);
        progress("text-encoder", 1.0, "caption encoded");

        progress("condition", 0.0, "encoding conditions");
        let enc = self.condition.encode(
            text_hidden,
            lyric_hidden,
            lyric_mask,
            self.silence.timbre_reference(),
        );
        progress("condition", 1.0, "conditions encoded");
        if let Some(trace) = trace.as_deref_mut() {
            let values: Vec<f32> = enc
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec()
                .map_err(|e| anyhow::anyhow!("failed to read encoder states: {e}"))?;
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let var =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
            trace.enc_mean = mean;
            trace.enc_std = var.sqrt();
        }

        // LM plan: caption + CoT → 5Hz audio codes → 25Hz hint latents.
        // Diagnostic override: MAOLAN_ACESTEP_CODES="1,2,3" skips the LM and
        // uses the given codes directly (for oracle comparisons).
        let codes = if let Some(raw) = std::env::var_os("MAOLAN_ACESTEP_CODES") {
            raw.to_string_lossy()
                .split(',')
                .filter_map(|part| part.trim().parse::<u32>().ok())
                .collect::<Vec<u32>>()
        } else {
            let (codes, cot, prompt) =
                self.plan_codes(caption, metadata, duration_ms, seed, &mut *progress)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.cot_block = cot;
                trace.lm_prompt = prompt;
                trace.codes = codes.clone();
            }
            codes
        };
        // The LM is expected to emit 5 codes per second; clamp to exactly that
        // (at least one code so the detokenizer never sees an empty input).
        let expected = lm::code_count_for_duration(duration_s).max(1);
        let codes = pad_or_truncate_codes(codes, expected);
        let codes_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(codes, [1, expected]), &device);
        let hints = self.condition.codes_to_hints(codes_tensor);
        // Short hints are padded with silence (the official `silence_latent`
        // filler), long ones cropped — never tiled.
        let [_, hint_frames, _] = hints.dims();
        let src_latents = if hint_frames >= latent_frames {
            hints.narrow(1, 0, latent_frames)
        } else {
            let padding = self.silence.slice(latent_frames - hint_frames);
            Tensor::cat(vec![hints, padding], 1)
        };

        if let Some(trace) = trace.as_deref_mut() {
            trace.hints_latent_rms = tensor_rms(&src_latents)?;
            // Decode the hint latents directly through the VAE, bypassing the
            // DiT: if this sounds like music, LM + FSQ + detokenizer work.
            progress("vae", 0.0, "decoding hint latents (trace)");
            let hints_audio = self.vae.decode(src_latents.clone());
            let interleaved = interleave_channel_major(hints_audio)?;
            trace.hints_audio = Some(interleaved);
            progress("vae", 0.3, "hint latents decoded (trace)");
        }

        // Diagnostic override: MAOLAN_ACESTEP_NOCODES=1 replaces the LM hint
        // latents with silence (the is_covers=0 base text2music task), taking
        // the LM plan out of the equation for DiT verification.
        let src_latents = if std::env::var_os("MAOLAN_ACESTEP_NOCODES").is_some() {
            self.silence.slice(latent_frames)
        } else {
            src_latents
        };

        // DiT context: src latents ++ all-ones chunk mask.
        let chunk_mask = Tensor::ones([1, latent_frames, acoustic_dim], &device);
        let context = Tensor::cat(vec![src_latents, chunk_mask], 2);

        // Seeded initial noise (splitmix64 + Box-Muller, same scheme as the
        // HeartMuLa decoder latent init).
        let noise = seeded_latent_noise(seed, latent_frames, acoustic_dim, &device);

        // Sampler schedule: turbo checkpoints use the distilled 8-step
        // shift-3.0 schedule; base/SFT checkpoints use 50 uniform steps
        // (shift 1.0). CFG stays off in both cases (official guidance 1.0).
        let timesteps: &[f32] = if self.dit_config.is_turbo {
            &TURBO_TIMESTEPS
        } else {
            &SFT_TIMESTEPS
        };

        let latents = if let Some(trace) = trace.as_deref_mut() {
            // Same explicit-Euler loop as `AceStepDiT::sample_turbo`, with
            // per-step xt/v statistics for the trace.
            let kv = self.dit.prepare_cross_kv(enc);
            let total = timesteps.len();
            let mut xt = noise;
            for (index, &t_cur) in timesteps.iter().enumerate() {
                let v = self
                    .dit
                    .forward_with_kv(xt.clone(), t_cur, context.clone(), &kv);
                trace.dit_steps.push(DitStepStat {
                    t: t_cur,
                    xt_rms: tensor_rms(&xt)?,
                    v_rms: tensor_rms(&v)?,
                });
                let dt = if index + 1 == total {
                    t_cur
                } else {
                    t_cur - timesteps[index + 1]
                };
                xt = xt - v * dt;
                progress("dit", (index + 1) as f32 / total as f32, "diffusing");
            }
            xt
        } else {
            let mut dit_progress = |done: usize, total: usize| {
                progress("dit", done as f32 / total as f32, "diffusing");
            };
            self.dit
                .sample_turbo(noise, context, enc, timesteps, Some(&mut dit_progress))
        };

        if let Some(trace) = trace {
            trace.final_latent_rms = tensor_rms(&latents)?;
        }

        progress("vae", 0.0, "decoding audio");
        let audio = self.vae.decode(latents);
        progress("vae", 1.0, "audio decoded");

        let [_, channels, frames] = audio.dims();
        let meta = GenerateAudioMeta {
            channels,
            frames,
            sample_rate_hz: self.vae_config.sampling_rate as u32,
            steps: timesteps.len(),
            prompt_tokens,
        };
        Ok((audio, meta))
    }
}

/// 25Hz latent frame count for a clip of `duration_ms` milliseconds
/// (rounded, at least 5 frames).
fn latent_frames_for_duration(duration_ms: usize) -> usize {
    ((duration_ms * 25 + 500) / 1000).max(5)
}

/// Host-side RMS of a rank-3 tensor.
fn tensor_rms<B: Backend>(tensor: &Tensor<B, 3>) -> Result<f32> {
    let values: Vec<f32> = tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("failed to read tensor: {e}"))?;
    Ok((values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt())
}

/// Convert `[1, channels, frames]` audio into interleaved host samples:
/// `(interleaved, channels, frames)`.
fn interleave_channel_major<B: Backend>(audio: Tensor<B, 3>) -> Result<(Vec<f32>, usize, usize)> {
    let [_, channels, frames] = audio.dims();
    let channel_major: Vec<f32> = audio
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("failed to read audio tensor: {e}"))?;
    let mut interleaved = vec![0.0_f32; channel_major.len()];
    for (channel, samples) in channel_major.chunks_exact(frames).enumerate() {
        for (frame, sample) in samples.iter().enumerate() {
            interleaved[frame * channels + channel] = *sample;
        }
    }
    Ok((interleaved, channels, frames))
}

/// Max tokens for the DiT caption branch (official truncation).
const MAX_TEXT_TOKENS: usize = 256;
/// Max tokens for the lyric branch (official truncation).
const MAX_LYRIC_TOKENS: usize = 2048;

/// The DiT instruction used when generating from LM audio codes
/// (`DIT_INSTR_COVER` in the official pipeline: with codes the task is the
/// "cover" task — `use_source_context ? DIT_INSTR_COVER : DIT_INSTR_TEXT2MUSIC`,
/// and our only path is codes-driven).
pub const DIT_INSTRUCTION: &str = "Generate audio semantic tokens based on the given conditions:";

/// Lyric-branch prompt for instrumental generation: the official
/// `# Languages / # Lyric` template with the `[Instrumental]` placeholder
/// and language `unknown` (the training distribution for instrumental
/// tracks — empty lyrics are out-of-distribution).
pub const INSTRUMENTAL_LYRIC_PROMPT: &str =
    "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>";

/// Official metadata block (`_dict_to_meta_string`): always four lines,
/// `"N/A"` for missing values, duration as `"N seconds"`.
pub fn build_metas_block(
    bpm: Option<f32>,
    key_scale: Option<&str>,
    time_signature: Option<&str>,
    duration_s: usize,
) -> String {
    let bpm = match bpm {
        Some(bpm) if bpm.fract() == 0.0 => format!("{}", bpm as i64),
        Some(bpm) => format!("{bpm}"),
        None => "N/A".to_string(),
    };
    let time_signature = time_signature.unwrap_or("N/A");
    let key_scale = key_scale.unwrap_or("N/A");
    format!(
        "- bpm: {bpm}\n- timesignature: {time_signature}\n- keyscale: {key_scale}\n- duration: {duration_s} seconds\n"
    )
}

/// Official caption-branch prompt (`SFT_GEN_PROMPT` with
/// `DEFAULT_DIT_INSTRUCTION`), consumed by the Qwen3 text encoder.
pub fn build_dit_text_prompt(caption: &str, metas_block: &str) -> String {
    format!(
        "# Instruction\n{DIT_INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{metas_block}<|endoftext|>\n"
    )
}

/// Clamp LM-emitted codes to exactly `expected` entries: pad short outputs by
/// repeating the last code (or zeros when empty), truncate long ones.
fn pad_or_truncate_codes(mut codes: Vec<u32>, expected: usize) -> Vec<u32> {
    if codes.len() < expected {
        let fill = codes.last().copied().unwrap_or(0);
        codes.resize(expected, fill);
    } else {
        codes.truncate(expected);
    }
    codes
}

/// Tile/repeat a `[1, T, D]` latent along dim 1 and crop to exactly `frames`
/// frames. Panics on a zero-length input (callers guarantee `T >= 1`).
pub fn tile_frames<B: Backend>(tensor: Tensor<B, 3>, frames: usize) -> Tensor<B, 3> {
    let len = tensor.dims()[1];
    assert!(len > 0, "cannot tile an empty latent");
    if len == frames {
        return tensor;
    }
    let repeats = frames.div_ceil(len);
    let tiled = if repeats > 1 {
        tensor.repeat_dim(1, repeats)
    } else {
        tensor
    };
    tiled.narrow(1, 0, frames)
}

/// Seeded standard-normal latent of shape `[1, frames, channels]`, using the
/// splitmix64 + Box-Muller scheme from the HeartMuLa runtime so the same seed
/// reproduces the same noise on every backend.
pub fn seeded_latent_noise<B: Backend>(
    seed: u64,
    frames: usize,
    channels: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let data = generate_gaussian_data(seed, frames * channels);
    Tensor::<B, 3>::from_data(TensorData::new(data, [1, frames, channels]), device)
}

fn generate_gaussian_data(seed: u64, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut state = seed;
    while out.len() < len {
        let u1 = uniform01_open(&mut state);
        let u2 = uniform01_open(&mut state);
        let radius = (-2.0_f64 * u1.ln()).sqrt();
        let theta = 2.0_f64 * std::f64::consts::PI * u2;
        out.push((radius * theta.cos()) as f32);
        if out.len() < len {
            out.push((radius * theta.sin()) as f32);
        }
    }
    out
}

fn uniform01_open(state: &mut u64) -> f64 {
    let value = splitmix64_next(state);
    let mantissa = (value >> 11) as f64;
    ((mantissa + 0.5) / ((1_u64 << 53) as f64)).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn metas_block_matches_official_format() {
        assert_eq!(
            build_metas_block(Some(120.0), Some("A minor"), Some("4/4"), 10),
            "- bpm: 120\n- timesignature: 4/4\n- keyscale: A minor\n- duration: 10 seconds\n"
        );
        assert_eq!(
            build_metas_block(None, None, None, 30),
            "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n"
        );
        assert_eq!(
            build_metas_block(Some(128.5), None, Some("6/8"), 5),
            "- bpm: 128.5\n- timesignature: 6/8\n- keyscale: N/A\n- duration: 5 seconds\n"
        );
    }

    #[test]
    fn dit_text_prompt_matches_official_sft_format() {
        let metas = build_metas_block(Some(120.0), Some("A minor"), Some("4/4"), 10);
        assert_eq!(
            build_dit_text_prompt("dark techno", &metas),
            "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n# Caption\ndark techno\n\n# Metas\n- bpm: 120\n- timesignature: 4/4\n- keyscale: A minor\n- duration: 10 seconds\n<|endoftext|>\n"
        );
    }

    #[test]
    fn instrumental_lyric_prompt_matches_official_format() {
        assert_eq!(
            INSTRUMENTAL_LYRIC_PROMPT,
            "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>"
        );
    }

    #[test]
    fn latent_frame_count_math() {
        assert_eq!(latent_frames_for_duration(0), 5, "minimum of 5 frames");
        assert_eq!(latent_frames_for_duration(100), 5);
        assert_eq!(latent_frames_for_duration(1000), 25);
        assert_eq!(latent_frames_for_duration(30_000), 750);
        // Rounds to nearest: 1020ms * 25 = 25.5 -> 26.
        assert_eq!(latent_frames_for_duration(1020), 26);
    }

    #[test]
    fn pad_or_truncate_codes_behaviour() {
        assert_eq!(pad_or_truncate_codes(vec![], 3), vec![0, 0, 0]);
        assert_eq!(pad_or_truncate_codes(vec![7, 9], 4), vec![7, 9, 9, 9]);
        assert_eq!(pad_or_truncate_codes(vec![1, 2, 3, 4], 2), vec![1, 2]);
        assert_eq!(pad_or_truncate_codes(vec![5], 1), vec![5]);
    }

    #[test]
    fn tile_frames_tiles_and_crops() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            [[
                [1.0, 10.0, 100.0, 1000.0],
                [2.0, 20.0, 200.0, 2000.0],
                [3.0, 30.0, 300.0, 3000.0],
            ]],
            &device,
        );

        // Tiling: 3 frames -> 7 frames repeats 0,1,2,0,1,2,0.
        let tiled = tile_frames(tensor.clone(), 7);
        assert_eq!(tiled.dims(), [1, 7, 4]);
        let values = tiled.to_data().to_vec::<f32>().expect("tiled values");
        for frame in 0..7 {
            let source = frame % 3;
            assert_eq!(values[frame * 4], (source + 1) as f32);
        }

        // Cropping: 3 frames -> 2 frames keeps the prefix.
        let cropped = tile_frames(tensor.clone(), 2);
        assert_eq!(cropped.dims(), [1, 2, 4]);
        let values = cropped.to_data().to_vec::<f32>().expect("cropped values");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[4], 2.0);

        // Exact length is returned unchanged.
        assert_eq!(tile_frames(tensor, 3).dims(), [1, 3, 4]);
    }

    #[test]
    fn silence_latent_slice_and_timbre_reference_tile() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            [[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]],
            &device,
        );
        let silence = SilenceLatent::<TestBackend> {
            silence_latent: Param::from_tensor(tensor),
        };

        let sliced = silence.slice(5);
        assert_eq!(sliced.dims(), [1, 5, 4]);
        let values = sliced.to_data().to_vec::<f32>().expect("slice values");
        // Frames repeat 0,1,2,0,1.
        assert_eq!(&values[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&values[12..16], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&values[16..20], &[5.0, 6.0, 7.0, 8.0]);

        let reference = silence.timbre_reference();
        assert_eq!(reference.dims(), [1, TIMBRE_REFERENCE_FRAMES, 4]);
        let values = reference
            .to_data()
            .to_vec::<f32>()
            .expect("reference values");
        // Frame 750 wraps back to frame 0 (750 % 3 == 0).
        let last = &values[(TIMBRE_REFERENCE_FRAMES - 1) * 4..];
        assert_eq!(last, &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn seeded_noise_is_deterministic_and_normal_shaped() {
        let device = Default::default();
        let first = seeded_latent_noise::<TestBackend>(42, 6, 4, &device);
        let second = seeded_latent_noise::<TestBackend>(42, 6, 4, &device);
        assert_eq!(first.dims(), [1, 6, 4]);
        let a = first.to_data().to_vec::<f32>().expect("first noise");
        let b = second.to_data().to_vec::<f32>().expect("second noise");
        assert_eq!(a, b, "same seed must reproduce the same noise");
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(
            a.iter().any(|v| v.abs() > 1e-6),
            "noise must not be all zeros"
        );

        let other = seeded_latent_noise::<TestBackend>(7, 6, 4, &device)
            .to_data()
            .to_vec::<f32>()
            .expect("other noise");
        assert_ne!(a, other, "different seeds must diverge");
    }
}
