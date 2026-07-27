//! ACE-Step 1.5 condition encoder stack + audio token detokenizer, ported to Burn.
//!
//! Covers the lyric encoder, timbre encoder, condition assembly (`pack_sequences`),
//! and the 5Hz-codes → 25Hz-latents detokenizer from
//! `modeling_acestep_v15_turbo.py`. All attention is BIDIRECTIONAL; layers whose
//! global `layer_types` entry is "sliding_attention" restrict each query to keys
//! with `|i - j| <= sliding_window`. RoPE (theta 1e6, duplicated halves) is applied
//! post q/k-norm; q/k/v/o and MLP projections carry no bias. The encoders honor the
//! lyric key-padding mask; the detokenizer and timbre encoder use no padding mask.
//!
//! The checkpoint's `timbre_encoder.special_token` is dead code upstream (its
//! prepend is commented out) and is intentionally NOT modeled here — the converter
//! drops it. Pooled timbre output is position 0 of the encoder output.
//!
//! Canonical burnpack tensor names (= module paths of `AceStepCondition`, what the
//! offline converter writes into the condition `.bpk`):
//!
//! - `text_projector.weight`                                   — Linear(1024→2048), no bias
//! - `lyric_encoder.embed_tokens.{weight,bias}`                — Linear(1024→2048)
//! - `lyric_encoder.norm.weight`
//! - `lyric_encoder.layers.{0..7}.input_layernorm.weight`
//! - `lyric_encoder.layers.{i}.post_attention_layernorm.weight`
//! - `lyric_encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! - `lyric_encoder.layers.{i}.self_attn.{q,k}_norm.weight`
//! - `lyric_encoder.layers.{i}.mlp.{gate,up,down}_proj.weight`
//! - `timbre_encoder.embed_tokens.{weight,bias}`               — Linear(64→2048)
//! - `timbre_encoder.norm.weight`
//! - `timbre_encoder.layers.{0..3}.<same per-layer keys>`      (NO `special_token`)
//! - `detokenizer.embed_tokens.{weight,bias}`                  — Linear(2048→2048)
//! - `detokenizer.special_tokens`                              — [1, 5, 2048]
//! - `detokenizer.norm.weight`
//! - `detokenizer.layers.{0..1}.<same per-layer keys>`
//! - `detokenizer.proj_out.{weight,bias}`                      — Linear(2048→64)
//! - `quantizer.project_in.{weight,bias}`                      — Linear(2048→6)
//! - `quantizer.project_out.{weight,bias}`                     — Linear(6→2048)

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, LinearLayout};
use burn::prelude::Backend;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

use crate::acestep::config::AceStepConfig;
use crate::acestep::fsq::ResidualFsq;

/// Weight-only RMSNorm (Qwen3 style): `x * rsqrt(mean(x^2) + eps) * weight`,
/// with the variance computed in fp32.
#[derive(Module, Debug)]
pub struct ConditionRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend> ConditionRmsNorm<B> {
    pub fn new(device: &B::Device, hidden_size: usize, epsilon: f64) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::<B, 1>::ones([hidden_size], device)),
            epsilon,
        }
    }

    pub fn forward<const D: usize>(&self, hidden: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = hidden.dtype();
        let rms = (hidden.clone().cast(DType::F32).square().mean_dim(D - 1) + self.epsilon).sqrt();
        (hidden / rms.cast(dtype)) * self.weight.val().unsqueeze()
    }
}

#[derive(Clone, Debug)]
struct AttentionMeta {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f32,
    /// `Some(window)` for sliding-attention layers (`|i - j| <= window`),
    /// `None` for full bidirectional attention.
    sliding_window: Option<usize>,
}

/// Bidirectional self-attention with per-head q/k RMSNorm, post-norm RoPE and GQA.
#[derive(Module, Debug)]
pub struct AceStepAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: ConditionRmsNorm<B>,
    pub k_norm: ConditionRmsNorm<B>,
    #[module(skip)]
    meta: AttentionMeta,
}

impl<B: Backend> AceStepAttention<B> {
    pub fn new(config: &AceStepConfig, layer_idx: usize, device: &B::Device) -> Self {
        let head_dim = config.head_dim;
        let sliding_window = config
            .is_sliding_layer(layer_idx)
            .then_some(config.sliding_window);
        Self {
            q_proj: linear_no_bias(
                device,
                config.hidden_size,
                config.num_attention_heads * head_dim,
            ),
            k_proj: linear_no_bias(
                device,
                config.hidden_size,
                config.num_key_value_heads * head_dim,
            ),
            v_proj: linear_no_bias(
                device,
                config.hidden_size,
                config.num_key_value_heads * head_dim,
            ),
            o_proj: linear_no_bias(
                device,
                config.num_attention_heads * head_dim,
                config.hidden_size,
            ),
            q_norm: ConditionRmsNorm::new(device, head_dim, config.rms_norm_eps),
            k_norm: ConditionRmsNorm::new(device, head_dim, config.rms_norm_eps),
            meta: AttentionMeta {
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim,
                scaling: (head_dim as f32).powf(-0.5),
                sliding_window,
            },
        }
    }

    /// `hidden`: [B, L, hidden]; `cos`/`sin`: [1, L, 1, head_dim];
    /// `additive_mask`: optional [B, 1, L, L] with 0 for allowed and a large
    /// negative value for disallowed query/key pairs.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        additive_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden.dims();
        let num_heads = self.meta.num_heads;
        let num_kv_heads = self.meta.num_kv_heads;
        let head_dim = self.meta.head_dim;

        let q = self
            .q_proj
            .forward(hidden.clone())
            .reshape([batch, seq_len, num_heads, head_dim]);
        let k =
            self.k_proj
                .forward(hidden.clone())
                .reshape([batch, seq_len, num_kv_heads, head_dim]);
        let v = self
            .v_proj
            .forward(hidden)
            .reshape([batch, seq_len, num_kv_heads, head_dim]);

        // Per-head RMSNorm on the head dim, RoPE applied after the norm.
        let q = apply_rotary_pos_emb(self.q_norm.forward(q), cos, sin).swap_dims(1, 2);
        let k = apply_rotary_pos_emb(self.k_norm.forward(k), cos, sin).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = if num_heads != num_kv_heads {
            let repeats = num_heads / num_kv_heads;
            (repeat_kv(k, repeats), repeat_kv(v, repeats))
        } else {
            (k, v)
        };

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.meta.scaling);
        let scores = match additive_mask {
            Some(mask) => scores + mask,
            None => scores,
        };
        // Softmax in fp32, then back to the value dtype.
        let dtype = scores.dtype();
        let weights = softmax(scores.cast(DType::F32), 3).cast(dtype);
        let attended =
            weights
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attended)
    }
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`, no biases.
#[derive(Module, Debug)]
pub struct AceStepMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> AceStepMlp<B> {
    pub fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: linear_no_bias(device, config.hidden_size, config.intermediate_size),
            up_proj: linear_no_bias(device, config.hidden_size, config.intermediate_size),
            down_proj: linear_no_bias(device, config.intermediate_size, config.hidden_size),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(hidden.clone()));
        let up = self.up_proj.forward(hidden);
        self.down_proj.forward(gate * up)
    }
}

/// Pre-norm bidirectional encoder layer shared by the lyric encoder, timbre
/// encoder and detokenizer:
/// `x += self_attn(input_layernorm(x)); x += mlp(post_attention_layernorm(x))`.
#[derive(Module, Debug)]
pub struct AceStepEncoderLayer<B: Backend> {
    pub self_attn: AceStepAttention<B>,
    pub mlp: AceStepMlp<B>,
    pub input_layernorm: ConditionRmsNorm<B>,
    pub post_attention_layernorm: ConditionRmsNorm<B>,
}

impl<B: Backend> AceStepEncoderLayer<B> {
    pub fn new(config: &AceStepConfig, layer_idx: usize, device: &B::Device) -> Self {
        Self {
            self_attn: AceStepAttention::new(config, layer_idx, device),
            mlp: AceStepMlp::new(config, device),
            input_layernorm: ConditionRmsNorm::new(device, config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: ConditionRmsNorm::new(
                device,
                config.hidden_size,
                config.rms_norm_eps,
            ),
        }
    }

    /// `padding_mask`: optional [B, L] integer mask (1 = valid key). It is
    /// combined with the sliding-window geometry mask into one additive mask.
    /// A row whose keys are all masked out yields NaN — never pass an all-zero
    /// mask row (upstream has the same caveat).
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        padding_mask: Option<&Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let sliding_window = self.self_attn.meta.sliding_window;
        let additive_mask = if sliding_window.is_some() || padding_mask.is_some() {
            let [batch, seq_len, _] = hidden.dims();
            let device = hidden.device();
            let padding = padding_mask.map(|mask| {
                mask.clone()
                    .to_data()
                    .convert::<i64>()
                    .to_vec::<i64>()
                    .expect("padding mask must materialize as i64")
            });
            Some(additive_attention_mask::<B>(
                batch,
                seq_len,
                sliding_window,
                padding,
                &device,
            ))
        } else {
            None
        };

        let residual = hidden.clone();
        let normed = self.input_layernorm.forward(hidden);
        let hidden = residual + self.self_attn.forward(normed, cos, sin, additive_mask);

        let residual = hidden.clone();
        let normed = self.post_attention_layernorm.forward(hidden);
        residual + self.mlp.forward(normed)
    }
}

/// Lyric encoder: projects precomputed Qwen3-Embedding lyric hidden states to
/// the model hidden size and runs `num_lyric_encoder_hidden_layers` encoder
/// layers (bidirectional, alternating sliding/full, padding mask honored).
#[derive(Module, Debug)]
pub struct LyricEncoder<B: Backend> {
    pub embed_tokens: Linear<B>,
    pub layers: Vec<AceStepEncoderLayer<B>>,
    pub norm: ConditionRmsNorm<B>,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> LyricEncoder<B> {
    pub fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        Self {
            embed_tokens: linear_with_bias(device, config.text_hidden_dim, config.hidden_size),
            layers: (0..config.num_lyric_encoder_hidden_layers)
                .map(|layer_idx| AceStepEncoderLayer::new(config, layer_idx, device))
                .collect(),
            norm: ConditionRmsNorm::new(device, config.hidden_size, config.rms_norm_eps),
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
        }
    }

    /// `lyric_hidden_states`: [B, Ll, text_hidden_dim]; `lyric_mask`: [B, Ll]
    /// integer (1 = valid). Returns [B, Ll, hidden_size].
    ///
    /// For instrumental tracks pass ONE dummy lyric token with mask [1]; an
    /// all-zero mask row produces NaN (all-masked softmax row), as upstream.
    pub fn forward(
        &self,
        lyric_hidden_states: Tensor<B, 3>,
        lyric_mask: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [_, seq_len, _] = lyric_hidden_states.dims();
        let device = lyric_hidden_states.device();
        let (cos, sin) = rotary_cos_sin::<B>(seq_len, self.head_dim, self.rope_theta, &device);

        let mut hidden = self.embed_tokens.forward(lyric_hidden_states);
        for layer in &self.layers {
            hidden = layer.forward(hidden, &cos, &sin, Some(&lyric_mask));
        }
        self.norm.forward(hidden)
    }
}

/// Timbre encoder: embeds 750-frame (timbre_fix_frame) VAE latents of one
/// reference clip per batch item and pools by taking position 0 of the final
/// normed output. The checkpoint's `special_token` prepend is dead code
/// upstream and is not modeled.
#[derive(Module, Debug)]
pub struct TimbreEncoder<B: Backend> {
    pub embed_tokens: Linear<B>,
    pub layers: Vec<AceStepEncoderLayer<B>>,
    pub norm: ConditionRmsNorm<B>,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> TimbreEncoder<B> {
    pub fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        Self {
            embed_tokens: linear_with_bias(
                device,
                config.audio_acoustic_hidden_dim,
                config.hidden_size,
            ),
            layers: (0..config.num_timbre_encoder_hidden_layers)
                .map(|layer_idx| AceStepEncoderLayer::new(config, layer_idx, device))
                .collect(),
            norm: ConditionRmsNorm::new(device, config.hidden_size, config.rms_norm_eps),
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
        }
    }

    /// `refer_latents`: [B, timbre_fix_frame, audio_acoustic_hidden_dim] — one
    /// reference clip per batch item (VAE-encoded silence for text2music).
    /// Returns the pooled embedding [B, 1, hidden_size].
    pub fn forward(&self, refer_latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = refer_latents.dims();
        let device = refer_latents.device();
        let (cos, sin) = rotary_cos_sin::<B>(seq_len, self.head_dim, self.rope_theta, &device);

        let mut hidden = self.embed_tokens.forward(refer_latents);
        for layer in &self.layers {
            hidden = layer.forward(hidden, &cos, &sin, None);
        }
        let hidden = self.norm.forward(hidden);
        let hidden_dim = self.norm.weight.dims()[0];
        hidden.slice([0..batch, 0..1, 0..hidden_dim])
    }
}

/// Audio token detokenizer: expands each 5Hz code embedding into
/// `pool_window_size` (5) consecutive 25Hz latent frames. Each code is
/// processed independently: embed → repeat 5× → add learned per-position
/// `special_tokens` → 2 encoder layers → norm → project to the acoustic dim.
#[derive(Module, Debug)]
pub struct AudioTokenDetokenizer<B: Backend> {
    pub embed_tokens: Linear<B>,
    pub special_tokens: Param<Tensor<B, 3>>,
    pub layers: Vec<AceStepEncoderLayer<B>>,
    pub norm: ConditionRmsNorm<B>,
    pub proj_out: Linear<B>,
    pool_window_size: usize,
    head_dim: usize,
    rope_theta: f64,
    acoustic_dim: usize,
}

impl<B: Backend> AudioTokenDetokenizer<B> {
    pub fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        Self {
            embed_tokens: linear_with_bias(device, config.hidden_size, config.hidden_size),
            special_tokens: Param::from_tensor(Tensor::<B, 3>::zeros(
                [1, config.pool_window_size, config.hidden_size],
                device,
            )),
            layers: (0..config.num_attention_pooler_hidden_layers)
                .map(|layer_idx| AceStepEncoderLayer::new(config, layer_idx, device))
                .collect(),
            norm: ConditionRmsNorm::new(device, config.hidden_size, config.rms_norm_eps),
            proj_out: linear_with_bias(
                device,
                config.hidden_size,
                config.audio_acoustic_hidden_dim,
            ),
            pool_window_size: config.pool_window_size,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            acoustic_dim: config.audio_acoustic_hidden_dim,
        }
    }

    /// `q`: [B, T5, hidden_size] (FSQ `project_out` output) →
    /// 25Hz hint latents [B, T5 * pool_window_size, audio_acoustic_hidden_dim].
    pub fn forward(&self, q: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, t5, _] = q.dims();
        let device = q.device();
        let (cos, sin) = rotary_cos_sin::<B>(
            self.pool_window_size,
            self.head_dim,
            self.rope_theta,
            &device,
        );

        let embedded = self.embed_tokens.forward(q);
        let mut hidden = self.expand_codes(embedded);
        for layer in &self.layers {
            hidden = layer.forward(hidden, &cos, &sin, None);
        }
        let hidden = self.norm.forward(hidden);
        self.proj_out.forward(hidden).reshape([
            batch,
            t5 * self.pool_window_size,
            self.acoustic_dim,
        ])
    }

    /// [B, T5, hidden] → [(B·T5), pool_window_size, hidden]: repeat each code
    /// 5× along a new axis and add the learned per-position special tokens.
    fn expand_codes(&self, embedded: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, t5, hidden_size] = embedded.dims();
        let repeated = embedded
            .unsqueeze_dim::<4>(2)
            .repeat_dim(2, self.pool_window_size);
        let special = self.special_tokens.val().unsqueeze_dim::<4>(1);
        (repeated + special).reshape([batch * t5, self.pool_window_size, hidden_size])
    }
}

/// Loadable root of the condition stack: text projector, lyric encoder,
/// timbre encoder, detokenizer and the FSQ quantizer (decode direction, used
/// to turn 5Hz LM codes into continuous hints for the detokenizer).
#[derive(Module, Debug)]
pub struct AceStepCondition<B: Backend> {
    pub text_projector: Linear<B>,
    pub lyric_encoder: LyricEncoder<B>,
    pub timbre_encoder: TimbreEncoder<B>,
    pub detokenizer: AudioTokenDetokenizer<B>,
    pub quantizer: ResidualFsq<B>,
}

impl<B: Backend> AceStepCondition<B> {
    pub fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        Self {
            text_projector: linear_no_bias(device, config.text_hidden_dim, config.hidden_size),
            lyric_encoder: LyricEncoder::new(config, device),
            timbre_encoder: TimbreEncoder::new(config, device),
            detokenizer: AudioTokenDetokenizer::new(config, device),
            quantizer: ResidualFsq::new(device),
        }
    }

    /// Load weights from a burnpack file whose keys are the canonical tensor
    /// names documented at the top of this module.
    pub fn from_burnpack(config: &AceStepConfig, path: &Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = BurnpackStore::from_file(path).zero_copy(true);
        model
            .load_from(&mut store)
            .with_context(|| format!("failed to load condition weights from {}", path.display()))?;
        Ok(model)
    }

    /// Assemble the conditioning sequence for the DiT.
    ///
    /// - `text_hidden`: [B, Lt, text_hidden_dim] — Qwen3-Embedding caption states
    /// - `lyric_hidden`: [B, Ll, text_hidden_dim] — Qwen3-Embedding lyric states
    /// - `lyric_mask`: [B, Ll] Int (1 = valid)
    /// - `silence_ref_latents`: [B, timbre_fix_frame, audio_acoustic_hidden_dim] —
    ///   timbre reference (VAE-encoded silence for text2music)
    ///
    /// Returns `encoder_hidden_states` [B, S, hidden_size] with
    /// S = Ll + 1 + Lt, layout [valid lyric…, valid timbre…, valid text…,
    /// (masked tail)]. The final mask is computed internally (it drives the
    /// lyric encoder's padding behavior via `pack_sequences`) but is NOT
    /// returned: the DiT discards it and cross-attends to the entire packed
    /// sequence, tail included.
    pub fn encode(
        &self,
        text_hidden: Tensor<B, 3>,
        lyric_hidden: Tensor<B, 3>,
        lyric_mask: Tensor<B, 2, Int>,
        silence_ref_latents: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, text_len, _] = text_hidden.dims();
        let device = text_hidden.device();

        let lyric = self.lyric_encoder.forward(lyric_hidden, lyric_mask.clone());
        let timbre = self.timbre_encoder.forward(silence_ref_latents);
        let text = self.text_projector.forward(text_hidden);

        let timbre_mask = Tensor::<B, 2, Int>::ones([batch, 1], &device);
        let text_mask = Tensor::<B, 2, Int>::ones([batch, text_len], &device);

        let (packed, mask) = pack_sequences(lyric, timbre, lyric_mask, timbre_mask);
        let (encoder_hidden_states, _final_mask) = pack_sequences(packed, text, mask, text_mask);
        encoder_hidden_states
    }

    /// 5Hz LM codes [B, T5] Int → 25Hz hint latents
    /// [B, T5 * pool_window_size, audio_acoustic_hidden_dim]
    /// (`quantizer.decode_indices` + detokenizer). These REPLACE `src_latents`
    /// in the DiT input when is_covers > 0.
    pub fn codes_to_hints(&self, audio_codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hints = self.quantizer.decode_indices(audio_codes);
        self.detokenizer.forward(hints)
    }
}

/// Concatenate two [B, L1/L2, D] sequences with their [B, L] masks along the
/// sequence dimension, then stable-sort each batch row so mask=1 positions
/// come first, preserving relative order within equal mask values. The new
/// mask marks the first `sum(mask)` positions of each row as valid.
///
/// Runs on the host (B is 1 in practice, and this is called twice per
/// generation); the returned hidden states keep the input dtype.
pub fn pack_sequences<B: Backend>(
    hidden1: Tensor<B, 3>,
    hidden2: Tensor<B, 3>,
    mask1: Tensor<B, 2, Int>,
    mask2: Tensor<B, 2, Int>,
) -> (Tensor<B, 3>, Tensor<B, 2, Int>) {
    let [batch, len1, dim] = hidden1.dims();
    let len2 = hidden2.dims()[1];
    let len = len1 + len2;
    let dtype = hidden1.dtype();
    let device = hidden1.device();

    let hidden1_data = hidden1
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("hidden1 must materialize as f32");
    let hidden2_data = hidden2
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("hidden2 must materialize as f32");
    let mask1_data = mask1
        .to_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("mask1 must materialize as i64");
    let mask2_data = mask2
        .to_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("mask2 must materialize as i64");

    let mut packed = vec![0.0f32; batch * len * dim];
    let mut new_mask = vec![0i64; batch * len];

    for b in 0..batch {
        // Stable descending sort by mask: valid (1) first, order preserved
        // within each group (Vec::sort_by_key is stable).
        let mut order: Vec<(i64, usize)> = (0..len1)
            .map(|i| (mask1_data[b * len1 + i], i))
            .chain((0..len2).map(|j| (mask2_data[b * len2 + j], len1 + j)))
            .collect();
        order.sort_by_key(|&(mask, _)| std::cmp::Reverse(mask));

        let valid_count = order.iter().filter(|&&(mask, _)| mask > 0).count();
        for (new_pos, &(_, old_pos)) in order.iter().enumerate() {
            let source = if old_pos < len1 {
                &hidden1_data[(b * len1 + old_pos) * dim..][..dim]
            } else {
                &hidden2_data[(b * len2 + old_pos - len1) * dim..][..dim]
            };
            packed[(b * len + new_pos) * dim..][..dim].copy_from_slice(source);
        }
        for (new_pos, slot) in new_mask[b * len..(b + 1) * len].iter_mut().enumerate() {
            *slot = i64::from(new_pos < valid_count);
        }
    }

    let packed =
        Tensor::<B, 3>::from_data(TensorData::new(packed, [batch, len, dim]), &device).cast(dtype);
    let new_mask = Tensor::<B, 2, Int>::from_data(TensorData::new(new_mask, [batch, len]), &device);
    (packed, new_mask)
}

/// Additive [B, 1, L, L] attention mask: 0 where query i may attend to key j,
/// `f32::MIN` elsewhere. A key is allowed when it is inside the sliding window
/// (`|i - j| <= window`, if any) AND not padding.
fn additive_attention_mask<B: Backend>(
    batch: usize,
    seq_len: usize,
    sliding_window: Option<usize>,
    padding: Option<Vec<i64>>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut values = vec![f32::MIN; batch * seq_len * seq_len];
    for b in 0..batch {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let within_window = sliding_window.is_none_or(|window| i.abs_diff(j) <= window);
                let key_valid = padding
                    .as_ref()
                    .is_none_or(|mask| mask[b * seq_len + j] != 0);
                if within_window && key_valid {
                    values[(b * seq_len + i) * seq_len + j] = 0.0;
                }
            }
        }
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(values, [batch, 1, seq_len, seq_len]),
        device,
    )
}

/// inv_freq[i] = 1 / theta^(2i / head_dim); cos/sin over positions 0..seq_len
/// with duplicated halves (cat([freqs, freqs])). Returned as [1, L, 1, head_dim]
/// so they broadcast against [batch, seq, heads, head_dim].
fn rotary_cos_sin<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
        .collect();
    let mut cos_values = Vec::with_capacity(seq_len * head_dim);
    let mut sin_values = Vec::with_capacity(seq_len * head_dim);
    for pos in 0..seq_len {
        for _ in 0..2 {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_values.push(angle.cos());
                sin_values.push(angle.sin());
            }
        }
    }
    let shape = [1, seq_len, 1, head_dim];
    (
        Tensor::<B, 4>::from_data(TensorData::new(cos_values, shape), device),
        Tensor::<B, 4>::from_data(TensorData::new(sin_values, shape), device),
    )
}

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, seq_len, heads, head_dim] = x.dims();
    let half = head_dim / 2;
    let x1 = x.clone().slice([0..batch, 0..seq_len, 0..heads, 0..half]);
    let x2 = x.slice([0..batch, 0..seq_len, 0..heads, half..head_dim]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

fn apply_rotary_pos_emb<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    x.clone() * cos.clone() + rotate_half(x) * sin.clone()
}

/// [batch, kv_heads, seq, head_dim] -> [batch, kv_heads * repeats, seq, head_dim]
fn repeat_kv<B: Backend>(tensor: Tensor<B, 4>, repeats: usize) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = tensor.dims();
    tensor
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repeats)
        .reshape([batch, heads * repeats, seq_len, head_dim])
}

fn linear_no_bias<B: Backend>(device: &B::Device, d_input: usize, d_output: usize) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init(device)
}

fn linear_with_bias<B: Backend>(device: &B::Device, d_input: usize, d_output: usize) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_layout(LinearLayout::Col)
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> AceStepConfig {
        AceStepConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            sliding_window: 2,
            in_channels: 12,
            audio_acoustic_hidden_dim: 4,
            patch_size: 2,
            text_hidden_dim: 8,
            num_lyric_encoder_hidden_layers: 2,
            num_timbre_encoder_hidden_layers: 2,
            timbre_fix_frame: 750,
            pool_window_size: 5,
            num_attention_pooler_hidden_layers: 2,
            fsq_dim: 32,
            fsq_input_levels: vec![8, 8, 8, 5, 5, 5],
            vocab_size: 64003,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            is_turbo: true,
        }
    }

    fn assert_finite(tensor: Tensor<TestBackend, 3>) {
        let values = tensor
            .to_data()
            .to_vec::<f32>()
            .expect("output should materialize as f32");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    #[test]
    fn encode_shapes_and_finiteness() {
        let config = tiny_config();
        let device = Default::default();
        let condition = AceStepCondition::<TestBackend>::new(&config, &device);

        let text_hidden =
            Tensor::from_data(TensorData::new(vec![0.01f32; 3 * 8], [1, 3, 8]), &device);
        let lyric_hidden =
            Tensor::from_data(TensorData::new(vec![-0.02f32; 2 * 8], [1, 2, 8]), &device);
        let lyric_mask = Tensor::<TestBackend, 2, Int>::from_data([[1, 1]], &device);
        let silence_ref_latents = Tensor::zeros([1, 750, 4], &device);

        let encoded = condition.encode(text_hidden, lyric_hidden, lyric_mask, silence_ref_latents);
        // S = Ll (2) + 1 (timbre) + Lt (3) = 6.
        assert_eq!(encoded.dims(), [1, 6, 32]);
        assert_finite(encoded);
    }

    #[test]
    fn pack_sequences_stable_sorts_valid_first() {
        let device = Default::default();
        // Batch row 0: masks [1,0,1] + [0,1] -> order 10,30,50,20,40, mask 11100.
        // Batch row 1: masks [0,1,0] + [1,0] -> order 70,90,60,80,100, mask 11000.
        let hidden1 = Tensor::<TestBackend, 3>::from_data(
            [[[10.0], [20.0], [30.0]], [[60.0], [70.0], [80.0]]],
            &device,
        );
        let hidden2 =
            Tensor::<TestBackend, 3>::from_data([[[40.0], [50.0]], [[90.0], [100.0]]], &device);
        let mask1 = Tensor::<TestBackend, 2, Int>::from_data([[1, 0, 1], [0, 1, 0]], &device);
        let mask2 = Tensor::<TestBackend, 2, Int>::from_data([[0, 1], [1, 0]], &device);

        let (packed, new_mask) = pack_sequences(hidden1, hidden2, mask1, mask2);
        assert_eq!(packed.dims(), [2, 5, 1]);
        let packed_values = packed.to_data().to_vec::<f32>().expect("packed values");
        assert_eq!(
            packed_values,
            vec![10.0, 30.0, 50.0, 20.0, 40.0, 70.0, 90.0, 60.0, 80.0, 100.0]
        );
        let mask_values: Vec<i64> = new_mask
            .to_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("mask values");
        assert_eq!(mask_values, vec![1, 1, 1, 0, 0, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn codes_to_hints_shape_and_finiteness() {
        let config = tiny_config();
        let device = Default::default();
        let mut condition = AceStepCondition::<TestBackend>::new(&config, &device);
        // The real quantizer is fixed at 2048 dims (FSQ contract); swap in a
        // tiny project_out so the detokenizer sees the tiny hidden size.
        condition.quantizer = ResidualFsq {
            project_in: LinearConfig::new(32, 6).with_bias(true).init(&device),
            project_out: LinearConfig::new(6, 32).with_bias(true).init(&device),
        };

        let codes = Tensor::<TestBackend, 2, Int>::from_data([[0, 63999]], &device);
        let hints = condition.codes_to_hints(codes);
        assert_eq!(hints.dims(), [1, 10, 4]);
        assert_finite(hints);
    }

    #[test]
    fn detokenizer_expand_repeats_and_adds_special_tokens() {
        let config = tiny_config();
        let device = Default::default();
        let mut detokenizer = AudioTokenDetokenizer::<TestBackend>::new(&config, &device);

        // Known special tokens: special[0, p, h] = (p * 32 + h) * 0.001.
        let special: Vec<f32> = (0..5 * 32).map(|i| i as f32 * 0.001).collect();
        detokenizer.special_tokens = Param::from_tensor(Tensor::from_data(
            TensorData::new(special, [1, 5, 32]),
            &device,
        ));

        let q: Vec<f32> = (0..2 * 32).map(|i| i as f32 * 0.01).collect();
        let q = Tensor::<TestBackend, 3>::from_data(TensorData::new(q, [1, 2, 32]), &device);
        let embedded = detokenizer.embed_tokens.forward(q);
        let expanded = detokenizer.expand_codes(embedded.clone());
        assert_eq!(expanded.dims(), [2, 5, 32]);

        let embedded_values = embedded.to_data().to_vec::<f32>().expect("embedded");
        let expanded_values = expanded.to_data().to_vec::<f32>().expect("expanded");
        for t in 0..2 {
            for p in 0..5 {
                for h in 0..32 {
                    let expected = embedded_values[t * 32 + h] + (p * 32 + h) as f32 * 0.001;
                    let actual = expanded_values[(t * 5 + p) * 32 + h];
                    assert!(
                        (actual - expected).abs() < 1e-5,
                        "mismatch at t={t} p={p} h={h}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn lyric_encoder_ignores_padded_values() {
        let config = tiny_config();
        let device = Default::default();
        let encoder = LyricEncoder::<TestBackend>::new(&config, &device);
        let mask = Tensor::<TestBackend, 2, Int>::from_data([[1, 0]], &device);

        let mut row_a = vec![0.05f32; 8];
        row_a.extend(vec![1.0f32; 8]);
        let mut row_b = vec![0.05f32; 8];
        row_b.extend(vec![-7.0f32; 8]);
        let input_a =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(row_a, [1, 2, 8]), &device);
        let input_b =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(row_b, [1, 2, 8]), &device);

        let out_a = encoder.forward(input_a, mask.clone());
        let out_b = encoder.forward(input_b, mask);
        let valid_a = out_a
            .slice([0..1, 0..1, 0..32])
            .to_data()
            .to_vec::<f32>()
            .expect("out_a");
        let valid_b = out_b
            .slice([0..1, 0..1, 0..32])
            .to_data()
            .to_vec::<f32>()
            .expect("out_b");
        for (a, b) in valid_a.iter().zip(valid_b.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "padded key values leaked into valid output: {a} vs {b}"
            );
        }
    }
}
