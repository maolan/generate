//! ACE-Step 1.5 turbo DiT (`AceStepDiTModel`) ported to Burn, plus the turbo
//! sampler (explicit Euler, no CFG, 8-step shift-3 schedule).
//!
//! Ground truth: `modeling_acestep_v15_turbo.py` (`AceStepDiTModel`,
//! `AceStepDiTLayer`, `AceStepAttention`, `TimestepEmbedding`). Key behaviors:
//!
//! - Patchify: `proj_in` Conv1d(in_channels → hidden, k=patch, stride=patch,
//!   bias) over frames zero-padded to a multiple of `patch_size`;
//!   `proj_out` ConvTranspose1d(hidden → acoustic_dim, k=patch, stride=patch,
//!   bias) de-patchifies and the output is cropped back to the input length.
//! - AdaLN chunk order per layer: (shift_msa, scale_msa, gate_msa,
//!   c_shift_msa, c_scale_msa, c_gate_msa) from
//!   `scale_shift_table + timestep_proj` (each [B, 1, hidden]); residual gate
//!   multiplies the sublayer output; cross-attention has NO modulation/gate.
//! - Final modulation uses the summed `temb` ([B, hidden], NOT the 6× proj)
//!   with table order (shift, scale).
//! - Timestep embedding: scale 1000, max_period 10000, COS-then-SIN halves.
//!   `time_embed_r` is always evaluated at `t - r == 0` at inference and
//!   contributes a nonzero constant — it must not be dropped.
//! - All attention is BIDIRECTIONAL. Sliding layers attend within the band
//!   |i − j| ≤ sliding_window; full layers attend to everything. The DiT
//!   ignores padding masks entirely, including the encoder tail.
//! - RoPE (theta 1e6, duplicated halves, rotate_half) is applied AFTER the
//!   per-head q/k RMSNorm, self-attention only. Cross-attention applies no
//!   RoPE and attends to the full encoder sequence.
//!
//! # Canonical burnpack tensor names
//!
//! The offline converter renames the official `decoder.*` checkpoint tensors
//! to exactly these module paths:
//!
//! - `proj_in.conv.{weight,bias}` — Conv1d [hidden, in_channels, patch], [hidden]
//! - `proj_out.conv.{weight,bias}` — ConvTranspose1d [hidden, acoustic, patch], [acoustic]
//! - `time_embed.linear_1.{weight,bias}` — 256 → hidden
//! - `time_embed.linear_2.{weight,bias}` — hidden → hidden
//! - `time_embed.time_proj.{weight,bias}` — hidden → 6 × hidden
//! - `time_embed_r.linear_1.{weight,bias}`, `time_embed_r.linear_2.{weight,bias}`,
//!   `time_embed_r.time_proj.{weight,bias}` — same trio
//! - `condition_embedder.{weight,bias}` — hidden → hidden
//! - `layers.{i}.self_attn_norm.weight`
//! - `layers.{i}.self_attn.{q,k,v,o}_proj.weight` — q/o: hidden↔hidden,
//!   k/v: hidden ↔ kv_heads × head_dim, all bias-free
//! - `layers.{i}.self_attn.{q,k}_norm.weight` — [head_dim]
//! - `layers.{i}.cross_attn_norm.weight`
//! - `layers.{i}.cross_attn.{q,k,v,o}_proj.weight`, `layers.{i}.cross_attn.{q,k}_norm.weight`
//! - `layers.{i}.mlp_norm.weight`
//! - `layers.{i}.mlp.{gate,up,down}_proj.weight` — hidden ↔ intermediate, bias-free
//! - `layers.{i}.scale_shift_table` — [1, 6, hidden]
//! - `norm_out.weight`
//! - `scale_shift_table` — [1, 2, hidden]

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{Linear, LinearConfig, LinearLayout};
use burn::prelude::Backend;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{DType, Distribution, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

use crate::acestep::config::AceStepConfig;
use crate::acestep::qwen3::Qwen3RmsNorm;

/// Turbo inference schedule (shift = 3.0 transform of uniform eighths), t: 1.0 → 0.3.
/// Spec values (f64): [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
/// 0.6428571428571429, 0.5, 0.3]; the literals below round to the same f32.
pub const TURBO_TIMESTEPS: [f32; 8] = [
    1.0,
    0.954_545_44,
    0.9,
    0.833_333_3,
    0.75,
    0.642_857_13,
    0.5,
    0.3,
];

/// SFT/base inference schedule (shift = 1.0, i.e. uniform), 50 steps,
/// t: 1.0 → 0.02 (`t_i = 1 − i/50`, i in 0..50).
pub const SFT_TIMESTEPS: [f32; 50] = {
    let mut schedule = [0.0_f32; 50];
    let mut i = 0;
    while i < 50 {
        schedule[i] = 1.0 - i as f32 / 50.0;
        i += 1;
    }
    schedule
};

/// Sinusoidal timestep embedding width (`TimestepEmbedding in_channels`).
const TIME_EMBED_CHANNELS: usize = 256;
/// Timestep scale applied before the sinusoidal projection.
const TIME_EMBED_SCALE: f32 = 1000.0;
/// Maximum period of the sinusoidal timestep frequencies.
const TIME_EMBED_MAX_PERIOD: f32 = 10_000.0;
/// Additive attention mask value for disallowed positions (fp32 "-inf").
const MASK_MIN: f32 = -1.0e30;

/// Per-head GQA attention with q/k RMSNorm; self-attn applies RoPE,
/// cross-attn consumes precomputed encoder K/V (no RoPE, no sliding).
#[derive(Module, Debug)]
pub struct AceStepAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: Qwen3RmsNorm<B>,
    pub k_norm: Qwen3RmsNorm<B>,
    #[module(skip)]
    meta: AttentionMeta,
}

#[derive(Clone, Debug)]
struct AttentionMeta {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f32,
}

impl<B: Backend> AceStepAttention<B> {
    fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        let head_dim = config.head_dim;
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
            q_norm: Qwen3RmsNorm::new(device, head_dim, config.rms_norm_eps),
            k_norm: Qwen3RmsNorm::new(device, head_dim, config.rms_norm_eps),
            meta: AttentionMeta {
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim,
                scaling: (head_dim as f32).powf(-0.5),
            },
        }
    }

    /// Cross-attention K/V projection of (already condition-embedded) encoder
    /// states: k_norm applied to K, heads swapped to [batch, heads, seq, dim]
    /// and KV heads repeated to the full query head count. Computed once per
    /// generation and reused across all sampler steps.
    pub fn project_kv(&self, encoder: &Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _] = encoder.dims();
        let kv_heads = self.meta.num_kv_heads;
        let head_dim = self.meta.head_dim;

        let k = self
            .k_norm
            .forward(
                self.k_proj
                    .forward(encoder.clone())
                    .reshape([batch, seq_len, kv_heads, head_dim]),
            )
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(encoder.clone())
            .reshape([batch, seq_len, kv_heads, head_dim])
            .swap_dims(1, 2);

        let repeats = self.meta.num_heads / self.meta.num_kv_heads;
        (repeat_kv(k, repeats), repeat_kv(v, repeats))
    }

    /// Bidirectional self-attention with RoPE and an optional additive
    /// sliding-window mask of shape [seq, seq].
    pub fn forward_self(
        &self,
        hidden: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        mask: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden.dims();
        let num_heads = self.meta.num_heads;
        let kv_heads = self.meta.num_kv_heads;
        let head_dim = self.meta.head_dim;

        let q = self.q_norm.forward(
            self.q_proj
                .forward(hidden.clone())
                .reshape([batch, seq_len, num_heads, head_dim]),
        );
        let k = self.k_norm.forward(
            self.k_proj
                .forward(hidden.clone())
                .reshape([batch, seq_len, kv_heads, head_dim]),
        );
        let v = self
            .v_proj
            .forward(hidden)
            .reshape([batch, seq_len, kv_heads, head_dim]);

        // RoPE post q/k-norm, self-attention only.
        let q = apply_rotary_pos_emb(q, cos, sin).swap_dims(1, 2);
        let k = apply_rotary_pos_emb(k, cos, sin).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let repeats = num_heads / kv_heads;
        let k = repeat_kv(k, repeats);
        let v = repeat_kv(v, repeats);

        let attended = self.attend(q, k, v, mask);
        let attended = attended
            .swap_dims(1, 2)
            .reshape([batch, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attended)
    }

    /// Bidirectional cross-attention over the full encoder sequence using
    /// precomputed K/V of shape [batch, heads, enc_seq, head_dim].
    pub fn forward_cross(
        &self,
        hidden: Tensor<B, 3>,
        key: &Tensor<B, 4>,
        value: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden.dims();
        let num_heads = self.meta.num_heads;
        let head_dim = self.meta.head_dim;

        let q = self
            .q_norm
            .forward(
                self.q_proj
                    .forward(hidden)
                    .reshape([batch, seq_len, num_heads, head_dim]),
            )
            .swap_dims(1, 2);

        let attended = self.attend(q, key.clone(), value.clone(), None);
        let attended = attended
            .swap_dims(1, 2)
            .reshape([batch, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attended)
    }

    /// scores = q @ kᵀ · head_dim^-0.5 (+ optional additive mask), softmax in
    /// fp32, then @ v. Inputs are [batch, heads, seq, head_dim].
    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let dtype = q.dtype();
        let mut scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.meta.scaling);
        if let Some(mask) = mask {
            let [rows, cols] = mask.dims();
            scores = scores + mask.clone().reshape([1, 1, rows, cols]);
        }
        let weights = softmax(scores.cast(DType::F32), 3).cast(dtype);
        weights.matmul(v)
    }
}

/// SwiGLU MLP: down(silu(gate(x)) * up(x)), all projections bias-free.
#[derive(Module, Debug)]
pub struct AceStepMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> AceStepMlp<B> {
    fn new(config: &AceStepConfig, device: &B::Device) -> Self {
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

/// One DiT layer: AdaLN self-attention (gated), plain-residual cross-attention,
/// AdaLN MLP (gated). Modulation chunk order is
/// (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa).
#[derive(Module, Debug)]
pub struct AceStepDiTLayer<B: Backend> {
    pub self_attn_norm: Qwen3RmsNorm<B>,
    pub self_attn: AceStepAttention<B>,
    pub cross_attn_norm: Qwen3RmsNorm<B>,
    pub cross_attn: AceStepAttention<B>,
    pub mlp_norm: Qwen3RmsNorm<B>,
    pub mlp: AceStepMlp<B>,
    /// AdaLN modulation table of shape [1, 6, hidden].
    pub scale_shift_table: Param<Tensor<B, 3>>,
    #[module(skip)]
    sliding: bool,
}

impl<B: Backend> AceStepDiTLayer<B> {
    fn new(config: &AceStepConfig, layer: usize, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        Self {
            self_attn_norm: Qwen3RmsNorm::new(device, hidden, config.rms_norm_eps),
            self_attn: AceStepAttention::new(config, device),
            cross_attn_norm: Qwen3RmsNorm::new(device, hidden, config.rms_norm_eps),
            cross_attn: AceStepAttention::new(config, device),
            mlp_norm: Qwen3RmsNorm::new(device, hidden, config.rms_norm_eps),
            mlp: AceStepMlp::new(config, device),
            scale_shift_table: Param::from_tensor(randn_scaled([1, 6, hidden], hidden, device)),
            sliding: config.is_sliding_layer(layer),
        }
    }

    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        rope: (&Tensor<B, 4>, &Tensor<B, 4>),
        timestep_proj: &Tensor<B, 3>,
        cross_kv: (&Tensor<B, 4>, &Tensor<B, 4>),
        mask: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let (cos, sin) = rope;
        let (cross_key, cross_value) = cross_kv;
        // [1, 6, hidden] + [1, 6, hidden] → six [1, 1, hidden] chunks.
        let modulation = self.scale_shift_table.val() + timestep_proj.clone();
        let chunks = modulation.chunk(6, 1);
        let (shift_msa, scale_msa, gate_msa) =
            (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());
        let (c_shift_msa, c_scale_msa, c_gate_msa) =
            (chunks[3].clone(), chunks[4].clone(), chunks[5].clone());

        let normed = self.self_attn_norm.forward(hidden.clone()) * (scale_msa + 1.0) + shift_msa;
        let attn = self.self_attn.forward_self(normed, cos, sin, mask);
        let hidden = hidden + attn * gate_msa;

        // Cross-attention: plain residual, no modulation and no gate.
        let normed = self.cross_attn_norm.forward(hidden.clone());
        let hidden = hidden
            + self
                .cross_attn
                .forward_cross(normed, cross_key, cross_value);

        let normed = self.mlp_norm.forward(hidden.clone()) * (c_scale_msa + 1.0) + c_shift_msa;
        hidden + self.mlp.forward(normed) * c_gate_msa
    }
}

/// TimestepEmbedding(256, hidden, scale=1000): sinusoidal COS-then-SIN
/// projection followed by linear_1 → SiLU → linear_2 (temb) and
/// time_proj(SiLU(temb)) reshaped to [1, 6, hidden] (timestep_proj).
#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1: Linear<B>,
    pub linear_2: Linear<B>,
    pub time_proj: Linear<B>,
    #[module(skip)]
    meta: TimestepMeta,
}

#[derive(Clone, Debug)]
struct TimestepMeta {
    in_channels: usize,
    embed_dim: usize,
}

impl<B: Backend> TimestepEmbedding<B> {
    fn new(embed_dim: usize, device: &B::Device) -> Self {
        Self {
            linear_1: linear_with_bias(device, TIME_EMBED_CHANNELS, embed_dim),
            linear_2: linear_with_bias(device, embed_dim, embed_dim),
            time_proj: linear_with_bias(device, embed_dim, 6 * embed_dim),
            meta: TimestepMeta {
                in_channels: TIME_EMBED_CHANNELS,
                embed_dim,
            },
        }
    }

    /// Returns (temb [1, hidden], timestep_proj [1, 6, hidden]).
    pub fn forward(&self, t: f32, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let sinusoid = sinusoidal_timestep::<B>(t, self.meta.in_channels, device);
        let temb = self.linear_2.forward(silu(self.linear_1.forward(sinusoid)));
        let proj = self
            .time_proj
            .forward(silu(temb.clone()))
            .reshape([1, 6, self.meta.embed_dim]);
        (temb, proj)
    }
}

/// `proj_in`: transposed strided Conv1d patchifier. The `conv` field name is
/// load-bearing for the canonical burnpack path `proj_in.conv.*`.
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub conv: Conv1d<B>,
}

/// `proj_out`: transposed ConvTranspose1d de-patchifier. The `conv` field name
/// is load-bearing for the canonical burnpack path `proj_out.conv.*`.
#[derive(Module, Debug)]
pub struct PatchUnembed<B: Backend> {
    pub conv: ConvTranspose1d<B>,
}

/// Cross-attention K/V pairs, one per DiT layer, computed once per generation
/// (encoder states are constant across sampler steps).
pub struct CrossKv<B: Backend> {
    keys: Vec<Tensor<B, 4>>,
    values: Vec<Tensor<B, 4>>,
}

/// The ACE-Step 1.5 turbo DiT decoder (`AceStepDiTModel`).
#[derive(Module, Debug)]
pub struct AceStepDiT<B: Backend> {
    pub proj_in: PatchEmbed<B>,
    pub time_embed: TimestepEmbedding<B>,
    pub time_embed_r: TimestepEmbedding<B>,
    pub condition_embedder: Linear<B>,
    pub layers: Vec<AceStepDiTLayer<B>>,
    pub norm_out: Qwen3RmsNorm<B>,
    pub proj_out: PatchUnembed<B>,
    /// Final AdaLN modulation table of shape [1, 2, hidden].
    pub scale_shift_table: Param<Tensor<B, 3>>,
    #[module(skip)]
    meta: DiTMeta,
}

#[derive(Clone, Debug)]
struct DiTMeta {
    head_dim: usize,
    rope_theta: f64,
    sliding_window: usize,
    patch_size: usize,
    any_sliding: bool,
}

impl<B: Backend> AceStepDiT<B> {
    pub fn new(config: &AceStepConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_size;
        Self {
            proj_in: PatchEmbed {
                conv: Conv1dConfig::new(config.in_channels, hidden, config.patch_size)
                    .with_stride(config.patch_size)
                    .with_bias(true)
                    .init(device),
            },
            time_embed: TimestepEmbedding::new(hidden, device),
            time_embed_r: TimestepEmbedding::new(hidden, device),
            condition_embedder: linear_with_bias(device, hidden, hidden),
            layers: (0..config.num_hidden_layers)
                .map(|layer| AceStepDiTLayer::new(config, layer, device))
                .collect(),
            norm_out: Qwen3RmsNorm::new(device, hidden, config.rms_norm_eps),
            proj_out: PatchUnembed {
                conv: ConvTranspose1dConfig::new(
                    [hidden, config.audio_acoustic_hidden_dim],
                    config.patch_size,
                )
                .with_stride(config.patch_size)
                .with_bias(true)
                .init(device),
            },
            scale_shift_table: Param::from_tensor(randn_scaled([1, 2, hidden], hidden, device)),
            meta: DiTMeta {
                head_dim: config.head_dim,
                rope_theta: config.rope_theta,
                sliding_window: config.sliding_window,
                patch_size: config.patch_size,
                any_sliding: (0..config.num_hidden_layers)
                    .any(|layer| config.is_sliding_layer(layer)),
            },
        }
    }

    pub fn from_burnpack(config: &AceStepConfig, path: &Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = BurnpackStore::from_file(path).zero_copy(true);
        model.load_from(&mut store).with_context(|| {
            format!("failed to load AceStep DiT weights from {}", path.display())
        })?;
        Ok(model)
    }

    /// Runs one DiT forward pass.
    ///
    /// - `xt`: current noisy latent [B, T, acoustic_dim]
    /// - `t`: scalar flow timestep (timestep_r == t, so time_embed_r sees 0)
    /// - `context`: [B, T, in_channels − acoustic_dim] (src_latents ++ chunk_masks)
    /// - `encoder_hidden_states`: [B, S, hidden] raw encoder output;
    ///   `condition_embedder` is applied inside.
    ///
    /// Returns the predicted velocity v [B, T, acoustic_dim].
    pub fn forward(
        &self,
        xt: Tensor<B, 3>,
        t: f32,
        context: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let kv = self.prepare_cross_kv(encoder_hidden_states);
        self.forward_with_kv(xt, t, context, &kv)
    }

    /// Applies `condition_embedder` and every layer's cross-attention K/V
    /// projection to the encoder states. Cache the result and call
    /// [`AceStepDiT::forward_with_kv`] to skip redundant encoder work.
    pub fn prepare_cross_kv(&self, encoder_hidden_states: Tensor<B, 3>) -> CrossKv<B> {
        let conditioned = self.condition_embedder.forward(encoder_hidden_states);
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (key, value) = layer.cross_attn.project_kv(&conditioned);
            keys.push(key);
            values.push(value);
        }
        CrossKv { keys, values }
    }

    /// DiT forward with precomputed cross-attention K/V (see
    /// [`AceStepDiT::prepare_cross_kv`]). Numerically identical to
    /// [`AceStepDiT::forward`].
    pub fn forward_with_kv(
        &self,
        xt: Tensor<B, 3>,
        t: f32,
        context: Tensor<B, 3>,
        kv: &CrossKv<B>,
    ) -> Tensor<B, 3> {
        let device = xt.device();
        let [batch, seq_len, _] = xt.dims();

        // x_in = cat([context_latents, xt], -1) → [B, T, in_channels].
        let x = Tensor::cat(vec![context, xt], 2);
        // Zero-pad the frame count to a multiple of patch_size.
        let remainder = seq_len % self.meta.patch_size;
        let x = if remainder != 0 {
            let pad_len = self.meta.patch_size - remainder;
            let in_channels = x.dims()[2];
            let pad = Tensor::zeros([batch, pad_len, in_channels], &device);
            Tensor::cat(vec![x, pad], 1)
        } else {
            x
        };

        // Patchify: [B, T, C] → conv over [B, C, T] → [B, T/patch, hidden].
        let mut hidden = self.proj_in.conv.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
        let patch_len = hidden.dims()[1];

        let (cos, sin) =
            rotary_cos_sin::<B>(patch_len, self.meta.head_dim, self.meta.rope_theta, &device);
        let sliding_mask = if self.meta.any_sliding {
            Some(sliding_window_mask::<B>(
                patch_len,
                self.meta.sliding_window,
                &device,
            ))
        } else {
            None
        };

        // timestep_r == timestep at inference → time_embed_r is evaluated at 0
        // (a nonzero constant that must not be dropped).
        let (temb_t, proj_t) = self.time_embed.forward(t, &device);
        let (temb_r, proj_r) = self.time_embed_r.forward(0.0, &device);
        let temb = temb_t + temb_r;
        let timestep_proj = proj_t + proj_r;

        for (index, layer) in self.layers.iter().enumerate() {
            let mask = if layer.sliding {
                sliding_mask.as_ref()
            } else {
                None
            };
            hidden = layer.forward(
                hidden,
                (&cos, &sin),
                &timestep_proj,
                (&kv.keys[index], &kv.values[index]),
                mask,
            );
        }

        // Final AdaLN uses temb (not the 6× proj), table order (shift, scale).
        let modulation = self.scale_shift_table.val() + temb.unsqueeze_dim::<3>(1);
        let chunks = modulation.chunk(2, 1);
        let shift = chunks[0].clone();
        let scale = chunks[1].clone();
        let out = self.norm_out.forward(hidden) * (scale + 1.0) + shift;

        // De-patchify and crop back to the original (unpadded) frame count.
        let out = self
            .proj_out
            .conv
            .forward(out.swap_dims(1, 2))
            .swap_dims(1, 2);
        out.narrow(1, 0, seq_len)
    }

    /// Turbo sampler: explicit Euler over `timesteps` (default
    /// [`TURBO_TIMESTEPS`], 8 steps, t descending 1.0 → 0.3), no CFG.
    ///
    /// Flow convention: x1 = noise, x0 = data, x_t = t·x1 + (1−t)·x0 and the
    /// model predicts v = x1 − x0. Step i applies
    /// `xt ← xt − v·(t_cur − t_next)`, and the last step `xt ← xt − v·t_cur`
    /// (recovering x0).
    ///
    /// - `noise`: initial latent x1 [B, T, acoustic_dim]
    /// - `context`: [B, T, in_channels − acoustic_dim]
    /// - `encoder_hidden_states`: [B, S, hidden] (cross K/V computed once)
    /// - `progress`: optional callback invoked as (steps_done, total_steps)
    ///
    /// Returns the final latent [B, T, acoustic_dim].
    pub fn sample_turbo(
        &self,
        noise: Tensor<B, 3>,
        context: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        timesteps: &[f32],
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Tensor<B, 3> {
        let kv = self.prepare_cross_kv(encoder_hidden_states);
        let total = timesteps.len();
        let mut xt = noise;
        for (index, &t_cur) in timesteps.iter().enumerate() {
            let v = self.forward_with_kv(xt.clone(), t_cur, context.clone(), &kv);
            let dt = if index + 1 == total {
                t_cur
            } else {
                t_cur - timesteps[index + 1]
            };
            xt = xt - v * dt;
            if let Some(callback) = progress.as_mut() {
                callback(index + 1, total);
            }
        }
        xt
    }
}

/// Sinusoidal timestep embedding [1, channels]: t scaled by 1000, frequencies
/// exp(-ln(10000) · k / half), COS first then SIN (matches the reference).
fn sinusoidal_timestep<B: Backend>(t: f32, channels: usize, device: &B::Device) -> Tensor<B, 2> {
    let half = channels / 2;
    let scaled = t * TIME_EMBED_SCALE;
    let mut data = Vec::with_capacity(channels);
    for k in 0..half {
        let freq = (-TIME_EMBED_MAX_PERIOD.ln() * k as f32 / half as f32).exp();
        data.push((scaled * freq).cos());
    }
    for k in 0..half {
        let freq = (-TIME_EMBED_MAX_PERIOD.ln() * k as f32 / half as f32).exp();
        data.push((scaled * freq).sin());
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [1, channels]), device)
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

/// [batch, kv_heads, seq, head_dim] → [batch, kv_heads × repeats, seq, head_dim]
/// (repeat-interleave along the head axis, matching HF `repeat_kv`).
fn repeat_kv<B: Backend>(tensor: Tensor<B, 4>, repeats: usize) -> Tensor<B, 4> {
    if repeats == 1 {
        return tensor;
    }
    let [batch, heads, seq_len, head_dim] = tensor.dims();
    tensor
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repeats)
        .reshape([batch, heads * repeats, seq_len, head_dim])
}

/// Additive bidirectional sliding-window mask [seq, seq]: 0 where
/// |i − j| ≤ window, MASK_MIN elsewhere.
fn sliding_window_mask<B: Backend>(
    seq_len: usize,
    window: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for row in 0..seq_len {
        for col in 0..seq_len {
            let visible = row.abs_diff(col) <= window;
            data.push(if visible { 0.0 } else { MASK_MIN });
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [seq_len, seq_len]), device)
}

/// randn(shape) / sqrt(hidden), matching the reference parameter init.
fn randn_scaled<B: Backend, const D: usize>(
    shape: [usize; D],
    hidden: usize,
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::random(
        shape,
        Distribution::Normal(0.0, 1.0 / (hidden as f64).sqrt()),
        device,
    )
}

fn linear_no_bias<B: Backend>(device: &B::Device, d_input: usize, d_output: usize) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init(device)
}

fn linear_with_bias<B: Backend>(device: &B::Device, d_input: usize, d_output: usize) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_bias(true)
        .with_layout(LinearLayout::Col)
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    /// Tiny DiT: hidden 64, 2 layers (layer 0 sliding, layer 1 full), 4 heads,
    /// 2 KV heads, head_dim 16, intermediate 128, in_channels 12 = 4 (xt) +
    /// 4 (src) + 4 (chunk_mask), acoustic latent dim 4, patch 2, window 2.
    fn tiny_config() -> AceStepConfig {
        let base = AceStepConfig::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/acestep/testdata/acestep_v15_turbo_config.json"),
        )
        .expect("reference config must parse");
        AceStepConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            sliding_window: 2,
            in_channels: 12,
            audio_acoustic_hidden_dim: 4,
            patch_size: 2,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            ..base
        }
    }

    fn tiny_model() -> AceStepDiT<TestBackend> {
        let device = Default::default();
        AceStepDiT::new(&tiny_config(), &device)
    }

    fn assert_finite(tensor: &Tensor<TestBackend, 3>, what: &str) {
        let values = tensor
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("output should materialize as f32");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{what} contains non-finite values"
        );
    }

    #[test]
    fn forward_shapes_and_finite() {
        let model = tiny_model();
        let device = Default::default();
        let xt =
            Tensor::<TestBackend, 3>::random([1, 10, 4], Distribution::Normal(0.0, 1.0), &device);
        let context =
            Tensor::<TestBackend, 3>::random([1, 10, 8], Distribution::Normal(0.0, 1.0), &device);
        let encoder =
            Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Normal(0.0, 1.0), &device);

        let v = model.forward(xt, 0.5, context, encoder);
        assert_eq!(v.dims(), [1, 10, 4]);
        assert_finite(&v, "velocity");
    }

    #[test]
    fn odd_sequence_length_is_padded_and_cropped() {
        let model = tiny_model();
        let device = Default::default();
        let xt =
            Tensor::<TestBackend, 3>::random([1, 9, 4], Distribution::Normal(0.0, 1.0), &device);
        let context =
            Tensor::<TestBackend, 3>::random([1, 9, 8], Distribution::Normal(0.0, 1.0), &device);
        let encoder =
            Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Normal(0.0, 1.0), &device);

        let v = model.forward(xt, 0.5, context, encoder);
        assert_eq!(v.dims(), [1, 9, 4]);
        assert_finite(&v, "velocity for odd T");
    }

    #[test]
    fn forward_with_cached_kv_matches_forward() {
        let model = tiny_model();
        let device = Default::default();
        let xt =
            Tensor::<TestBackend, 3>::random([1, 10, 4], Distribution::Normal(0.0, 1.0), &device);
        let context =
            Tensor::<TestBackend, 3>::random([1, 10, 8], Distribution::Normal(0.0, 1.0), &device);
        let encoder =
            Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Normal(0.0, 1.0), &device);

        let direct = model.forward(xt.clone(), 0.5, context.clone(), encoder.clone());
        let kv = model.prepare_cross_kv(encoder);
        let cached = model.forward_with_kv(xt, 0.5, context, &kv);
        let diff = (direct - cached).abs().max().to_data().to_vec::<f32>();
        assert!(
            diff.expect("max diff")[0] <= 0.0,
            "cached-KV forward must be numerically identical"
        );
    }

    #[test]
    fn sliding_window_mask_band_pattern() {
        let device = Default::default();
        let mask = sliding_window_mask::<TestBackend>(5, 2, &device);
        assert_eq!(mask.dims(), [5, 5]);
        let values = mask.to_data().to_vec::<f32>().expect("mask values");
        for row in 0..5usize {
            for col in 0..5usize {
                let expected = if row.abs_diff(col) <= 2 {
                    0.0
                } else {
                    MASK_MIN
                };
                assert_eq!(
                    values[row * 5 + col],
                    expected,
                    "mask[{row}][{col}] should be {expected}"
                );
            }
        }
    }

    #[test]
    fn timestep_embedding_shapes_and_cos_sin_order() {
        let model = tiny_model();
        let device = Default::default();

        let (temb, proj) = model.time_embed.forward(0.5, &device);
        assert_eq!(temb.dims(), [1, 64]);
        assert_eq!(proj.dims(), [1, 6, 64]);

        // At t = 0 the sinusoid is cos(0)=1 in the first half, sin(0)=0 in the
        // second half (COS-then-SIN order).
        let sinusoid = sinusoidal_timestep::<TestBackend>(0.0, TIME_EMBED_CHANNELS, &device);
        let values = sinusoid.to_data().to_vec::<f32>().expect("sinusoid values");
        let half = TIME_EMBED_CHANNELS / 2;
        assert!(values[..half].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(values[half..].iter().all(|v| v.abs() < 1e-6));

        // time_embed_r(0) contributes a nonzero constant; it must not be dropped.
        let (temb_r, proj_r) = model.time_embed_r.forward(0.0, &device);
        let temb_r_values = temb_r.to_data().to_vec::<f32>().expect("temb_r values");
        let proj_r_values = proj_r.to_data().to_vec::<f32>().expect("proj_r values");
        assert!(temb_r_values.iter().any(|v| v.abs() > 1e-6));
        assert!(proj_r_values.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn sample_turbo_euler_loop() {
        let model = tiny_model();
        let device = Default::default();
        let noise =
            Tensor::<TestBackend, 3>::random([1, 10, 4], Distribution::Normal(0.0, 1.0), &device);
        let context =
            Tensor::<TestBackend, 3>::random([1, 10, 8], Distribution::Normal(0.0, 1.0), &device);
        let encoder =
            Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Normal(0.0, 1.0), &device);
        let timesteps = [1.0, 0.75, 0.5, 0.3];

        let mut calls = 0usize;
        let mut progress = |done: usize, total: usize| {
            calls += 1;
            assert_eq!(done, calls);
            assert_eq!(total, timesteps.len());
        };
        let latent = model.sample_turbo(noise, context, encoder, &timesteps, Some(&mut progress));

        assert_eq!(calls, timesteps.len());
        assert_eq!(latent.dims(), [1, 10, 4]);
        assert_finite(&latent, "sampled latent");
    }

    #[test]
    fn default_schedule_has_eight_descending_steps() {
        assert_eq!(TURBO_TIMESTEPS.len(), 8);
        assert_eq!(TURBO_TIMESTEPS[0], 1.0);
        assert_eq!(TURBO_TIMESTEPS[7], 0.3);
        assert!(TURBO_TIMESTEPS.windows(2).all(|w| w[0] > w[1]));
    }
}
