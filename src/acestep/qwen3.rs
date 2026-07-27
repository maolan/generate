//! Qwen3 transformer ported to Burn (ACE-Step 1.5 text encoder / 5Hz LM planner).
//!
//! Canonical burnpack tensor names (the offline converter renames the official
//! Qwen3 checkpoint tensors to exactly these paths):
//!
//! - `embed_tokens.weight`
//! - `layers.{i}.self_attn.q_proj.weight`
//! - `layers.{i}.self_attn.k_proj.weight`
//! - `layers.{i}.self_attn.v_proj.weight`
//! - `layers.{i}.self_attn.o_proj.weight`
//! - `layers.{i}.self_attn.q_norm.weight`
//! - `layers.{i}.self_attn.k_norm.weight`
//! - `layers.{i}.mlp.gate_proj.weight`
//! - `layers.{i}.mlp.up_proj.weight`
//! - `layers.{i}.mlp.down_proj.weight`
//! - `layers.{i}.input_layernorm.weight`
//! - `layers.{i}.post_attention_layernorm.weight`
//! - `norm.weight`

use anyhow::{Context, Result};
use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout};
use burn::prelude::Backend;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, DType, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

impl Qwen3Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read Qwen3 config from {}", path.display()))?;
        let config: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse Qwen3 config from {}", path.display()))?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        if self.head_dim == 0 {
            self.hidden_size / self.num_attention_heads
        } else {
            self.head_dim
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen3RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend> Qwen3RmsNorm<B> {
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

#[derive(Module, Debug)]
pub struct Qwen3Attention<B: Backend> {
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

impl<B: Backend> Qwen3Attention<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        let head_dim = config.head_dim();
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

    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        causal: bool,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
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

        // Per-head RMSNorm on the head dim, applied before RoPE (Qwen3 style).
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        let q = apply_rotary_pos_emb(q, cos, sin).swap_dims(1, 2);
        let k = apply_rotary_pos_emb(k, cos, sin).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = if num_heads != num_kv_heads {
            let repeats = num_heads / num_kv_heads;
            (repeat_kv(k, repeats), repeat_kv(v, repeats))
        } else {
            (k, v)
        };

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.meta.scaling);
        let scores = if causal {
            let mask = causal_mask::<B>(seq_len, &scores.device());
            scores.mask_fill(mask, -1.0e9)
        } else {
            scores
        };
        let weights = softmax(scores, 3);
        let attended =
            weights
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attended)
    }

    /// Incremental forward: `hidden` holds the new positions only; keys/values
    /// are appended to `cache`. `cos`/`sin` must cover exactly the new
    /// positions (i.e. be built with the running `position_offset`), and
    /// `position_offset` is the absolute position of the first new token.
    pub fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        cache: &mut Qwen3AttentionCache<B>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        position_offset: usize,
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

        // Per-head RMSNorm on the head dim, applied before RoPE (Qwen3 style).
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        let q = apply_rotary_pos_emb(q, cos, sin).swap_dims(1, 2);
        let k = apply_rotary_pos_emb(k, cos, sin).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let k = match cache.key.take() {
            Some(prev) => Tensor::cat(vec![prev, k], 2),
            None => k,
        };
        let v = match cache.value.take() {
            Some(prev) => Tensor::cat(vec![prev, v], 2),
            None => v,
        };
        let kv_len = k.dims()[2];
        cache.key = Some(k.clone());
        cache.value = Some(v.clone());

        let (k, v) = if num_heads != num_kv_heads {
            let repeats = num_heads / num_kv_heads;
            (repeat_kv(k, repeats), repeat_kv(v, repeats))
        } else {
            (k, v)
        };

        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.meta.scaling);
        // New query row r sits at absolute position `position_offset + r` and
        // may attend to every cached key up to and including itself.
        let mask = incremental_causal_mask::<B>(seq_len, kv_len, position_offset, &scores.device());
        let scores = scores.mask_fill(mask, -1.0e9);
        let weights = softmax(scores, 3);
        let attended =
            weights
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq_len, num_heads * head_dim]);
        self.o_proj.forward(attended)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Qwen3Mlp<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
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

/// Per-layer key/value cache for incremental decoding.
///
/// Tensors are stored post-RoPE as `[batch, num_kv_heads, seq, head_dim]`
/// (before grouped-query expansion).
#[derive(Clone, Debug)]
pub struct Qwen3AttentionCache<B: Backend> {
    pub key: Option<Tensor<B, 4>>,
    pub value: Option<Tensor<B, 4>>,
}

impl<B: Backend> Qwen3AttentionCache<B> {
    fn empty() -> Self {
        Self {
            key: None,
            value: None,
        }
    }
}

/// Whole-model KV cache, one entry per decoder layer.
#[derive(Clone, Debug)]
pub struct Qwen3KvCache<B: Backend> {
    pub layers: Vec<Qwen3AttentionCache<B>>,
}

#[derive(Module, Debug)]
pub struct Qwen3DecoderLayer<B: Backend> {
    pub self_attn: Qwen3Attention<B>,
    pub mlp: Qwen3Mlp<B>,
    pub input_layernorm: Qwen3RmsNorm<B>,
    pub post_attention_layernorm: Qwen3RmsNorm<B>,
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            self_attn: Qwen3Attention::new(config, device),
            mlp: Qwen3Mlp::new(config, device),
            input_layernorm: Qwen3RmsNorm::new(device, config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: Qwen3RmsNorm::new(
                device,
                config.hidden_size,
                config.rms_norm_eps,
            ),
        }
    }

    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        causal: bool,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let normed = self.input_layernorm.forward(hidden);
        let hidden = residual + self.self_attn.forward(normed, causal, cos, sin);

        let residual = hidden.clone();
        let normed = self.post_attention_layernorm.forward(hidden);
        residual + self.mlp.forward(normed)
    }

    pub fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        cache: &mut Qwen3AttentionCache<B>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        position_offset: usize,
    ) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let normed = self.input_layernorm.forward(hidden);
        let hidden = residual
            + self
                .self_attn
                .forward_cached(normed, cache, cos, sin, position_offset);

        let residual = hidden.clone();
        let normed = self.post_attention_layernorm.forward(hidden);
        residual + self.mlp.forward(normed)
    }
}

#[derive(Module, Debug)]
pub struct Qwen3Model<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen3DecoderLayer<B>>,
    pub norm: Qwen3RmsNorm<B>,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> Qwen3Model<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers: (0..config.num_hidden_layers)
                .map(|_| Qwen3DecoderLayer::new(config, device))
                .collect(),
            norm: Qwen3RmsNorm::new(device, config.hidden_size, config.rms_norm_eps),
            head_dim: config.head_dim(),
            rope_theta: config.rope_theta,
        }
    }

    pub fn from_burnpack(config: &Qwen3Config, path: &Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = BurnpackStore::from_file(path).zero_copy(true);
        model
            .load_from(&mut store)
            .with_context(|| format!("failed to load Qwen3 weights from {}", path.display()))?;
        Ok(model)
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
        causal: bool,
    ) -> Tensor<B, 3> {
        let [_, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let (cos, sin) = rotary_cos_sin::<B>(seq_len, self.head_dim, self.rope_theta, &device);

        let mut hidden = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            hidden = layer.forward(hidden, causal, &cos, &sin);
        }
        self.norm.forward(hidden)
    }

    /// Fresh KV cache sized to the model's layer count.
    pub fn new_cache(&self) -> Qwen3KvCache<B> {
        Qwen3KvCache {
            layers: (0..self.layers.len())
                .map(|_| Qwen3AttentionCache::empty())
                .collect(),
        }
    }

    /// Incremental causal forward. `token_ids` holds only the new tokens
    /// (the full prompt on prefill, one token per decode step afterwards);
    /// `position_offset` is the absolute position of the first new token.
    /// Attention is always causal against the cached prefix plus the new
    /// tokens, matching `forward(.., causal = true)`.
    pub fn forward_cached(
        &self,
        token_ids: Tensor<B, 2, burn::tensor::Int>,
        cache: &mut Qwen3KvCache<B>,
        position_offset: usize,
    ) -> Tensor<B, 3> {
        let [_, seq_len] = token_ids.dims();
        let device = token_ids.device();
        let (cos, sin) = rotary_cos_sin_at::<B>(
            seq_len,
            position_offset,
            self.head_dim,
            self.rope_theta,
            &device,
        );

        let mut hidden = self.embed_tokens.forward(token_ids);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_cached(hidden, layer_cache, &cos, &sin, position_offset);
        }
        self.norm.forward(hidden)
    }

    /// Tied lm_head weight (`tie_word_embeddings = true`): the
    /// `[vocab_size, hidden_size]` embedding table. Logits for a hidden state
    /// are `hidden @ weight.swap_dims(0, 1)`.
    pub fn embedding_weight(&self) -> Tensor<B, 2> {
        self.embed_tokens.weight.val()
    }
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
    rotary_cos_sin_at::<B>(seq_len, 0, head_dim, theta, device)
}

/// Same as [`rotary_cos_sin`] but over absolute positions
/// `offset..offset + seq_len` (used by the KV-cache incremental path).
fn rotary_cos_sin_at<B: Backend>(
    seq_len: usize,
    offset: usize,
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
    for pos in offset..offset + seq_len {
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

fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4, Bool> {
    let mut mask = Vec::with_capacity(seq_len * seq_len);
    for row in 0..seq_len {
        for col in 0..seq_len {
            mask.push(col > row);
        }
    }
    Tensor::<B, 4, Bool>::from_data(TensorData::new(mask, [1, 1, seq_len, seq_len]), device)
}

/// Mask of shape [1, 1, query_len, key_len] for the KV-cache path: query row r
/// (absolute position `offset + r`) may not attend to keys past itself.
fn incremental_causal_mask<B: Backend>(
    query_len: usize,
    key_len: usize,
    offset: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let mut mask = Vec::with_capacity(query_len * key_len);
    for row in 0..query_len {
        for col in 0..key_len {
            mask.push(col > offset + row);
        }
    }
    Tensor::<B, 4, Bool>::from_data(TensorData::new(mask, [1, 1, query_len, key_len]), device)
}

fn linear_no_bias<B: Backend>(device: &B::Device, d_input: usize, d_output: usize) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Int;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab_size: 128,
            max_position_embeddings: 512,
            tie_word_embeddings: true,
        }
    }

    fn tiny_model() -> Qwen3Model<TestBackend> {
        let device = Default::default();
        Qwen3Model::new(&tiny_config(), &device)
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
    fn forward_causal_and_bidirectional_shapes() {
        let model = tiny_model();
        let device = Default::default();
        let input_ids = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3, 4, 5]], &device);

        let causal_out = model.forward(input_ids.clone(), true);
        assert_eq!(causal_out.dims(), [1, 5, 64]);
        assert_finite(causal_out);

        let bidirectional_out = model.forward(input_ids, false);
        assert_eq!(bidirectional_out.dims(), [1, 5, 64]);
        assert_finite(bidirectional_out);
    }

    #[test]
    fn causal_mask_changes_output() {
        let model = tiny_model();
        let device = Default::default();
        let input_ids = Tensor::<TestBackend, 2, Int>::from_data([[7, 8, 9, 10]], &device);
        let causal = model.forward(input_ids.clone(), true);
        let bidirectional = model.forward(input_ids, false);
        let causal_first = causal
            .slice([0..1, 0..1, 0..64])
            .to_data()
            .to_vec::<f32>()
            .expect("causal slice");
        let bidirectional_first = bidirectional
            .slice([0..1, 0..1, 0..64])
            .to_data()
            .to_vec::<f32>()
            .expect("bidirectional slice");
        // Position 0 attends to all tokens when bidirectional, only itself when causal.
        assert_ne!(causal_first, bidirectional_first);
    }

    #[test]
    fn rotate_half_negates_and_swaps() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::from_data([[[[1.0, 2.0, 3.0, 4.0]]]], &device);
        let rotated = rotate_half(x);
        let values = rotated.to_data().to_vec::<f32>().expect("rotated values");
        assert_eq!(values, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn rotary_cos_sin_shapes_and_origin() {
        let device = Default::default();
        let (cos, sin) = rotary_cos_sin::<TestBackend>(7, 16, 1_000_000.0, &device);
        assert_eq!(cos.dims(), [1, 7, 1, 16]);
        assert_eq!(sin.dims(), [1, 7, 1, 16]);
        let cos_values = cos.to_data().to_vec::<f32>().expect("cos values");
        let sin_values = sin.to_data().to_vec::<f32>().expect("sin values");
        // Position 0: cos == 1, sin == 0 for every frequency.
        assert!(cos_values[..16].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(sin_values[..16].iter().all(|v| v.abs() < 1e-6));
        // Halves are duplicated: emb = cat([freqs, freqs]).
        let pos1 = &cos_values[16..32];
        assert_eq!(&pos1[..8], &pos1[8..16]);
        assert!(cos_values.iter().all(|v| v.is_finite()));
        assert!(sin_values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn load_embedding_model_config() {
        let config = Qwen3Config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/acestep/testdata/qwen3_embedding_config.json"),
        )
        .expect("should parse Qwen3-Embedding-0.6B config");
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim(), 128);
        assert!((config.rms_norm_eps - 1e-6).abs() < f64::EPSILON);
        assert!((config.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
        assert_eq!(config.vocab_size, 151669);
        assert_eq!(config.max_position_embeddings, 32768);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn forward_cached_matches_full_forward() {
        let model = tiny_model();
        let device = Default::default();
        let ids: [i64; 6] = [3, 5, 7, 11, 13, 17];

        let full = model.forward(
            Tensor::<TestBackend, 2, Int>::from_data([ids], &device),
            true,
        );
        let expected_last = full
            .slice([0..1, 5..6, 0..64])
            .to_data()
            .to_vec::<f32>()
            .expect("full forward last position");

        // Fully incremental: one token per step.
        let mut cache = model.new_cache();
        let mut last: Option<Tensor<TestBackend, 3>> = None;
        for (pos, &tok) in ids.iter().enumerate() {
            let step = Tensor::<TestBackend, 2, Int>::from_data([[tok]], &device);
            last = Some(model.forward_cached(step, &mut cache, pos));
        }
        let cached_last = last
            .expect("at least one step")
            .to_data()
            .to_vec::<f32>()
            .expect("cached last position");
        for (a, b) in expected_last.iter().zip(cached_last.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "token-by-token cache diverged: {a} vs {b}"
            );
        }

        // Prefill 4 tokens, then decode the remaining 2 one at a time.
        let mut cache = model.new_cache();
        let prefill = Tensor::<TestBackend, 2, Int>::from_data([[3, 5, 7, 11]], &device);
        model.forward_cached(prefill, &mut cache, 0);
        let step = Tensor::<TestBackend, 2, Int>::from_data([[13]], &device);
        model.forward_cached(step, &mut cache, 4);
        let step = Tensor::<TestBackend, 2, Int>::from_data([[17]], &device);
        let decoded = model
            .forward_cached(step, &mut cache, 5)
            .to_data()
            .to_vec::<f32>()
            .expect("prefill+decode last position");
        for (a, b) in expected_last.iter().zip(decoded.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "prefill+decode cache diverged: {a} vs {b}"
            );
        }
    }
}
