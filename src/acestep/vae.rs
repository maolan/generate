//! Burn port of the decoder side of the diffusers `AutoencoderOobleck` VAE
//! used by ACE-Step 1.5 (see `vae/config.json` in the checkpoint).
//!
//! Maps DiT output latents of shape `[B, T, decoder_input_channels]` (25 Hz,
//! time-major) to 48 kHz stereo audio of shape `[B, audio_channels,
//! T * hop_length]`, where `hop_length` is the product of
//! `downsampling_ratios` (1920 for the released model).
//!
//! All convolutions are weight-normalized, matching
//! `torch.nn.utils.weight_norm` in the reference: each conv stores
//! `weight_g` `[out, 1, 1]` and `weight_v` (the unnormalized kernel) and the
//! effective weight is `g * v / (||v|| + 1e-12)` with the norm taken per
//! output channel over the remaining dims.
//!
//! # Canonical burnpack tensor names
//!
//! Tensor names are the module paths of [`OobleckDecoder`]. The offline
//! converter produces them from the official
//! `vae/diffusion_pytorch_model.safetensors` names by stripping the leading
//! `decoder.` prefix (e.g. `decoder.block.0.conv_t1.weight_v` becomes
//! `block.0.conv_t1.weight_v`):
//!
//! - `conv1.{weight_g, weight_v, bias}` — input conv, k=7, p=3
//! - `block.{i}.snake1.{alpha, beta}`
//! - `block.{i}.conv_t1.{weight_g, weight_v, bias}` — upsampling transpose
//!   conv, k = 2*stride, p = ceil(stride/2)
//! - `block.{i}.res_unit{1,2,3}.snake{1,2}.{alpha, beta}`
//! - `block.{i}.res_unit{1,2,3}.conv1.{weight_g, weight_v, bias}` — k=7,
//!   dilation 1/3/9, p = 3*dilation
//! - `block.{i}.res_unit{1,2,3}.conv2.{weight_g, weight_v, bias}` — k=1
//! - `snake1.{alpha, beta}`
//! - `conv2.{weight_g, weight_v}` — output conv, k=7, p=3, **no bias**
//!
//! `block.{i}` enumerates the decoder blocks in decode order, i.e. using
//! `downsampling_ratios` reversed (`[10, 6, 4, 4, 2]` for the released
//! model).

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::module::{conv_transpose1d, conv1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn_store::{BurnpackStore, ModuleSnapshot};
use serde::{Deserialize, Serialize};

/// Configuration for the Oobleck VAE, parsed from `vae/config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OobleckVaeConfig {
    /// Latent dimension produced by the DiT (input channels of the decoder).
    pub decoder_input_channels: usize,
    /// Base channel count of the decoder.
    pub decoder_channels: usize,
    /// Per-stage channel multiples applied to `decoder_channels`.
    pub channel_multiples: Vec<usize>,
    /// Encoder downsampling ratios; the decoder upsamples by these reversed.
    pub downsampling_ratios: Vec<usize>,
    /// Number of audio channels in the decoded output (2 = stereo).
    pub audio_channels: usize,
    /// Sampling rate of the decoded audio in Hz.
    pub sampling_rate: usize,
}

impl OobleckVaeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read VAE config from {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse VAE config from {}", path.display()))
    }

    /// Number of audio samples produced per latent frame.
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }
}

/// Snake activation with learned per-channel `alpha` and `beta` in log scale:
/// `x + sin(exp(alpha) * x)^2 / (exp(beta) + 1e-9)`.
#[derive(Module, Debug)]
pub struct Snake1d<B: Backend> {
    pub alpha: Param<Tensor<B, 3>>,
    pub beta: Param<Tensor<B, 3>>,
}

impl<B: Backend> Snake1d<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::zeros([1, channels, 1], device)),
            beta: Param::from_tensor(Tensor::zeros([1, channels, 1], device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha = self.alpha.val().exp();
        let beta = self.beta.val().exp();
        x.clone() + (alpha * x).sin().powf_scalar(2.0) / (beta + 1e-9)
    }
}

/// Weight-normalized 1D convolution (stride 1 "same" convs and the k=7
/// input/output convs of the decoder).
#[derive(Module, Debug)]
pub struct WnConv1d<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Option<Param<Tensor<B, 1>>>,
    padding: usize,
    dilation: usize,
}

impl<B: Backend> WnConv1d<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            weight_g: Param::from_tensor(Tensor::ones([out_channels, 1, 1], device)),
            weight_v: Param::from_tensor(Tensor::zeros(
                [out_channels, in_channels, kernel_size],
                device,
            )),
            bias: bias.then(|| Param::from_tensor(Tensor::zeros([out_channels], device))),
            padding,
            dilation,
        }
    }

    /// Effective weight `g * v / (||v|| + 1e-12)`, norm per output channel.
    fn weight(&self) -> Tensor<B, 3> {
        let g = self.weight_g.val();
        let v = self.weight_v.val();
        let out_channels = v.dims()[0];
        let v_norm = v
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sum_dim(1)
            .sqrt()
            .reshape([out_channels, 1, 1]);
        g * v / (v_norm + 1e-12)
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let bias = self.bias.as_ref().map(|bias| bias.val());
        conv1d(
            x,
            self.weight(),
            bias,
            ConvOptions::new([1], [self.padding], [self.dilation], 1),
        )
    }
}

/// Weight-normalized 1D transpose convolution used for upsampling.
#[derive(Module, Debug)]
pub struct WnConvTranspose1d<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
}

impl<B: Backend> WnConvTranspose1d<B> {
    fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            weight_g: Param::from_tensor(Tensor::ones([in_channels, 1, 1], device)),
            weight_v: Param::from_tensor(Tensor::zeros(
                [in_channels, out_channels, 2 * stride],
                device,
            )),
            bias: Param::from_tensor(Tensor::zeros([out_channels], device)),
            stride,
            padding: stride.div_ceil(2),
        }
    }

    /// Effective weight `g * v / (||v|| + 1e-12)`, norm per input channel
    /// (dim 0 of the transpose-conv kernel, as in `weight_norm` with dim=0).
    fn weight(&self) -> Tensor<B, 3> {
        let g = self.weight_g.val();
        let v = self.weight_v.val();
        let in_channels = v.dims()[0];
        let v_norm = v
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sum_dim(1)
            .sqrt()
            .reshape([in_channels, 1, 1]);
        g * v / (v_norm + 1e-12)
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv_transpose1d(
            x,
            self.weight(),
            Some(self.bias.val()),
            ConvTransposeOptions::new([self.stride], [self.padding], [0], [1], 1),
        )
    }
}

/// Residual unit: `x + conv2(snake2(conv1(snake1(x))))` with a dilated k=7
/// conv and a k=1 conv. With same-padding the sequence length is preserved,
/// so the length-mismatch crop in the reference is a no-op and is omitted.
#[derive(Module, Debug)]
pub struct OobleckResidualUnit<B: Backend> {
    pub snake1: Snake1d<B>,
    pub conv1: WnConv1d<B>,
    pub snake2: Snake1d<B>,
    pub conv2: WnConv1d<B>,
}

impl<B: Backend> OobleckResidualUnit<B> {
    pub fn new(channels: usize, dilation: usize, device: &B::Device) -> Self {
        Self {
            snake1: Snake1d::new(channels, device),
            conv1: WnConv1d::new(channels, channels, 7, 3 * dilation, dilation, true, device),
            snake2: Snake1d::new(channels, device),
            conv2: WnConv1d::new(channels, channels, 1, 0, 1, true, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let out = self.conv1.forward(self.snake1.forward(x));
        let out = self.conv2.forward(self.snake2.forward(out));
        residual + out
    }
}

/// Decoder block: Snake -> ConvTranspose1d upsampling -> 3 residual units
/// with dilations 1, 3, 9.
#[derive(Module, Debug)]
pub struct OobleckDecoderBlock<B: Backend> {
    pub snake1: Snake1d<B>,
    pub conv_t1: WnConvTranspose1d<B>,
    pub res_unit1: OobleckResidualUnit<B>,
    pub res_unit2: OobleckResidualUnit<B>,
    pub res_unit3: OobleckResidualUnit<B>,
}

impl<B: Backend> OobleckDecoderBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            snake1: Snake1d::new(in_channels, device),
            conv_t1: WnConvTranspose1d::new(in_channels, out_channels, stride, device),
            res_unit1: OobleckResidualUnit::new(out_channels, 1, device),
            res_unit2: OobleckResidualUnit::new(out_channels, 3, device),
            res_unit3: OobleckResidualUnit::new(out_channels, 9, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv_t1.forward(self.snake1.forward(x));
        let x = self.res_unit1.forward(x);
        let x = self.res_unit2.forward(x);
        self.res_unit3.forward(x)
    }
}

/// Oobleck VAE decoder. See the module-level docs for the canonical burnpack
/// tensor names.
#[derive(Module, Debug)]
pub struct OobleckDecoder<B: Backend> {
    pub conv1: WnConv1d<B>,
    pub block: Vec<OobleckDecoderBlock<B>>,
    pub snake1: Snake1d<B>,
    pub conv2: WnConv1d<B>,
    /// Audio samples per latent frame (product of `downsampling_ratios`).
    pub hop_length: usize,
    latent_channels: usize,
}

impl<B: Backend> OobleckDecoder<B> {
    pub fn new(config: &OobleckVaeConfig, device: &B::Device) -> Self {
        // Channel multiples with the prepended base multiple, as in the
        // reference: [1] + channel_multiples.
        let mut multiples = Vec::with_capacity(config.channel_multiples.len() + 1);
        multiples.push(1);
        multiples.extend_from_slice(&config.channel_multiples);

        let num_stages = config.downsampling_ratios.len();
        let top_channels = config.decoder_channels * multiples[num_stages];

        let conv1 = WnConv1d::new(
            config.decoder_input_channels,
            top_channels,
            7,
            3,
            1,
            true,
            device,
        );

        // Decode order uses the downsampling ratios reversed.
        let block = (0..num_stages)
            .map(|i| {
                let stride = config.downsampling_ratios[num_stages - 1 - i];
                let in_channels = config.decoder_channels * multiples[num_stages - i];
                let out_channels = config.decoder_channels * multiples[num_stages - i - 1];
                OobleckDecoderBlock::new(in_channels, out_channels, stride, device)
            })
            .collect();

        let snake1 = Snake1d::new(config.decoder_channels, device);
        let conv2 = WnConv1d::new(
            config.decoder_channels,
            config.audio_channels,
            7,
            3,
            1,
            false,
            device,
        );

        Self {
            conv1,
            block,
            snake1,
            conv2,
            hop_length: config.hop_length(),
            latent_channels: config.decoder_input_channels,
        }
    }

    /// Load decoder weights from a burnpack file using the canonical tensor
    /// names documented at the top of this module.
    pub fn from_burnpack(
        config: &OobleckVaeConfig,
        path: &Path,
        device: &B::Device,
    ) -> Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = BurnpackStore::from_file(path).zero_copy(true);
        model.load_from(&mut store).map_err(|err| {
            anyhow::anyhow!("failed to load VAE decoder from {}: {err}", path.display())
        })?;
        Ok(model)
    }

    /// Forward pass on channel-major latents `[B, decoder_input_channels, T]`
    /// returning audio `[B, audio_channels, T * hop_length]`.
    pub fn forward(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv1.forward(latents);
        let x = self
            .block
            .iter()
            .fold(x, |hidden, block| block.forward(hidden));
        let x = self.snake1.forward(x);
        self.conv2.forward(x)
    }

    /// Decode time-major DiT latents `[B, T, decoder_input_channels]` into
    /// audio `[B, audio_channels, T * hop_length]`.
    ///
    /// Every released ratio is even, in which case each transpose conv
    /// upsamples by exactly its stride and the output length is already
    /// `T * hop_length`; the final trim/pad only matters for hypothetical
    /// odd ratios, where ConvTranspose1d falls one sample short per stage.
    pub fn decode(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, frames, channels] = latents.dims();
        assert_eq!(
            channels, self.latent_channels,
            "expected {} latent channels, got {channels}",
            self.latent_channels
        );
        let audio = self.forward(latents.swap_dims(1, 2));

        let target = frames * self.hop_length;
        let [_, _, length] = audio.dims();
        if length > target {
            audio.slice([0..batch, 0..self.conv2_channels(), 0..target])
        } else if length < target {
            let device = audio.device();
            let padding = Tensor::zeros([batch, self.conv2_channels(), target - length], &device);
            Tensor::cat(vec![audio, padding], 2)
        } else {
            audio
        }
    }

    fn conv2_channels(&self) -> usize {
        self.conv2.weight_v.dims()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> OobleckVaeConfig {
        OobleckVaeConfig {
            decoder_input_channels: 4,
            decoder_channels: 8,
            channel_multiples: vec![1, 2],
            downsampling_ratios: vec![2, 3],
            audio_channels: 2,
            sampling_rate: 48_000,
        }
    }

    #[test]
    fn decode_produces_stereo_audio_of_expected_length() {
        let device = NdArrayDevice::default();
        let decoder = OobleckDecoder::<TestBackend>::new(&tiny_config(), &device);

        let latents = Tensor::<TestBackend, 3>::random(
            [1, 5, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let audio = decoder.decode(latents);

        assert_eq!(audio.dims(), [1, 2, 5 * 6]);
        let values: Vec<f32> = audio.into_data().to_vec().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_upsamples_each_stage() {
        let device = NdArrayDevice::default();
        let config = tiny_config();
        let decoder = OobleckDecoder::<TestBackend>::new(&config, &device);

        // Stage strides are the downsampling ratios reversed: [3, 2].
        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 5], &device);
        let x = decoder.conv1.forward(x);
        assert_eq!(x.dims(), [1, 16, 5]);
        let x = decoder.block[0].forward(x);
        assert_eq!(x.dims(), [1, 8, 14]); // odd stride 3: (5-1)*3 - 4 + 6
        let x = decoder.block[1].forward(x);
        assert_eq!(x.dims(), [1, 8, 28]); // even stride 2: 14 * 2
    }

    #[test]
    fn residual_unit_preserves_shape() {
        let device = NdArrayDevice::default();
        let unit = OobleckResidualUnit::<TestBackend>::new(8, 3, &device);
        let x = Tensor::<TestBackend, 3>::random(
            [2, 8, 11],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        assert_eq!(unit.forward(x).dims(), [2, 8, 11]);
    }

    #[test]
    fn loads_real_vae_config() {
        let config = OobleckVaeConfig::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/acestep/testdata/oobleck_vae_config.json"),
        )
        .unwrap();
        assert_eq!(config.decoder_input_channels, 64);
        assert_eq!(config.decoder_channels, 128);
        assert_eq!(config.channel_multiples, vec![1, 2, 4, 8, 16]);
        assert_eq!(config.downsampling_ratios, vec![2, 4, 4, 6, 10]);
        assert_eq!(config.audio_channels, 2);
        assert_eq!(config.sampling_rate, 48_000);
        assert_eq!(config.hop_length(), 1920);
    }
}
