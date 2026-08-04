pub mod conv;
pub mod loader;

use anyhow::{Context, Result};
use burn::module::{Module, Param};
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, LinearLayout};
use burn::prelude::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot, ModuleStore};
use oxideav_core::{
    CodecId, CodecParameters, MediaType, Packet, RuntimeContext, SampleFormat, StreamInfo, TimeBase,
};
use rayon::prelude::*;

pub use conv::PostProcessor;
pub use conv::{PlainConv1d, WNConv1d, WNConvTranspose1d};

const HEARTMULA_SAMPLE_RATE: usize = 48_000;
const HEARTCODEC_WINDOW_FRAMES: usize = 93;
const HEARTCODEC_SEGMENT_DURATION_SECONDS: f32 = 29.76;
type TensorLookup = dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>;

#[derive(Debug, Clone)]
pub struct HeartCodecConfig {
    pub dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub num_quantizers: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_layers_2: usize,
    pub out_channels: usize,
    pub sample_rate: usize,
    pub latent_hidden_dim: usize,
    pub init_channel: usize,
    pub num_bands: usize,
    pub num_samples: usize,
    pub downsample_factors: [usize; 5],
    pub downsample_kernel_sizes: [usize; 5],
    pub upsample_factors: [usize; 5],
    pub upsample_kernel_sizes: [usize; 5],
    pub default_kernel_size: usize,
    pub delay_kernel_size: usize,
    pub res_kernel_size: usize,
    pub causal: bool,

    pub ode_steps: usize,
}

impl Default for HeartCodecConfig {
    fn default() -> Self {
        Self {
            dim: 512,
            codebook_size: 8192,
            codebook_dim: 32,
            num_quantizers: 8,
            attention_head_dim: 64,
            in_channels: 1024,
            num_attention_heads: 24,
            num_layers: 24,
            num_layers_2: 6,
            out_channels: 256,
            sample_rate: 48000,
            latent_hidden_dim: 128,
            init_channel: 64,
            num_bands: 1,
            num_samples: 2,
            downsample_factors: [3, 4, 4, 4, 5],
            downsample_kernel_sizes: [6, 8, 8, 8, 10],
            upsample_factors: [5, 4, 4, 4, 3],
            upsample_kernel_sizes: [10, 8, 8, 8, 6],
            default_kernel_size: 7,
            delay_kernel_size: 5,
            res_kernel_size: 7,
            causal: true,
            ode_steps: 10,
        }
    }
}

#[derive(Module, Debug)]
pub struct HeartCodecModel<B: Backend> {
    pub flow_matching: FlowMatching<B>,
    pub scalar_model: ScalarModel<B>,
    pub ode_steps: usize,
    pub guidance_scale: f32,
}

#[derive(Debug)]
pub struct ScalarDecodePlan<B: Backend> {
    pub target_len: usize,
    pub audio_target_len: usize,
    pub windows: Vec<Tensor<B, 3>>,
}

impl<B: Backend> HeartCodecModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let config = HeartCodecConfig::default();
        Self {
            flow_matching: FlowMatching::new(device, &config),
            scalar_model: ScalarModel::new(device, &config),
            ode_steps: config.ode_steps,
            guidance_scale: 1.0,
        }
    }

    pub fn with_ode_steps(mut self, steps: usize) -> Self {
        self.ode_steps = steps.clamp(1, 50);
        self
    }

    pub fn with_guidance_scale(mut self, scale: f32) -> Self {
        self.guidance_scale = scale.max(1.0);
        self
    }

    pub fn from_burnpack(path: &std::path::Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(path).zero_copy(true);

        if let Err(_e) = model.load_from(&mut store) {
            model = Self::load_with_mapping(path, device)?;
        }

        Ok(model)
    }

    fn load_flow_matching_manually<F>(
        _flow_matching: &mut FlowMatching<B>,
        _path: &std::path::Path,
        device: &B::Device,
        get_tensor: &F,
    ) -> Result<FlowMatching<B>>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let config = HeartCodecConfig::default();

        let cond_feature_emb = load_linear_from_tensors(
            device,
            get_tensor,
            "flow_matching.cond_feature_emb",
            config.dim,
            config.dim,
        )?;
        let zero_cond_embedding1 =
            if let Some((data, shape)) = get_tensor("flow_matching.zero_cond_embedding1") {
                Tensor::<B, 1>::from_data(TensorData::new(data, [shape[0]]), device)
            } else {
                Tensor::zeros([config.dim], device)
            };

        let vq_embed = ResidualVQ::load_from_dot_notation(device, get_tensor)?;

        let estimator = LlamaTransformer::load_from_burnpack(device, get_tensor)?;

        Ok(FlowMatching {
            cond_feature_emb,
            zero_cond_embedding1: Param::from_tensor(zero_cond_embedding1),
            estimator,
            vq_embed,
            debug_latent_steps: false,
        })
    }

    fn load_with_mapping(path: &std::path::Path, device: &B::Device) -> Result<Self> {
        use burn::tensor::DType;
        use burn_store::ModuleStore;

        let snapshots = BurnpackStore::from_file(path)
            .zero_copy(true)
            .get_all_snapshots()
            .with_context(|| "failed to read burnpack snapshots")?
            .clone();

        let mut model = Self::new(device);

        let get_tensor = |name: &str| -> Option<(Vec<f32>, Vec<usize>)> {
            snapshots.iter().find_map(|(_, snap)| {
                if snap.full_path() == name {
                    let data = snap.to_data().ok()?;
                    if data.dtype == DType::F32 {
                        let shape = data.shape.to_vec();
                        let values: Vec<f32> = data.to_vec::<f32>().ok()?;
                        Some((values, shape))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        };

        match Self::load_flow_matching_manually(&mut model.flow_matching, path, device, &get_tensor)
        {
            Ok(flow_matching) => {
                model.flow_matching = flow_matching;
            }
            Err(_e) => {}
        }

        match ScalarModel::load_from_dot_notation(path, device, &get_tensor) {
            Ok(scalar_model) => {
                model.scalar_model = scalar_model;
            }
            Err(_e) => {}
        }

        Ok(model)
    }

    fn snapshots_to_f32_lookup(path: &std::path::Path) -> Result<Box<TensorLookup>> {
        let snapshots = BurnpackStore::from_file(path)
            .zero_copy(true)
            .get_all_snapshots()
            .with_context(|| "failed to read burnpack snapshots")?
            .clone();

        Ok(Box::new(
            move |name: &str| -> Option<(Vec<f32>, Vec<usize>)> {
                snapshots.iter().find_map(|(_, snap)| {
                    if snap.full_path() == name {
                        let data = snap.to_data().ok()?;
                        if data.dtype == DType::F32 {
                            let shape = data.shape.to_vec();
                            let values: Vec<f32> = data.to_vec::<f32>().ok()?;
                            Some((values, shape))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            },
        ))
    }

    pub(crate) fn build_scalar_decode_plan_impl(
        flow_matching: &FlowMatching<B>,
        guidance_scale: f32,
        ode_steps: usize,
        codes: Tensor<B, 3, Int>,
        first_latent: Tensor<B, 3>,
    ) -> ScalarDecodePlan<B> {
        let device = codes.device();
        let [batch, num_quantizers, seq_len] = codes.dims();
        assert_eq!(batch, 1, "HeartCodec decode expects a single code batch");

        let duration_seconds = seq_len as f32 / 12.5;
        let segment_duration_seconds = HEARTCODEC_SEGMENT_DURATION_SECONDS;
        let latent_length = (segment_duration_seconds * 25.0) as usize;
        assert_eq!(
            first_latent.dims(),
            [batch, latent_length, 256],
            "initial latent shape must match [batch, latent_length, 256]"
        );
        let min_samples = ((segment_duration_seconds * 12.5) as usize).max(1);
        let mut hop_samples = (min_samples / HEARTCODEC_WINDOW_FRAMES) * 80;
        let mut ovlp_samples = min_samples.saturating_sub(hop_samples);
        if hop_samples == 0 {
            hop_samples = 1;
            ovlp_samples = min_samples;
        }
        let ovlp_frames = ovlp_samples * 2;
        let target_len = (duration_seconds * HEARTMULA_SAMPLE_RATE as f32) as usize;
        let audio_target_len = target_len;
        let mut codes = codes;
        if seq_len < min_samples {
            while codes.dims()[2] < min_samples {
                codes = Tensor::cat(vec![codes.clone(), codes], 2);
            }
            codes = codes.slice([0..batch, 0..num_quantizers, 0..min_samples]);
        }

        let mut codes_len = codes.dims()[2];
        if !(codes_len.saturating_sub(ovlp_frames)).is_multiple_of(hop_samples) {
            let len_codes = codes_len.saturating_sub(ovlp_samples).div_ceil(hop_samples)
                * hop_samples
                + ovlp_samples;
            while codes.dims()[2] < len_codes {
                codes = Tensor::cat(vec![codes.clone(), codes], 2);
            }
            codes = codes.slice([0..batch, 0..num_quantizers, 0..len_codes]);
            codes_len = len_codes;
        }

        let mut windows = Vec::new();
        let mut previous_latent: Option<Tensor<B, 3>> = None;
        let mut sinx = 0usize;
        while sinx + min_samples <= codes_len {
            let window_end = (sinx + min_samples).min(codes_len);
            let codes_input = codes
                .clone()
                .slice([0..batch, 0..num_quantizers, sinx..window_end]);
            let window_latent_length = codes_input.dims()[2] * 2;

            if sinx == 0 || ovlp_frames == 0 {
                let initial_window_latent = first_latent.clone().slice([
                    0..batch,
                    0..window_latent_length.min(first_latent.dims()[1]),
                    0..first_latent.dims()[2],
                ]);
                let latent = Self::run_flow_matching_window(
                    flow_matching,
                    guidance_scale,
                    ode_steps,
                    codes_input,
                    initial_window_latent.clone(),
                    window_latent_length,
                    0,
                    Some(initial_window_latent),
                );
                windows.push(Self::latent_to_scalar_input(latent.clone()));
                previous_latent = Some(latent);
            } else {
                let prev_latent = previous_latent
                    .as_ref()
                    .expect("previous latent window is required when overlap is enabled");
                let true_latent = prev_latent.clone().slice([
                    0..batch,
                    prev_latent.dims()[1].saturating_sub(ovlp_frames)..prev_latent.dims()[1],
                    0..prev_latent.dims()[2],
                ]);
                let len_add_to_latent = window_latent_length.saturating_sub(true_latent.dims()[1]);
                let true_latent = if len_add_to_latent == 0 {
                    true_latent
                } else {
                    Tensor::cat(
                        vec![
                            true_latent.clone(),
                            Tensor::<B, 3>::random(
                                [batch, len_add_to_latent, true_latent.dims()[2]],
                                burn::tensor::Distribution::Normal(0.0, 1.0),
                                &device,
                            ),
                        ],
                        1,
                    )
                };
                let latent = Self::run_flow_matching_window(
                    flow_matching,
                    guidance_scale,
                    ode_steps,
                    codes_input,
                    true_latent,
                    window_latent_length,
                    ovlp_frames,
                    None,
                );
                windows.push(Self::latent_to_scalar_input(latent.clone()));
                previous_latent = Some(latent);
            }
            sinx += hop_samples.max(1);
        }

        ScalarDecodePlan {
            target_len,
            audio_target_len,
            windows,
        }
    }

    pub(crate) fn decode_scalar_plan_impl(
        scalar_model: &ScalarModel<B>,
        plan: ScalarDecodePlan<B>,
    ) -> Tensor<B, 3> {
        let device = plan
            .windows
            .first()
            .expect("expected at least one scalar decode window")
            .device();
        let min_samples = plan
            .windows
            .first()
            .map(|window| window.dims()[2] * HEARTMULA_SAMPLE_RATE / 25)
            .expect("expected at least one scalar decode window");
        let hop_samples = ((min_samples / HEARTCODEC_WINDOW_FRAMES) * 80).max(1);
        let ovlp_samples = min_samples.saturating_sub(hop_samples);
        let mut output: Option<Tensor<B, 3>> = None;

        for scalar_input in plan.windows {
            let mut cur_output = scalar_model.decode_with_sync(scalar_input);
            let cur_output_dims = cur_output.dims();
            cur_output = cur_output.slice([
                0..cur_output_dims[0],
                0..1,
                0..min_samples.min(cur_output_dims[2]),
            ]);
            if let Some(prev) = output {
                if ovlp_samples == 0 {
                    output = Some(Tensor::cat(vec![prev, cur_output], 2));
                } else {
                    let ov_win = {
                        let mut v = Vec::with_capacity(ovlp_samples);
                        for i in 0..ovlp_samples {
                            let denom = (ovlp_samples.saturating_sub(1)).max(1) as f32;
                            v.push(i as f32 / denom);
                        }
                        Tensor::<B, 3>::from_data(TensorData::new(v, [1, 1, ovlp_samples]), &device)
                    };
                    let prev_dims = prev.dims();
                    let prev_len = prev_dims[2];
                    let prev_head =
                        prev.clone()
                            .slice([0..prev_dims[0], 0..1, 0..prev_len - ovlp_samples]);
                    let prev_tail =
                        prev.slice([0..prev_dims[0], 0..1, prev_len - ovlp_samples..prev_len]);
                    let cur_dims = cur_output.dims();
                    let cur_head =
                        cur_output
                            .clone()
                            .slice([0..cur_dims[0], 0..1, 0..ovlp_samples]);
                    let prev_energy = prev_tail.clone().square();
                    let cur_energy = cur_head.clone().square();
                    let energy_sum = prev_energy.clone() + cur_energy.clone() + 1.0e-8;
                    let transient_cur = cur_energy / energy_sum;
                    let cur_weight = (ov_win.clone() + transient_cur) * 0.5;
                    let prev_weight =
                        Tensor::<B, 3>::ones([1, 1, ovlp_samples], &device) - cur_weight.clone();
                    let blended = prev_tail * prev_weight + cur_head * cur_weight;
                    output = Some(Tensor::cat(
                        vec![
                            prev_head,
                            blended,
                            cur_output.slice([0..cur_dims[0], 0..1, ovlp_samples..cur_dims[2]]),
                        ],
                        2,
                    ));
                }
            } else {
                output = Some(cur_output);
            }
        }

        output
            .expect("expected at least one decoded window")
            .slice([0..2, 0..1, 0..plan.target_len])
    }

    #[allow(clippy::too_many_arguments)]
    fn run_flow_matching_window(
        flow_matching: &FlowMatching<B>,
        guidance_scale: f32,
        ode_steps: usize,
        codes: Tensor<B, 3, Int>,
        true_latents: Tensor<B, 3>,
        latent_length: usize,
        incontext_length: usize,
        initial_latent_override: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        flow_matching.inference_codes(
            vec![codes],
            true_latents,
            latent_length,
            incontext_length,
            guidance_scale,
            ode_steps,
            false,
            "other_seg",
            initial_latent_override,
        )
    }

    fn latent_to_scalar_input(latent: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = latent.dims();
        assert_eq!(channels, 256, "Expected 256 channels from flow matching");

        let latent_reshaped = latent.reshape([batch, seq_len, 2, 128]);
        let latent_permuted = latent_reshaped.swap_dims(1, 2);
        let latent_split = latent_permuted.reshape([batch * 2, seq_len, 128]);

        latent_split.swap_dims(1, 2)
    }

    pub fn build_scalar_decode_plan(
        &self,
        codes: Tensor<B, 3, Int>,
        first_latent: Tensor<B, 3>,
    ) -> ScalarDecodePlan<B> {
        Self::build_scalar_decode_plan_impl(
            &self.flow_matching,
            self.guidance_scale,
            self.ode_steps,
            codes,
            first_latent,
        )
    }

    pub fn decode_scalar_plan(&self, plan: ScalarDecodePlan<B>) -> Tensor<B, 3> {
        Self::decode_scalar_plan_impl(&self.scalar_model, plan)
    }

    pub fn decode(&self, codes: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [batch, _num_quantizers, _seq_len] = codes.dims();
        let latent_length = (HEARTCODEC_SEGMENT_DURATION_SECONDS * 25.0) as usize;
        let first_latent = Tensor::<B, 3>::random(
            [batch, latent_length, 256],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &codes.device(),
        );
        self.decode_with_initial_latent(codes, first_latent)
    }

    pub fn decode_with_initial_latent(
        &self,
        codes: Tensor<B, 3, Int>,
        first_latent: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let plan = self.build_scalar_decode_plan(codes, first_latent);
        self.decode_scalar_plan(plan)
    }
}

#[derive(Module, Debug)]
pub struct FlowMatching<B: Backend> {
    pub cond_feature_emb: Linear<B>,
    pub zero_cond_embedding1: Param<Tensor<B, 1>>,
    pub estimator: LlamaTransformer<B>,
    pub vq_embed: ResidualVQ<B>,
    pub debug_latent_steps: bool,
}

impl<B: Backend> FlowMatching<B> {
    pub fn new(device: &B::Device, config: &HeartCodecConfig) -> Self {
        Self {
            cond_feature_emb: LinearConfig::new(config.dim, config.dim)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
            zero_cond_embedding1: Param::from_tensor(Tensor::zeros([config.dim], device)),
            estimator: LlamaTransformer::new(device, config),
            vq_embed: ResidualVQ::new(device, config),
            debug_latent_steps: false,
        }
    }

    pub fn load_from_burnpack(path: &std::path::Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(device, &HeartCodecConfig::default());
        let mut store = BurnpackStore::from_file(path).zero_copy(true);

        if let Err(load_err) = model.load_from(&mut store) {
            let get_tensor = HeartCodecModel::<B>::snapshots_to_f32_lookup(path)?;
            return HeartCodecModel::<B>::load_flow_matching_manually(
                &mut model, path, device, &get_tensor,
            )
            .map_err(|fallback_err| {
                anyhow::anyhow!(
                    "Failed to load FlowMatching: {load_err}; fallback mapping also failed: {fallback_err}"
                )
            });
        }

        Ok(model)
    }

    fn interpolate_1d(x: &Tensor<B, 3>, scale_factor: usize) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = x.dims();
        if scale_factor <= 1 {
            return x.clone();
        }

        let mut repeated_steps = Vec::with_capacity(seq_len * scale_factor);
        for step in 0..seq_len {
            let slice = x.clone().slice([0..batch, step..step + 1, 0..channels]);
            for _ in 0..scale_factor {
                repeated_steps.push(slice.clone());
            }
        }
        Tensor::cat(repeated_steps, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_ode(
        &self,
        conditioning: Tensor<B, 3>,
        true_latents: Tensor<B, 3>,
        _latent_length: usize,
        incontext_length: usize,
        num_steps: usize,
        guidance_scale: f32,
        initial_latent_override: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let device = conditioning.device();
        let [batch, _seq_len, _cond_dim] = conditioning.dims();

        let latent_dim = 256;
        let num_steps = num_steps.clamp(1, 50);

        let cond_interp = Self::interpolate_1d(&conditioning, 2);
        let [_batch, seq_len_interp, _] = cond_interp.dims();
        let latent_masks =
            Self::build_latent_masks(seq_len_interp, _latent_length, incontext_length);
        let masked_incontext_length = latent_masks.iter().filter(|&&mask| mask == 1).count();
        let zero_cond = self
            .zero_cond_embedding1
            .val()
            .clone()
            .reshape([1, 1, 512])
            .repeat(&[batch, seq_len_interp, 1]);
        let active_mask = Tensor::<B, 3>::from_data(
            TensorData::new(
                latent_masks
                    .iter()
                    .map(|&mask| if mask > 0 { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
                [1, seq_len_interp, 1],
            ),
            &device,
        )
        .repeat(&[batch, 1, 512]);
        let inactive_mask =
            Tensor::<B, 3>::ones([batch, seq_len_interp, 512], &device) - active_mask.clone();
        let cond_with_mask = cond_interp.clone() * active_mask + zero_cond.clone() * inactive_mask;
        let uncond_mask = Tensor::<B, 3>::zeros([batch, seq_len_interp, 512], &device);

        let mut latent = if let Some(initial_latent) = initial_latent_override {
            assert_eq!(
                initial_latent.dims(),
                [batch, seq_len_interp, latent_dim],
                "initial latent override shape must match [batch, seq_len_interp, latent_dim]"
            );
            initial_latent
        } else {
            Tensor::<B, 3>::random(
                [batch, seq_len_interp, latent_dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )
        };
        let incontext_mask = latent_masks
            .iter()
            .map(|&m| if m == 1 { 1_i64 } else { 0_i64 })
            .collect::<Vec<_>>();
        let incontext_mask = Tensor::<B, 3>::from_data(
            TensorData::new(
                incontext_mask
                    .into_iter()
                    .map(|v| v as f32)
                    .collect::<Vec<_>>(),
                [1, seq_len_interp, 1],
            ),
            &device,
        );
        let incontext_x = true_latents * incontext_mask;

        let dt = 1.0 / num_steps as f32;

        let sync_interval = if num_steps <= 5 { 5 } else { 3 };
        for step in 0..num_steps {
            let t = step as f32 * dt;

            if masked_incontext_length > 0 {
                let noise = latent.clone();
                let prefix =
                    noise
                        .clone()
                        .slice([0..batch, 0..masked_incontext_length, 0..latent_dim]);
                let incontext_prefix = incontext_x.clone().slice([
                    0..batch,
                    0..masked_incontext_length,
                    0..latent_dim,
                ]);
                let anchored = prefix * (1.0 - (1.0 - 1e-6) * t) + incontext_prefix * t;
                let suffix = noise.slice([
                    0..batch,
                    masked_incontext_length..seq_len_interp,
                    0..latent_dim,
                ]);
                latent = Tensor::cat(vec![anchored, suffix], 1);
            }

            let velocity = if guidance_scale > 1.0 {
                let uncond_input = Tensor::cat(
                    vec![latent.clone(), incontext_x.clone(), uncond_mask.clone()],
                    2,
                );
                let uncond_vel = self.estimator.forward(&uncond_input, t, step);

                let cond_input = Tensor::cat(
                    vec![latent.clone(), incontext_x.clone(), cond_with_mask.clone()],
                    2,
                );
                let cond_vel = self.estimator.forward(&cond_input, t, step);

                uncond_vel.clone() + (cond_vel - uncond_vel) * guidance_scale
            } else {
                let estimator_input = Tensor::cat(
                    vec![latent.clone(), incontext_x.clone(), cond_with_mask.clone()],
                    2,
                );
                self.estimator.forward(&estimator_input, t, step)
            };

            latent = latent + velocity * dt;

            if step > 0 && step % sync_interval == 0 {
                let _ = latent.to_data();
            }
        }

        let _ = latent.to_data();

        if masked_incontext_length > 0 {
            let prefix =
                incontext_x
                    .clone()
                    .slice([0..batch, 0..masked_incontext_length, 0..latent_dim]);
            let suffix = latent.slice([
                0..batch,
                masked_incontext_length..seq_len_interp,
                0..latent_dim,
            ]);
            latent = Tensor::cat(vec![prefix, suffix], 1);
        }

        latent
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inference_codes(
        &self,
        codes: Vec<Tensor<B, 3, Int>>,
        true_latents: Tensor<B, 3>,
        latent_length: usize,
        incontext_length: usize,
        guidance_scale: f32,
        num_steps: usize,
        disable_progress: bool,
        scenario: &str,
        initial_latent_override: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let _ = disable_progress;
        let codes_bestrq_emb = codes
            .into_iter()
            .next()
            .expect("inference_codes expects at least one codes tensor");
        let conditioning = self.get_output_from_indices(codes_bestrq_emb);
        let conditioning = self.cond_feature_emb.forward(conditioning);
        let _ = scenario;
        self.solve_ode(
            conditioning,
            true_latents,
            latent_length,
            incontext_length,
            num_steps,
            guidance_scale,
            initial_latent_override,
        )
    }

    fn gather_codebook(
        &self,
        embed: Tensor<B, 3>,
        indices: Tensor<B, 2, Int>,
        codebook_size: usize,
        dim: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len] = indices.dims();
        let embed_2d: Tensor<B, 2> = embed.squeeze_dim(0);
        let indices_flat: Tensor<B, 1, Int> = indices.reshape([batch * seq_len]);
        let max_idx = (codebook_size as i64) - 1;
        let indices_clamped: Tensor<B, 1, Int> = indices_flat.clamp(0, max_idx);
        let gathered = embed_2d.select(0, indices_clamped);
        gathered.reshape([batch, seq_len, dim])
    }

    fn get_output_from_indices(&self, codes: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [batch, num_quantizers, seq_len] = codes.dims();

        let mut quantized_sum = Tensor::<B, 3>::zeros([batch, seq_len, 32], &codes.device());
        for q in 0..num_quantizers.min(self.vq_embed.layers.len()) {
            let q_codes_3d = codes.clone().slice([0..batch, q..q + 1, 0..seq_len]);
            let q_codes = q_codes_3d.reshape([batch, seq_len]);
            let embed = self.vq_embed.layers[q]._codebook.embed.val();
            let embed_dim = embed.dims()[2];
            let codebook_size = embed.dims()[1];
            let q_emb = self.gather_codebook(embed, q_codes, codebook_size, embed_dim);
            quantized_sum = quantized_sum + q_emb;
        }
        self.vq_embed.project_out.forward(quantized_sum)
    }

    fn build_latent_masks(
        seq_len: usize,
        latent_length: usize,
        incontext_length: usize,
    ) -> Vec<i64> {
        let mut masks = vec![0_i64; seq_len];
        for mask in masks.iter_mut().take(seq_len.min(latent_length)) {
            *mask = 2;
        }
        for mask in masks.iter_mut().take(seq_len.min(incontext_length)) {
            *mask = 1;
        }
        masks
    }
}

#[derive(Module, Debug)]
pub struct ResidualVQ<B: Backend> {
    pub layers: Vec<VQCodebook<B>>,
    pub project_in: Linear<B>,
    pub project_out: Linear<B>,
}

impl<B: Backend> ResidualVQ<B> {
    pub fn new(device: &B::Device, config: &HeartCodecConfig) -> Self {
        let layers: Vec<_> = (0..config.num_quantizers)
            .map(|_| VQCodebook::new(device, config.codebook_size, config.codebook_dim))
            .collect();

        Self {
            layers,
            project_in: LinearConfig::new(512, 32)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
            project_out: LinearConfig::new(32, 512)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
        }
    }

    pub fn load_from_dot_notation<F>(device: &B::Device, get_tensor: &F) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let mut layers = Vec::new();
        for i in 0..8 {
            let prefix = format!("flow_matching.vq_embed.layers.{}._codebook", i);

            let embed_name = format!("{}.embed", prefix);
            let embed = if let Some((data, shape)) = get_tensor(&embed_name) {
                Tensor::<B, 3>::from_data(
                    TensorData::new(data, [shape[0], shape[1], shape[2]]),
                    device,
                )
            } else {
                Tensor::zeros([1, 8192, 32], device)
            };

            let cluster_size_name = format!("{}.cluster_size", prefix);
            let cluster_size = if let Some((data, shape)) = get_tensor(&cluster_size_name) {
                Tensor::<B, 2>::from_data(TensorData::new(data, [shape[0], shape[1]]), device)
            } else {
                Tensor::zeros([1, 8192], device)
            };

            let embed_avg_name = format!("{}.embed_avg", prefix);
            let embed_avg = if let Some((data, shape)) = get_tensor(&embed_avg_name) {
                Tensor::<B, 3>::from_data(
                    TensorData::new(data, [shape[0], shape[1], shape[2]]),
                    device,
                )
            } else {
                Tensor::zeros([1, 8192, 32], device)
            };

            layers.push(VQCodebook {
                _codebook: VQCodebookInner {
                    embed: Param::from_tensor(embed),
                    cluster_size: Param::from_tensor(cluster_size),
                    embed_avg: Param::from_tensor(embed_avg),
                },
            });
        }

        let project_in = load_linear_from_tensors(
            device,
            get_tensor,
            "flow_matching.vq_embed.project_in",
            512,
            32,
        )?;

        let project_out = load_linear_from_tensors(
            device,
            get_tensor,
            "flow_matching.vq_embed.project_out",
            32,
            512,
        )?;

        Ok(Self {
            layers,
            project_in,
            project_out,
        })
    }
}

#[derive(Module, Debug)]
pub struct VQCodebook<B: Backend> {
    pub _codebook: VQCodebookInner<B>,
}

impl<B: Backend> VQCodebook<B> {
    pub fn new(device: &B::Device, codebook_size: usize, codebook_dim: usize) -> Self {
        Self {
            _codebook: VQCodebookInner::new(device, codebook_size, codebook_dim),
        }
    }
}

#[derive(Module, Debug)]
pub struct VQCodebookInner<B: Backend> {
    pub cluster_size: Param<Tensor<B, 2>>,
    pub embed: Param<Tensor<B, 3>>,
    pub embed_avg: Param<Tensor<B, 3>>,
}

impl<B: Backend> VQCodebookInner<B> {
    pub fn new(device: &B::Device, codebook_size: usize, codebook_dim: usize) -> Self {
        Self {
            cluster_size: Param::from_tensor(Tensor::zeros([1, codebook_size], device)),
            embed: Param::from_tensor(Tensor::zeros([1, codebook_size, codebook_dim], device)),
            embed_avg: Param::from_tensor(Tensor::zeros([1, codebook_size, codebook_dim], device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct LlamaTransformer<B: Backend> {
    pub proj_in: ProjectLayer<B>,
    pub proj_out: ProjectLayer<B>,
    pub connection_proj: ProjectLayer<B>,
    pub transformer_blocks: Vec<TransformerBlock<B>>,
    pub transformer_blocks_2: Vec<TransformerBlock<B>>,
    pub norm_out: LayerNorm<B>,
    pub norm_out_2: LayerNorm<B>,
    pub adaln_single: AdaLayerNormSingle<B>,
    pub adaln_single_2: AdaLayerNormSingle<B>,
    pub scale_shift_table: Param<Tensor<B, 2>>,
    pub scale_shift_table_2: Param<Tensor<B, 2>>,
}

impl<B: Backend> LlamaTransformer<B> {
    pub fn new(device: &B::Device, config: &HeartCodecConfig) -> Self {
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let inner_dim_2 = inner_dim * 2;
        let _latent_dim = config.latent_hidden_dim;

        let transformer_blocks: Vec<_> = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    device,
                    inner_dim,
                    config.num_attention_heads,
                    config.attention_head_dim,
                )
            })
            .collect();

        let transformer_blocks_2: Vec<_> = (0..config.num_layers_2)
            .map(|_| {
                TransformerBlock::new(
                    device,
                    inner_dim_2,
                    config.num_attention_heads,
                    config.attention_head_dim * 2,
                )
            })
            .collect();

        let in_channels = 1024;
        let connection_in = 2560;

        Self {
            proj_in: ProjectLayer::new(device, in_channels, inner_dim, 3),
            proj_out: ProjectLayer::new(device, inner_dim_2, config.out_channels, 3),
            connection_proj: ProjectLayer::new(device, connection_in, inner_dim_2, 3),
            transformer_blocks,
            transformer_blocks_2,
            norm_out: LayerNormConfig::new(inner_dim)
                .with_epsilon(1e-6)
                .with_bias(false)
                .init(device),
            norm_out_2: LayerNormConfig::new(inner_dim_2)
                .with_epsilon(1e-6)
                .with_bias(false)
                .init(device),
            adaln_single: AdaLayerNormSingle::new(device, inner_dim),
            adaln_single_2: AdaLayerNormSingle::new(device, inner_dim_2),
            scale_shift_table: Param::from_tensor(Tensor::zeros([2, inner_dim], device)),
            scale_shift_table_2: Param::from_tensor(Tensor::zeros([2, inner_dim_2], device)),
        }
    }

    pub fn load_from_burnpack<F>(device: &B::Device, get_tensor: &F) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let config = HeartCodecConfig::default();
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let inner_dim_2 = inner_dim * 2;
        let in_channels = 1024;
        let connection_in = 2560;

        let proj_in = ProjectLayer::load_from_tensors(
            device,
            get_tensor,
            "flow_matching.estimator.proj_in",
            in_channels,
            inner_dim,
        )?;

        let proj_out = ProjectLayer::load_from_tensors(
            device,
            get_tensor,
            "flow_matching.estimator.proj_out",
            inner_dim_2,
            config.out_channels,
        )?;

        let connection_proj = ProjectLayer::load_from_tensors(
            device,
            get_tensor,
            "flow_matching.estimator.connection_proj",
            connection_in,
            inner_dim_2,
        )?;

        let mut transformer_blocks = Vec::new();
        for i in 0..config.num_layers {
            let block = TransformerBlock::load_from_tensors(
                device,
                get_tensor,
                &format!("flow_matching.estimator.transformer_blocks.{}", i),
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
            )?;
            transformer_blocks.push(block);
        }

        let mut transformer_blocks_2 = Vec::new();
        for i in 0..config.num_layers_2 {
            let block = TransformerBlock::load_from_tensors(
                device,
                get_tensor,
                &format!("flow_matching.estimator.transformer_blocks_2.{}", i),
                inner_dim_2,
                config.num_attention_heads,
                config.attention_head_dim * 2,
            )?;
            transformer_blocks_2.push(block);
        }

        let adaln_single = AdaLayerNormSingle::load_from_tensors(
            device,
            get_tensor,
            "flow_matching.estimator.adaln_single",
            inner_dim,
        )?;

        let adaln_single_2 = AdaLayerNormSingle::load_from_tensors(
            device,
            get_tensor,
            "flow_matching.estimator.adaln_single_2",
            inner_dim_2,
        )?;

        let scale_shift_table = load_param_tensor(
            device,
            get_tensor,
            "flow_matching.estimator.scale_shift_table",
            [2, inner_dim],
        )?;

        let scale_shift_table_2 = load_param_tensor(
            device,
            get_tensor,
            "flow_matching.estimator.scale_shift_table_2",
            [2, inner_dim_2],
        )?;

        Ok(Self {
            proj_in,
            proj_out,
            connection_proj,
            transformer_blocks,
            transformer_blocks_2,
            norm_out: LayerNormConfig::new(inner_dim)
                .with_epsilon(1e-6)
                .with_bias(false)
                .init(device),
            norm_out_2: LayerNormConfig::new(inner_dim_2)
                .with_epsilon(1e-6)
                .with_bias(false)
                .init(device),
            adaln_single,
            adaln_single_2,
            scale_shift_table,
            scale_shift_table_2,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor<B, 3>, t: f32, step: usize) -> Tensor<B, 3> {
        let mut s = self.proj_in.forward(hidden_states.clone(), step);
        let (timestep_mod, embedded_timestep) = self.adaln_single.forward(t, s.dtype());

        for block in &self.transformer_blocks {
            s = block.forward(s, Some(timestep_mod.clone()), false, step);
        }

        let shift_scale_1 = self.scale_shift_table.val().clone().unsqueeze_dim(0)
            + embedded_timestep.unsqueeze_dim(1);
        let shift_1 = shift_scale_1.clone().slice([0..1, 0..1, 0..s.dims()[2]]);
        let scale_1 = shift_scale_1.slice([0..1, 1..2, 0..s.dims()[2]]);
        let s_norm = self.norm_out.forward(s);
        let s = s_norm * (scale_1 + 1.0) + shift_1;

        let x = Tensor::cat(vec![hidden_states.clone(), s.clone()], 2);

        let x = self.connection_proj.forward(x, step);

        let mut x = x;
        let (timestep_mod_2, embedded_timestep_2) = self.adaln_single_2.forward(t, x.dtype());
        for block in &self.transformer_blocks_2 {
            x = block.forward(x, Some(timestep_mod_2.clone()), false, step);
        }

        let shift_scale_2 = self.scale_shift_table_2.val().clone().unsqueeze_dim(0)
            + embedded_timestep_2.unsqueeze_dim(1);
        let shift_2 = shift_scale_2.clone().slice([0..1, 0..1, 0..x.dims()[2]]);
        let scale_2 = shift_scale_2.slice([0..1, 1..2, 0..x.dims()[2]]);
        let x_norm = self.norm_out_2.forward(x);
        let x = x_norm * (scale_2 + 1.0) + shift_2;

        self.proj_out.forward(x, step)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attn: Attention<B>,
    pub attn_norm: RmsNorm<B>,
    pub mlp: Mlp<B>,
    pub mlp_norm: RmsNorm<B>,
    pub scale_shift_table: Param<Tensor<B, 2>>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(device: &B::Device, dim: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            attn: Attention::new(device, dim, num_heads, head_dim),
            attn_norm: RmsNorm::new(device, dim, 1e-6),
            mlp: Mlp::new(device, dim),
            mlp_norm: RmsNorm::new(device, dim, 1e-6),
            scale_shift_table: Param::from_tensor(Tensor::zeros([6, dim], device)),
        }
    }

    pub fn load_from_tensors<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let inner_dim = num_heads * head_dim;
        let hidden_dim = Mlp::<B>::compute_hidden_dim(dim);

        let attn = Attention {
            q_proj: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.attn.q_proj", prefix),
                dim,
                inner_dim,
            )?,
            k_proj: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.attn.k_proj", prefix),
                dim,
                inner_dim,
            )?,
            v_proj: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.attn.v_proj", prefix),
                dim,
                inner_dim,
            )?,
            o_proj: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.attn.o_proj", prefix),
                inner_dim,
                dim,
            )?,
            num_heads,
            head_dim,
            rope_dim: head_dim,
        };

        let mlp = Mlp {
            gate: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.mlp.gate", prefix),
                dim,
                hidden_dim,
            )?,
            up: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.mlp.up", prefix),
                dim,
                hidden_dim,
            )?,
            down: load_linear_from_tensors(
                device,
                get_tensor,
                &format!("{}.mlp.down", prefix),
                hidden_dim,
                dim,
            )?,
        };

        let attn_norm =
            load_rmsnorm_from_tensors(device, get_tensor, &format!("{}.attn_norm", prefix), dim)?;

        let mlp_norm =
            load_rmsnorm_from_tensors(device, get_tensor, &format!("{}.mlp_norm", prefix), dim)?;

        let scale_shift_table = load_param_tensor(
            device,
            get_tensor,
            &format!("{}.scale_shift_table", prefix),
            [6, dim],
        )?;

        Ok(Self {
            attn,
            attn_norm,
            mlp,
            mlp_norm,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        timestep: Option<Tensor<B, 3>>,
        _dump_attention: bool,
        _step: usize,
    ) -> Tensor<B, 3> {
        if let Some(timestep) = timestep {
            let [batch, _seq, dim] = x.dims();
            let shift_scale = self.scale_shift_table.val().clone().unsqueeze_dim(0)
                + timestep.reshape([batch, 6, dim]);
            let mut parts = shift_scale.chunk(6, 1);
            let shift_msa = parts.remove(0);
            let scale_msa = parts.remove(0);
            let gate_msa = parts.remove(0);
            let shift_mlp = parts.remove(0);
            let scale_mlp = parts.remove(0);
            let gate_mlp = parts.remove(0);

            let normed = self.attn_norm.forward(x.clone());
            let normed = normed * (scale_msa + 1.0) + shift_msa;
            let attn_out = self.attn.forward(normed, false, 0);
            let x = x + gate_msa * attn_out;

            let normed = self.mlp_norm.forward(x.clone());
            let normed = normed * (scale_mlp + 1.0) + shift_mlp;
            let mlp_out = self.mlp.forward(normed);
            x + gate_mlp * mlp_out
        } else {
            let normed = self.attn_norm.forward(x.clone());
            let attn_out = self.attn.forward(normed, false, 0);
            let x = x + attn_out;
            let normed = self.mlp_norm.forward(x.clone());
            let mlp_out = self.mlp.forward(normed);
            x + mlp_out
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(device: &B::Device, dim: usize, num_heads: usize, head_dim: usize) -> Self {
        let inner_dim = num_heads * head_dim;
        Self {
            q_proj: LinearConfig::new(dim, inner_dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            k_proj: LinearConfig::new(dim, inner_dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            v_proj: LinearConfig::new(dim, inner_dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            o_proj: LinearConfig::new(inner_dim, dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            num_heads,
            head_dim,
            rope_dim: head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, _dump_attention: bool, _step: usize) -> Tensor<B, 3> {
        let [batch, seq_len, dim] = x.dims();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, num_heads, head_dim])
            .swap_dims(1, 2);

        let (q, k) = Self::apply_rope(q, k, self.rope_dim.min(head_dim));

        let scores = q.matmul(k.swap_dims(2, 3)) / (head_dim as f32).sqrt();

        use burn::tensor::activation::softmax;
        let attn_weights = softmax(scores, 3);

        let out = attn_weights.matmul(v);

        let out = out.swap_dims(1, 2).reshape([batch, seq_len, dim]);

        self.o_proj.forward(out)
    }

    fn apply_rope(
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        rope_dim: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if rope_dim == 0 {
            return (q, k);
        }

        let [batch, heads, seq_len, head_dim] = q.dims();
        let rope_pairs = rope_dim / 2;
        let device = q.device();
        let dtype = q.dtype();
        let base = 10_000.0_f32;

        let mut inv_freq = Vec::with_capacity(rope_pairs);
        for i in 0..rope_pairs {
            let exponent = (2 * i) as f32 / rope_dim as f32;
            inv_freq.push(1.0 / base.powf(exponent));
        }

        let inv_freq = Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [rope_pairs]), &device);
        let positions = {
            let mut data = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                data.push(i as f32);
            }
            Tensor::<B, 2>::from_data(TensorData::new(data, [seq_len, 1]), &device).cast(dtype)
        };
        let freqs = positions.matmul(inv_freq.reshape([1, rope_pairs]));
        let sin = freqs.clone().sin().reshape([1, 1, seq_len, rope_pairs]);
        let cos = freqs.cos().reshape([1, 1, seq_len, rope_pairs]);

        let rotate = |x: Tensor<B, 4>| {
            let head = x
                .clone()
                .slice([0..batch, 0..heads, 0..seq_len, 0..rope_dim]);
            let tail = x.slice([0..batch, 0..heads, 0..seq_len, rope_dim..head_dim]);
            let head = head.reshape([batch, heads, seq_len, rope_pairs, 2]);
            let x1 = head
                .clone()
                .slice([0..batch, 0..heads, 0..seq_len, 0..rope_pairs, 0..1])
                .reshape([batch, heads, seq_len, rope_pairs]);
            let x2 = head
                .slice([0..batch, 0..heads, 0..seq_len, 0..rope_pairs, 1..2])
                .reshape([batch, heads, seq_len, rope_pairs]);
            let rot_a = x1.clone() * cos.clone() - x2.clone() * sin.clone();
            let rot_b = x1 * sin.clone() + x2 * cos.clone();
            let rot = Tensor::cat(vec![rot_a, rot_b], 3);
            Tensor::cat(vec![rot, tail], 3)
        };

        (rotate(q), rotate(k))
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub gate: Linear<B>,
    pub up: Linear<B>,
    pub down: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(device: &B::Device, dim: usize) -> Self {
        let hidden_dim = Self::compute_hidden_dim(dim);
        Self {
            gate: LinearConfig::new(dim, hidden_dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            up: LinearConfig::new(dim, hidden_dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            down: LinearConfig::new(hidden_dim, dim)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
        }
    }

    fn compute_hidden_dim(dim: usize) -> usize {
        let multiple_of = 256;
        let hidden_dim = (4 * dim * 2) / 3;
        multiple_of * hidden_dim.div_ceil(multiple_of)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        use burn::tensor::activation::silu;

        let gate = self.gate.forward(x.clone());
        let up = self.up.forward(x.clone());

        let gate_activated = silu(gate);

        let hidden = gate_activated * up;

        self.down.forward(hidden)
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(device: &B::Device, dim: usize, _eps: f64) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let eps = 1e-6;
        let weight = self.weight.val();

        let x_sq = x.clone().powf_scalar(2.0);

        let mean_sq = x_sq.mean_dim(2);

        let rms = (mean_sq + eps).sqrt();

        let [batch, seq_len, dim] = x.dims();
        let rms_expanded = rms.expand([batch, seq_len, dim]);
        let normalized = x / rms_expanded;

        let weight_expanded = weight.reshape([1, 1, dim]).expand([batch, seq_len, dim]);
        normalized * weight_expanded
    }
}

#[derive(Module, Debug)]
pub struct AdaLayerNormSingle<B: Backend> {
    pub emb: PixArtAlphaCombinedFlowEmbeddings<B>,
    pub linear: Linear<B>,
}

impl<B: Backend> AdaLayerNormSingle<B> {
    pub fn new(device: &B::Device, embedding_dim: usize) -> Self {
        Self {
            emb: PixArtAlphaCombinedFlowEmbeddings::new(device, embedding_dim),
            linear: LinearConfig::new(embedding_dim, 6 * embedding_dim)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
        }
    }

    pub fn load_from_tensors<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        embedding_dim: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let emb_prefix = format!("{}.emb", prefix);
        let emb = PixArtAlphaCombinedFlowEmbeddings::load_from_tensors(
            device,
            get_tensor,
            &emb_prefix,
            embedding_dim,
        )?;

        let linear = load_linear_from_tensors(
            device,
            get_tensor,
            &format!("{}.linear", prefix),
            embedding_dim,
            6 * embedding_dim,
        )?;

        Ok(Self { emb, linear })
    }

    pub fn forward(&self, timestep: f32, hidden_dtype: DType) -> (Tensor<B, 3>, Tensor<B, 2>) {
        use burn::tensor::activation::silu;

        let embedded_timestep = self.emb.forward(timestep, hidden_dtype);
        let timestep_mod = self.linear.forward(silu(embedded_timestep.clone()));
        let [batch, features] = timestep_mod.dims();
        (
            timestep_mod.reshape([batch, 6, features / 6]),
            embedded_timestep,
        )
    }
}

#[derive(Module, Debug)]
pub struct PixArtAlphaCombinedFlowEmbeddings<B: Backend> {
    pub timestep_embedder: TimestepEmbedding<B>,
}

impl<B: Backend> PixArtAlphaCombinedFlowEmbeddings<B> {
    pub fn new(device: &B::Device, embedding_dim: usize) -> Self {
        Self {
            timestep_embedder: TimestepEmbedding::new(device, 512, embedding_dim),
        }
    }

    pub fn load_from_tensors<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        embedding_dim: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let timestep_embedder = TimestepEmbedding::load_from_tensors(
            device,
            get_tensor,
            &format!("{}.timestep_embedder", prefix),
            512,
            embedding_dim,
        )?;

        Ok(Self { timestep_embedder })
    }

    pub fn forward(&self, timestep: f32, _hidden_dtype: DType) -> Tensor<B, 2> {
        let device = self.timestep_embedder.linear_1.weight.device();
        let flow_t_size = 512;
        let half = flow_t_size / 2;
        let mut data = Vec::with_capacity(flow_t_size);
        for i in 0..half {
            let freq = 10_000_f32.powf(-(i as f32) / half.max(1) as f32);
            let angle = timestep * freq * 1000.0;
            data.push(angle.cos());
        }
        for i in 0..half {
            let freq = 10_000_f32.powf(-(i as f32) / half.max(1) as f32);
            let angle = timestep * freq * 1000.0;
            data.push(angle.sin());
        }
        let timestep = Tensor::<B, 2>::from_data(TensorData::new(data, [1, flow_t_size]), &device);
        self.timestep_embedder.forward(timestep)
    }
}

#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1: Linear<B>,
    pub linear_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn new(device: &B::Device, in_channels: usize, time_embed_dim: usize) -> Self {
        Self {
            linear_1: LinearConfig::new(in_channels, time_embed_dim)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
            linear_2: LinearConfig::new(time_embed_dim, time_embed_dim)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
        }
    }

    pub fn load_from_tensors<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        in_channels: usize,
        time_embed_dim: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let linear_1 = load_linear_from_tensors(
            device,
            get_tensor,
            &format!("{}.linear_1", prefix),
            in_channels,
            time_embed_dim,
        )?;

        let linear_2 = load_linear_from_tensors(
            device,
            get_tensor,
            &format!("{}.linear_2", prefix),
            time_embed_dim,
            time_embed_dim,
        )?;

        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        use burn::tensor::activation::silu;

        let emb = self.linear_1.forward(x);
        self.linear_2.forward(silu(emb))
    }
}

#[derive(Module, Debug)]
pub struct ProjectLayer<B: Backend> {
    pub ffn_1: Conv1d<B>,
    pub ffn_2: Linear<B>,
    pub kernel_size: usize,
    pub out_channels: usize,
}

impl<B: Backend> ProjectLayer<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Self {
        let padding = kernel_size / 2;
        Self {
            ffn_1: Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .init(device),
            ffn_2: LinearConfig::new(out_channels, out_channels)
                .with_bias(true)
                .with_layout(LinearLayout::Col)
                .init(device),
            kernel_size,
            out_channels,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, _step: usize) -> Tensor<B, 3> {
        let x_t = x.swap_dims(1, 2);

        let conv_out = self.forward_conv1d_exact(x_t);

        let conv_out_t = conv_out.swap_dims(1, 2) * (self.kernel_size as f32).powf(-0.5);

        self.ffn_2.forward(conv_out_t)
    }

    fn forward_conv1d_exact(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, in_channels, seq_len] = input.dims();
        let [out_channels, weight_in_channels, kernel_size] = self.ffn_1.weight.dims();
        assert_eq!(
            in_channels, weight_in_channels,
            "project conv input channels mismatch"
        );

        let padding = kernel_size / 2;
        let device = input.device();
        let left = Tensor::<B, 3>::zeros([batch, in_channels, padding], &device);
        let right = Tensor::<B, 3>::zeros([batch, in_channels, padding], &device);
        let padded = Tensor::cat(vec![left, input, right], 2);

        let mut out = Tensor::<B, 3>::zeros([batch, seq_len, out_channels], &device);
        for k in 0..kernel_size {
            let x_k = padded
                .clone()
                .slice([0..batch, 0..in_channels, k..k + seq_len])
                .swap_dims(1, 2);
            let w_k = self
                .ffn_1
                .weight
                .val()
                .slice([0..out_channels, 0..in_channels, k..k + 1])
                .reshape([out_channels, in_channels])
                .swap_dims(0, 1);
            let x_k_flat = x_k.reshape([batch * seq_len, in_channels]);
            let projected = x_k_flat.matmul(w_k).reshape([batch, seq_len, out_channels]);
            out = out + projected;
        }

        if let Some(bias) = &self.ffn_1.bias {
            let bias =
                bias.val()
                    .reshape([1, 1, out_channels])
                    .expand([batch, seq_len, out_channels]);
            out = out + bias;
        }

        out.swap_dims(1, 2)
    }

    pub fn load_from_tensors<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        use burn::nn::conv::Conv1dConfig;

        let kernel_size = 3;
        let padding = kernel_size / 2;

        let ffn_1_weight_name = format!("{}.ffn_1.weight", prefix);
        let ffn_1_bias_name = format!("{}.ffn_1.bias", prefix);

        let ffn_1 = if let (Some((w_data, w_shape)), Some((b_data, b_shape))) =
            (get_tensor(&ffn_1_weight_name), get_tensor(&ffn_1_bias_name))
        {
            if w_shape.len() == 3
                && w_shape[0] == out_channels
                && w_shape[1] == in_channels
                && w_shape[2] == kernel_size
                && b_shape.len() == 1
                && b_shape[0] == out_channels
            {
                let weight = Tensor::<B, 3>::from_data(
                    TensorData::new(w_data, [out_channels, in_channels, kernel_size]),
                    device,
                );
                let bias =
                    Tensor::<B, 1>::from_data(TensorData::new(b_data, [out_channels]), device);

                Conv1d {
                    weight: Param::from_tensor(weight),
                    bias: Some(Param::from_tensor(bias)),
                    stride: 1,
                    kernel_size,
                    dilation: 1,
                    groups: 1,
                    padding: PaddingConfig1d::Explicit(padding, padding),
                }
            } else {
                Conv1dConfig::new(in_channels, out_channels, kernel_size)
                    .with_padding(PaddingConfig1d::Explicit(padding, padding))
                    .init(device)
            }
        } else {
            Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .init(device)
        };

        let ffn_2 = load_linear_from_tensors(
            device,
            get_tensor,
            &format!("{}.ffn_2", prefix),
            out_channels,
            out_channels,
        )?;

        Ok(Self {
            ffn_1,
            ffn_2,
            kernel_size,
            out_channels,
        })
    }
}

#[derive(Module, Debug)]
pub struct ScalarModel<B: Backend> {
    pub decoder_0: WNConv1d<B>,
    pub decoder_1: ResDecoderBlock<B>,
    pub decoder_2: ResDecoderBlock<B>,
    pub decoder_3: ResDecoderBlock<B>,
    pub decoder_4: ResDecoderBlock<B>,
    pub decoder_5: ResDecoderBlock<B>,
    pub decoder_6: PostProcessor<B>,
    pub decoder_7: WNConv1d<B>,
}

impl<B: Backend> ScalarModel<B> {
    pub fn from_burnpack(path: &std::path::Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(device, &HeartCodecConfig::default());
        let mut store = BurnpackStore::from_file(path).zero_copy(true);

        if model.load_from(&mut store).is_ok() {
            return Ok(model);
        }

        let get_tensor = HeartCodecModel::<B>::snapshots_to_f32_lookup(path)?;
        Self::load_from_dot_notation(path, device, &get_tensor)
    }

    pub fn new(device: &B::Device, _config: &HeartCodecConfig) -> Self {
        let config = HeartCodecConfig::default();
        Self {
            decoder_0: WNConv1d::new(
                device,
                128,
                2048,
                config.delay_kernel_size,
                1,
                config.delay_kernel_size / 2,
                1,
                1,
                false,
            ),

            decoder_1: ResDecoderBlock::new(device, 2048, 1024),
            decoder_2: ResDecoderBlock::new(device, 1024, 512),
            decoder_3: ResDecoderBlock::new(device, 512, 256),
            decoder_4: ResDecoderBlock::new(device, 256, 128),
            decoder_5: ResDecoderBlock::new(device, 128, 64),

            decoder_6: PostProcessor::new(device, 64, 2),

            decoder_7: WNConv1d::new(device, 64, 1, 7, 1, 3, 1, 1, true),
        }
    }

    pub fn load_from_dot_notation<F>(
        _path: &std::path::Path,
        device: &B::Device,
        get_tensor: &F,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        use crate::heartcodec::conv::{WNConv1dLoadArgs, load_wnconv_from_tensors};

        let load_conv = |prefix: &str,
                         in_ch: usize,
                         out_ch: usize,
                         ksize: usize,
                         padding: usize,
                         causal: bool|
         -> WNConv1d<B> {
            let weight_g_name = format!("{}.weight_g", prefix);
            let weight_v_name = format!("{}.weight_v", prefix);
            let bias_name = format!("{}.bias", prefix);

            if let (Some(weight_g), Some(weight_v)) =
                (get_tensor(&weight_g_name), get_tensor(&weight_v_name))
            {
                let bias = get_tensor(&bias_name);
                return load_wnconv_from_tensors(
                    device,
                    WNConv1dLoadArgs {
                        in_channels: in_ch,
                        out_channels: out_ch,
                        kernel_size: ksize,
                        dilation: 1,
                        causal,
                        weight_g,
                        weight_v,
                        bias,
                    },
                )
                .unwrap_or_else(|_| {
                    WNConv1d::new(device, in_ch, out_ch, ksize, 1, padding, 1, 1, causal)
                });
            }

            let g_name = format!("{}.parametrizations.weight.original0", prefix);
            let v_name = format!("{}.parametrizations.weight.original1", prefix);

            if let (Some(weight_g), Some(weight_v)) = (get_tensor(&g_name), get_tensor(&v_name)) {
                let bias = get_tensor(&bias_name);
                return load_wnconv_from_tensors(
                    device,
                    WNConv1dLoadArgs {
                        in_channels: in_ch,
                        out_channels: out_ch,
                        kernel_size: ksize,
                        dilation: 1,
                        causal,
                        weight_g,
                        weight_v,
                        bias,
                    },
                )
                .unwrap_or_else(|_| {
                    WNConv1d::new(device, in_ch, out_ch, ksize, 1, padding, 1, 1, causal)
                });
            }

            WNConv1d::new(device, in_ch, out_ch, ksize, 1, padding, 1, 1, causal)
        };

        let decoder_0 = load_conv("scalar_model.decoder.0", 128, 2048, 5, 2, false);
        let decoder_1 = ResDecoderBlock::load_from_dot_notation(
            device,
            get_tensor,
            "scalar_model.decoder.1",
            2048,
            1024,
        )?;
        let decoder_2 = ResDecoderBlock::load_from_dot_notation(
            device,
            get_tensor,
            "scalar_model.decoder.2",
            1024,
            512,
        )?;
        let decoder_3 = ResDecoderBlock::load_from_dot_notation(
            device,
            get_tensor,
            "scalar_model.decoder.3",
            512,
            256,
        )?;
        let decoder_4 = ResDecoderBlock::load_from_dot_notation(
            device,
            get_tensor,
            "scalar_model.decoder.4",
            256,
            128,
        )?;
        let decoder_5 = ResDecoderBlock::load_from_dot_notation(
            device,
            get_tensor,
            "scalar_model.decoder.5",
            128,
            64,
        )?;
        let decoder_6 = if let Ok(pp) =
            PostProcessor::load_from_tensors(device, get_tensor, "scalar_model.decoder.6", 64, 2)
        {
            pp
        } else {
            PostProcessor::new(device, 64, 2)
        };

        let decoder_7 = load_conv("scalar_model.decoder.7", 64, 1, 7, 3, true);

        Ok(Self {
            decoder_0,
            decoder_1,
            decoder_2,
            decoder_3,
            decoder_4,
            decoder_5,
            decoder_6,
            decoder_7,
        })
    }

    pub fn decode(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_quantized = (x.clone() * 9.0).round() / 9.0;

        let h = self.decoder_0.forward(x_quantized);
        let h = self.decoder_1.forward(h);
        let h = self.decoder_2.forward(h);
        let h = self.decoder_3.forward(h);
        let h = self.decoder_4.forward(h);
        let h = self.decoder_5.forward(h);
        let h = self.decoder_6.forward(h);
        self.decoder_7.forward(h)
    }

    pub fn decode_with_sync(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_quantized = (x.clone() * 9.0).round() / 9.0;

        let h = self.decoder_0.forward(x_quantized);
        let _ = h.to_data();

        let h = self.decoder_1.forward(h);
        let _ = h.to_data();

        let h = self.decoder_2.forward(h);
        let _ = h.to_data();

        let h = self.decoder_3.forward(h);
        let _ = h.to_data();

        let h = self.decoder_4.forward(h);
        let _ = h.to_data();

        let h = self.decoder_5.forward(h);
        let _ = h.to_data();

        let h = self.decoder_6.forward(h);
        let _ = h.to_data();

        self.decoder_7.forward(h)
    }

    pub fn decode_latent_with_sync(&self, latent: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = latent.dims();
        assert_eq!(
            channels, 256,
            "Expected 256 channels from flow matching latent"
        );

        let latent_reshaped = latent.reshape([batch, seq_len, 2, 128]);
        let latent_permuted = latent_reshaped.swap_dims(1, 2);
        let latent_split = latent_permuted.reshape([batch * 2, seq_len, 128]);

        let scalar_input = latent_split.swap_dims(1, 2);
        self.decode_with_sync(scalar_input)
    }
}

#[derive(Module, Debug)]
pub struct ResDecoderBlock<B: Backend> {
    pub up_conv: WNConvTranspose1d<B>,
    pub convs: Vec<ResidualUnit<B>>,
}

impl<B: Backend> ResDecoderBlock<B> {
    pub fn new(device: &B::Device, in_ch: usize, out_ch: usize) -> Self {
        let (kernel_size, stride) = match (in_ch, out_ch) {
            (2048, 1024) => (10, 5),
            (1024, 512) => (8, 4),
            (512, 256) => (8, 4),
            (256, 128) => (8, 4),
            (128, 64) => (6, 3),
            _ => (8, 4),
        };

        let up_conv = WNConvTranspose1d::new(
            device,
            in_ch,
            out_ch,
            kernel_size,
            stride,
            kernel_size / 2,
            0,
            1,
            1,
            true,
        );

        let dilations = [1, 3, 5, 7, 9];
        let convs: Vec<_> = dilations
            .into_iter()
            .map(|dilation| ResidualUnit::new(device, out_ch, dilation))
            .collect();

        Self { up_conv, convs }
    }

    pub fn load_from_dot_notation<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        use crate::heartcodec::conv::{
            WNConvTranspose1dLoadArgs, load_wnconv_transpose_from_tensors,
        };

        let (kernel_size, stride) = match (in_ch, out_ch) {
            (2048, 1024) => (10, 5),
            (1024, 512) => (8, 4),
            (512, 256) => (8, 4),
            (256, 128) => (8, 4),
            (128, 64) => (6, 3),
            _ => (8, 4),
        };

        let up_conv_prefix = format!("{}.up_conv", prefix);

        let up_conv = {
            let weight_g_name = format!("{}.weight_g", up_conv_prefix);
            let weight_v_name = format!("{}.weight_v", up_conv_prefix);
            let bias_name = format!("{}.layer.bias", up_conv_prefix);

            if let (Some(weight_g), Some(weight_v)) =
                (get_tensor(&weight_g_name), get_tensor(&weight_v_name))
            {
                let bias = get_tensor(&bias_name);
                load_wnconv_transpose_from_tensors(
                    device,
                    WNConvTranspose1dLoadArgs {
                        out_channels: out_ch,
                        kernel_size,
                        stride,
                        causal: true,
                        weight_g,
                        weight_v,
                        bias,
                    },
                )
                .unwrap_or_else(|_| {
                    WNConvTranspose1d::new(
                        device,
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride,
                        kernel_size / 2,
                        0,
                        1,
                        1,
                        true,
                    )
                })
            } else {
                let g_name = format!("{}.layer.parametrizations.weight.original0", up_conv_prefix);
                let v_name = format!("{}.layer.parametrizations.weight.original1", up_conv_prefix);

                if let (Some(weight_g), Some(weight_v)) = (get_tensor(&g_name), get_tensor(&v_name))
                {
                    let bias = get_tensor(&bias_name);
                    load_wnconv_transpose_from_tensors(
                        device,
                        WNConvTranspose1dLoadArgs {
                            out_channels: out_ch,
                            kernel_size,
                            stride,
                            causal: true,
                            weight_g,
                            weight_v,
                            bias,
                        },
                    )
                    .unwrap_or_else(|_| {
                        WNConvTranspose1d::new(
                            device,
                            in_ch,
                            out_ch,
                            kernel_size,
                            stride,
                            kernel_size / 2,
                            0,
                            1,
                            1,
                            true,
                        )
                    })
                } else {
                    WNConvTranspose1d::new(
                        device,
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride,
                        kernel_size / 2,
                        0,
                        1,
                        1,
                        true,
                    )
                }
            }
        };

        let mut convs = Vec::new();
        let dilations = [1, 3, 5, 7, 9];
        for (i, dilation) in dilations.into_iter().enumerate() {
            let unit_prefix = format!("{}.convs.{}", prefix, i);
            convs.push(ResidualUnit::load_from_dot_notation(
                device,
                get_tensor,
                &unit_prefix,
                out_ch,
                dilation,
            )?);
        }

        Ok(Self { up_conv, convs })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = self.up_conv.forward(x);
        for block in &self.convs {
            h = block.forward(h);
        }
        h
    }
}

#[derive(Module, Debug)]
pub struct ResidualUnit<B: Backend> {
    pub conv1: WNConv1d<B>,
    pub conv2: WNConv1d<B>,
    pub activation1: PReLU<B>,
    pub activation2: PReLU<B>,
}

impl<B: Backend> ResidualUnit<B> {
    pub fn new(device: &B::Device, channels: usize, dilation: usize) -> Self {
        Self {
            conv1: WNConv1d::new(
                device,
                channels,
                channels,
                7,
                1,
                3 * dilation,
                dilation,
                1,
                true,
            ),
            conv2: WNConv1d::new(device, channels, channels, 1, 1, 0, 1, 1, true),
            activation1: PReLU::new(device),
            activation2: PReLU::new(device),
        }
    }

    pub fn load_from_dot_notation<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        use crate::heartcodec::conv::{
            WNConv1dLoadArgs, load_prelu_from_tensor, load_wnconv_from_tensors,
        };

        let load_conv = |conv_prefix: &str, ksize: usize, dilation: usize| -> WNConv1d<B> {
            let weight_g_name = format!("{}.weight_g", conv_prefix);
            let weight_v_name = format!("{}.weight_v", conv_prefix);
            let bias_name = format!("{}.bias", conv_prefix);

            if let (Some(weight_g), Some(weight_v)) =
                (get_tensor(&weight_g_name), get_tensor(&weight_v_name))
            {
                let bias = get_tensor(&bias_name);
                return load_wnconv_from_tensors(
                    device,
                    WNConv1dLoadArgs {
                        in_channels: channels,
                        out_channels: channels,
                        kernel_size: ksize,
                        dilation,
                        causal: true,
                        weight_g,
                        weight_v,
                        bias,
                    },
                )
                .unwrap_or_else(|_| {
                    WNConv1d::new(
                        device,
                        channels,
                        channels,
                        ksize,
                        1,
                        (ksize / 2) * dilation,
                        dilation,
                        1,
                        true,
                    )
                });
            }

            let g_name = format!("{}.parametrizations.weight.original0", conv_prefix);
            let v_name = format!("{}.parametrizations.weight.original1", conv_prefix);
            let bias_name = format!("{}.bias", conv_prefix);

            if let (Some(weight_g), Some(weight_v)) = (get_tensor(&g_name), get_tensor(&v_name)) {
                let bias = get_tensor(&bias_name);
                return load_wnconv_from_tensors(
                    device,
                    WNConv1dLoadArgs {
                        in_channels: channels,
                        out_channels: channels,
                        kernel_size: ksize,
                        dilation,
                        causal: true,
                        weight_g,
                        weight_v,
                        bias,
                    },
                )
                .unwrap_or_else(|_| {
                    WNConv1d::new(
                        device,
                        channels,
                        channels,
                        ksize,
                        1,
                        (ksize / 2) * dilation,
                        dilation,
                        1,
                        true,
                    )
                });
            }

            WNConv1d::new(
                device,
                channels,
                channels,
                ksize,
                1,
                (ksize / 2) * dilation,
                dilation,
                1,
                true,
            )
        };

        let conv1_prefix = format!("{}.conv1", prefix);
        let conv2_prefix = format!("{}.conv2", prefix);

        let conv1 = load_conv(&conv1_prefix, 7, dilation);
        let conv2 = load_conv(&conv2_prefix, 1, 1);

        let act1_name = format!("{}.activation1.weight", prefix);
        let act2_name = format!("{}.activation2.weight", prefix);

        let activation1 = if let Some((data, shape)) = get_tensor(&act1_name) {
            load_prelu_from_tensor(device, data, shape).unwrap_or_else(|_| PReLU::new(device))
        } else {
            PReLU::new(device)
        };

        let activation2 = if let Some((data, shape)) = get_tensor(&act2_name) {
            load_prelu_from_tensor(device, data, shape).unwrap_or_else(|_| PReLU::new(device))
        } else {
            PReLU::new(device)
        };

        Ok(Self {
            conv1,
            conv2,
            activation1,
            activation2,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.activation1.forward(self.conv1.forward(x.clone()));
        let h = self.activation2.forward(self.conv2.forward(h));
        x + h
    }
}

#[derive(Module, Debug)]
pub struct PReLU<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> PReLU<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([1], device) * 0.25),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        use burn::tensor::activation::relu;

        let weight = self.weight.val().reshape([1, 1, 1]);
        let positive = relu(x.clone());
        let negative = relu(x.neg()).neg() * weight;
        positive + negative
    }
}

pub fn write_wav_from_f32(samples: &[f32], sample_rate: u32, path: &std::path::Path) -> Result<()> {
    write_wav_f32_oxideav(path, samples, 1, sample_rate)
        .with_context(|| format!("failed to write mono WAV to {}", path.display()))
}

pub fn write_wav_from_f32_interleaved(
    samples: &[f32],
    channels: usize,
    frames: usize,
    sample_rate: u32,
    path: &std::path::Path,
) -> Result<()> {
    let sample_count = channels
        .checked_mul(frames)
        .context("interleaved float WAV sample count overflow")?;
    let mut interleaved = vec![0.0_f32; sample_count];
    interleaved
        .par_iter_mut()
        .enumerate()
        .for_each(|(sample_index, slot)| {
            let frame = sample_index / channels;
            let channel = sample_index % channels;
            let source_index = channel
                .checked_mul(frames)
                .and_then(|base| base.checked_add(frame))
                .unwrap_or(0);
            *slot = samples.get(source_index).copied().unwrap_or_default();
        });
    write_wav_f32_oxideav(path, &interleaved, channels, sample_rate)
        .with_context(|| format!("failed to write interleaved WAV to {}", path.display()))
}

fn write_wav_f32_oxideav(
    path: &std::path::Path,
    samples: &[f32],
    channels: usize,
    sample_rate: u32,
) -> std::io::Result<()> {
    let channels = channels.max(1);
    if sample_rate == 0 {
        return Err(std::io::Error::other(
            "write_wav_f32: sample_rate must be > 0",
        ));
    }
    if channels > 8 {
        return Err(std::io::Error::other(format!(
            "write_wav_f32: channel count {channels} exceeds the supported maximum of 8"
        )));
    }
    if !samples.len().is_multiple_of(channels) {
        return Err(std::io::Error::other(
            "write_wav_f32: sample slice length is not a multiple of channels",
        ));
    }

    let mut ctx = RuntimeContext::new();
    oxideav_basic::register(&mut ctx);

    let stream = wav_f32_stream_info(channels, sample_rate);
    let file = std::fs::File::create(path)?;
    let output: Box<dyn oxideav_core::WriteSeek> = Box::new(file);
    let mut mux = ctx
        .containers
        .open_muxer("wav", output, std::slice::from_ref(&stream))
        .map_err(|e| std::io::Error::other(format!("OxideAV error: {e}")))?;
    mux.write_header()
        .map_err(|e| std::io::Error::other(format!("OxideAV error: {e}")))?;

    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let packet = Packet::new(0, TimeBase::new(1, sample_rate as i64), bytes);
    mux.write_packet(&packet)
        .map_err(|e| std::io::Error::other(format!("OxideAV error: {e}")))?;
    mux.write_trailer()
        .map_err(|e| std::io::Error::other(format!("OxideAV error: {e}")))?;
    Ok(())
}

fn wav_f32_stream_info(channels: usize, sample_rate: u32) -> StreamInfo {
    let mut params = CodecParameters::audio(CodecId::new("pcm_f32le"));
    params.media_type = MediaType::Audio;
    params.channels = Some(channels as u16);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::F32);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, sample_rate as i64),
        duration: None,
        start_time: Some(0),
        params,
    }
}

pub fn frames_to_tensor<B: Backend>(frames: &[Vec<i64>], device: &B::Device) -> Tensor<B, 3, Int> {
    let num_frames = frames.len();
    let num_codebooks = if num_frames > 0 { frames[0].len() } else { 8 };

    let mut data = Vec::with_capacity(num_frames * num_codebooks);
    for codebook in 0..num_codebooks {
        for frame in frames {
            if frame.len() != num_codebooks {
                panic!(
                    "frames_to_tensor: inconsistent codebook count {}, expected {}",
                    frame.len(),
                    num_codebooks
                );
            }
            data.push(frame[codebook]);
        }
    }

    Tensor::from_data(
        TensorData::new(data, [1, num_codebooks, num_frames]),
        device,
    )
}

fn load_linear_from_tensors<B: Backend, F>(
    device: &B::Device,
    get_tensor: &F,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Linear<B>>
where
    F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
{
    let weight_name = format!("{}.weight", prefix);
    let bias_name = format!("{}.bias", prefix);

    let weight = if let Some((data, shape)) = get_tensor(&weight_name) {
        if shape.len() == 2 && shape[0] == out_dim && shape[1] == in_dim {
            let mut transposed = vec![0.0f32; in_dim * out_dim];
            for i in 0..out_dim {
                for j in 0..in_dim {
                    transposed[j * out_dim + i] = data[i * in_dim + j];
                }
            }
            Tensor::<B, 2>::from_data(TensorData::new(transposed, [in_dim, out_dim]), device)
        } else if shape.len() == 2 && shape[0] == in_dim && shape[1] == out_dim {
            Tensor::<B, 2>::from_data(TensorData::new(data, [in_dim, out_dim]), device)
        } else {
            Tensor::zeros([in_dim, out_dim], device)
        }
    } else {
        Tensor::zeros([in_dim, out_dim], device)
    };

    let bias = if let Some((data, shape)) = get_tensor(&bias_name) {
        if shape.len() == 1 && shape[0] == out_dim {
            Some(Tensor::<B, 1>::from_data(
                TensorData::new(data, [out_dim]),
                device,
            ))
        } else {
            Some(Tensor::zeros([out_dim], device))
        }
    } else {
        None
    };

    Ok(Linear {
        weight: Param::from_tensor(weight),
        bias: bias.map(Param::from_tensor),
    })
}

fn load_rmsnorm_from_tensors<B: Backend, F>(
    device: &B::Device,
    get_tensor: &F,
    prefix: &str,
    dim: usize,
) -> Result<RmsNorm<B>>
where
    F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
{
    let weight_name = format!("{}.weight", prefix);

    let weight = if let Some((data, shape)) = get_tensor(&weight_name) {
        if shape.len() == 1 && shape[0] == dim {
            Tensor::<B, 1>::from_data(TensorData::new(data, [dim]), device)
        } else {
            Tensor::ones([dim], device)
        }
    } else {
        Tensor::ones([dim], device)
    };

    Ok(RmsNorm {
        weight: Param::from_tensor(weight),
    })
}

fn load_param_tensor<B: Backend, F, const D: usize>(
    device: &B::Device,
    get_tensor: &F,
    name: &str,
    shape: [usize; D],
) -> Result<Param<Tensor<B, D>>>
where
    F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
{
    use burn::module::Param;
    use burn::tensor::TensorData;

    let tensor = if let Some((data, _)) = get_tensor(name) {
        Tensor::from_data(TensorData::new(data, shape), device)
    } else {
        Tensor::zeros(shape, device)
    };

    Ok(Param::from_tensor(tensor))
}

#[cfg(test)]
mod tests {
    use super::{Tensor, TensorData, write_wav_from_f32, write_wav_from_f32_interleaved};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_wav_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("maolan_{name}_{nanos}.wav"))
    }

    #[derive(Debug, Clone)]
    struct WavInfo {
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        is_float: bool,
        samples: Vec<f32>,
    }

    fn read_wav_f32(path: &std::path::Path) -> WavInfo {
        use symphonia::core::audio::SampleBuffer;
        use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
        use symphonia::core::errors::Error as SymphoniaError;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        let file = std::fs::File::open(path).expect("open wav");
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .expect("probe wav");
        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .or_else(|| format.tracks().first())
            .expect("audio track");
        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
        let sample_rate = track.codec_params.sample_rate.unwrap_or(48_000);
        let bits_per_sample = track.codec_params.bits_per_sample.unwrap_or(32);
        let sample_format = track.codec_params.sample_format;
        let is_float = sample_format
            .map(|f| {
                matches!(
                    f,
                    symphonia::core::sample::SampleFormat::F32
                        | symphonia::core::sample::SampleFormat::F64
                )
            })
            .unwrap_or(true);
        let track_id = track.id;
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .expect("create decoder");
        let mut sample_buf = None;
        let mut samples = Vec::new();
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => panic!("read error: {e}"),
            };
            if packet.track_id() != track_id {
                continue;
            }
            let decoded = decoder.decode(&packet).expect("decode packet");
            if sample_buf.is_none() {
                let spec = *decoded.spec();
                sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
            }
            let buf = sample_buf.as_mut().unwrap();
            buf.copy_interleaved_ref(decoded);
            samples.extend_from_slice(buf.samples());
        }
        WavInfo {
            channels: channels as u16,
            sample_rate,
            bits_per_sample: bits_per_sample as u16,
            is_float,
            samples,
        }
    }

    #[test]
    fn writes_mono_wav_as_f32() {
        let path = temp_wav_path("mono_f32");
        write_wav_from_f32(&[0.25, -0.5, 1.25], 48_000, &path).expect("write mono wav");

        let info = read_wav_f32(&path);
        assert_eq!(info.channels, 1);
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.bits_per_sample, 32);
        assert!(info.is_float);
        assert_eq!(info.samples, vec![0.25, -0.5, 1.25]);

        std::fs::remove_file(&path).expect("remove mono wav");
    }

    #[test]
    fn writes_interleaved_wav_as_f32() {
        let path = temp_wav_path("stereo_f32");
        let planar_samples = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3];
        write_wav_from_f32_interleaved(&planar_samples, 2, 3, 48_000, &path)
            .expect("write stereo wav");

        let info = read_wav_f32(&path);
        assert_eq!(info.channels, 2);
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.bits_per_sample, 32);
        assert!(info.is_float);
        assert_eq!(info.samples, vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3]);

        std::fs::remove_file(&path).expect("remove stereo wav");
    }

    #[test]
    fn heartcodec_config_default_values() {
        let config = super::HeartCodecConfig::default();
        assert_eq!(config.dim, 512);
        assert_eq!(config.codebook_size, 8192);
        assert_eq!(config.codebook_dim, 32);
        assert_eq!(config.num_quantizers, 8);
        assert_eq!(config.attention_head_dim, 64);
        assert_eq!(config.in_channels, 1024);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_layers_2, 6);
        assert_eq!(config.out_channels, 256);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.latent_hidden_dim, 128);
        assert_eq!(config.init_channel, 64);
        assert_eq!(config.num_bands, 1);
        assert_eq!(config.num_samples, 2);
        assert_eq!(config.downsample_factors, [3, 4, 4, 4, 5]);
        assert_eq!(config.downsample_kernel_sizes, [6, 8, 8, 8, 10]);
        assert_eq!(config.upsample_factors, [5, 4, 4, 4, 3]);
        assert_eq!(config.upsample_kernel_sizes, [10, 8, 8, 8, 6]);
        assert_eq!(config.default_kernel_size, 7);
        assert_eq!(config.delay_kernel_size, 5);
        assert_eq!(config.res_kernel_size, 7);
        assert!(config.causal);
        assert_eq!(config.ode_steps, 10);
    }

    #[test]
    fn frames_to_tensor_converts_correctly() {
        use burn::backend::ndarray::NdArray;
        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let frames: Vec<Vec<i64>> = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![9, 10, 11, 12, 13, 14, 15, 16],
            vec![17, 18, 19, 20, 21, 22, 23, 24],
        ];

        let tensor = super::frames_to_tensor::<NdArray<f32>>(&frames, &device);
        let dims = tensor.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 8);
        assert_eq!(dims[2], 3);
    }

    #[test]
    fn frames_to_tensor_empty_frames() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let frames: Vec<Vec<i64>> = vec![];

        let tensor = super::frames_to_tensor::<NdArray<f32>>(&frames, &device);
        let dims = tensor.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 8);
        assert_eq!(dims[2], 0);
    }

    #[test]
    #[should_panic(expected = "inconsistent codebook count")]
    fn frames_to_tensor_rejects_inconsistent_codebooks() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let frames: Vec<Vec<i64>> = vec![vec![1, 2, 3, 4, 5, 6, 7, 8], vec![9, 10, 11]];

        let _tensor = super::frames_to_tensor::<NdArray<f32>>(&frames, &device);
    }

    #[test]
    fn prelu_activation_forward() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let prelu = super::PReLU::<NdArray<f32>>::new(&device);

        let input = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0], [1, 1, 3]),
            &device,
        );
        let output = prelu.forward(input);
        let data = output.to_data().to_vec::<f32>().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn prelu_activation_negative_values() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let prelu = super::PReLU::<NdArray<f32>>::new(&device);

        let input = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![-4.0, -8.0], [1, 1, 2]),
            &device,
        );
        let output = prelu.forward(input);
        let data = output.to_data().to_vec::<f32>().unwrap();

        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn mlp_compute_hidden_dim() {
        let dim_1536 = super::Mlp::<burn::backend::ndarray::NdArray<f32>>::compute_hidden_dim(1536);
        let dim_3072 = super::Mlp::<burn::backend::ndarray::NdArray<f32>>::compute_hidden_dim(3072);

        assert_eq!(dim_1536, 4096);

        assert_eq!(dim_3072, 8192);
    }

    #[test]
    fn rms_norm_forward_preserves_shape() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let rms_norm = super::RmsNorm::<NdArray<f32>>::new(&device, 16, 1e-6);

        let input = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![1.0; 128], [2, 4, 16]),
            &device,
        );
        let output = rms_norm.forward(input);
        assert_eq!(output.dims(), [2, 4, 16]);
    }

    #[test]
    fn heartcodec_model_new_creates_valid_model() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let config = super::HeartCodecConfig::default();
        let model = super::HeartCodecModel::<NdArray<f32>>::new(&device);

        assert_eq!(model.ode_steps, config.ode_steps);
        assert_eq!(model.guidance_scale, 1.0);
    }

    #[test]
    fn heartcodec_model_with_ode_steps() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let model = super::HeartCodecModel::<NdArray<f32>>::new(&device).with_ode_steps(20);

        assert_eq!(model.ode_steps, 20);
    }

    #[test]
    fn heartcodec_model_with_guidance_scale() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let model = super::HeartCodecModel::<NdArray<f32>>::new(&device).with_guidance_scale(2.5);

        assert_eq!(model.guidance_scale, 2.5);
    }

    #[test]
    fn heartcodec_model_clamps_ode_steps() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();

        let model_high = super::HeartCodecModel::<NdArray<f32>>::new(&device).with_ode_steps(100);
        assert_eq!(model_high.ode_steps, 50);

        let model_low = super::HeartCodecModel::<NdArray<f32>>::new(&device).with_ode_steps(0);
        assert_eq!(model_low.ode_steps, 1);
    }

    #[test]
    fn residual_unit_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let unit = super::ResidualUnit::<NdArray<f32>>::new(&device, 128, 3);

        assert_eq!(unit.conv1.dilation, 3);
        assert_eq!(unit.conv2.dilation, 1);
    }

    #[test]
    fn resdecoder_block_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let block = super::ResDecoderBlock::<NdArray<f32>>::new(&device, 2048, 1024);

        assert_eq!(block.up_conv.stride, 5);
        assert_eq!(block.convs.len(), 5);
    }

    #[test]
    fn resdecoder_block_kernel_stride_pairs() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();

        let test_cases = [
            ((2048, 1024), (10, 5)),
            ((1024, 512), (8, 4)),
            ((512, 256), (8, 4)),
            ((256, 128), (8, 4)),
            ((128, 64), (6, 3)),
            ((100, 50), (8, 4)),
        ];

        for ((in_ch, out_ch), (_expected_k, expected_s)) in test_cases {
            let block = super::ResDecoderBlock::<NdArray<f32>>::new(&device, in_ch, out_ch);
            assert_eq!(
                block.up_conv.stride, expected_s,
                "stride mismatch for ({}, {})",
                in_ch, out_ch
            );
        }
    }

    #[test]
    fn project_layer_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let proj = super::ProjectLayer::<NdArray<f32>>::new(&device, 512, 256, 3);

        assert_eq!(proj.kernel_size, 3);
        assert_eq!(proj.out_channels, 256);
    }

    #[test]
    fn scalar_model_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let config = super::HeartCodecConfig::default();
        let model = super::ScalarModel::<NdArray<f32>>::new(&device, &config);

        assert_eq!(model.decoder_0.kernel_size(), 5);
        assert_eq!(model.decoder_7.kernel_size(), 7);
    }

    #[test]
    fn interpolate_1d_scale_factor_1() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let input = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        );

        let output = super::FlowMatching::<NdArray<f32>>::interpolate_1d(&input, 1);
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn interpolate_1d_scale_factor_2() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let input = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(vec![1.0, 2.0], [1, 1, 2]),
            &device,
        );

        let output = super::FlowMatching::<NdArray<f32>>::interpolate_1d(&input, 2);
        assert_eq!(output.dims(), [1, 2, 2]);
    }

    #[test]
    fn build_latent_masks_basic() {
        let masks = super::FlowMatching::<burn::backend::ndarray::NdArray<f32>>::build_latent_masks(
            10, 8, 5,
        );

        assert_eq!(masks.len(), 10);

        assert_eq!(masks[0], 1);
        assert_eq!(masks[4], 1);

        assert_eq!(masks[5], 2);
        assert_eq!(masks[7], 2);

        assert_eq!(masks[8], 0);
        assert_eq!(masks[9], 0);
    }

    #[test]
    fn build_latent_masks_incontext_larger_than_seq() {
        let masks = super::FlowMatching::<burn::backend::ndarray::NdArray<f32>>::build_latent_masks(
            5, 10, 10,
        );

        assert_eq!(masks.len(), 5);

        for mask in &masks {
            assert_eq!(*mask, 1);
        }
    }

    #[test]
    fn transformer_block_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let block = super::TransformerBlock::<NdArray<f32>>::new(&device, 512, 8, 64);

        assert_eq!(block.attn.num_heads, 8);
        assert_eq!(block.attn.head_dim, 64);
    }

    #[test]
    fn attention_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let attn = super::Attention::<NdArray<f32>>::new(&device, 512, 8, 64);

        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.head_dim, 64);
        assert_eq!(attn.rope_dim, 64);
    }

    #[test]
    fn llama_transformer_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let config = super::HeartCodecConfig::default();
        let transformer = super::LlamaTransformer::<NdArray<f32>>::new(&device, &config);

        assert_eq!(transformer.transformer_blocks.len(), 24);
        assert_eq!(transformer.transformer_blocks_2.len(), 6);
    }

    #[test]
    fn adalayer_norm_single_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let adaln = super::AdaLayerNormSingle::<NdArray<f32>>::new(&device, 512);

        let (output, embedded) = adaln.forward(0.5, burn::tensor::DType::F32);
        assert_eq!(output.dims(), [1, 6, 512]);
        assert_eq!(embedded.dims(), [1, 512]);
    }

    #[test]
    fn timestep_embedding_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let emb = super::TimestepEmbedding::<NdArray<f32>>::new(&device, 128, 512);

        let input = Tensor::<NdArray<f32>, 2>::zeros([1, 128], &device);
        let output = emb.forward(input);
        assert_eq!(output.dims(), [1, 512]);
    }

    #[test]
    fn pixart_alpha_combined_flow_embeddings() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let emb = super::PixArtAlphaCombinedFlowEmbeddings::<NdArray<f32>>::new(&device, 512);

        let output = emb.forward(0.5, burn::tensor::DType::F32);
        assert_eq!(output.dims(), [1, 512]);
    }

    #[test]
    fn vq_codebook_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let codebook = super::VQCodebook::<NdArray<f32>>::new(&device, 1024, 64);

        assert_eq!(codebook._codebook.embed.val().dims()[0], 1);
        assert_eq!(codebook._codebook.embed.val().dims()[1], 1024);
        assert_eq!(codebook._codebook.embed.val().dims()[2], 64);
    }

    #[test]
    fn residual_vq_new() {
        use burn::backend::ndarray::NdArray;

        let device = burn::prelude::Device::<NdArray<f32>>::default();
        let config = super::HeartCodecConfig::default();
        let rvq = super::ResidualVQ::<NdArray<f32>>::new(&device, &config);

        assert_eq!(rvq.layers.len(), 8);
    }
}
