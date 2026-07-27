//! Probe: feed controlled FSQ code sequences through codes_to_hints + VAE to
//! test whether the FSQ/detokenizer chain produces lively latents.
//!
//! Usage: cargo run --release --example fsq_probe -- <model_dir>

use burn::tensor::{Int, Tensor, TensorData};
use maolan_generate::acestep::condition::AceStepCondition;
use maolan_generate::acestep::config::AceStepConfig;
use maolan_generate::acestep::pipeline::SilenceLatent;
use maolan_generate::acestep::vae::{OobleckDecoder, OobleckVaeConfig};
use std::path::Path;

type B = burn::backend::NdArray<f32>;

fn rms(values: &[f32]) -> f32 {
    (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt()
}

fn run_case(
    name: &str,
    codes: &[u32],
    condition: &AceStepCondition<B>,
    vae: &OobleckDecoder<B>,
    silence: &SilenceLatent<B>,
    device: &burn::tensor::Device<B>,
) -> anyhow::Result<()> {
    let n = codes.len();
    let tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(codes.to_vec(), [1, n]), device);
    let hints = condition.codes_to_hints(tensor);
    let silence_frames = silence.slice(n * 5);
    let diff = (hints.clone() - silence_frames)
        .abs()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let hint_values: Vec<f32> = hints
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let audio = vae.decode(hints);
    let audio_values: Vec<f32> = audio
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let peak = audio_values.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
    let mean_abs_diff = diff.iter().sum::<f32>() / diff.len() as f32;
    println!(
        "{name}: hints rms {:.4}  mean|hints-silence| {:.4}  audio rms {:.4}  audio peak {:.4}",
        rms(&hint_values),
        mean_abs_diff,
        rms(&audio_values),
        peak
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/meka/repos/ace".to_string());
    let model_dir = Path::new(&model_dir);
    let device = Default::default();

    let dit_config = AceStepConfig::load(&model_dir.join("dit_config.json"))?;
    let condition = AceStepCondition::<B>::from_burnpack(
        &dit_config,
        &model_dir.join("acestep-condition.bpk"),
        &device,
    )?;
    let vae_config = OobleckVaeConfig::load(&model_dir.join("vae_config.json"))?;
    let vae = OobleckDecoder::<B>::from_burnpack(
        &vae_config,
        &model_dir.join("acestep-vae.bpk"),
        &device,
    )?;
    let silence =
        SilenceLatent::<B>::from_burnpack(&model_dir.join("silence_latent.bpk"), &device)?;

    // Splitmix64 random codes.
    let mut state = 42_u64;
    let mut rand_codes = Vec::new();
    for _ in 0..20 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rand_codes.push(((state >> 33) % 64000) as u32);
    }

    let lm_codes: Vec<u32> = vec![
        61890, 51649, 53753, 53753, 56314, 53754, 55802, 4538, 4538, 4538, 5050, 5050, 5050, 5050,
        5050, 5050, 5050, 5050, 5050, 43513,
    ];

    // Structure check: for a few distinct codes, dump the detokenizer output
    // frames and measure how much outputs differ between codes.
    for code in [0u32, 12345, 53754] {
        let tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![code], [1, 1]), &device);
        let hints = condition.codes_to_hints(tensor); // [1, 5, 64]
        let values: Vec<f32> = hints
            .into_data()
            .convert::<f32>()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let frame_means: Vec<f32> = values
            .as_chunks::<64>()
            .0
            .iter()
            .map(|f| f.iter().sum::<f32>() / 64.0)
            .collect();
        let frame_rms: Vec<f32> = values.as_chunks::<64>().0.iter().map(|f| rms(f)).collect();
        println!(
            "code {code:>6}: frame means {:?}",
            frame_means
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
        );
        println!(
            "           frame rms   {:?}",
            frame_rms
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect::<Vec<_>>()
        );
    }

    // Time-variance vs channel-variance of hint latents for random codes.
    let tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(rand_codes.clone(), [1, 20]), &device);
    let hints = condition.codes_to_hints(tensor); // [1, 100, 64]
    let values: Vec<f32> = hints
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (frames, _) = values.as_chunks::<64>();
    // variance of per-frame means (time structure) vs mean per-frame variance (channel structure)
    let frame_means: Vec<f32> = frames
        .iter()
        .map(|f| f.iter().sum::<f32>() / 64.0)
        .collect();
    let tm = frame_means.iter().sum::<f32>() / frame_means.len() as f32;
    let time_var =
        frame_means.iter().map(|v| (v - tm) * (v - tm)).sum::<f32>() / frame_means.len() as f32;
    let chan_var = frames
        .iter()
        .map(|f| {
            let m = f.iter().sum::<f32>() / 64.0;
            f.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / 64.0
        })
        .sum::<f32>()
        / frames.len() as f32;
    println!("random-code hints: time var {time_var:.6}  channel var {chan_var:.6}");

    run_case(
        "random codes ",
        &rand_codes,
        &condition,
        &vae,
        &silence,
        &device,
    )?;
    run_case(
        "LM codes     ",
        &lm_codes,
        &condition,
        &vae,
        &silence,
        &device,
    )?;
    run_case(
        "constant 0   ",
        &[0u32; 20],
        &condition,
        &vae,
        &silence,
        &device,
    )?;
    Ok(())
}
