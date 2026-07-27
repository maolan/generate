//! Diagnostic: decode the VAE-encoded silence latent through the ported
//! Oobleck decoder. The output should be (near-)digital silence; a large or
//! noisy result means the VAE decoder port is wrong.
//!
//! Usage: cargo run --release --example decode_silence -- <model_dir>

use maolan_generate::acestep::pipeline::SilenceLatent;
use maolan_generate::acestep::vae::{OobleckDecoder, OobleckVaeConfig};
use std::path::Path;

type B = burn::backend::NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/home/meka/repos/ace"));
    let device = Default::default();

    let vae_config = OobleckVaeConfig::load(&model_dir.join("vae_config.json"))?;
    let vae = OobleckDecoder::<B>::from_burnpack(
        &vae_config,
        &model_dir.join("acestep-vae.bpk"),
        &device,
    )?;
    let silence =
        SilenceLatent::<B>::from_burnpack(&model_dir.join("silence_latent.bpk"), &device)?;
    let [_, frames, _] = silence.silence_latent.dims();
    println!("silence latent: {frames} frames");

    let latents = silence.slice(750);
    let audio = vae.decode(latents);
    let [_, channels, samples] = audio.dims();
    let values: Vec<f32> = audio
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let peak = values.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
    let rms = (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt();
    println!("decoded: {channels} ch x {samples} samples");
    println!("peak {peak:.6}  rms {rms:.6}");
    println!(
        "{}",
        if peak < 1e-3 {
            "OK: VAE decodes silence correctly"
        } else {
            "SUSPECT: VAE decoder output is far from silence"
        }
    );

    // Also dump the latent stats so we can eyeball them.
    let latents = SilenceLatent::<B>::from_burnpack(
        Path::new(&model_dir).join("silence_latent.bpk").as_path(),
        &device,
    )?
    .slice(4);
    let v: Vec<f32> = latents
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let lpeak = v.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
    let lrms = (v.iter().map(|v| v * v).sum::<f32>() / v.len() as f32).sqrt();
    println!("latent peak {lpeak:.6} rms {lrms:.6}");
    Ok(())
}
