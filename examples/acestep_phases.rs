//! Phased ACE-Step diagnostic: runs one generation and dumps/analyzes every
//! phase artifact, writing three WAVs:
//!
//! - `<out_prefix>_hints.wav` — LM hint latents decoded by the VAE, DiT
//!   bypassed. Music here means LM + FSQ + detokenizer + VAE all work and any
//!   remaining problem lives in the DiT.
//! - `<out_prefix>_final.wav` — the normal full-pipeline output.
//! - `<out_prefix>_silence.wav` — silence latent decoded by the VAE (should be
//!   near-zero).
//!
//! Usage:
//!   cargo run --release --example acestep_phases -- \
//!       <model_dir> <out_prefix> [caption] [bpm] [key_scale] [time_sig] [length_ms]

use maolan_generate::acestep::{
    AceStepModelPaths, AceStepPipeline, AceStepTrace, GenerateMetadata, SilenceLatent,
};
use maolan_generate::heartcodec::write_wav_from_f32_interleaved;
use std::path::Path;

type B = burn::backend::Wgpu<f32, i64, u32>;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "/home/meka/repos/ace".to_string());
    let out_prefix = args
        .next()
        .unwrap_or_else(|| "/tmp/acestep_phase".to_string());
    let caption = args
        .next()
        .unwrap_or_else(|| "Metal guitar with a lot of distortion".to_string());
    let bpm: Option<f32> = args.next().and_then(|v| v.parse().ok()).or(Some(120.0));
    let key_scale = args.next().or_else(|| Some("A minor".to_string()));
    let time_signature = args.next().or_else(|| Some("4/4".to_string()));
    let length_ms: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4000);

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        burn::backend::wgpu::RuntimeOptions {
            memory_config: burn::backend::wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        },
    );

    let variant = if std::env::var("MAOLAN_ACESTEP_VARIANT")
        .map(|v| v == "sft")
        .unwrap_or(false)
    {
        maolan_generate::acestep::AceStepVariant::Sft
    } else {
        maolan_generate::acestep::AceStepVariant::Turbo
    };
    let paths = AceStepModelPaths::resolve(Path::new(&model_dir), variant)?;
    let mut progress = |phase: &str, p: f32, op: &str| {
        eprintln!("[{phase}] {:.0}% {op}", p * 100.0);
    };
    let pipeline = AceStepPipeline::<B>::load(&paths, &device, &mut progress)?;

    let metadata = GenerateMetadata {
        bpm,
        key_scale: key_scale.as_deref(),
        time_signature: time_signature.as_deref(),
    };
    let mut trace = AceStepTrace::default();
    let (audio, meta) =
        pipeline.generate_traced(&caption, &metadata, length_ms, 0, &mut progress, &mut trace)?;

    // ---- Phase dumps ----
    println!(
        "\n===== TEXT PROMPT (caption branch) =====\n{}",
        trace.text_prompt
    );
    println!("===== LYRIC PROMPT =====\n{}", trace.lyric_prompt);
    println!("===== LM CoT BLOCK =====\n{}", trace.cot_block);
    println!("===== LM CODES ({} total) =====", trace.codes.len());
    println!("all codes: {:?}", trace.codes);
    let mut sorted = trace.codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    println!(
        "unique: {}, min: {}, max: {}",
        sorted.len(),
        sorted.first().unwrap_or(&0),
        sorted.last().unwrap_or(&0)
    );
    println!(
        "\n===== CONDITIONING =====\nenc mean {:.6}  enc std {:.6}",
        trace.enc_mean, trace.enc_std
    );
    println!("hints latent rms: {:.6}", trace.hints_latent_rms);
    println!("final latent rms: {:.6}", trace.final_latent_rms);
    println!("\n===== DIT STEPS =====\nstep  t        xt_rms   v_rms");
    for (i, step) in trace.dit_steps.iter().enumerate() {
        println!(
            "{i:>4}  {:.4}   {:.4}   {:.4}",
            step.t, step.xt_rms, step.v_rms
        );
    }

    // ---- WAVs ----
    if let Some((interleaved, channels, frames)) = &trace.hints_audio {
        let path = format!("{out_prefix}_hints.wav");
        write_wav_from_f32_interleaved(
            interleaved,
            *channels,
            *frames,
            meta.sample_rate_hz,
            Path::new(&path),
        )?;
        println!("\nwrote {path} (VAE decode of LM hints, DiT bypassed)");
    }

    let [_, channels, frames] = audio.dims();
    let channel_major: Vec<f32> = audio
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut interleaved = vec![0.0_f32; channel_major.len()];
    for (ch, samples) in channel_major.chunks_exact(frames).enumerate() {
        for (frame, sample) in samples.iter().enumerate() {
            interleaved[frame * channels + ch] = *sample;
        }
    }
    let path = format!("{out_prefix}_final.wav");
    write_wav_from_f32_interleaved(
        &interleaved,
        channels,
        frames,
        meta.sample_rate_hz,
        Path::new(&path),
    )?;
    println!("wrote {path} (full pipeline)");

    // Silence decode reference.
    let silence = SilenceLatent::<B>::from_burnpack(&paths.silence_latent_bpk, &device)?;
    let silence_audio = pipeline.vae.decode(silence.slice(frames / 1920));
    let [_, sch, sframes] = silence_audio.dims();
    let smaj: Vec<f32> = silence_audio
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut sinter = vec![0.0_f32; smaj.len()];
    for (ch, samples) in smaj.chunks_exact(sframes).enumerate() {
        for (frame, sample) in samples.iter().enumerate() {
            sinter[frame * sch + ch] = *sample;
        }
    }
    let path = format!("{out_prefix}_silence.wav");
    write_wav_from_f32_interleaved(&sinter, sch, sframes, meta.sample_rate_hz, Path::new(&path))?;
    println!("wrote {path} (VAE decode of silence latent)");

    Ok(())
}
