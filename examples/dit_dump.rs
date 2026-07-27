//! Dump our pipeline intermediates in acestep.cpp's debug format for direct
//! numerical comparison against `ace-synth --dump` output.
//!
//! Usage:
//!   cargo run --release --example dit_dump -- <model_dir> <out_dir> [noise.bin]
//!
//! Uses the oracle's 19 codes and (optionally) the oracle's noise.bin so that
//! dit_step*_vt/xt can be compared exactly.

use burn::tensor::{Int, Tensor, TensorData};
use maolan_generate::acestep::condition::AceStepCondition;
use maolan_generate::acestep::config::AceStepConfig;
use maolan_generate::acestep::dit::{AceStepDiT, SFT_TIMESTEPS, TURBO_TIMESTEPS};
use maolan_generate::acestep::pipeline::{SilenceLatent, build_dit_text_prompt, build_metas_block};
use maolan_generate::acestep::qwen3::{Qwen3Config, Qwen3Model};
use maolan_generate::acestep::vae::{OobleckDecoder, OobleckVaeConfig};
use std::path::{Path, PathBuf};

type B = burn::backend::Wgpu<f32, i64, u32>;

const ORACLE_CODES: [u32; 19] = [
    37162, 24482, 53681, 61696, 48912, 36323, 57341, 14786, 18321, 50090, 36760, 36697, 36056,
    38160, 35080, 56000, 32927, 35847, 37127,
];

fn dump_tensor(path: &Path, name: &str, data: &[f32], shape: &[i32]) {
    let mut out = Vec::new();
    out.extend_from_slice(&(shape.len() as i32).to_le_bytes());
    for &d in shape {
        out.extend_from_slice(&d.to_le_bytes());
    }
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path.join(format!("{name}.bin")), out).expect("dump write");
}

fn load_bin(path: &Path) -> (Vec<i32>, Vec<f32>) {
    let raw = std::fs::read(path).expect("read bin");
    let ndims = i32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::new();
    for i in 0..ndims {
        shape.push(i32::from_le_bytes(
            raw[4 + 4 * i..8 + 4 * i].try_into().unwrap(),
        ));
    }
    let data: Vec<f32> = raw[4 + 4 * ndims..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (shape, data)
}

fn vec_of<B: burn::tensor::backend::Backend>(t: Tensor<B, 3>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .to_vec()
        .expect("tensor to vec")
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().unwrap_or_else(|| "/home/meka/repos/ace".into()));
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "/var/tmp/dump_ours".into()));
    let noise_path = args.next().map(PathBuf::from);
    std::fs::create_dir_all(&out_dir)?;

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        burn::backend::wgpu::RuntimeOptions {
            memory_config: burn::backend::wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        },
    );

    let variant_sft = std::env::var("MAOLAN_ACESTEP_VARIANT")
        .map(|v| v == "sft")
        .unwrap_or(false);
    let prefix = if variant_sft { "sft-" } else { "" };

    let dit_config = AceStepConfig::load(&model_dir.join(format!("{prefix}dit_config.json")))?;
    let text_config = Qwen3Config::load(&model_dir.join("qwen3_config.json"))?;
    let vae_config = OobleckVaeConfig::load(&model_dir.join("vae_config.json"))?;
    eprintln!("loading components (prefix '{prefix}')...");
    let text_encoder = Qwen3Model::<B>::from_burnpack(
        &text_config,
        &model_dir.join("qwen3-encoder.bpk"),
        &device,
    )?;
    let condition = AceStepCondition::<B>::from_burnpack(
        &dit_config,
        &model_dir.join(format!("{prefix}acestep-condition.bpk")),
        &device,
    )?;
    let dit = AceStepDiT::<B>::from_burnpack(
        &dit_config,
        &model_dir.join(format!("{prefix}acestep-dit.bpk")),
        &device,
    )?;
    let vae = OobleckDecoder::<B>::from_burnpack(
        &vae_config,
        &model_dir.join("acestep-vae.bpk"),
        &device,
    )?;
    let silence =
        SilenceLatent::<B>::from_burnpack(&model_dir.join("silence_latent.bpk"), &device)?;
    let tokenizer = tokie::Tokenizer::from_json(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // ---- text + lyric encoding (causal, official prompts) ----
    let metas = build_metas_block(Some(120.0), Some("A minor"), Some("4/4"), 4);
    let text_prompt = build_dit_text_prompt("Metal guitar with a lot of distortion", &metas);
    let ids = tokenizer.encode(&text_prompt, false).ids;
    let mut ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
    ids.push(151643); // explicit EOS (official add_eos=true)
    let n_text = ids.len();
    let ids_t = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, n_text]), &device);
    let text_hidden = text_encoder.forward(ids_t, true);
    let text_hidden_vec = vec_of(text_hidden.clone());
    dump_tensor(
        &out_dir,
        "text_hidden",
        &text_hidden_vec,
        &[n_text as i32, 1024],
    );

    let lyric_ids = tokenizer
        .encode(
            maolan_generate::acestep::pipeline::INSTRUMENTAL_LYRIC_PROMPT,
            false,
        )
        .ids;
    let mut lyric_ids: Vec<i64> = lyric_ids.into_iter().map(i64::from).collect();
    lyric_ids.push(151643); // explicit EOS (official add_eos=true)
    let n_lyric = lyric_ids.len();
    let lyric_ids_t =
        Tensor::<B, 2, Int>::from_data(TensorData::new(lyric_ids, [1, n_lyric]), &device);
    // Official: lyric branch is a raw embed_tokens lookup (no transformer).
    let lyric_hidden = text_encoder.embed_tokens.forward(lyric_ids_t);
    let lyric_embed_vec = vec_of(lyric_hidden.clone());
    dump_tensor(
        &out_dir,
        "lyric_embed",
        &lyric_embed_vec,
        &[n_lyric as i32, 1024],
    );
    let lyric_mask = Tensor::<B, 2, Int>::ones([1, n_lyric], &device);

    // ---- conditioning: use the oracle's enc_hidden/context when provided ----
    let mut oracle_context: Option<Tensor<B, 3>> = None;
    let enc = if let Some(dir) = std::env::var_os("MAOLAN_ORACLE_DUMP_DIR") {
        let dir = PathBuf::from(dir);
        let (enc_shape, enc_data) = load_bin(&dir.join("enc_hidden.bin"));
        eprintln!("oracle enc_hidden: {enc_shape:?}");
        let (ctx_shape, ctx_data) = load_bin(&dir.join("context.bin"));
        eprintln!("oracle context: {ctx_shape:?}");
        let enc = Tensor::<B, 3>::from_data(
            TensorData::new(enc_data, [1, enc_shape[0] as usize, enc_shape[1] as usize]),
            &device,
        );
        let ctx = Tensor::<B, 3>::from_data(TensorData::new(ctx_data, [1, 96, 128]), &device);
        oracle_context = Some(ctx);
        enc
    } else {
        condition.encode(
            text_hidden,
            lyric_hidden,
            lyric_mask,
            silence.timbre_reference(),
        )
    };
    let [_, enc_len, enc_dim] = enc.dims();
    let enc_vec = vec_of(enc.clone());
    dump_tensor(
        &out_dir,
        "enc_hidden",
        &enc_vec,
        &[enc_len as i32, enc_dim as i32],
    );

    // ---- FSQ hints from the given codes (env MAOLAN_ACESTEP_CODES or the
    // built-in oracle sequence) ----
    let codes: Vec<u32> = std::env::var_os("MAOLAN_ACESTEP_CODES")
        .map(|raw| {
            raw.to_string_lossy()
                .split(',')
                .filter_map(|part| part.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_else(|| ORACLE_CODES.to_vec());
    let n_codes = codes.len();
    let codes_tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(codes, [1, n_codes]), &device);
    let hints = condition.codes_to_hints(codes_tensor);
    let [_, hint_len, _] = hints.dims();
    let hints_vec = vec_of(hints.clone());
    dump_tensor(&out_dir, "detok_output", &hints_vec, &[hint_len as i32, 64]);

    // ---- noise ----
    let noise = if let Some(path) = noise_path {
        let (shape, data) = load_bin(&path);
        let frames = shape[0] as usize;
        Tensor::<B, 3>::from_data(TensorData::new(data, [1, frames, 64]), &device)
    } else {
        maolan_generate::acestep::pipeline::seeded_latent_noise(0, hint_len, 64, &device)
    };
    let [_, frames, _] = noise.dims();

    // src latents: hints cropped or silence-padded to the noise's frame count
    let src = if hint_len >= frames {
        hints.narrow(1, 0, frames)
    } else {
        let padding = silence.slice(frames - hint_len);
        Tensor::cat(vec![hints, padding], 1)
    };
    let chunk_mask = Tensor::ones([1, frames, 64], &device);
    let context = Tensor::cat(vec![src.clone(), chunk_mask], 2);
    let context = oracle_context.unwrap_or(context);
    let context_vec = vec_of(context.clone());
    dump_tensor(&out_dir, "context", &context_vec, &[frames as i32, 128]);

    let noise_vec = vec_of(noise.clone());
    dump_tensor(&out_dir, "noise", &noise_vec, &[frames as i32, 64]);

    // ---- DiT loop with per-step dumps (schedule from is_turbo) ----
    let timesteps: &[f32] = if dit_config.is_turbo {
        &TURBO_TIMESTEPS
    } else {
        &SFT_TIMESTEPS
    };
    let kv = dit.prepare_cross_kv(enc);
    let mut xt = noise;
    let total = timesteps.len();
    for (index, &t_cur) in timesteps.iter().enumerate() {
        let xt_vec = vec_of(xt.clone());
        dump_tensor(
            &out_dir,
            &format!("dit_step{index}_xt"),
            &xt_vec,
            &[frames as i32, 64],
        );
        let v = dit.forward_with_kv(xt.clone(), t_cur, context.clone(), &kv);
        let v_vec = vec_of(v.clone());
        dump_tensor(
            &out_dir,
            &format!("dit_step{index}_vt"),
            &v_vec,
            &[frames as i32, 64],
        );
        let dt = if index + 1 == total {
            t_cur
        } else {
            t_cur - timesteps[index + 1]
        };
        xt = xt - v * dt;
    }
    let x0_vec = vec_of(xt.clone());
    dump_tensor(&out_dir, "dit_x0", &x0_vec, &[frames as i32, 64]);

    // ---- VAE decode ----
    let audio = vae.decode(xt);
    let [_, channels, samples] = audio.dims();
    let audio_vec = vec_of(audio);
    dump_tensor(
        &out_dir,
        "vae_audio",
        &audio_vec,
        &[channels as i32, samples as i32],
    );
    println!("dumps written to {}", out_dir.display());
    Ok(())
}
