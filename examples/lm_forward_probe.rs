//! Probe: run the LM forward on the real prompt and inspect next-token
//! logits (top-10), then compare greedy vs sampled code generation.
//!
//! Usage: cargo run --release --example lm_forward_probe -- <model_dir>

use burn::tensor::{Int, Tensor, TensorData};
use maolan_generate::acestep::lm::{self, AceStepLm, AudioCodeVocab, SamplingConfig};
use maolan_generate::acestep::qwen3::{Qwen3Config, Qwen3Model};
use std::path::Path;

type B = burn::backend::NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/meka/repos/ace".to_string());
    let model_dir = Path::new(&model_dir);
    let device = Default::default();

    let config = Qwen3Config::load(&model_dir.join("lm_config.json"))?;
    let lm = AceStepLm::<B>::from_burnpack(&config, &model_dir.join("acestep-lm.bpk"), &device)?;
    let vocab = AudioCodeVocab::from_tokenizer_json(&model_dir.join("lm_tokenizer.json"))?;

    let cot = lm::build_cot_block(
        "Metal guitar with a lot of distortion",
        Some(120.0),
        Some("A minor"),
        Some("4/4"),
        4,
    );
    let prompt = lm::build_codes_prompt("Metal guitar with a lot of distortion", &cot);
    let ids = lm::tokenize_prompt(&model_dir.join("lm_tokenizer.json"), &prompt)?;
    println!("prompt: {} tokens", ids.len());

    // One full forward; inspect logits at the last position.
    let model: &Qwen3Model<B> = &lm.model;
    let ids_i64: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
    let len = ids_i64.len();
    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i64, [1, len]), &device);
    let hidden = model.forward(input, true);
    let [_, _, hidden_dim] = hidden.dims();
    let last = hidden.narrow(1, len - 1, 1).reshape([1, hidden_dim]);
    let weight = model.embedding_weight();
    let logits = last
        .matmul(weight.clone().transpose())
        .reshape([config.vocab_size as usize]);
    let values: Vec<f32> = logits
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    println!("\ntop-10 next-token predictions:");
    for &idx in order.iter().take(10) {
        let id = idx as u32;
        let label = vocab
            .token_id_to_code(id)
            .map(|c| format!("audio_code_{c}"))
            .unwrap_or_else(|| format!("token {id}"));
        println!("  id {id:>7} logit {:>8.3}  {label}", values[idx]);
    }

    // Greedy generation (temperature ~0) vs sampled (official defaults).
    let greedy = SamplingConfig {
        temperature: 1e-5,
        top_k: 1,
        cfg_scale: 1.0,
        ..SamplingConfig::new(28, 0)
    };
    for (name, sampling) in [
        ("greedy", greedy),
        ("sampled official", SamplingConfig::new(28, 0)),
    ] {
        let codes = lm.generate_codes(&ids, None, &vocab, &sampling, None);
        println!("\n{name}: {} codes: {codes:?}", codes.len());
    }

    // Greedy generation via FULL forward each step (no KV cache) to isolate
    // the incremental path.
    let mut full_ids = ids.clone();
    let mut full_codes = Vec::new();
    for _ in 0..20 {
        let ids_i64: Vec<i64> = full_ids.iter().map(|&id| i64::from(id)).collect();
        let len = ids_i64.len();
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i64, [1, len]), &device);
        let hidden = model.forward(input, true);
        let last = hidden.narrow(1, len - 1, 1).reshape([1, hidden_dim]);
        let logits = last
            .matmul(weight.clone().transpose())
            .reshape([config.vocab_size as usize]);
        let values: Vec<f32> = logits
            .into_data()
            .convert::<f32>()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let next = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap();
        if next == AudioCodeVocab::IM_END_ID {
            break;
        }
        if let Some(code) = vocab.token_id_to_code(next) {
            full_codes.push(code);
        }
        full_ids.push(next);
    }
    println!(
        "\nfull-forward greedy: {} codes: {full_codes:?}",
        full_codes.len()
    );
    Ok(())
}
