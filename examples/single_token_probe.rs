//! Single-token text encoder probe: run the Qwen3 encoder on token id 2 ("#")
//! and dump the hidden state; with causal attention this must equal
//! text_hidden.bin row 0 of the oracle.

use burn::tensor::{Int, Tensor, TensorData};
use maolan_generate::acestep::qwen3::{Qwen3Config, Qwen3Model};
use std::path::Path;

type B = burn::backend::NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/meka/repos/ace".to_string());
    let model_dir = Path::new(&model_dir);
    let device = Default::default();

    let config = Qwen3Config::load(&model_dir.join("qwen3_config.json"))?;
    let encoder =
        Qwen3Model::<B>::from_burnpack(&config, &model_dir.join("qwen3-encoder.bpk"), &device)?;

    let ids: Vec<i64> = std::env::args()
        .skip(2)
        .map(|a| a.parse().unwrap())
        .collect();
    let ids = if ids.is_empty() { vec![2i64] } else { ids };
    let n = ids.len();
    let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, n]), &device);
    let hidden = encoder.forward(ids, true);
    let values: Vec<f32> = hidden
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // dump in acestep debug format [n, 1024]
    let mut out = Vec::new();
    out.extend_from_slice(&2i32.to_le_bytes());
    out.extend_from_slice(&(n as i32).to_le_bytes());
    out.extend_from_slice(&1024i32.to_le_bytes());
    for v in &values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("/var/tmp/dump_ours/enc_probe_out.bin", out)?;
    let rms = (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt();
    println!(
        "tokens {n} hidden rms {rms:.4}, first 8: {:?}",
        &values[..8]
    );
    Ok(())
}
