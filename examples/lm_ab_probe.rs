//! A/B probe: f32 vs f16 4B LM forward on the same prompt — checks for
//! NaN/Inf (overflow) vs wrong values (broken load).

use burn::tensor::{Int, Tensor, TensorData};
use maolan_generate::acestep::lm::AceStepLm;
use maolan_generate::acestep::qwen3::{Qwen3Config, Qwen3Model};
use std::path::Path;

fn top5(model: &Qwen3Model, label: &str, device: &burn::tensor::Device) {
    let ids = vec![151644i64, 8948, 198, 2, 29051, 198, 31115, 7699];
    let n = ids.len();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, n]), device);
    let hidden = model.forward(input, true);
    let [_, _, hidden_dim] = hidden.dims();
    let last = hidden.narrow(1, n - 1, 1).reshape([1, hidden_dim]);
    let weight = model.embedding_weight();
    let vocab = weight.dims()[0];
    let logits = last.matmul(weight.transpose()).reshape([vocab]);
    let values: Vec<f32> = logits
        .into_data()
        .convert::<f32>()
        .try_to_vec()
        .expect("logits");
    let nan = values.iter().filter(|v| v.is_nan()).count();
    let inf = values.iter().filter(|v| v.is_infinite()).count();
    let max = values.iter().fold(f32::NEG_INFINITY, |a, v| a.max(*v));
    let min = values.iter().fold(f32::INFINITY, |a, v| a.min(*v));
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    println!("{label}: NaN {nan}, Inf {inf}, min {min:.1}, max {max:.1}");
    for &i in order.iter().take(5) {
        println!("  id {i} logit {:.3}", values[i]);
    }
}

fn main() -> anyhow::Result<()> {
    let model_dir = Path::new("/home/meka/repos/ace-sft");
    let config = Qwen3Config::load(&model_dir.join("sft-lm_config.json"))?;
    let bpk = model_dir.join("sft-acestep-lm.bpk");

    let dev32 = burn::tensor::Device::flex();
    let lm32 = AceStepLm::from_burnpack(&config, &bpk, &dev32)?;
    top5(&lm32.model, "f32", &dev32);

    let dev16 = burn::tensor::Device::wgpu_options()
        .graphics_api(burn::tensor::wgpu::WgpuBackend::Vulkan)
        .memory_config(burn::tensor::wgpu::MemoryConfiguration::ExclusivePages)
        .init()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let lm16 = AceStepLm::from_burnpack_cast(&config, &bpk, &dev16)?;
    top5(&lm16.model, "f16", &dev16);
    Ok(())
}
