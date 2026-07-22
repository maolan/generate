use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub fn load_weight_norm_conv1d<B: Backend>(
    device: &B::Device,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: PaddingConfig1d,
    dilation: usize,
    groups: usize,
    _bias_data: Option<Vec<f32>>,
    original0: Option<Vec<f32>>,
    original1: Option<Vec<f32>>,
) -> Conv1d<B> {
    let in_channels_per_group = in_channels / groups;

    let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
        .with_stride(stride)
        .with_padding(padding)
        .with_dilation(dilation)
        .with_groups(groups)
        .init(device);

    if let (Some(g_data), Some(v_data)) = (original0, original1) {
        let g = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(g_data, [out_channels, 1, 1]),
            device,
        );

        let v = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(
                v_data,
                [out_channels, in_channels_per_group, kernel_size],
            ),
            device,
        );

        let v_norm_sq = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1);
        let v_norm = v_norm_sq.sqrt();
        let v_norm = v_norm.unsqueeze_dim::<3>(2).unsqueeze_dim::<3>(2);

        let _weight = g * v / (v_norm + 1e-12);
    }

    conv
}

pub fn compute_weight_norm_weight<B: Backend>(g: &Tensor<B, 3>, v: &Tensor<B, 3>) -> Tensor<B, 3> {
    let v_norm_sq = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1);
    let v_norm = v_norm_sq.sqrt();
    let v_norm = v_norm.unsqueeze_dim::<3>(2).unsqueeze_dim::<3>(2);

    g.clone() * v.clone() / (v_norm + 1e-12)
}
