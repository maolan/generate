use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use burn::tensor::module::{conv_transpose1d, conv1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, PadMode};

#[derive(Module, Debug)]
pub struct WNConv1d<B: Backend> {
    pub bias: Param<Tensor<B, 1>>,

    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,

    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub causal: bool,
}

impl<B: Backend> WNConv1d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
    ) -> Self {
        let in_channels_per_group = in_channels / groups;

        Self {
            bias: Param::from_tensor(Tensor::zeros([out_channels], device)),
            weight_g: Param::from_tensor(Tensor::ones([out_channels, 1, 1], device)),
            weight_v: Param::from_tensor(Tensor::zeros(
                [out_channels, in_channels_per_group, kernel_size],
                device,
            )),
            stride,
            padding,
            dilation,
            groups,
            causal,
        }
    }

    fn compute_weight(&self) -> Tensor<B, 3> {
        let g = self.weight_g.val();
        let v = self.weight_v.val();

        let v_norm_sq = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1);
        let v_norm = v_norm_sq.sqrt();

        let out_ch = v_norm.dims()[0];
        let v_norm = v_norm.reshape([out_ch, 1, 1]);

        g * v / (v_norm + 1e-12)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let weight = self.compute_weight();
        let bias = self.bias.val();

        let left_padding = if self.causal {
            self.dilation * (self.kernel_size().saturating_sub(1))
        } else {
            self.padding
        };
        let x = if self.causal && left_padding > 0 {
            x.pad((left_padding, 0, 0, 0), PadMode::Constant(0.0))
        } else {
            x
        };
        let options = ConvOptions::new(
            [self.stride],
            [if self.causal { 0 } else { self.padding }],
            [self.dilation],
            self.groups,
        );

        conv1d(x, weight, Some(bias), options)
    }

    pub fn kernel_size(&self) -> usize {
        self.weight_v.dims()[2]
    }
}

#[derive(Module, Debug)]
pub struct WNConvTranspose1d<B: Backend> {
    pub bias: Param<Tensor<B, 1>>,

    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,

    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub causal: bool,
}

impl<B: Backend> WNConvTranspose1d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
    ) -> Self {
        Self {
            bias: Param::from_tensor(Tensor::zeros([out_channels], device)),
            weight_g: Param::from_tensor(Tensor::ones([in_channels, 1, 1], device)),
            weight_v: Param::from_tensor(Tensor::zeros(
                [in_channels, out_channels / groups, kernel_size],
                device,
            )),
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            causal,
        }
    }

    fn compute_weight(&self) -> Tensor<B, 3> {
        let g = self.weight_g.val();
        let v = self.weight_v.val();

        let v_norm_sq = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1);
        let v_norm = v_norm_sq.sqrt();

        let in_ch = v_norm.dims()[0];
        let v_norm = v_norm.reshape([in_ch, 1, 1]);

        g * v / (v_norm + 1e-12)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let weight = self.compute_weight();
        let bias = self.bias.val();

        let options = ConvTransposeOptions::new(
            [self.stride],
            [if self.causal { 0 } else { self.padding }],
            [self.output_padding],
            [self.dilation],
            self.groups,
        );

        let x = conv_transpose1d(x, weight, Some(bias), options);
        if self.causal {
            let [batch, channels, time] = x.dims();
            let crop = self.stride.min(time);
            x.slice([0..batch, 0..channels, 0..time.saturating_sub(crop)])
        } else {
            x
        }
    }
}

pub struct WNConv1dLoadArgs {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub causal: bool,
    pub weight_g: (Vec<f32>, Vec<usize>),
    pub weight_v: (Vec<f32>, Vec<usize>),
    pub bias: Option<(Vec<f32>, Vec<usize>)>,
}

pub fn load_wnconv_from_tensors<B: Backend>(
    device: &B::Device,
    args: WNConv1dLoadArgs,
) -> anyhow::Result<WNConv1d<B>> {
    use burn::tensor::TensorData;

    let (g_data, g_shape) = args.weight_g;
    let weight_g_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(g_data, [g_shape[0], g_shape[1], g_shape[2]]),
        device,
    );

    let (v_data, v_shape) = args.weight_v;
    let weight_v_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(v_data, [v_shape[0], v_shape[1], v_shape[2]]),
        device,
    );

    let bias_tensor = if let Some((b_data, b_shape)) = args.bias {
        Tensor::<B, 1>::from_data(TensorData::new(b_data, [b_shape[0]]), device)
    } else {
        Tensor::zeros([args.out_channels], device)
    };

    Ok(WNConv1d {
        bias: Param::from_tensor(bias_tensor),
        weight_g: Param::from_tensor(weight_g_tensor),
        weight_v: Param::from_tensor(weight_v_tensor),
        stride: 1,
        padding: (args.kernel_size / 2) * args.dilation,
        dilation: args.dilation,
        groups: 1,
        causal: args.causal,
    })
}

pub struct WNConvTranspose1dLoadArgs {
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub causal: bool,
    pub weight_g: (Vec<f32>, Vec<usize>),
    pub weight_v: (Vec<f32>, Vec<usize>),
    pub bias: Option<(Vec<f32>, Vec<usize>)>,
}

pub fn load_wnconv_transpose_from_tensors<B: Backend>(
    device: &B::Device,
    args: WNConvTranspose1dLoadArgs,
) -> anyhow::Result<WNConvTranspose1d<B>> {
    use burn::tensor::TensorData;

    let (g_data, g_shape) = args.weight_g;
    let weight_g_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(g_data, [g_shape[0], g_shape[1], g_shape[2]]),
        device,
    );

    let (v_data, v_shape) = args.weight_v;
    let weight_v_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(v_data, [v_shape[0], v_shape[1], v_shape[2]]),
        device,
    );

    let bias_tensor = if let Some((b_data, b_shape)) = args.bias {
        Tensor::<B, 1>::from_data(TensorData::new(b_data, [b_shape[0]]), device)
    } else {
        Tensor::zeros([args.out_channels], device)
    };

    Ok(WNConvTranspose1d {
        bias: Param::from_tensor(bias_tensor),
        weight_g: Param::from_tensor(weight_g_tensor),
        weight_v: Param::from_tensor(weight_v_tensor),
        stride: args.stride,
        padding: args.kernel_size / 2,
        output_padding: 0,
        dilation: 1,
        groups: 1,
        causal: args.causal,
    })
}

pub fn load_prelu_from_tensor<B: Backend>(
    device: &B::Device,
    data: Vec<f32>,
    shape: Vec<usize>,
) -> anyhow::Result<super::PReLU<B>> {
    use burn::module::Param;
    use burn::tensor::TensorData;

    let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, [shape[0]]), device);

    Ok(super::PReLU {
        weight: Param::from_tensor(tensor),
    })
}

#[derive(Module, Debug)]
pub struct PlainConv1d<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 1>>,

    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub causal: bool,
}

#[derive(Module, Debug)]
pub struct PostProcessor<B: Backend> {
    pub conv: PlainConv1d<B>,
    pub activation: super::PReLU<B>,
    pub num_samples: usize,
}

impl<B: Backend> PostProcessor<B> {
    pub fn new(device: &B::Device, channels: usize, num_samples: usize) -> Self {
        Self {
            conv: PlainConv1d::new(device, channels, channels, 7, 1, 3, 1, 1, true),
            activation: super::PReLU::new(device),
            num_samples,
        }
    }

    pub fn load_from_tensors<F>(
        device: &B::Device,
        get_tensor: &F,
        prefix: &str,
        channels: usize,
        num_samples: usize,
    ) -> anyhow::Result<Self>
    where
        F: Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
    {
        let conv = if let Some((w_data, w_shape)) = get_tensor(&format!("{}.conv.weight", prefix)) {
            let b = get_tensor(&format!("{}.conv.bias", prefix));
            load_conv1d_from_tensors(
                device,
                Conv1dLoadArgs {
                    in_channels: channels,
                    out_channels: channels,
                    kernel_size: 7,
                    causal: true,
                    weight: (w_data, w_shape),
                    bias: b,
                },
            )?
        } else {
            PlainConv1d::new(device, channels, channels, 7, 1, 3, 1, 1, true)
        };
        let activation =
            if let Some((data, shape)) = get_tensor(&format!("{}.activation.weight", prefix)) {
                load_prelu_from_tensor(device, data, shape)?
            } else {
                super::PReLU::new(device)
            };
        Ok(Self {
            conv,
            activation,
            num_samples,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();
        let x = x.swap_dims(1, 2);
        let x = if self.num_samples <= 1 {
            x
        } else {
            let repeated: Tensor<B, 4> = x.unsqueeze_dim::<4>(2).repeat_dim(2, self.num_samples);
            repeated.reshape([batch, time * self.num_samples, channels])
        };
        let x = x.swap_dims(1, 2);
        let x = self.conv.forward(x);
        self.activation.forward(x)
    }
}

impl<B: Backend> PlainConv1d<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
    ) -> Self {
        let in_channels_per_group = in_channels / groups;

        Self {
            weight: Param::from_tensor(Tensor::zeros(
                [out_channels, in_channels_per_group, kernel_size],
                device,
            )),
            bias: Param::from_tensor(Tensor::zeros([out_channels], device)),
            stride,
            padding,
            dilation,
            groups,
            causal,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let weight = self.weight.val();
        let bias = self.bias.val();
        let left_padding = if self.causal {
            self.dilation * (self.kernel_size().saturating_sub(1))
        } else {
            self.padding
        };
        let x = if self.causal && left_padding > 0 {
            x.pad((left_padding, 0, 0, 0), PadMode::Constant(0.0))
        } else {
            x
        };
        let options = ConvOptions::new(
            [self.stride],
            [if self.causal { 0 } else { self.padding }],
            [self.dilation],
            self.groups,
        );
        conv1d(x, weight, Some(bias), options)
    }

    pub fn kernel_size(&self) -> usize {
        self.weight.dims()[2]
    }
}

pub struct Conv1dLoadArgs {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub causal: bool,
    pub weight: (Vec<f32>, Vec<usize>),
    pub bias: Option<(Vec<f32>, Vec<usize>)>,
}

pub fn load_conv1d_from_tensors<B: Backend>(
    device: &B::Device,
    args: Conv1dLoadArgs,
) -> anyhow::Result<PlainConv1d<B>> {
    use burn::tensor::TensorData;

    let (w_data, w_shape) = args.weight;
    let weight_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(w_data, [w_shape[0], w_shape[1], w_shape[2]]),
        device,
    );

    let bias_tensor = if let Some((b_data, b_shape)) = args.bias {
        Tensor::<B, 1>::from_data(TensorData::new(b_data, [b_shape[0]]), device)
    } else {
        Tensor::zeros([args.out_channels], device)
    };

    Ok(PlainConv1d {
        weight: Param::from_tensor(weight_tensor),
        bias: Param::from_tensor(bias_tensor),
        stride: 1,
        padding: args.kernel_size / 2,
        dilation: 1,
        groups: 1,
        causal: args.causal,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn transposed_conv_uses_burn_channel_layout() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let layer = super::WNConvTranspose1d::<burn::backend::ndarray::NdArray<f32>>::new(
            &device, 128, 2048, 5, 1, 2, 0, 1, 1, false,
        );

        assert_eq!(layer.weight_g.val().dims(), [128, 1, 1]);
        assert_eq!(layer.weight_v.val().dims(), [128, 2048, 5]);
    }
}
