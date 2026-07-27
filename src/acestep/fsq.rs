//! Finite Scalar Quantization (FSQ) for the ACE-Step 1.5 audio tokenizer.
//!
//! Semantics per `dit_spec.md` §3:
//! - levels L = [8, 8, 8, 5, 5, 5], codebook size = 8·8·8·5·5·5 = 64000
//! - little-endian mixed-radix basis = [1, 8, 64, 512, 2560, 12800]
//! - decode: `level_idx_d = (index // basis_d) % levels_d`,
//!   `code_d = level_idx_d * 2/(levels_d - 1) - 1`, then `project_out(code)`.
//!
//! Canonical burnpack tensor names (loaded verbatim by `ResidualFsq::from_burnpack`):
//! - `project_in.{weight,bias}`  — Linear(2048→6, bias=True)
//! - `project_out.{weight,bias}` — Linear(6→2048, bias=True)
//!
//! Note: the checkpoint converter places these under a `quantizer.` prefix in the
//! condition `.bpk` (`quantizer.project_in.{weight,bias}`, `quantizer.project_out.{weight,bias}`);
//! callers loading from that file must strip/select the prefix before calling
//! `from_burnpack` (or pass a burnpack whose keys already match the bare names above).

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, LinearLayout};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

/// FSQ levels per code dimension.
pub const FSQ_LEVELS: [u32; 6] = [8, 8, 8, 5, 5, 5];
/// Little-endian mixed-radix basis: cumprod([1, 8, 8, 8, 5, 5]).
pub const FSQ_BASIS: [u32; 6] = [1, 8, 64, 512, 2560, 12800];
/// Total number of codes: 8·8·8·5·5·5.
pub const FSQ_CODEBOOK_SIZE: u32 = 64000;

const FSQ_HIDDEN_DIM: usize = 2048;
const FSQ_CODE_DIM: usize = 6;

/// Decompose a code index (0..64000) into per-dimension level indices.
///
/// `level_idx_d = (index // basis_d) % levels_d`, values in `0..levels_d`.
pub fn index_to_level_indices(index: u32) -> [u32; 6] {
    let mut level_indices = [0u32; 6];
    for (d, level_index) in level_indices.iter_mut().enumerate() {
        *level_index = (index / FSQ_BASIS[d]) % FSQ_LEVELS[d];
    }
    level_indices
}

/// Map per-dimension level indices to continuous code values in [-1, 1].
///
/// `code_d = level_idx_d * 2/(levels_d - 1) - 1`
/// (L=8: {-1, -5/7, ..., 1}; L=5: {-1, -0.5, 0, 0.5, 1}).
pub fn level_values(level_indices: &[u32; 6]) -> [f32; 6] {
    let mut values = [0.0f32; 6];
    for (d, value) in values.iter_mut().enumerate() {
        let levels = FSQ_LEVELS[d] as f32;
        *value = level_indices[d] as f32 * 2.0 / (levels - 1.0) - 1.0;
    }
    values
}

/// ResidualFSQ wrapper: `project_in` (2048→6) and `project_out` (6→2048), both with bias.
///
/// Only the decode direction (LM codes → continuous 2048-dim hints) is needed for
/// text2music with LM codes; the encode direction is used by cover/extract flows.
#[derive(Module, Debug)]
pub struct ResidualFsq<B: Backend> {
    pub project_in: Linear<B>,
    pub project_out: Linear<B>,
}

impl<B: Backend> ResidualFsq<B> {
    pub fn new(device: &B::Device) -> Self {
        // LinearLayout::Col matches the checkpoint's PyTorch [out, in] weights.
        Self {
            project_in: LinearConfig::new(FSQ_HIDDEN_DIM, FSQ_CODE_DIM)
                .with_layout(LinearLayout::Col)
                .with_bias(true)
                .init(device),
            project_out: LinearConfig::new(FSQ_CODE_DIM, FSQ_HIDDEN_DIM)
                .with_layout(LinearLayout::Col)
                .with_bias(true)
                .init(device),
        }
    }

    /// Load weights from a burnpack file whose keys match the canonical tensor names
    /// documented at the top of this module.
    pub fn from_burnpack(path: &Path, device: &B::Device) -> Result<Self> {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(path).zero_copy(true);
        model
            .load_from(&mut store)
            .with_context(|| format!("failed to load FSQ weights from {}", path.display()))?;
        Ok(model)
    }

    /// Decode LM code indices into continuous 2048-dim hints.
    ///
    /// `indices`: [B, T5] integer codes in 0..64000 → returns [B, T5, 2048]
    /// (`project_out(code)`). The mixed-radix decomposition runs on the host;
    /// this is called once per generation, so clarity beats cleverness.
    pub fn decode_indices(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, t5] = indices.dims();
        let device = indices.device();
        let raw = indices
            .to_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("FSQ indices must be convertible to i64");

        let mut codes = Vec::with_capacity(batch * t5 * FSQ_CODE_DIM);
        for index in raw {
            // The official `_parse_audio_code_string` clamps out-of-range
            // codes into [0, 64000) instead of failing.
            let index = index.clamp(0, FSQ_CODEBOOK_SIZE as i64 - 1);
            let level_indices = index_to_level_indices(index as u32);
            codes.extend_from_slice(&level_values(&level_indices));
        }

        let code_tensor =
            Tensor::<B, 3>::from_data(TensorData::new(codes, [batch, t5, FSQ_CODE_DIM]), &device);
        self.project_out.forward(code_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn index_zero_maps_to_all_zero_levels() {
        assert_eq!(index_to_level_indices(0), [0; 6]);
    }

    #[test]
    fn max_index_maps_to_max_levels() {
        assert_eq!(
            index_to_level_indices(FSQ_CODEBOOK_SIZE - 1),
            [7, 7, 7, 4, 4, 4]
        );
    }

    #[test]
    fn mid_range_indices_decompose_correctly() {
        // 1 -> only dim 0
        assert_eq!(index_to_level_indices(1), [1, 0, 0, 0, 0, 0]);
        // 8 -> dim 1
        assert_eq!(index_to_level_indices(8), [0, 1, 0, 0, 0, 0]);
        // 64 + 8 + 1 -> 1 in dims 0,1,2
        assert_eq!(index_to_level_indices(73), [1, 1, 1, 0, 0, 0]);
        // 12800 -> dim 5
        assert_eq!(index_to_level_indices(12800), [0, 0, 0, 0, 0, 1]);
        // 2*12800 + 3*2560 + 4*512 + 5*64 + 6*8 + 7
        let index = 2 * 12800 + 3 * 2560 + 4 * 512 + 5 * 64 + 6 * 8 + 7;
        assert_eq!(index_to_level_indices(index), [7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn level_values_edges() {
        assert_eq!(level_values(&[0; 6]), [-1.0; 6]);
        assert_eq!(level_values(&[7, 7, 7, 4, 4, 4]), [1.0; 6]);
    }

    #[test]
    fn level_values_midpoints() {
        // level 4 of 8 -> 4*2/7 - 1 = 1/7
        let values = level_values(&[4, 0, 0, 0, 0, 0]);
        assert!((values[0] - 1.0 / 7.0).abs() < 1e-6);
        // level 2 of 5 -> 2*2/4 - 1 = 0
        let values = level_values(&[0, 0, 0, 2, 0, 0]);
        assert_eq!(values[3], 0.0);
        // level 1 of 5 -> -0.5
        let values = level_values(&[0, 0, 0, 0, 1, 0]);
        assert_eq!(values[4], -0.5);
    }

    #[test]
    fn decode_indices_shape_and_finiteness() {
        type B = NdArray<f32>;
        let device = burn::prelude::Device::<B>::default();
        let fsq = ResidualFsq::<B>::new(&device);
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![0i64, 12345, 63999], [1, 3]),
            &device,
        );
        let decoded = fsq.decode_indices(indices);
        assert_eq!(decoded.dims(), [1, 3, 2048]);
        let values: Vec<f32> = decoded.to_data().to_vec::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }
}
