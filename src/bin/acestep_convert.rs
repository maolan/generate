//! Offline converter from official ACE-Step 1.5 checkpoints to burnpack
//! (`.bpk`) files with the canonical tensor names the runtime modules in
//! `src/acestep/` expect.
//!
//! Inputs:
//! - Qwen3-Embedding-0.6B `model.safetensors`      → text encoder
//! - acestep-5Hz-lm `model.safetensors`            → 5Hz LM (tied embeddings)
//! - acestep-v15-turbo `model.safetensors`         → DiT + condition encoder
//! - vae `diffusion_pytorch_model.safetensors`     → Oobleck VAE decoder
//! - `silence_latent.pt` (torch.save)              → silence latent
//!
//! Checkpoints are bf16; runtime tensors are f32. bf16 is the top 16 bits of
//! f32, so the conversion is an exact bit shift. F16 and F32 inputs are also
//! accepted.
//!
//! Writing uses burn-store's lower-level `BurnpackWriter` API directly with
//! `TensorSnapshot::from_data`, so no runtime module is instantiated (the DiT
//! alone is ~4.8 GB of weights). The runtime reads the result with
//! `BurnpackStore::from_file(path).zero_copy(true)` + `load_from`, which
//! matches tensors by their dotted path.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use burn::module::ParamId;
use burn::tensor::TensorData;
use burn_store::{BurnpackWriter, TensorSnapshot};
use safetensors::{Dtype, SafeTensors};

fn main() {
    if let Err(err) = run(std::env::args_os()) {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
}

/// Which model component a checkpoint belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    TextEncoder,
    Lm,
    Dit,
    Condition,
    Vae,
    Silence,
}

impl Component {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "text-encoder" => Ok(Self::TextEncoder),
            "lm" => Ok(Self::Lm),
            "dit" => Ok(Self::Dit),
            "condition" => Ok(Self::Condition),
            "vae" => Ok(Self::Vae),
            "silence" => Ok(Self::Silence),
            _ => bail!(
                "unsupported component '{value}', expected one of: \
                 text-encoder, lm, dit, condition, vae, silence"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::TextEncoder => "text-encoder",
            Self::Lm => "lm",
            Self::Dit => "dit",
            Self::Condition => "condition",
            Self::Vae => "vae",
            Self::Silence => "silence",
        }
    }
}

#[derive(Debug)]
struct Options {
    component: Component,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    list: bool,
    silence_raw: Option<PathBuf>,
    silence_shape: Option<Vec<usize>>,
}

fn help_text() -> &'static str {
    "\
acestep_convert

Usage:
  acestep_convert --component <text-encoder|lm|dit|condition|vae|silence> \\
      --input <path> --output <path.bpk>
  acestep_convert --component <...> --input <path> --list
  acestep_convert --component silence --silence-raw <path> \\
      --silence-shape <1,S,64> --output silence_latent.bpk

Options:
  --component <name>      Which model component the checkpoint belongs to
  --input <path>          Official checkpoint (safetensors, or silence_latent.pt
                          for the silence component)
  --output <path.bpk>     Destination burnpack file
  --list                  Print input tensor names/dtypes/shapes and exit
  --silence-raw <path>    Raw little-endian f32 silence latent, used instead of
                          parsing --input as a torch.save archive
  --silence-shape <dims>  Comma-separated shape for --silence-raw, e.g. 1,4096,64
  -h, --help
"
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Options> {
    let mut args = args.into_iter();
    let _program = args.next();
    let mut component = None;
    let mut input = None;
    let mut output = None;
    let mut list = false;
    let mut silence_raw = None;
    let mut silence_shape = None;

    while let Some(arg) = args.next() {
        let arg = arg
            .into_string()
            .map_err(|_| anyhow!("arguments must be valid UTF-8"))?;

        if matches!(arg.as_str(), "-h" | "--help") {
            bail!(help_text());
        }

        if arg == "--component" {
            let value = args
                .next()
                .ok_or_else(|| anyhow!("missing value after --component"))?
                .into_string()
                .map_err(|_| anyhow!("component value must be valid UTF-8"))?;
            component = Some(Component::parse(&value)?);
            continue;
        }

        if arg == "--input" {
            input = Some(PathBuf::from(
                args.next()
                    .ok_or_else(|| anyhow!("missing value after --input"))?,
            ));
            continue;
        }

        if arg == "--output" {
            output = Some(PathBuf::from(
                args.next()
                    .ok_or_else(|| anyhow!("missing value after --output"))?,
            ));
            continue;
        }

        if arg == "--list" {
            list = true;
            continue;
        }

        if arg == "--silence-raw" {
            silence_raw =
                Some(PathBuf::from(args.next().ok_or_else(|| {
                    anyhow!("missing value after --silence-raw")
                })?));
            continue;
        }

        if arg == "--silence-shape" {
            let value = args
                .next()
                .ok_or_else(|| anyhow!("missing value after --silence-shape"))?
                .into_string()
                .map_err(|_| anyhow!("silence-shape value must be valid UTF-8"))?;
            let shape = value
                .split(',')
                .map(|dim| {
                    dim.trim().parse::<usize>().map_err(|_| {
                        anyhow!("silence-shape dimension '{dim}' is not a whole number")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if shape.is_empty() || shape.contains(&0) {
                bail!("silence-shape must be a non-empty list of positive dimensions");
            }
            silence_shape = Some(shape);
            continue;
        }

        bail!("unknown argument '{arg}'");
    }

    let component = component.ok_or_else(|| anyhow!("missing required --component"))?;
    if component != Component::Silence && (silence_raw.is_some() || silence_shape.is_some()) {
        bail!("--silence-raw/--silence-shape only apply to the silence component");
    }

    Ok(Options {
        component,
        input,
        output,
        list,
        silence_raw,
        silence_shape,
    })
}

fn run(args: impl IntoIterator<Item = OsString>) -> Result<()> {
    let options = parse_args(args)?;
    if options.component == Component::Silence {
        return run_silence(&options);
    }

    let input = options
        .input
        .as_deref()
        .ok_or_else(|| anyhow!("missing required --input"))?;
    if options.list {
        return list_safetensors(input);
    }
    let output = options
        .output
        .as_deref()
        .ok_or_else(|| anyhow!("missing required --output"))?;

    let (tensors, skipped) = convert_safetensors(options.component, input)?;
    report(options.component, &tensors, &skipped, output);
    write_burnpack(output, tensors)
}

fn run_silence(options: &Options) -> Result<()> {
    let (shape, data) = if let Some(raw) = &options.silence_raw {
        let shape = options
            .silence_shape
            .clone()
            .ok_or_else(|| anyhow!("--silence-shape is required with --silence-raw"))?;
        let bytes = std::fs::read(raw)
            .with_context(|| format!("failed to read raw silence latent from {}", raw.display()))?;
        let values = f32_bytes_to_vec(&bytes)
            .with_context(|| format!("invalid raw f32 data in {}", raw.display()))?;
        let numel: usize = shape.iter().product();
        if values.len() != numel {
            bail!(
                "raw silence latent has {} elements but shape {shape:?} implies {numel}",
                values.len()
            );
        }
        (shape, values)
    } else {
        let input = options
            .input
            .as_deref()
            .ok_or_else(|| anyhow!("missing required --input (or use --silence-raw)"))?;
        let (spec, data) = read_torch_save(input)?;
        (spec.shape, data)
    };

    // The official silence_latent.pt stores the latent channel-major as
    // [1, 64, S]; the runtime expects time-major [1, S, 64] like every other
    // latent in the pipeline. Transpose on the host (small, one-off).
    let (shape, data) = if shape.len() == 3 && shape[0] == 1 && shape[1] == 64 {
        let frames = shape[2];
        let mut transposed = vec![0.0_f32; data.len()];
        for frame in 0..frames {
            for channel in 0..64 {
                transposed[frame * 64 + channel] = data[channel * frames + frame];
            }
        }
        (vec![1, frames, 64], transposed)
    } else {
        (shape, data)
    };

    if options.list {
        println!("silence_latent\tF32\t{shape:?}");
        return Ok(());
    }

    let output = options
        .output
        .as_deref()
        .ok_or_else(|| anyhow!("missing required --output"))?;
    let tensor = ConvertedTensor {
        name: "silence_latent".to_string(),
        shape,
        data,
    };
    println!(
        "silence: wrote 1 tensor ({} elements) to {}",
        tensor.data.len(),
        output.display()
    );
    write_burnpack(output, vec![tensor])
}

fn report(component: Component, tensors: &[ConvertedTensor], skipped: &[String], output: &Path) {
    let elements: usize = tensors.iter().map(|tensor| tensor.data.len()).sum();
    println!(
        "{}: converted {} tensors ({} elements) to {}",
        component.name(),
        tensors.len(),
        elements,
        output.display()
    );
    if !skipped.is_empty() {
        println!("  skipped {} input tensors:", skipped.len());
        for name in skipped.iter().take(8) {
            println!("    {name}");
        }
        if skipped.len() > 8 {
            println!("    ... and {} more", skipped.len() - 8);
        }
    }
}

fn list_safetensors(input: &Path) -> Result<()> {
    for file in safetensors_files(input)? {
        let bytes = std::fs::read(&file)
            .with_context(|| format!("failed to read checkpoint from {}", file.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("failed to parse safetensors from {}", file.display()))?;
        for (name, view) in tensors.tensors() {
            println!("{name}\t{:?}\t{:?}", view.dtype(), view.shape());
        }
    }
    Ok(())
}

/// Collect the safetensors files to convert: `input` itself when it is a
/// file, or every `*.safetensors` file (sorted by name) when it is a
/// directory (sharded checkpoints like `model-0000N-of-0000M.safetensors`).
fn safetensors_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if input.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(input)
            .with_context(|| format!("failed to list checkpoint dir {}", input.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files found in {}", input.display());
        }
        return Ok(files);
    }
    bail!(
        "checkpoint {} is neither a file nor a directory",
        input.display()
    )
}

// ---------------------------------------------------------------------------
// Name mapping (official checkpoint names → canonical burnpack names)
// ---------------------------------------------------------------------------

/// Qwen3 text encoder / 5Hz LM. Keys may carry a `model.` prefix in the
/// checkpoint; `lm_head.weight` is skipped because embeddings are tied.
fn map_qwen3(name: &str) -> Option<String> {
    let name = name.strip_prefix("model.").unwrap_or(name);
    if name == "lm_head.weight" {
        None
    } else {
        Some(name.to_string())
    }
}

/// ACE-Step DiT. Only the `decoder.*` tree of the turbo checkpoint is kept.
/// `proj_in`/`proj_out` are `nn.Sequential(conv, ...)` upstream where the conv
/// is index 1, hence `proj_in.1.*` → `proj_in.conv.*`.
fn map_dit(name: &str) -> Option<String> {
    let name = name.strip_prefix("decoder.")?;
    if let Some(rest) = name.strip_prefix("proj_in.1.") {
        return Some(format!("proj_in.conv.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("proj_out.1.") {
        return Some(format!("proj_out.conv.{rest}"));
    }
    Some(name.to_string())
}

/// Condition encoder stack from the same turbo checkpoint. Everything outside
/// the listed subtrees (including `decoder.*`, `null_condition_emb`,
/// `tokenizer.audio_acoustic_proj.*`, `tokenizer.attention_pooler.*` and
/// `encoder.timbre_encoder.special_token`) is dropped.
fn map_condition(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("encoder.text_projector.") {
        return Some(format!("text_projector.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("encoder.lyric_encoder.") {
        return Some(format!("lyric_encoder.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("encoder.timbre_encoder.") {
        if rest == "special_token" {
            return None;
        }
        return Some(format!("timbre_encoder.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("detokenizer.") {
        return Some(format!("detokenizer.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("tokenizer.quantizer.project_in.") {
        return Some(format!("quantizer.project_in.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("tokenizer.quantizer.project_out.") {
        return Some(format!("quantizer.project_out.{rest}"));
    }
    None
}

/// Oobleck VAE decoder: strip the leading `decoder.` prefix and keep the rest
/// (`block.{i}.*`, `conv1.*`, `snake1.*`, `conv2.*`); skip the encoder and
/// quantization convs.
fn map_vae(name: &str) -> Option<String> {
    name.strip_prefix("decoder.").map(str::to_string)
}

fn name_mapper(component: Component) -> fn(&str) -> Option<String> {
    match component {
        Component::TextEncoder | Component::Lm => map_qwen3,
        Component::Dit => map_dit,
        Component::Condition => map_condition,
        Component::Vae => map_vae,
        Component::Silence => unreachable!("silence component has no safetensors input"),
    }
}

// ---------------------------------------------------------------------------
// safetensors → f32 conversion
// ---------------------------------------------------------------------------

/// A converted tensor ready to be written into a burnpack file.
#[derive(Debug)]
struct ConvertedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Read a safetensors checkpoint, rename tensors to their canonical burnpack
/// names and decode them to f32. Returns the converted tensors (sorted by
/// name) and the names that were skipped by the mapping.
fn convert_safetensors(
    component: Component,
    input: &Path,
) -> Result<(Vec<ConvertedTensor>, Vec<String>)> {
    let files = safetensors_files(input)?;
    let mut buffers = Vec::with_capacity(files.len());
    for file in &files {
        buffers.push(
            std::fs::read(file)
                .with_context(|| format!("failed to read checkpoint from {}", file.display()))?,
        );
    }
    let map = name_mapper(component);

    let mut converted = Vec::new();
    let mut skipped = Vec::new();
    for (file, bytes) in files.iter().zip(buffers.iter()) {
        let tensors = SafeTensors::deserialize(bytes)
            .with_context(|| format!("failed to parse safetensors from {}", file.display()))?;
        for (name, view) in tensors.tensors() {
            let Some(canonical) = map(&name) else {
                skipped.push(name);
                continue;
            };
            let data = decode_safetensor(view.dtype(), view.data())
                .with_context(|| format!("failed to decode tensor '{name}'"))?;
            let numel: usize = view.shape().iter().product();
            if numel != data.len() {
                bail!(
                    "tensor '{name}': shape {:?} implies {numel} elements but {} were decoded",
                    view.shape(),
                    data.len()
                );
            }
            converted.push(ConvertedTensor {
                name: canonical,
                shape: view.shape().to_vec(),
                data,
            });
        }
    }

    converted.sort_by(|a, b| a.name.cmp(&b.name));
    skipped.sort();
    for pair in converted.windows(2) {
        if pair[0].name == pair[1].name {
            bail!(
                "name collision: multiple input tensors map to '{}'",
                pair[0].name
            );
        }
    }
    Ok((converted, skipped))
}

fn decode_safetensor(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::BF16 => bf16_bytes_to_f32(data),
        Dtype::F16 => f16_bytes_to_f32(data),
        Dtype::F32 => f32_bytes_to_vec(data),
        other => bail!("unsupported safetensors dtype {other:?} (expected BF16, F16 or F32)"),
    }
}

/// Exact bf16 → f32 conversion: bf16 is the top 16 bits of f32.
fn bf16_bytes_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        bail!("odd byte count for bf16 tensor data");
    }
    Ok(data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16))
        .collect())
}

fn f16_bytes_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        bail!("odd byte count for f16 tensor data");
    }
    Ok(data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect())
}

fn f32_bytes_to_vec(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        bail!("byte count is not a multiple of 4 for f32 tensor data");
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

// ---------------------------------------------------------------------------
// burnpack writing
// ---------------------------------------------------------------------------

/// Write tensors to a `.bpk` file using burn-store's low-level
/// `BurnpackWriter`. Tensor paths are matched by name on load, so each
/// snapshot gets a fresh `ParamId` (the reader regenerates ids anyway).
fn write_burnpack(output: &Path, tensors: Vec<ConvertedTensor>) -> Result<()> {
    let mut snapshots = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let segments: Vec<String> = tensor.name.split('.').map(str::to_string).collect();
        let containers = segments[..segments.len().saturating_sub(1)].to_vec();
        let data = TensorData::new(tensor.data, tensor.shape);
        snapshots.push(TensorSnapshot::from_data(
            data,
            segments,
            containers,
            ParamId::new(),
        ));
    }
    BurnpackWriter::new(snapshots)
        .with_metadata("producer", "acestep_convert")
        .write_to_file(output)
        .map_err(|err| anyhow!("failed to write burnpack {}: {err}", output.display()))
}

// ---------------------------------------------------------------------------
// torch.save (.pt) reading: zip archive with STORED entries + pickle
// ---------------------------------------------------------------------------

/// Location and dtype of the single tensor inside a torch.save archive, as
/// recovered from the pickle stream.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TorchTensorSpec {
    storage_type: String,
    storage_key: String,
    storage_offset: usize,
    shape: Vec<usize>,
}

/// Parse a torch.save archive containing a single tensor. Returns the tensor
/// spec and its data converted to f32.
fn read_torch_save(path: &Path) -> Result<(TorchTensorSpec, Vec<f32>)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read torch archive from {}", path.display()))?;
    let entries = zip_entries(&bytes)
        .with_context(|| format!("{} is not a readable zip archive", path.display()))?;

    let pickle_entry = entries
        .iter()
        .find(|entry| entry.name.ends_with("data.pkl"))
        .ok_or_else(|| {
            anyhow!(
                "{} has no data.pkl entry (not a torch.save file?)",
                path.display()
            )
        })?;
    let pickle_bytes = zip_entry_data(&bytes, pickle_entry)?;
    let value = PickleReader::new(pickle_bytes)
        .run()
        .with_context(|| format!("failed to parse pickle stream in {}", path.display()))?;
    let Pickle::Tensor(spec) = value else {
        bail!(
            "unsupported pickle payload in {} (expected a single tensor saved with torch.save); \
             re-export the raw data and use --silence-raw instead",
            path.display()
        )
    };
    if spec.storage_offset != 0 {
        bail!(
            "tensor in {} has a nonzero storage offset ({}), which is not supported; \
             use --silence-raw instead",
            path.display(),
            spec.storage_offset
        );
    }

    let data_suffix = format!("/data/{}", spec.storage_key);
    let data_entry = entries
        .iter()
        .find(|entry| entry.name.ends_with(&data_suffix))
        .ok_or_else(|| {
            anyhow!(
                "storage '{}' not found in {} (expected entry '*{data_suffix}')",
                spec.storage_key,
                path.display()
            )
        })?;
    let storage = zip_entry_data(&bytes, data_entry)?;
    let data = decode_torch_storage(&spec.storage_type, storage).with_context(|| {
        format!(
            "failed to decode storage '{}' in {}",
            spec.storage_key,
            path.display()
        )
    })?;
    let numel: usize = spec.shape.iter().product();
    if data.len() != numel {
        bail!(
            "tensor in {} has {} elements but shape {:?} implies {numel}",
            path.display(),
            data.len(),
            spec.shape
        );
    }
    Ok((spec, data))
}

fn decode_torch_storage(storage_type: &str, data: &[u8]) -> Result<Vec<f32>> {
    match storage_type {
        "FloatStorage" => f32_bytes_to_vec(data),
        "DoubleStorage" => {
            if !data.len().is_multiple_of(8) {
                bail!("byte count is not a multiple of 8 for f64 storage");
            }
            Ok(data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect())
        }
        "HalfStorage" => f16_bytes_to_f32(data),
        "BFloat16Storage" => bf16_bytes_to_f32(data),
        other => bail!(
            "unsupported torch storage type '{other}' (expected Float/Double/Half/BFloat16); \
             use --silence-raw instead"
        ),
    }
}

// --- minimal zip reader (STORED entries only, no ZIP64) ---

#[derive(Debug)]
struct ZipEntry {
    name: String,
    method: u16,
    compressed_size: u32,
    local_offset: u32,
}

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| anyhow!("unexpected end of zip data at offset {offset}"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("unexpected end of zip data at offset {offset}"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntry>> {
    const EOCD_SIG: u32 = 0x0605_4b50;
    const EOCD_LEN: usize = 22;
    const MAX_COMMENT: usize = 65_535;

    let search_start = bytes.len().saturating_sub(EOCD_LEN + MAX_COMMENT);
    let search_end = bytes.len().saturating_sub(EOCD_LEN);
    let sig = EOCD_SIG.to_le_bytes();
    let eocd = (search_start..=search_end)
        .rev()
        .find(|&pos| bytes.get(pos..pos + 4) == Some(sig.as_slice()))
        .ok_or_else(|| anyhow!("zip end of central directory not found"))?;

    let total = read_u16_le(bytes, eocd + 10)?;
    let cd_offset = read_u32_le(bytes, eocd + 16)?;
    if total == u16::MAX || cd_offset == u32::MAX {
        bail!("ZIP64 archives are not supported; use --silence-raw instead");
    }

    let mut entries = Vec::with_capacity(usize::from(total));
    let mut pos = cd_offset as usize;
    for _ in 0..total {
        if read_u32_le(bytes, pos)? != 0x0201_4b50 {
            bail!("corrupt zip central directory at offset {pos}");
        }
        let method = read_u16_le(bytes, pos + 10)?;
        let compressed_size = read_u32_le(bytes, pos + 20)?;
        let name_len = usize::from(read_u16_le(bytes, pos + 28)?);
        let extra_len = usize::from(read_u16_le(bytes, pos + 30)?);
        let comment_len = usize::from(read_u16_le(bytes, pos + 32)?);
        let local_offset = read_u32_le(bytes, pos + 42)?;
        if compressed_size == u32::MAX || local_offset == u32::MAX {
            bail!("ZIP64 archives are not supported; use --silence-raw instead");
        }
        let name_bytes = bytes
            .get(pos + 46..pos + 46 + name_len)
            .ok_or_else(|| anyhow!("unexpected end of zip central directory"))?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        entries.push(ZipEntry {
            name,
            method,
            compressed_size,
            local_offset,
        });
        pos += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

fn zip_entry_data<'a>(bytes: &'a [u8], entry: &ZipEntry) -> Result<&'a [u8]> {
    if entry.method != 0 {
        bail!(
            "zip entry '{}' is compressed (method {}); only uncompressed torch.save archives \
             are supported — re-export the raw data and use --silence-raw instead",
            entry.name,
            entry.method
        );
    }
    let pos = entry.local_offset as usize;
    if read_u32_le(bytes, pos)? != 0x0403_4b50 {
        bail!("corrupt zip local header for entry '{}'", entry.name);
    }
    let name_len = usize::from(read_u16_le(bytes, pos + 26)?);
    let extra_len = usize::from(read_u16_le(bytes, pos + 28)?);
    let start = pos + 30 + name_len + extra_len;
    let end = start + entry.compressed_size as usize;
    bytes
        .get(start..end)
        .ok_or_else(|| anyhow!("unexpected end of zip data for entry '{}'", entry.name))
}

// --- minimal pickle protocol-2 reader for a single rebuilt tensor ---

/// Simplified pickle value model covering the opcodes torch.save emits for a
/// bare tensor (protocol 2, persistent storage ids).
#[derive(Debug, Clone, PartialEq)]
enum Pickle {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Pickle>),
    List(Vec<Pickle>),
    Global(String, String),
    PersistentId(Box<Pickle>),
    /// Result of `REDUCE` on an unsupported callable (e.g. `OrderedDict()`).
    Object(String, Vec<Pickle>),
    /// Result of `REDUCE` on `torch._utils._rebuild_tensor_v2`.
    Tensor(TorchTensorSpec),
}

struct PickleReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    stack: Vec<Pickle>,
    marks: Vec<usize>,
    memo: BTreeMap<u32, Pickle>,
}

impl<'a> PickleReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: BTreeMap::new(),
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let byte = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| anyhow!("unexpected end of pickle stream"))?;
        self.pos += 1;
        Ok(byte)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + len)
            .ok_or_else(|| anyhow!("unexpected end of pickle stream"))?;
        self.pos += len;
        Ok(slice)
    }

    fn line(&mut self) -> Result<String> {
        let rest = &self.bytes[self.pos.min(self.bytes.len())..];
        let end = rest
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or_else(|| anyhow!("unterminated pickle line"))?;
        let line = String::from_utf8_lossy(&rest[..end]).into_owned();
        self.pos += end + 1;
        Ok(line)
    }

    fn pop(&mut self) -> Result<Pickle> {
        self.stack
            .pop()
            .ok_or_else(|| anyhow!("pickle stack underflow"))
    }

    fn memoize(&mut self, index: u32) -> Result<()> {
        let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("pickle stack underflow"))?;
        self.memo.insert(index, top);
        Ok(())
    }

    fn drain_mark(&mut self) -> Result<Vec<Pickle>> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| anyhow!("pickle mark stack underflow"))?;
        Ok(self.stack.split_off(mark))
    }

    fn run(mut self) -> Result<Pickle> {
        loop {
            match self.byte()? {
                0x80 => {
                    self.byte()?;
                } // PROTO
                0x95 => {
                    self.take(8)?;
                } // FRAME
                b'c' => {
                    // GLOBAL
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Pickle::Global(module, name));
                }
                b'N' => self.stack.push(Pickle::None),
                0x88 => self.stack.push(Pickle::Bool(true)), // NEWTRUE
                0x89 => self.stack.push(Pickle::Bool(false)), // NEWFALSE
                b'K' => {
                    let value = self.byte()?;
                    self.stack.push(Pickle::Int(i64::from(value)));
                }
                b'M' => {
                    let bytes = self.take(2)?;
                    self.stack.push(Pickle::Int(i64::from(u16::from_le_bytes([
                        bytes[0], bytes[1],
                    ]))));
                }
                b'J' => {
                    let bytes = self.take(4)?;
                    self.stack.push(Pickle::Int(i64::from(i32::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                    ]))));
                }
                b'L' => {
                    // LONG: decimal text, possibly with a trailing 'L'
                    let line = self.line()?;
                    let text = line.strip_suffix('L').unwrap_or(&line);
                    let value = text
                        .parse::<i64>()
                        .map_err(|_| anyhow!("invalid pickle LONG '{line}'"))?;
                    self.stack.push(Pickle::Int(value));
                }
                0x8a => {
                    // LONG1
                    let len = usize::from(self.byte()?);
                    let value = self.long_from_bytes(len)?;
                    self.stack.push(Pickle::Int(value));
                }
                0x8b => {
                    // LONG4
                    let bytes = self.take(4)?;
                    let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                    let value = self.long_from_bytes(len)?;
                    self.stack.push(Pickle::Int(value));
                }
                b'U' => {
                    // SHORT_BINSTRING
                    let len = usize::from(self.byte()?);
                    let bytes = self.take(len)?;
                    self.stack
                        .push(Pickle::Str(String::from_utf8_lossy(bytes).into_owned()));
                }
                b'T' => {
                    // BINSTRING
                    let bytes = self.take(4)?;
                    let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                    let bytes = self.take(len)?;
                    self.stack
                        .push(Pickle::Str(String::from_utf8_lossy(bytes).into_owned()));
                }
                b'X' => {
                    // BINUNICODE
                    let bytes = self.take(4)?;
                    let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                    let bytes = self.take(len)?;
                    let text = String::from_utf8(bytes.to_vec())
                        .map_err(|_| anyhow!("invalid UTF-8 in pickle BINUNICODE"))?;
                    self.stack.push(Pickle::Str(text));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let len = usize::from(self.byte()?);
                    let bytes = self.take(len)?;
                    let text = String::from_utf8(bytes.to_vec())
                        .map_err(|_| anyhow!("invalid UTF-8 in pickle SHORT_BINUNICODE"))?;
                    self.stack.push(Pickle::Str(text));
                }
                b'B' => {
                    // BINBYTES
                    let bytes = self.take(4)?;
                    let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                    let bytes = self.take(len)?;
                    self.stack.push(Pickle::Bytes(bytes.to_vec()));
                }
                b'C' => {
                    // SHORT_BINBYTES
                    let len = usize::from(self.byte()?);
                    let bytes = self.take(len)?;
                    self.stack.push(Pickle::Bytes(bytes.to_vec()));
                }
                b'(' => self.marks.push(self.stack.len()), // MARK
                b')' => self.stack.push(Pickle::Tuple(Vec::new())), // EMPTY_TUPLE
                b']' => self.stack.push(Pickle::List(Vec::new())), // EMPTY_LIST
                b't' => {
                    // TUPLE
                    let items = self.drain_mark()?;
                    self.stack.push(Pickle::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Pickle::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Pickle::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Pickle::Tuple(vec![a, b, c]));
                }
                b'q' => {
                    let index = u32::from(self.byte()?);
                    self.memoize(index)?;
                }
                b'r' => {
                    let bytes = self.take(4)?;
                    let index = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    self.memoize(index)?;
                }
                0x94 => {
                    // MEMOIZE
                    let index = self.memo.keys().max().map_or(0, |max| max + 1);
                    self.memoize(index)?;
                }
                b'h' => {
                    let index = u32::from(self.byte()?);
                    let value = self
                        .memo
                        .get(&index)
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle memo index {index} not set"))?;
                    self.stack.push(value);
                }
                b'j' => {
                    let bytes = self.take(4)?;
                    let index = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    let value = self
                        .memo
                        .get(&index)
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle memo index {index} not set"))?;
                    self.stack.push(value);
                }
                b'Q' => {
                    // BINPERSID
                    let id = self.pop()?;
                    self.stack.push(Pickle::PersistentId(Box::new(id)));
                }
                b'R' => self.reduce()?,
                b'.' => {
                    // STOP
                    return self
                        .stack
                        .pop()
                        .ok_or_else(|| anyhow!("empty pickle stack at STOP"));
                }
                opcode => {
                    bail!(
                        "unsupported pickle opcode 0x{opcode:02x} at byte {}; \
                         use --silence-raw instead",
                        self.pos - 1
                    )
                }
            }
        }
    }

    fn long_from_bytes(&mut self, len: usize) -> Result<i64> {
        if len > 8 {
            bail!("pickle LONG with {len} bytes does not fit in i64");
        }
        let bytes = self.take(len)?;
        let mut value = 0_i64;
        for (shift, byte) in bytes.iter().enumerate() {
            value |= i64::from(*byte) << (shift * 8);
        }
        // Two's complement sign extension.
        if len > 0 && bytes[len - 1] & 0x80 != 0 && len < 8 {
            value |= -1_i64 << (len * 8);
        }
        Ok(value)
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let callable = self.pop()?;
        let Pickle::Global(module, name) = callable else {
            bail!("pickle REDUCE on a non-global callable is not supported");
        };
        let args = match args {
            Pickle::Tuple(items) => items,
            other => vec![other],
        };
        if module == "torch._utils" && name == "_rebuild_tensor_v2" {
            self.stack.push(rebuild_tensor_v2(&args)?);
        } else {
            self.stack
                .push(Pickle::Object(format!("{module}.{name}"), args));
        }
        Ok(())
    }
}

/// Interpret the argument tuple of `torch._utils._rebuild_tensor_v2`:
/// `(storage_persistent_id, storage_offset, size, stride, requires_grad,
/// backward_hooks)`.
fn rebuild_tensor_v2(args: &[Pickle]) -> Result<Pickle> {
    let Some(Pickle::PersistentId(storage_id)) = args.first() else {
        bail!("malformed _rebuild_tensor_v2 call (missing storage persistent id)");
    };
    let Pickle::Tuple(storage) = storage_id.as_ref() else {
        bail!("malformed _rebuild_tensor_v2 call (storage id is not a tuple)");
    };
    if storage.len() < 3 || storage.first() != Some(&Pickle::Str("storage".to_string())) {
        bail!("malformed _rebuild_tensor_v2 call (unexpected storage tuple)");
    }
    let storage_type = match &storage[1] {
        Pickle::Global(_, name) => name.clone(),
        other => bail!("unexpected storage class in pickle: {other:?}"),
    };
    let storage_key = match &storage[2] {
        Pickle::Str(key) => key.clone(),
        other => bail!("unexpected storage key in pickle: {other:?}"),
    };
    let storage_offset = match args.get(1) {
        Some(Pickle::Int(offset)) if *offset >= 0 => *offset as usize,
        other => bail!("unexpected storage offset in pickle: {other:?}"),
    };
    let shape = match args.get(2) {
        Some(Pickle::Tuple(dims)) => dims
            .iter()
            .map(|dim| match dim {
                Pickle::Int(value) if *value >= 0 => Ok(*value as usize),
                other => bail!("unexpected tensor dimension in pickle: {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?,
        other => bail!("unexpected tensor size in pickle: {other:?}"),
    };
    Ok(Pickle::Tensor(TorchTensorSpec {
        storage_type,
        storage_key,
        storage_offset,
        shape,
    }))
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn_store::{BurnpackStore, ModuleStore};

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("acestep_convert_test_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write a minimal safetensors file: 8-byte LE header length, JSON header,
    /// then raw tensor data back to back.
    fn write_test_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
        let mut entries = Vec::new();
        let mut offset = 0_usize;
        for (name, dtype, shape, data) in tensors {
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\
                 \"data_offsets\":[{offset},{}]}}",
                offset + data.len()
            ));
            offset += data.len();
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        for (_, _, _, data) in tensors {
            file.extend_from_slice(data);
        }
        std::fs::write(path, file).unwrap();
    }

    /// Build a zip archive with STORED (uncompressed) entries.
    fn build_stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let offset = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50_u32.to_le_bytes());
            out.extend_from_slice(&20_u16.to_le_bytes()); // version needed
            out.extend_from_slice(&0_u16.to_le_bytes()); // flags
            out.extend_from_slice(&0_u16.to_le_bytes()); // method: stored
            out.extend_from_slice(&0_u16.to_le_bytes()); // mod time
            out.extend_from_slice(&0_u16.to_le_bytes()); // mod date
            out.extend_from_slice(&0_u32.to_le_bytes()); // crc32 (unchecked)
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0_u16.to_le_bytes()); // extra len
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);

            central.extend_from_slice(&0x0201_4b50_u32.to_le_bytes());
            central.extend_from_slice(&20_u16.to_le_bytes()); // version made by
            central.extend_from_slice(&20_u16.to_le_bytes()); // version needed
            central.extend_from_slice(&0_u16.to_le_bytes()); // flags
            central.extend_from_slice(&0_u16.to_le_bytes()); // method
            central.extend_from_slice(&0_u16.to_le_bytes());
            central.extend_from_slice(&0_u16.to_le_bytes());
            central.extend_from_slice(&0_u32.to_le_bytes()); // crc32
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&0_u16.to_le_bytes()); // extra len
            central.extend_from_slice(&0_u16.to_le_bytes()); // comment len
            central.extend_from_slice(&0_u16.to_le_bytes()); // disk number
            central.extend_from_slice(&0_u16.to_le_bytes()); // internal attrs
            central.extend_from_slice(&0_u32.to_le_bytes()); // external attrs
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_offset = out.len() as u32;
        let cd_size = central.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50_u32.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0_u16.to_le_bytes()); // cd disk
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // comment len
        out
    }

    /// Pickle protocol-2 stream for `torch.save(tensor)` of a FloatStorage
    /// tensor with storage key "0", shape (1, 2, 3), 6 elements.
    fn torch_tensor_pickle() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"\x80\x02"); // PROTO 2
        out.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n"); // GLOBAL
        out.push(b'('); // MARK (args)
        out.push(b'('); // MARK (storage tuple)
        out.extend_from_slice(b"X\x07\x00\x00\x00storage"); // BINUNICODE
        out.extend_from_slice(b"ctorch\nFloatStorage\n"); // GLOBAL
        out.extend_from_slice(b"X\x01\x00\x00\x000"); // storage key
        out.extend_from_slice(b"X\x03\x00\x00\x00cpu"); // location
        out.extend_from_slice(b"K\x06"); // numel = 6
        out.push(b't'); // TUPLE
        out.push(b'Q'); // BINPERSID
        out.extend_from_slice(b"K\x00"); // storage_offset = 0
        out.extend_from_slice(b"(K\x01K\x02K\x03t"); // size (1, 2, 3)
        out.extend_from_slice(b"(K\x06K\x03K\x01t"); // stride (6, 3, 1)
        out.push(0x89); // NEWFALSE (requires_grad)
        out.extend_from_slice(b"ccollections\nOrderedDict\n"); // GLOBAL
        out.push(b')'); // EMPTY_TUPLE
        out.push(b'R'); // REDUCE -> OrderedDict()
        out.push(b't'); // TUPLE (args)
        out.push(b'R'); // REDUCE -> rebuilt tensor
        out.push(b'.'); // STOP
        out
    }

    /// Load a burnpack back the way the runtime does and return
    /// name → (shape, values).
    fn read_burnpack(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let mut store = BurnpackStore::from_file(path).zero_copy(true);
        store
            .get_all_snapshots()
            .unwrap()
            .iter()
            .map(|(name, snapshot)| {
                let data = snapshot.to_data().unwrap();
                let shape = data.shape.to_vec();
                let values = data.to_vec::<f32>().unwrap();
                (name.clone(), (shape, values))
            })
            .collect()
    }

    #[test]
    fn bf16_conversion_is_exact() {
        // 1.0 = 0x3F800000, -2.5 = 0xC0200000, NaN-ish payload preserved.
        let bytes = [0x80, 0x3F, 0x20, 0xC0, 0x01, 0x7F];
        let values = bf16_bytes_to_f32(&bytes).unwrap();
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -2.5);
        assert_eq!(values[2].to_bits(), 0x7F01_0000);
    }

    #[test]
    fn f16_and_f32_conversion() {
        let f16_bytes = [
            half::f16::from_f32(1.5).to_le_bytes(),
            half::f16::from_f32(-0.25).to_le_bytes(),
        ]
        .concat();
        assert_eq!(f16_bytes_to_f32(&f16_bytes).unwrap(), vec![1.5, -0.25]);

        let f32_bytes = [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
        assert_eq!(f32_bytes_to_vec(&f32_bytes).unwrap(), vec![1.25, -3.5]);
    }

    #[test]
    fn qwen3_mapping_strips_model_prefix_and_skips_lm_head() {
        assert_eq!(
            map_qwen3("model.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            map_qwen3("embed_tokens.weight").as_deref(),
            Some("embed_tokens.weight")
        );
        assert_eq!(
            map_qwen3("model.norm.weight").as_deref(),
            Some("norm.weight")
        );
        assert_eq!(map_qwen3("lm_head.weight"), None);
        assert_eq!(map_qwen3("model.lm_head.weight"), None);
    }

    #[test]
    fn dit_mapping_renames_decoder_tree() {
        assert_eq!(
            map_dit("decoder.proj_in.1.weight").as_deref(),
            Some("proj_in.conv.weight")
        );
        assert_eq!(
            map_dit("decoder.proj_out.1.bias").as_deref(),
            Some("proj_out.conv.bias")
        );
        assert_eq!(
            map_dit("decoder.time_embed.linear_1.weight").as_deref(),
            Some("time_embed.linear_1.weight")
        );
        assert_eq!(
            map_dit("decoder.time_embed_r.time_proj.bias").as_deref(),
            Some("time_embed_r.time_proj.bias")
        );
        assert_eq!(
            map_dit("decoder.condition_embedder.weight").as_deref(),
            Some("condition_embedder.weight")
        );
        assert_eq!(
            map_dit("decoder.layers.23.scale_shift_table").as_deref(),
            Some("layers.23.scale_shift_table")
        );
        assert_eq!(
            map_dit("decoder.norm_out.weight").as_deref(),
            Some("norm_out.weight")
        );
        assert_eq!(
            map_dit("decoder.scale_shift_table").as_deref(),
            Some("scale_shift_table")
        );
        assert_eq!(map_dit("encoder.text_projector.weight"), None);
        assert_eq!(map_dit("null_condition_emb"), None);
    }

    #[test]
    fn condition_mapping_drops_dead_tensors() {
        assert_eq!(
            map_condition("encoder.text_projector.weight").as_deref(),
            Some("text_projector.weight")
        );
        assert_eq!(
            map_condition("encoder.lyric_encoder.embed_tokens.bias").as_deref(),
            Some("lyric_encoder.embed_tokens.bias")
        );
        assert_eq!(
            map_condition("encoder.timbre_encoder.layers.3.mlp.gate_proj.weight").as_deref(),
            Some("timbre_encoder.layers.3.mlp.gate_proj.weight")
        );
        assert_eq!(map_condition("encoder.timbre_encoder.special_token"), None);
        assert_eq!(
            map_condition("detokenizer.special_tokens").as_deref(),
            Some("detokenizer.special_tokens")
        );
        assert_eq!(
            map_condition("tokenizer.quantizer.project_in.weight").as_deref(),
            Some("quantizer.project_in.weight")
        );
        assert_eq!(
            map_condition("tokenizer.quantizer.project_out.bias").as_deref(),
            Some("quantizer.project_out.bias")
        );
        assert_eq!(map_condition("tokenizer.audio_acoustic_proj.weight"), None);
        assert_eq!(map_condition("tokenizer.attention_pooler.weight"), None);
        assert_eq!(map_condition("null_condition_emb"), None);
        assert_eq!(
            map_condition("decoder.layers.0.self_attn.q_proj.weight"),
            None
        );
    }

    #[test]
    fn vae_mapping_strips_decoder_prefix() {
        assert_eq!(
            map_vae("decoder.block.0.conv_t1.weight_v").as_deref(),
            Some("block.0.conv_t1.weight_v")
        );
        assert_eq!(
            map_vae("decoder.conv1.weight_g").as_deref(),
            Some("conv1.weight_g")
        );
        assert_eq!(
            map_vae("decoder.snake1.alpha").as_deref(),
            Some("snake1.alpha")
        );
        assert_eq!(
            map_vae("decoder.conv2.weight_v").as_deref(),
            Some("conv2.weight_v")
        );
        assert_eq!(
            map_vae("decoder.block.4.res_unit3.conv2.bias").as_deref(),
            Some("block.4.res_unit3.conv2.bias")
        );
        assert_eq!(map_vae("encoder.conv1.weight_v"), None);
        assert_eq!(map_vae("quant_conv.weight"), None);
        assert_eq!(map_vae("post_quant_conv.weight"), None);
    }

    #[test]
    fn dit_safetensors_roundtrip() {
        let dir = temp_dir("dit_roundtrip");
        let input = dir.join("model.safetensors");
        let output = dir.join("acestep-dit.bpk");

        let bf16_values = [1.0_f32, -2.5, 0.5, 2.0, 3.25, -1.0];
        let bf16_bytes: Vec<u8> = bf16_values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f32_values = [0.125_f32, -0.5, 4.0, 8.0];
        let f32_bytes: Vec<u8> = f32_values
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes)
            .collect();
        let f16_values = [1.5_f32, -0.25];
        let f16_bytes: Vec<u8> = f16_values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
            .collect();

        write_test_safetensors(
            &input,
            &[
                ("decoder.proj_in.1.weight", "BF16", &[2, 3], &bf16_bytes),
                (
                    "decoder.layers.0.self_attn.q_proj.weight",
                    "F32",
                    &[2, 2],
                    &f32_bytes,
                ),
                ("decoder.scale_shift_table", "F16", &[1, 2], &f16_bytes),
                // belongs to the condition component, must be skipped here
                ("encoder.text_projector.weight", "F32", &[2, 2], &f32_bytes),
                ("null_condition_emb", "F32", &[2, 2], &f32_bytes),
            ],
        );

        let (tensors, skipped) = convert_safetensors(Component::Dit, &input).unwrap();
        assert_eq!(tensors.len(), 3);
        assert_eq!(
            skipped,
            vec![
                "encoder.text_projector.weight".to_string(),
                "null_condition_emb".to_string()
            ]
        );
        write_burnpack(&output, tensors).unwrap();

        let loaded = read_burnpack(&output);
        assert_eq!(loaded.len(), 3);
        assert_eq!(
            loaded["proj_in.conv.weight"],
            (vec![2, 3], bf16_values.to_vec())
        );
        assert_eq!(
            loaded["layers.0.self_attn.q_proj.weight"],
            (vec![2, 2], f32_values.to_vec())
        );
        assert_eq!(
            loaded["scale_shift_table"],
            (vec![1, 2], f16_values.to_vec())
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn text_encoder_roundtrip_via_component() {
        let dir = temp_dir("lm_roundtrip");
        let input = dir.join("model.safetensors");
        let output = dir.join("acestep-lm.bpk");

        let f32_bytes: Vec<u8> = [7.0_f32, -1.0]
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes)
            .collect();
        write_test_safetensors(
            &input,
            &[
                ("model.embed_tokens.weight", "F32", &[1, 2], &f32_bytes),
                ("model.norm.weight", "F32", &[1, 2], &f32_bytes),
                ("lm_head.weight", "F32", &[1, 2], &f32_bytes),
            ],
        );

        let (tensors, skipped) = convert_safetensors(Component::Lm, &input).unwrap();
        assert_eq!(skipped, vec!["lm_head.weight".to_string()]);
        write_burnpack(&output, tensors).unwrap();

        let loaded = read_burnpack(&output);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("embed_tokens.weight"));
        assert!(loaded.contains_key("norm.weight"));
        assert!(!loaded.contains_key("lm_head.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn torch_save_roundtrip() {
        let dir = temp_dir("torch_save");
        let input = dir.join("silence_latent.pt");
        let output = dir.join("silence_latent.bpk");

        let values = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let storage: Vec<u8> = values.iter().copied().flat_map(f32::to_le_bytes).collect();
        let zip = build_stored_zip(&[
            ("archive/data.pkl", &torch_tensor_pickle()),
            ("archive/data/0", &storage),
        ]);
        std::fs::write(&input, zip).unwrap();

        let (spec, data) = read_torch_save(&input).unwrap();
        assert_eq!(
            spec,
            TorchTensorSpec {
                storage_type: "FloatStorage".to_string(),
                storage_key: "0".to_string(),
                storage_offset: 0,
                shape: vec![1, 2, 3],
            }
        );
        assert_eq!(data, values.to_vec());

        write_burnpack(
            &output,
            vec![ConvertedTensor {
                name: "silence_latent".to_string(),
                shape: spec.shape,
                data,
            }],
        )
        .unwrap();
        let loaded = read_burnpack(&output);
        assert_eq!(loaded["silence_latent"], (vec![1, 2, 3], values.to_vec()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compressed_zip_entry_is_a_clear_error() {
        let mut zip = build_stored_zip(&[("archive/data.pkl", &torch_tensor_pickle())]);
        // Flip the compression method byte in both the local and central headers.
        zip[8] = 8;
        let central = zip
            .windows(4)
            .position(|window| window == 0x0201_4b50_u32.to_le_bytes())
            .unwrap();
        zip[central + 10] = 8;
        let err = read_torch_save_bytes(&zip).unwrap_err();
        assert!(format!("{err:#}").contains("compressed"), "{err:#}");
    }

    fn read_torch_save_bytes(bytes: &[u8]) -> Result<(TorchTensorSpec, Vec<f32>)> {
        let dir = temp_dir("torch_save_bytes");
        let path = dir.join("tensor.pt");
        std::fs::write(&path, bytes)?;
        let result = read_torch_save(&path);
        std::fs::remove_dir_all(&dir).ok();
        result
    }

    #[test]
    fn silence_raw_conversion() {
        let dir = temp_dir("silence_raw");
        let raw = dir.join("silence.raw");
        let output = dir.join("silence_latent.bpk");

        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().copied().flat_map(f32::to_le_bytes).collect();
        std::fs::write(&raw, &bytes).unwrap();

        let decoded = f32_bytes_to_vec(&std::fs::read(&raw).unwrap()).unwrap();
        write_burnpack(
            &output,
            vec![ConvertedTensor {
                name: "silence_latent".to_string(),
                shape: vec![1, 2, 2],
                data: decoded,
            }],
        )
        .unwrap();

        let loaded = read_burnpack(&output);
        assert_eq!(loaded["silence_latent"], (vec![1, 2, 2], values.to_vec()));

        std::fs::remove_dir_all(&dir).ok();
    }
}
