# maolan-generate

[![crates.io](https://img.shields.io/crates/v/maolan-generate.svg)](https://crates.io/crates/maolan-generate)

`maolan-generate` is the HeartMuLa generation crate from the Maolan project.
It provides a CLI for prompt-driven music generation and exposes the runtime
pieces the main Maolan application uses for in-process generation and decode.

This directory is a focused package, not the full DAW. The desktop application
and engine live in the repository root and sibling crates.

## What the crate provides

- `maolan-generate`: the main CLI for generating audio from a text prompt or
  lyrics prompt.
- `heartmula_runtime`: runtime helpers used by the CLI and the main app for
  HeartMuLa token generation and HeartCodec decode.
- `heartcodec`: model loading and decode support for the packaged HeartCodec
  path.

The crate currently supports:

- text or lyrics prompts with optional style tags
- CPU or Vulkan backends
- adjustable CFG scale, duration, top-k, temperature, and ODE step count
- decode-only mode from a saved frames JSON
- local model directory overrides or Hugging Face cache resolution
- ACE-Step 1.5 (turbo DiT) instrumental generation with BPM, key/scale, and
  time-signature conditioning (`--model acestep-turbo`)

## ACE-Step 1.5

`--model acestep-turbo` runs the turbo DiT (8 steps) with the 0.6B LM planner;
`--model acestep-sft` runs the SFT DiT (50 steps, shift 1.0) with the 4B LM
planner. The sampler schedule is selected automatically from `is_turbo` in the
checkpoint config, and the planner is loaded on demand in f16 and dropped
after planning, so peak VRAM stays around max(planner/2, everything else) —
about 8–12 GB for the big configuration. The pipeline:

1. `Qwen3-Embedding-0.6B` text encoder (causal) embeds the caption in the
   official SFT prompt format, with the metadata block from the request.
2. The 5 Hz LM planner (`acestep-5Hz-lm`) turns the caption plus metadata into
   FSQ audio codes. BPM, key/scale, and time signature are injected through a
   constructed `<think>` metadata block (the CoT generation phase is skipped
   because the values are always known); sampling uses CFG 2.0 and top-p 0.9
   with the official code-only mask, and the planner runs in f16.
3. The DiT (24 layers, sliding+full attention, AdaLN) renders 25 Hz latents
   with the LM codes as source hints — 8 Euler steps (shift 3.0) for turbo or
   50 steps (shift 1.0) for SFT, selected from the checkpoint config.
4. The Oobleck VAE decoder upsamples latents to 48 kHz stereo.

```bash
cargo run --release -- \
  --model acestep-turbo \
  --backend vulkan \
  --bpm 128 \
  --key-scale "A minor" \
  --time-signature "4/4" \
  --length 10000 \
  --output loop.wav \
  "dark rolling techno groove"
```

Weights are converted offline from the official safetensors checkpoints into
BurnPack files with the bundled converter:

```bash
cargo run --release --bin acestep_convert -- \
  --component dit --input model.safetensors --output acestep-dit.bpk
```

(`--component` is one of `text-encoder`, `lm`, `dit`, `condition`, `vae`,
`silence`; see `acestep_convert --help`.) To download the official checkpoints
from Hugging Face and convert them in one go (pure Rust + curl, no Python
needed):

```bash
bin/convert_acestep.sh /path/to/out                 # 0.6B LM planner (default)
bin/convert_acestep.sh /path/to/out --lm 1.7B       # larger planner
bin/convert_acestep.sh /path/to/out --snapshot-dir /data/Ace-Step1.5  # local checkout
```

The pre-converted files are expected
in a single model directory (or Hugging Face repo) as:

- `qwen3-encoder.bpk`, `qwen3_config.json`, `tokenizer.json`
- `acestep-lm.bpk`, `lm_config.json`, `lm_tokenizer.json`
- `acestep-dit.bpk`, `dit_config.json`
- `acestep-condition.bpk`
- `acestep-vae.bpk`, `vae_config.json`
- `silence_latent.bpk`

Like the HeartMuLa burn repos (which vendor their own `convert.sh` and
exporter sources), `maolandaw/ACE-Step-1.5-burn` should be published with
`src/bin/acestep_convert.rs` and `bin/convert_acestep.sh` copied in, so the
repo stays self-describing.

Lyrics/vocal conditioning is out of scope: the lyric encoder always receives a
single dummy token, and generation is instrumental only.


## Model assets

By default the CLI resolves model files through `hf-hub`. The current expected
repositories are:

- `maolandaw/HeartMuLa-happy-new-year-burn`
- `maolandaw/HeartMuLa-RL-oss-3B-20260123`
- `maolandaw/HeartCodec-oss-20260123-burn`

The HeartMuLa repository is expected to provide:

- `heartmula.bpk`
- `tokenizer.json`
- `gen_config.json`

The HeartCodec repository is expected to provide:

- `heartcodec.bpk`

You can bypass Hugging Face cache lookup with `--model-dir <path>` when using a
local Burn export layout.

## CLI usage

Basic generation (standalone):

```bash
cargo run --release -- "warm pads, slow build, distant vocal"
```

When running from the Maolan workspace root instead, add `-p maolan-generate`:

```bash
cargo run -p maolan-generate --release -- "warm pads, slow build, distant vocal"
```

Generation with explicit options:

```bash
cargo run --release -- \
  --model happy-new-year \
  --backend vulkan \
  --tags "ambient, cinematic, downtempo" \
  --length 12000 \
  --cfg-scale 1.5 \
  --topk 50 \
  --temperature 1.0 \
  --ode-steps 10 \
  --output output.wav \
  --lyrics "stars drift over the late train home"
```

Decode-only mode from a saved frames JSON:

```bash
cargo run --release -- \
  --decode-only \
  --backend cpu \
  --frames-json output.frames.json \
  --output output.wav
```

Run `maolan-generate --help` for the current full option list.

## Development

Standalone build:

```bash
cargo build
cargo clippy --all-targets
```

From the Maolan workspace root:

```bash
cargo build -p maolan-generate
cargo clippy -p maolan-generate --all-targets
```

### Windows

Building on Windows requires MSVC and a few environment variables.

`powershell -ExecutionPolicy Bypass -File "\\172.16.0.254\repos\maolan\generate\build.ps1"`

### Standalone usage note

This crate is published as a standalone package, but it depends on pre-release
versions of `burn` and related crates. Because pre-release semver ranges are
fluid, a fresh `Cargo.lock` may resolve to newer, API-incompatible versions.

## Repository

- Repository: <https://github.com/maolan/generate>
- Project site: <https://maolan.github.io>
