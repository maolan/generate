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
