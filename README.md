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

#### 1. Install dependencies

1. **Rust** — Install via [rustup](https://rustup.rs/):
   ```powershell
   winget install Rustlang.Rustup
   rustup target add x86_64-pc-windows-msvc
   ```

2. **Visual Studio 2022** — Install the *Desktop development with C++* workload.

3. **LLVM** — Required by `burn` dependencies (bindgen). Install from [llvm.org](https://releases.llvm.org/) or winget:
   ```powershell
   winget install LLVM.LLVM
   ```

4. **NSIS** — Required to build the installer:
   ```powershell
   # Download https://prdownloads.sourceforge.net/nsis/nsis-3.10.zip
   # Extract to C:\nsis-3.10 (or anywhere local)
   ```

#### 2. Set environment variables

```powershell
$env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
```

#### 3. Build the binary

```powershell
cargo build --release --target x86_64-pc-windows-msvc
```

If building from a network share, use a local target directory:

```powershell
cargo build --release --target x86_64-pc-windows-msvc --target-dir C:\cargo-target
```

#### 4. Build the installer

The installer bundles the executable and the VC++ Redistributable.

1. Download the VC++ Redistributable to the repo root:
   ```powershell
   Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '..\vc_redist.x64.exe'
   ```

2. Compile the installer:
   ```powershell
   C:\nsis-3.10\makensis.exe installer.nsi
   ```

The output is `maolan-generate-setup.exe` in the `generate/` directory.

> **Note:** This crate depends on `burn` → `tokenizers` → `esaxx-rs`. For standalone builds, the `[patch.crates-io]` entry in `Cargo.toml` points to a patched fork that removes a Windows CRT conflict. When building through the DAW, the DAW's patch applies automatically.

### Standalone usage note

This crate is published as a standalone package, but it depends on pre-release
versions of `burn` and related crates. Because pre-release semver ranges are
fluid, a fresh `Cargo.lock` may resolve to newer, API-incompatible versions.

## Repository

- Repository: <https://github.com/maolan/generate>
- Project site: <https://maolan.github.io>
