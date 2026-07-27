#!/usr/bin/env bash
# Download the official ACE-Step 1.5 checkpoints from Hugging Face and convert
# them into the BurnPack model directory expected by
# `maolan-generate --model acestep-turbo`.
#
# Usage:
#   bin/convert_acestep.sh <out_dir> [options]
#
# Options:
#   --lm <0.6B|1.7B|4B>     5Hz LM planner size (default: 0.6B)
#   --download-dir <dir>    Where to put the downloaded checkpoints
#                           (default: <out_dir>/checkpoints)
#   --snapshot-dir <dir>    Skip downloading and convert an existing local
#                           checkout of ACE-Step/Ace-Step1.5 instead
#   -h, --help
#
# Examples:
#   bin/convert_acestep.sh ~/models/acestep-burn
#   bin/convert_acestep.sh ~/models/acestep-burn --lm 1.7B
#   bin/convert_acestep.sh ~/models/acestep-burn --snapshot-dir /data/Ace-Step1.5
#
# Note: downloads are ~2-8 GB depending on the LM size, and the converted
# .bpk files are f32 (roughly twice the bf16 checkpoint size), so plan for
# ~25 GB of free disk in the worst case.
set -euo pipefail

usage() {
    sed -n '2,27p' "${BASH_SOURCE[0]}"
}

OUT_DIR=""
SNAPSHOT_DIR=""
DOWNLOAD_DIR=""
LM_SIZE="0.6B"

while [ $# -gt 0 ]; do
    case "$1" in
        --lm)
            LM_SIZE="${2:?missing value after --lm}"
            shift 2
            ;;
        --download-dir)
            DOWNLOAD_DIR="${2:?missing value after --download-dir}"
            shift 2
            ;;
        --snapshot-dir)
            SNAPSHOT_DIR="${2:?missing value after --snapshot-dir}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$OUT_DIR" ]; then
                OUT_DIR="$1"
            else
                echo "unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$OUT_DIR" ]; then
    usage >&2
    exit 1
fi

case "$LM_SIZE" in
    0.6B|1.7B|4B) ;;
    *)
        echo "unsupported --lm size '$LM_SIZE' (expected 0.6B, 1.7B or 4B)" >&2
        exit 1
        ;;
esac

CRATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$OUT_DIR/checkpoints}"
MAIN_REPO="$DOWNLOAD_DIR/Ace-Step1.5"

mkdir -p "$OUT_DIR" "$DOWNLOAD_DIR"

download() {
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    echo ">> downloading $url"
    curl -fSL --retry 3 --retry-delay 5 -C - "$url" -o "$dest"
}

hf_file() { # <repo> <relative_path> <dest_root>
    download "https://huggingface.co/$1/resolve/main/$2" "$3/$2"
}

if [ -z "$SNAPSHOT_DIR" ]; then
    SNAPSHOT_DIR="$MAIN_REPO"
    echo ">> downloading ACE-Step/Ace-Step1.5 components to $MAIN_REPO"
    for f in model.safetensors config.json tokenizer.json; do
        hf_file "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/$f" "$MAIN_REPO"
    done
    for f in model.safetensors config.json silence_latent.pt; do
        hf_file "ACE-Step/Ace-Step1.5" "acestep-v15-turbo/$f" "$MAIN_REPO"
    done
    for f in diffusion_pytorch_model.safetensors config.json; do
        hf_file "ACE-Step/Ace-Step1.5" "vae/$f" "$MAIN_REPO"
    done
fi

# The 1.7B planner only ships inside the main snapshot; other sizes have
# their own repos.
if [ "$LM_SIZE" = "1.7B" ]; then
    LM_DIR="$SNAPSHOT_DIR/acestep-5Hz-lm-1.7B"
    if [ ! -f "$LM_DIR/model.safetensors" ]; then
        for f in model.safetensors config.json tokenizer.json; do
            hf_file "ACE-Step/Ace-Step1.5" "acestep-5Hz-lm-1.7B/$f" "$MAIN_REPO"
        done
    fi
else
    LM_DIR="$DOWNLOAD_DIR/acestep-5Hz-lm-$LM_SIZE"
    if [ ! -f "$LM_DIR/model.safetensors" ]; then
        for f in model.safetensors config.json tokenizer.json; do
            hf_file "ACE-Step/acestep-5Hz-lm-$LM_SIZE" "$f" "$LM_DIR"
        done
    fi
fi

convert() {
    cargo run --release --manifest-path "$CRATE_DIR/Cargo.toml" \
        --bin acestep_convert -- "$@"
}

echo ">> converting text encoder (Qwen3-Embedding-0.6B)"
convert --component text-encoder \
    --input "$SNAPSHOT_DIR/Qwen3-Embedding-0.6B/model.safetensors" \
    --output "$OUT_DIR/qwen3-encoder.bpk"
cp "$SNAPSHOT_DIR/Qwen3-Embedding-0.6B/config.json" "$OUT_DIR/qwen3_config.json"
cp "$SNAPSHOT_DIR/Qwen3-Embedding-0.6B/tokenizer.json" "$OUT_DIR/tokenizer.json"

echo ">> converting 5Hz LM planner ($LM_SIZE)"
convert --component lm \
    --input "$LM_DIR/model.safetensors" \
    --output "$OUT_DIR/acestep-lm.bpk"
cp "$LM_DIR/config.json" "$OUT_DIR/lm_config.json"
cp "$LM_DIR/tokenizer.json" "$OUT_DIR/lm_tokenizer.json"

echo ">> converting DiT (turbo)"
convert --component dit \
    --input "$SNAPSHOT_DIR/acestep-v15-turbo/model.safetensors" \
    --output "$OUT_DIR/acestep-dit.bpk"

echo ">> converting condition stack"
convert --component condition \
    --input "$SNAPSHOT_DIR/acestep-v15-turbo/model.safetensors" \
    --output "$OUT_DIR/acestep-condition.bpk"
cp "$SNAPSHOT_DIR/acestep-v15-turbo/config.json" "$OUT_DIR/dit_config.json"

echo ">> converting silence latent"
convert --component silence \
    --input "$SNAPSHOT_DIR/acestep-v15-turbo/silence_latent.pt" \
    --output "$OUT_DIR/silence_latent.bpk"

echo ">> converting VAE decoder"
convert --component vae \
    --input "$SNAPSHOT_DIR/vae/diffusion_pytorch_model.safetensors" \
    --output "$OUT_DIR/acestep-vae.bpk"
cp "$SNAPSHOT_DIR/vae/config.json" "$OUT_DIR/vae_config.json"

echo ">> done: ACE-Step BurnPack model directory written to $OUT_DIR"
echo "   run with: maolan-generate --model acestep-turbo --model-dir $OUT_DIR ..."
