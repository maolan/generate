#!/usr/bin/env bash
# Download the official ACE-Step 1.5 checkpoints from Hugging Face and convert
# them into the BurnPack model directory expected by
# `maolan-generate --model acestep-turbo`.
#
# Usage:
#   bin/convert_acestep.sh <out_dir> [options]
#
# Options:
#   --download-dir <dir>    Where to put the downloaded checkpoints
#                           (default: <out_dir>/checkpoints)
#   --snapshot-dir <dir>    Skip downloading and convert an existing local
#                           checkout of ACE-Step/Ace-Step1.5 instead
#   -h, --help
#
# Examples:
#   bin/convert_acestep.sh ~/models/acestep-burn
#   bin/convert_acestep.sh ~/models/acestep-burn --snapshot-dir /data/Ace-Step1.5
#
# Note: downloads include all three turbo LM planners (0.6B, 1.7B, 4B), and
# the converted .bpk files are f32 (roughly twice the bf16 checkpoint size), so
# plan for ~25 GB of free disk.
set -euo pipefail

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}"
}

OUT_DIR=""
SNAPSHOT_DIR=""
DOWNLOAD_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
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

convert() {
    cargo run --release --manifest-path "$CRATE_DIR/Cargo.toml" \
        --bin acestep_convert -- "$@"
}

lm_dir_for_size() {
    local size="$1"
    if [ "$size" = "1.7B" ]; then
        echo "$SNAPSHOT_DIR/acestep-5Hz-lm-1.7B"
    else
        echo "$DOWNLOAD_DIR/acestep-5Hz-lm-$size"
    fi
}

download_lm() {
    local size="$1" lm_dir
    lm_dir="$(lm_dir_for_size "$size")"
    if [ "$size" = "1.7B" ]; then
        if [ -f "$lm_dir/model.safetensors" ]; then
            return
        fi
        for f in model.safetensors config.json tokenizer.json; do
            hf_file "ACE-Step/Ace-Step1.5" "acestep-5Hz-lm-1.7B/$f" "$MAIN_REPO"
        done
    elif [ "$size" = "4B" ]; then
        if [ -f "$lm_dir/model-00001-of-00002.safetensors" ] &&
            [ -f "$lm_dir/model-00002-of-00002.safetensors" ] &&
            [ -f "$lm_dir/config.json" ] &&
            [ -f "$lm_dir/tokenizer.json" ]; then
            return
        fi
        for f in \
            model-00001-of-00002.safetensors \
            model-00002-of-00002.safetensors \
            model.safetensors.index.json \
            config.json \
            tokenizer.json
        do
            hf_file "ACE-Step/acestep-5Hz-lm-$size" "$f" "$lm_dir"
        done
    else
        if [ -f "$lm_dir/model.safetensors" ]; then
            return
        fi
        for f in model.safetensors config.json tokenizer.json; do
            hf_file "ACE-Step/acestep-5Hz-lm-$size" "$f" "$lm_dir"
        done
    fi
}

lm_suffix_for_size() {
    case "$1" in
        0.6B) echo "" ;;
        1.7B) echo "-1.7b" ;;
        4B) echo "-4b" ;;
    esac
}

convert_lm() {
    local size="$1" suffix lm_dir
    suffix="$(lm_suffix_for_size "$size")"
    lm_dir="$(lm_dir_for_size "$size")"
    download_lm "$size"

    echo ">> converting 5Hz LM planner ($size)"
    local input="$lm_dir/model.safetensors"
    if [ "$size" = "4B" ]; then
        input="$lm_dir"
    fi
    convert --component lm \
        --input "$input" \
        --output "$OUT_DIR/acestep-lm$suffix.bpk"
    cp "$lm_dir/config.json" "$OUT_DIR/lm_config$suffix.json"
    cp "$lm_dir/tokenizer.json" "$OUT_DIR/lm_tokenizer$suffix.json"
}

echo ">> converting text encoder (Qwen3-Embedding-0.6B)"
convert --component text-encoder \
    --input "$SNAPSHOT_DIR/Qwen3-Embedding-0.6B/model.safetensors" \
    --output "$OUT_DIR/qwen3-encoder.bpk"
cp "$SNAPSHOT_DIR/Qwen3-Embedding-0.6B/config.json" "$OUT_DIR/qwen3_config.json"
cp "$SNAPSHOT_DIR/Qwen3-Embedding-0.6B/tokenizer.json" "$OUT_DIR/tokenizer.json"

for size in 0.6B 1.7B 4B; do
    convert_lm "$size"
done

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
echo "   run with: maolan-generate --model acestep-turbo --model-dir $OUT_DIR --acestep-lm <0.6B|1.7B|4B> ..."
