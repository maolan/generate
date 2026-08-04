#!/usr/bin/env bash
set -euo pipefail

# build-debian.sh — Build a .deb package for Maolan Generate on Debian.
#
# Usage:
#   ./scripts/build-debian.sh [OPTIONS]
#
# Options:
#   -s, --source-dir DIR     Path to maolan-generate source directory (default: parent of this script)
#   -o, --output-dir DIR     Where to write the .deb file (default: ./dist)
#   -v, --version VERSION    Override package version (default: read from Cargo.toml)
#   -t, --target-dir DIR     Local target directory (useful when source is on NFS)
#   -h, --help               Show this help message
#
# The script installs build dependencies via apt, installs Rust via rustup if missing,
# builds the release binaries, and produces a .deb package using dpkg-deb.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SOURCE_DIR/dist"
OVERRIDE_VERSION=""
TARGET_DIR=""

usage() {
    sed -n '2,14p' "$0" | sed 's/^# //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--source-dir)
            SOURCE_DIR="$(realpath "$2")"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$(realpath "$2")"
            shift 2
            ;;
        -v|--version)
            OVERRIDE_VERSION="$2"
            shift 2
            ;;
        -t|--target-dir)
            TARGET_DIR="$(realpath "$2")"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

CARGO_TOML="$SOURCE_DIR/Cargo.toml"
if [[ ! -f "$CARGO_TOML" ]]; then
    echo "Error: Cargo.toml not found at $CARGO_TOML" >&2
    exit 1
fi

# Extract version from Cargo.toml or use override
if [[ -n "$OVERRIDE_VERSION" ]]; then
    PKG_VERSION="$OVERRIDE_VERSION"
else
    PKG_VERSION="$(grep -m1 '^version' "$CARGO_TOML" | sed 's/.*= *"\(.*\)".*/\1/')"
fi

DEB_ARCH="$(dpkg --print-architecture)"
PKG_NAME="maolan-generate"
DEB_NAME="${PKG_NAME}-${PKG_VERSION}-debian.${DEB_ARCH}.deb"

echo "========================================"
echo "Building Maolan Generate .deb package"
echo "Version: $PKG_VERSION"
echo "Architecture: $DEB_ARCH"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR/$DEB_NAME"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Install system build dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    pkg-config \
    build-essential \
    libvulkan1 \
    curl \
    ca-certificates \
    git

# ---------------------------------------------------------------------------
# 2. Install Rust if missing
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking Rust toolchain..."
if ! command -v cargo &>/dev/null; then
    echo "Rust not found. Installing via rustup..."
    export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
    export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$CARGO_HOME/env"
else
    echo "Rust already installed: $(rustc --version)"
fi

# Ensure cargo is in PATH for the rest of the script
if [[ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
    source "${CARGO_HOME:-$HOME/.cargo}/env"
fi

# ---------------------------------------------------------------------------
# 3. Build release binaries
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Building release binaries..."
cd "$SOURCE_DIR"

CARGO_ARGS=("--release")
CARGO_ARGS+=("--all-targets")
if [[ -n "$TARGET_DIR" ]]; then
    mkdir -p "$TARGET_DIR"
    CARGO_ARGS+=("--target-dir" "$TARGET_DIR")
    echo "Using local target directory: $TARGET_DIR"
fi

cargo build "${CARGO_ARGS[@]}"

# Determine where binaries ended up
if [[ -n "$TARGET_DIR" ]]; then
    BIN_DIR="$TARGET_DIR/release"
else
    BIN_DIR="$SOURCE_DIR/target/release"
fi

# Verify binaries exist
for bin in maolan-generate acestep_convert; do
    if [[ ! -f "$BIN_DIR/$bin" ]]; then
        echo "Error: Binary '$BIN_DIR/$bin' not found after build" >&2
        exit 1
    fi
done

echo "Build completed successfully."

# ---------------------------------------------------------------------------
# 4. Prepare Debian package staging area
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Preparing Debian package structure..."

STAGING_DIR="$(mktemp -d)"
trap "rm -rf '$STAGING_DIR'" EXIT

mkdir -p "$STAGING_DIR/DEBIAN"
mkdir -p "$STAGING_DIR/usr/bin"
mkdir -p "$STAGING_DIR/usr/share/doc/$PKG_NAME"

# Binaries
cp "$BIN_DIR/maolan-generate" "$STAGING_DIR/usr/bin/"
cp "$BIN_DIR/acestep_convert" "$STAGING_DIR/usr/bin/"
strip "$STAGING_DIR/usr/bin/"*
chmod 755 "$STAGING_DIR/usr/bin/"*

# Documentation
cp "$SOURCE_DIR/README.md" "$STAGING_DIR/usr/share/doc/$PKG_NAME/"
cp "$SOURCE_DIR/LICENSE"   "$STAGING_DIR/usr/share/doc/$PKG_NAME/"
gzip -9 -n -c > "$STAGING_DIR/usr/share/doc/$PKG_NAME/changelog.gz" /dev/null 2>/dev/null || true

# DEBIAN/control
cat > "$STAGING_DIR/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $PKG_VERSION
Section: sound
Priority: optional
Architecture: $DEB_ARCH
Depends: libvulkan1
Maintainer: Maolan Team <maolan@github.io>
Description: Maolan AI Music Generation CLI
 maolan-generate is the HeartMuLa generation crate from the Maolan project.
 It provides a CLI for prompt-driven music generation and exposes the
 runtime pieces the main Maolan application uses for in-process generation.
EOF

cat > "$STAGING_DIR/DEBIAN/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: Maolan Generate
Source: https://github.com/maolan/generate

Files: *
Copyright: Maolan Team
License: BSD-2-Clause
EOF

# ---------------------------------------------------------------------------
# 5. Build the .deb package
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Building .deb package..."
mkdir -p "$OUTPUT_DIR"
fakeroot dpkg-deb --build "$STAGING_DIR" "$OUTPUT_DIR/$DEB_NAME"

# ---------------------------------------------------------------------------
# 6. Verify the package
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying package..."
dpkg-deb --info "$OUTPUT_DIR/$DEB_NAME"
dpkg-deb --contents "$OUTPUT_DIR/$DEB_NAME"

echo ""
echo "========================================"
echo "Package built successfully:"
echo "  $OUTPUT_DIR/$DEB_NAME"
echo "========================================"
