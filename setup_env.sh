#!/usr/bin/env bash
# setup_env.sh
# Builds the LARQL binary from cronos3k/larql (cross-platform fork).
# Uses cronos3k fork instead of chrishayuk/larql because the original
# is macOS/Apple Silicon only. cronos3k adds Linux OpenBLAS + CUDA backends.
set -euo pipefail

log() { echo "[setup] $*"; }

# ── Rust toolchain ───────────────────────────────────────────────────────────
if ! command -v rustc &>/dev/null; then
    log "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
# shellcheck source=/dev/null
source "${HOME}/.cargo/env"
log "Rust: $(rustc --version)"

# ── System deps (Linux) ──────────────────────────────────────────────────────
if [[ "$(uname)" == "Linux" ]]; then
    log "Installing OpenBLAS dev libs (required for Linux build)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libopenblas-dev pkg-config
fi

# ── Clone cronos3k/larql (cross-platform fork) ───────────────────────────────
# NOTE: Use cronos3k fork, NOT chrishayuk/larql.
# The original hardcodes Apple Accelerate (AMX) BLAS — it will not build on Linux.
if [ ! -d larql ]; then
    log "Cloning cronos3k/larql (Linux/CUDA cross-platform fork)..."
    git clone https://github.com/cronos3k/larql.git
else
    log "larql/ already present — pulling latest."
    ( cd larql && git pull --ff-only )
fi

# ── Compile ──────────────────────────────────────────────────────────────────
log "Building in release mode (first build takes 3-5 minutes)..."
( cd larql && cargo build --release )

BINARY="larql/target/release/larql"
[ -f "${BINARY}" ] || { echo "ERROR: binary missing at ${BINARY}" >&2; exit 1; }
log "Binary ready: ${BINARY}"

# ── Verify the query subcommand exists ───────────────────────────────────────
log "Checking available subcommands:"
"${BINARY}" --help | grep -E 'query|extract|serve' || true

log ""
log "Setup complete. Next: bash fetch_vindex.sh"
