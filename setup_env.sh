#!/usr/bin/env bash
# setup_env.sh — Task 1 + Task 2
# Bootstraps Rust toolchain, builds LARQL, downloads gemma-3-4b-it-vindex.
set -euo pipefail

log() { echo "[setup] $*"; }

# ── Task 1: Rust toolchain ─────────────────────────────────────────────────
if ! command -v rustc &>/dev/null; then
    log "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
# shellcheck source=/dev/null
source "${HOME}/.cargo/env"

# ── Task 1: HuggingFace CLI ────────────────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
    log "Installing huggingface_hub CLI..."
    pip install -q -U "huggingface_hub[cli]"
fi

# ── Task 1: Clone LARQL ────────────────────────────────────────────────────
if [ ! -d larql ]; then
    log "Cloning chrishayuk/larql..."
    git clone https://github.com/chrishayuk/larql.git
else
    log "larql/ already present — skipping clone."
fi

# ── Task 1: Compile release binary ────────────────────────────────────────
log "Building LARQL in release mode (this takes a few minutes)..."
( cd larql && cargo build --release )

BINARY="larql/target/release/larql"
if [ ! -f "${BINARY}" ]; then
    echo "ERROR: binary not found at ${BINARY}" >&2
    exit 1
fi
log "LARQL binary ready: ${BINARY}"

# ── Task 2: Download vindex ────────────────────────────────────────────────
VINDEX_DIR="./data/gemma3-4b.vindex"
mkdir -p "./data"

if [ -d "${VINDEX_DIR}" ] && find "${VINDEX_DIR}" -name '*.bin' -quit 2>/dev/null | grep -q .; then
    log "Vindex already present at ${VINDEX_DIR} — skipping download."
else
    log "Downloading chrishayuk/gemma-3-4b-it-vindex to ${VINDEX_DIR}..."
    huggingface-cli download chrishayuk/gemma-3-4b-it-vindex \
        --local-dir "${VINDEX_DIR}"
fi

# ── Verification ──────────────────────────────────────────────────────────
BIN_COUNT=$(find "${VINDEX_DIR}" -name '*.bin' 2>/dev/null | wc -l)
if [ "${BIN_COUNT}" -eq 0 ]; then
    echo "ERROR: no .bin files found in ${VINDEX_DIR} — download may have failed." >&2
    exit 1
fi

log "Setup complete. Vindex contains ${BIN_COUNT} .bin shard(s)."
log "Next: python server_manager.py"
