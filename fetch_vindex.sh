#!/usr/bin/env bash
# fetch_vindex.sh
# Downloads the pre-extracted gemma-3-4b-it-vindex from HuggingFace Hub.
# The vindex is the pre-compiled graph representation of the model's FFN —
# you do NOT need the original model weights to run LQL queries.
set -euo pipefail

log() { echo "[fetch] $*"; }

VINDEX_DIR="./data/gemma3-4b.vindex"

# ── HuggingFace CLI (new: hf, old: huggingface-cli) ─────────────────────────
if ! command -v hf &>/dev/null; then
    log "Installing huggingface_hub..."
    pip install -q -U "huggingface_hub"
fi

# ── Auth check ───────────────────────────────────────────────────────────────
if ! hf auth whoami &>/dev/null; then
    echo "WARNING: not logged in to HuggingFace."
    echo "Run: hf auth login"
fi

# ── Download ─────────────────────────────────────────────────────────────────
mkdir -p "./data"

if [ -d "${VINDEX_DIR}" ] && find "${VINDEX_DIR}" -name '*.bin' -quit 2>/dev/null | grep -q .; then
    log "Vindex already present at ${VINDEX_DIR} — skipping download."
else
    log "Downloading chrishayuk/gemma-3-4b-it-vindex (~4-5 GB)..."
    hf download chrishayuk/gemma-3-4b-it-vindex \
        --local-dir "${VINDEX_DIR}"
fi

# ── Verify ───────────────────────────────────────────────────────────────────
BIN_COUNT=$(find "${VINDEX_DIR}" -name '*.bin' 2>/dev/null | wc -l)
[ "${BIN_COUNT}" -gt 0 ] || { echo "ERROR: no .bin files in ${VINDEX_DIR}" >&2; exit 1; }
log "Vindex ready: ${BIN_COUNT} .bin shard(s) in ${VINDEX_DIR}"
log ""
log "Next: python lql_repl.py"
