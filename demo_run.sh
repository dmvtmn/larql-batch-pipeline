#!/usr/bin/env bash
# demo_run.sh
# Replicates the Chris Hay YouTube demo using confirmed cronos3k/larql CLI.
# Subcommands: walk, lql, repl, serve
# Run: bash demo_run.sh
set -euo pipefail

BIN="./larql/target/release/larql"
VINDEX="./data/gemma3-4b.vindex"
DIVIDER="================================================"

log() { echo ""; echo "${DIVIDER}"; echo "  $*"; echo "${DIVIDER}"; }

# ── 1. Verify binary + vindex ─────────────────────────────────────────────────
[ -f "${BIN}" ]      || { echo "ERROR: run bash setup_env.sh first"; exit 1; }
[ -d "${VINDEX}" ]   || { echo "ERROR: run bash fetch_vindex.sh first"; exit 1; }

log "STEP 1: Binary help (confirms available subcommands)"
"${BIN}" --help

# ── 2. LQL graph queries ──────────────────────────────────────────────────────
log "STEP 2: LQL — capital of France"
"${BIN}" lql 'SELECT ?x WHERE { <France> <capital> ?x }' --graph "${VINDEX}"

log "STEP 2b: LQL — all known facts about France"
"${BIN}" lql 'SELECT ?r ?x WHERE { <France> ?r ?x }' --graph "${VINDEX}"

log "STEP 2c: LQL — capital of Germany"
"${BIN}" lql 'SELECT ?x WHERE { <Germany> <capital> ?x }' --graph "${VINDEX}"

# ── 3. FFN walk (the core interpretability demo) ──────────────────────────────
log "STEP 3: FFN walk — which features activate for this prompt?"
"${BIN}" walk "${VINDEX}" --prompt "The capital of France is" --top 10

log "STEP 3b: FFN walk — compare Germany"
"${BIN}" walk "${VINDEX}" --prompt "The capital of Germany is" --top 10

log "STEP 3c: FFN walk with layer scope (if --layers supported)"
"${BIN}" walk "${VINDEX}" --prompt "The capital of France is" --top 10 --layers 20-23 \
    || echo "(--layers flag not supported in this build — skipping)"

log "Demo complete."
