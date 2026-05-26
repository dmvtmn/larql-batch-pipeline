#!/usr/bin/env bash
# demo_run.sh
# Replicates the Chris Hay YouTube demo sequence:
#   1. LQL graph queries (knowledge lookup)
#   2. Inference / next-token prediction (larql run)
#   3. Shannon slot probes (interpretability — which token slots fire?)
#   4. FFN walk (dev subcommand)
#
# Run: bash demo_run.sh
set -euo pipefail

BIN="./larql/target/release/larql"
VINDEX="./data/gemma3-4b.vindex"
DIVIDER="================================================"

log() { echo ""; echo "${DIVIDER}"; echo "  $*"; echo "${DIVIDER}"; }

# ── 1. Vindex metadata ────────────────────────────────────────────────────────
log "STEP 1: Vindex metadata (larql show)"
"${BIN}" show "${VINDEX}"

# ── 2. LQL graph queries ─────────────────────────────────────────────────────
log "STEP 2: LQL graph query — capital of France"
"${BIN}" lql 'SELECT ?x WHERE { <France> <capital> ?x }' --graph "${VINDEX}"

log "STEP 2b: LQL graph query — all facts about France"
"${BIN}" lql 'SELECT ?r ?x WHERE { <France> ?r ?x }' --graph "${VINDEX}"

# ── 3. Inference (next-token prediction) ──────────────────────────────────────
log "STEP 3: Inference — next-token prediction"
"${BIN}" run "${VINDEX}" "The capital of France is" -n 5 --top 5

log "STEP 3b: Inference — compare Germany"
"${BIN}" run "${VINDEX}" "The capital of Germany is" -n 5 --top 5

# ── 4. Shannon slot probe (interpretability) ─────────────────────────────────
log "STEP 4: Shannon slot probe — which token positions carry the most information?"
"${BIN}" shannon slot-probe "${VINDEX}" --prompt "The capital of France is"

# ── 5. FFN walk (dev subcommand) ────────────────────────────────────────────────
log "STEP 5: FFN walk — which features activate for this prompt?"
"${BIN}" dev walk --index "${VINDEX}" --prompt "The capital of France is" --predict

log "Demo complete."
