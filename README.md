# larql-lql-demo

Replication of Chris Hay's "I Decoupled Attention from Weights — Gemma 4 26B" YouTube demo,
using the cross-platform [cronos3k/larql](https://github.com/cronos3k/larql) fork
and the `chrishayuk/gemma-3-4b-it-vindex`.

**What this is:** mechanistic interpretability via LQL (Lazarus Query Language).
You are NOT running the LLM for text generation. You are querying the model's
FFN weight structure as a graph — inspecting which features activate, comparing
concept representations across layers, and optionally editing knowledge directly
in the vindex.

---

## Quickstart

```bash
# 1. Build the LARQL binary (cross-platform fork, Linux/macOS/Windows)
bash setup_env.sh

# 2. Download the Gemma 3 4B vindex (~4-5 GB)
bash fetch_vindex.sh

# 3. Launch the interactive LQL REPL
python lql_repl.py

# 4. Or run the YouTube demo queries non-interactively
python lql_repl.py --script demo_queries.lql
```

---

## What LQL Queries Look Like

```sql
-- Which features fire most strongly at layers 20-23 for this prompt?
WALK "The capital of France is" TOP 10;

-- Compare concept representations side-by-side at the same layer
PROBE "France is" vs "Germany is" vs "Japan is" AT LAYER 22;

-- Predict what the model thinks comes next
INFER "The capital of France is" TOP 5;

-- Edit the model's knowledge directly (not a fine-tune — writes to the graph)
INSERT INTO EDGES (entity, relation, target)
VALUES ("Atlantis", "capital-of", "Poseidon");
```

---

## Repository Layout

```
.
├── setup_env.sh        # Rust toolchain + clone cronos3k/larql + cargo build --release
├── fetch_vindex.sh     # HuggingFace CLI download of gemma-3-4b-it-vindex
├── lql_repl.py         # Interactive REPL + --script mode for batch query files
├── demo_queries.lql    # The exact query sequence from Chris Hay's YouTube demo
└── data/
    └── gemma3-4b.vindex/   # Populated by fetch_vindex.sh
```

---

## VM Sizing

LQL queries do **sparse KNN lookups**, not full autoregressive generation.
A DigitalOcean Basic **8 GB / 4 vCPU** droplet (~$0.07/hr on-demand) is sufficient.
SSH in from Codespaces, run `bash setup_env.sh`, done.

---

## Credits

- Original LARQL + LQL language: [chrishayuk/larql](https://github.com/chrishayuk/larql)
- Cross-platform fork (Linux/CUDA builds): [cronos3k/larql](https://github.com/cronos3k/larql)
- HuggingFace Space (zero-install browser demo): https://huggingface.co/spaces/cronos3k/LARQL-Explorer
