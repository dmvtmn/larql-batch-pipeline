# larql-batch-pipeline

Standalone, CPU-only batch inference pipeline built on [LARQL](https://github.com/chrishayuk/larql) (the "model-as-a-database" engine) and the `gemma-3-4b-it-vindex`.

**No GPU. No Kubernetes. No eBPF.** Pure mmap'd vector index queries over CPU BLAS.

---

## Quickstart

```bash
# 1. Bootstrap environment + download vindex
bash setup_env.sh

# 2. Launch the LARQL server (blocks until ready)
python server_manager.py &

# 3. Drop .txt files into data/input/, run the batch
python batch_processor.py

# Outputs land in output/ as structured JSON
```

---

## Repository Layout

```
.
├── setup_env.sh          # Task 1+2: Toolchain + vindex hydration
├── server_manager.py     # Task 3: Subprocess wrapper + health-check
├── batch_processor.py    # Task 4+5: Async batch client + output sink
├── ARCHITECTURE.md       # Detailed component blueprint
├── data/
│   ├── input/            # Drop .txt prompt files here
│   └── gemma3-4b.vindex/ # Populated by setup_env.sh
└── output/               # Structured JSON results land here
```

---

## Jules Task Reference

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full architectural blueprint and the sequential task list designed for autonomous coding-agent execution.
