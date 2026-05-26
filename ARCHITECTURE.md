# LARQL Batch Pipeline — Architecture Document

**Target:** Standalone, CPU-only batch processor using LARQL `gemma-3-4b-it-vindex`.  
**Constraint set:** No eBPF, no Kubernetes, no GPU VRAM dependency, no eager weight materialisation.

---

## 1. Core Paradigm: Model-as-a-Database

LARQL decouples transformer inference from matrix multiplication. Instead of materialising weight matrices and running `matmul` on GPU, it pre-compiles the model's Feed-Forward Network (FFN) gate vectors and feature embeddings into a `.vindex` — a directory of memory-mapped (mmap) binary files stored in `f16` precision.

At inference time the Rust engine performs **K-Nearest Neighbour (KNN) graph walks** over the vindex: for each input token representation, it traverses the FFN edges to find the top-K activated experts, aggregates their outputs, and emits decoded tokens. CPU BLAS kernels handle the dot-products. No GPU required.

This changes the bottleneck from VRAM bandwidth → OS page-cache management, which is why mmap lifecycle control is the critical operational parameter.

---

## 2. Component Topology

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer (Python Orchestrator)                │
│                                                         │
│  batch_processor.py          server_manager.py          │
│  ─ reads data/input/*.txt    ─ spawns larql serve PID   │
│  ─ asyncio + aiohttp         ─ health-check loop        │
│  ─ POST /v1/stream           ─ madvise lifecycle mgmt   │
│  ─ writes output/*.json      ─ SIGTERM on shutdown      │
└───────────────────┬─────────────────────────────────────┘
                    │  HTTP  (localhost:8080)
┌───────────────────▼─────────────────────────────────────┐
│  Compute Layer (LARQL Engine — Rust binary)             │
│                                                         │
│  larql serve <vindex_path>                              │
│  ─ HTTP/WebSocket server on :8080                       │
│  ─ KNN walks over FFN layers 0..33 (Gemma 3 4B)         │
│  ─ Streams tokens as JSON frames                        │
│  ─ --ffn-only: skip eager gate warmup                   │
│  ─ --release-mmap-after-request: madvise(DONTNEED)      │
└───────────────────┬─────────────────────────────────────┘
                    │  mmap  (zero-copy)
┌───────────────────▼─────────────────────────────────────┐
│  Storage Layer (The Vindex)                             │
│                                                         │
│  data/gemma3-4b.vindex/                                 │
│  ─ gate_vectors.bin    (f16, mmap'd)                    │
│  ─ feature_embeddings.bin                               │
│  ─ edge_relations.bin                                   │
│  ─ manifest.json       (layer/shard metadata)           │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Layer Specifications

### 3.1 Storage Layer — The Vindex

| Property | Value |
|---|---|
| Format | `.vindex` directory (pre-compiled by `chrishayuk/larql`) |
| Precision | `f16` (half-precision float, 2 bytes/param) |
| Access pattern | Memory-mapped (`mmap`), zero-copy |
| Source | `chrishayuk/gemma-3-4b-it-vindex` on HuggingFace Hub |
| Approximate size | ~4–5 GB on disk (f16 of Gemma 3 4B FFN weights) |

The vindex is **read-only** at inference time. The OS page cache is the only dynamic resource. Because Linux will happily page in the entire vindex and pin it in RAM during a long batch run, the server flag `--release-mmap-after-request` must be passed at startup — it forces `madvise(MADV_DONTNEED)` after each inference cycle, releasing the pages back to the kernel and preventing OOM on machines with < 16 GB RAM.

### 3.2 Compute Layer — LARQL Rust Engine

| Property | Value |
|---|---|
| Language | Rust (compiled with `cargo build --release`) |
| Transport | HTTP/1.1 + WebSocket on `localhost:8080` |
| Inference endpoint | `POST /v1/stream` |
| Token stream format | `{"type": "token", "text": "..."}` frames, NDJSON |
| Completion sentinel | `{"type": "done"}` |
| KNN graph depth | 34 layers (Gemma 3 4B architecture) |
| Startup flag 1 | `--ffn-only` — skips eager gate warmup, reduces base RSS |
| Startup flag 2 | `--release-mmap-after-request` — `madvise(DONTNEED)` after each cycle |

The `--ffn-only` flag is critical for batch workloads: without it the server pre-warms the gate vectors for all layers at startup, spiking RSS to near-full-model size before the first request arrives.

### 3.3 Application Layer — Python Orchestrator

Two scripts:

**`server_manager.py`** — lifecycle manager. Uses `subprocess.Popen` to start the compiled LARQL binary, exposes a blocking `wait_for_ready()` function that polls `GET /health` with exponential backoff, and registers a `SIGTERM` handler to cleanly kill the child process.

**`batch_processor.py`** — async batch client. Reads `data/input/*.txt`, maintains a concurrency-limited `asyncio` semaphore (default: 8 concurrent requests), posts to `POST /v1/stream`, reconstructs token streams, and writes structured JSON to `output/`. Failed requests are re-queued with exponential backoff (2 s base, 3 retries max).

---

## 4. Data Flow — One Inference Cycle

```
[data/input/file_001.txt]
        │
        │  1. batch_processor reads raw text
        ▼
[Prompt formatter]
  payload = {
    "type": "generate",
    "prompt": "<SYSTEM>\n...",
    "max_tokens": 500
  }
        │
        │  2. POST /v1/stream  (aiohttp, async)
        ▼
[LARQL Rust Server :8080]
  ─ Tokenise input
  ─ For each layer 0..33:
      KNN walk over gate_vectors.bin
      Activate top-K experts
      Aggregate FFN outputs
  ─ Decode token
  ─ Stream: {"type":"token","text":" The"}
        │
        │  3. Streaming NDJSON frames  →  client reconstructs string
        ▼
[batch_processor accumulates tokens]
        │
        │  4. On {"type":"done"}: write to disk
        ▼
[output/file_001.json]
  {
    "source": "file_001.txt",
    "prompt": "...",
    "response": "...",
    "tokens": 312,
    "elapsed_s": 4.7,
    "timestamp": "2026-05-26T..."
  }
```

---

## 5. Memory Budget (Reference)

| Scenario | Approximate RSS |
|---|---|
| Server idle, `--ffn-only`, no requests served | ~800 MB |
| During active inference (pages hot) | ~6–8 GB |
| After `--release-mmap-after-request` flush | Back to ~800 MB |
| Without `--release-mmap-after-request` (long batch) | Grows to full model size → OOM risk |

On a 16 GB machine, running at concurrency=1 with both flags is safe. At concurrency=8, ensure 32 GB+ RAM or reduce `MAX_CONCURRENCY` in `batch_processor.py`.

---

## 6. Sequential Task List for Jules

Pass these tasks **in order**. Each task is a prerequisite for the next.

---

### Task 1 — Bootstrap the Environment

**Goal:** Ensure Rust toolchain, HuggingFace CLI, LARQL source, and compiled binary all exist.

**Jules instruction:**  
Write and execute `setup_env.sh`. The script must:
1. Install the Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
2. Source the Cargo env: `. "$HOME/.cargo/env"`
3. Install HuggingFace CLI: `pip install -U "huggingface_hub[cli]"`
4. Clone LARQL if not already present: `git clone https://github.com/chrishayuk/larql.git` (idempotent: check `[ -d larql ]` first)
5. Compile in release mode: `cd larql && cargo build --release`
6. Assert binary exists: `[ -f larql/target/release/larql ] || exit 1`

---

### Task 2 — Hydrate the Vector Index

**Goal:** Download the pre-compiled `gemma-3-4b-it-vindex` from HuggingFace Hub into `./data/gemma3-4b.vindex/`.

**Jules instruction:**  
Append to `setup_env.sh` (or create `fetch_vindex.sh`):
1. Create directory: `mkdir -p ./data`
2. Download: `huggingface-cli download chrishayuk/gemma-3-4b-it-vindex --local-dir ./data/gemma3-4b.vindex`
3. Verify: check that `./data/gemma3-4b.vindex/` is non-empty and contains at least one `.bin` file. Exit non-zero if not.

---

### Task 3 — Build the Server Orchestrator

**Goal:** A Python script that starts the LARQL server with correct flags and blocks until it is ready to serve requests.

**Jules instruction:**  
Write `server_manager.py`. Requirements:
- Use `subprocess.Popen` to launch: `./larql/target/release/larql serve ./data/gemma3-4b.vindex --port 8080 --ffn-only --release-mmap-after-request`
- Route stdout/stderr to a log file (`larql_server.log`)
- Implement `wait_for_ready(timeout_s=120)`: polls `GET http://localhost:8080/health` every 2 seconds; raises `TimeoutError` if not reachable within timeout
- Register `atexit` and `signal.SIGTERM` handlers that call `proc.terminate()` then `proc.wait(timeout=5)`
- Export `start_server()` and `stop_server()` functions for use by `batch_processor.py`

---

### Task 4 — Develop the Batch Processor

**Goal:** An `asyncio` client that reads `data/input/*.txt`, POSTs each to the LARQL server, and reconstructs the streamed response.

**Jules instruction:**  
Write `batch_processor.py`. Requirements:
- Use `aiohttp.ClientSession` with a connection timeout of 30 s
- `MAX_CONCURRENCY = 8` — enforce via `asyncio.Semaphore`
- For each input file, send:
  ```
  POST http://localhost:8080/v1/stream
  Content-Type: application/json
  Body: {"type": "generate", "prompt": "<file contents>", "max_tokens": 500}
  ```
- Read the streaming NDJSON response line-by-line; accumulate `frame["text"]` for all `{"type": "token"}` frames; stop on `{"type": "done"}`
- Return the reconstructed string and elapsed time

---

### Task 5 — Output Sink and Retry Logic

**Goal:** Persist results and handle transient server failures without aborting the full batch.

**Jules instruction:**  
Refine `batch_processor.py`:
- On success: write `output/<stem>.json`:
  ```json
  {
    "source": "<filename>",
    "prompt": "<original text>",
    "response": "<reconstructed string>",
    "tokens": <int>,
    "elapsed_s": <float>,
    "timestamp": "<ISO-8601>"
  }
  ```
- On `aiohttp.ClientConnectionError` or HTTP 5xx: implement exponential backoff retry — wait `2 ** attempt` seconds, max 3 attempts
- Log each retry with `logging.warning(f"Retry {attempt}/3 for {filename}: {err}")`
- After 3 failures: log `logging.error(...)` and write a `output/<stem>.error.json` with the failure metadata — **do not raise, do not abort the batch**
- Final summary: print total processed, succeeded, failed counts to stdout
