"""batch_processor.py — Task 4 + Task 5
Async batch inference client for the LARQL server.

Usage:
    python batch_processor.py [--input data/input] [--output output] [--concurrency 8]
                              [--temperature 0.0] [--max-tokens 500]
                              [--stop '\\n' '.'] [--logit-lens]

Notes on temperature / sampling:
    temperature=0.0  → deterministic argmax (default, recommended for batch jobs)
    temperature>0.0  → softmax sampling; increases diversity but reduces reproducibility

Notes on --logit-lens:
    When enabled, each prompt is first sent to /v1/walk to retrieve per-layer
    top-predicted tokens (the evolving logit-lens view of the residual stream).
    The walk result is stored alongside the generation output in the JSON sink.
    This is a debug/inspection mode — it roughly doubles latency per file.
    Only enable it when diagnosing unexpected outputs.
"""
import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [batch] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:8080"
STREAM_ENDPOINT = f"{SERVER_URL}/v1/stream"
WALK_ENDPOINT = f"{SERVER_URL}/v1/walk"     # logit-lens / feature walk
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 300  # long-running inference on CPU
MAX_RETRIES = 3
BASE_RETRY_WAIT_S = 2


# ── Core inference call ────────────────────────────────────────────────────

async def infer(
    session: aiohttp.ClientSession,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
) -> tuple[str, int]:
    """POST to /v1/stream, reconstruct NDJSON token stream.

    The payload now includes:
      - temperature: controls sampling sharpness.
          0.0 → argmax (deterministic).  >0.0 → softmax sampling.
      - stop: optional list of stop sequences.  The engine halts generation
          when any sequence is matched in the decoded output, preventing the
          model from running to max_tokens when a natural boundary is hit.

    Returns (response_text, token_count).
    """
    payload: dict = {
        "type": "generate",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop

    response_parts: list[str] = []
    token_count = 0

    timeout = aiohttp.ClientTimeout(connect=CONNECT_TIMEOUT_S, total=READ_TIMEOUT_S)
    async with session.post(STREAM_ENDPOINT, json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                log.debug("Non-JSON line skipped: %r", line)
                continue

            if frame.get("type") == "token":
                response_parts.append(frame.get("text", ""))
                token_count += 1
            elif frame.get("type") == "done":
                break

    return "".join(response_parts), token_count


# ── Optional logit-lens walk ───────────────────────────────────────────────

async def walk_logit_lens(
    session: aiohttp.ClientSession,
    prompt: str,
    top_k: int = 5,
) -> list[dict] | None:
    """POST to /v1/walk to get per-layer top-predicted tokens (logit lens).

    This exposes how the residual stream evolves layer-by-layer before the
    final unembedding.  The response is expected to be a JSON array of layer
    objects, each containing the top-k token predictions at that depth.

    Returns the parsed layer list, or None if the endpoint is unavailable
    (older larql builds may not expose /v1/walk as REST — fall back silently).
    """
    payload = {"prompt": prompt, "top_k": top_k}
    timeout = aiohttp.ClientTimeout(connect=CONNECT_TIMEOUT_S, total=READ_TIMEOUT_S)
    try:
        async with session.post(WALK_ENDPOINT, json=payload, timeout=timeout) as resp:
            if resp.status == 404:
                log.debug("/v1/walk not available on this build — skipping logit lens.")
                return None
            resp.raise_for_status()
            body = await resp.json(content_type=None)
            # body is expected to be a list of layer dicts
            return body if isinstance(body, list) else body.get("layers")
    except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError) as exc:
        log.debug("logit-lens walk failed (%s) — continuing without it.", exc)
        return None


# ── Per-file orchestration ─────────────────────────────────────────────────

async def process_file(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    input_path: Path,
    output_dir: Path,
    *,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
    logit_lens: bool,
) -> bool:
    """Read one .txt file, infer, optionally walk, write JSON result."""
    prompt = input_path.read_text(encoding="utf-8").strip()
    stem = input_path.stem

    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            t0 = time.monotonic()
            try:
                # ── Generation ──────────────────────────────────────────
                response_text, token_count = await infer(
                    session, prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                elapsed = time.monotonic() - t0

                # ── Optional logit-lens walk ─────────────────────────────
                layers = None
                if logit_lens:
                    layers = await walk_logit_lens(session, prompt)

                # ── Output sink ──────────────────────────────────────────
                result: dict = {
                    "source": input_path.name,
                    "prompt": prompt,
                    "response": response_text,
                    "tokens": token_count,
                    "elapsed_s": round(elapsed, 3),
                    "temperature": temperature,
                    "stop": stop,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if layers is not None:
                    result["logit_lens"] = layers

                out_path = output_dir / f"{stem}.json"
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                log.info(
                    "OK  %s → %s  (%d tokens, %.1fs, temp=%.2f%s)",
                    input_path.name, out_path.name, token_count, elapsed, temperature,
                    ", +walk" if layers is not None else "",
                )
                return True

            except (aiohttp.ClientConnectionError, aiohttp.ServerConnectionError) as exc:
                wait = BASE_RETRY_WAIT_S ** attempt
                log.warning(
                    "Retry %d/%d for %s after connection error: %s (wait %ds)",
                    attempt, MAX_RETRIES, input_path.name, exc, wait,
                )
                await asyncio.sleep(wait)

            except aiohttp.ClientResponseError as exc:
                if exc.status >= 500:
                    wait = BASE_RETRY_WAIT_S ** attempt
                    log.warning(
                        "Retry %d/%d for %s after HTTP %d (wait %ds)",
                        attempt, MAX_RETRIES, input_path.name, exc.status, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error("HTTP %d for %s — not retrying.", exc.status, input_path.name)
                    break

    # ── All retries exhausted ────────────────────────────────────────────────
    log.error("FAIL %s — max retries exceeded.", input_path.name)
    error_path = output_dir / f"{stem}.error.json"
    error_path.write_text(
        json.dumps(
            {
                "source": input_path.name,
                "error": "max_retries_exceeded",
                "attempts": MAX_RETRIES,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return False


# ── Batch runner ───────────────────────────────────────────────────────────

async def run_batch(
    input_dir: Path,
    output_dir: Path,
    max_concurrency: int,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
    logit_lens: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.txt"))

    if not files:
        log.warning("No .txt files found in %s", input_dir)
        return

    log.info(
        "Starting batch: %d files, concurrency=%d, temp=%.2f, max_tokens=%d%s",
        len(files), max_concurrency, temperature, max_tokens,
        ", logit-lens ON" if logit_lens else "",
    )

    sem = asyncio.Semaphore(max_concurrency)
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_file(
                session, sem, f, output_dir,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                logit_lens=logit_lens,
            )
            for f in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    succeeded = sum(results)
    failed = len(results) - succeeded
    log.info(
        "Batch complete — total: %d, succeeded: %d, failed: %d",
        len(results), succeeded, failed,
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LARQL async batch processor")
    parser.add_argument("--input", default="data/input", help="Directory of .txt prompt files")
    parser.add_argument("--output", default="output", help="Directory for JSON output")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument(
        "--max-tokens", type=int, default=500, dest="max_tokens",
        help="Max tokens per response (default: 500)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. 0.0=deterministic argmax (default). >0.0=softmax sampling.",
    )
    parser.add_argument(
        "--stop", nargs="*", default=None,
        help="Stop sequences (space-separated). Generation halts on first match. "
             "Example: --stop '\\n' '.'",
    )
    parser.add_argument(
        "--logit-lens", action="store_true", dest="logit_lens",
        help="Also call /v1/walk per prompt to capture per-layer top token predictions. "
             "Debug mode — roughly doubles latency.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_batch(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            max_concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=args.stop,
            logit_lens=args.logit_lens,
        )
    )


if __name__ == "__main__":
    main()
