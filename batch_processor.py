"""batch_processor.py — Task 4 + Task 5
Async batch inference client for the LARQL server.

Usage:
    python batch_processor.py [--input data/input] [--output output] [--concurrency 8]
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
MAX_TOKENS = 500
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 300  # long-running inference on CPU
MAX_RETRIES = 3
BASE_RETRY_WAIT_S = 2


async def infer(session: aiohttp.ClientSession, prompt: str) -> tuple[str, int]:
    """POST to /v1/stream, reconstruct NDJSON token stream.
    Returns (response_text, token_count).
    """
    payload = {"type": "generate", "prompt": prompt, "max_tokens": MAX_TOKENS}

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


async def process_file(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    input_path: Path,
    output_dir: Path,
) -> bool:
    """Read one .txt file, infer, write JSON result. Returns True on success."""
    prompt = input_path.read_text(encoding="utf-8").strip()
    stem = input_path.stem

    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            t0 = time.monotonic()
            try:
                response_text, token_count = await infer(session, prompt)
                elapsed = time.monotonic() - t0

                result = {
                    "source": input_path.name,
                    "prompt": prompt,
                    "response": response_text,
                    "tokens": token_count,
                    "elapsed_s": round(elapsed, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                out_path = output_dir / f"{stem}.json"
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                log.info("OK  %s → %s  (%d tokens, %.1fs)", input_path.name, out_path.name, token_count, elapsed)
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
                    # 4xx — don't retry
                    log.error("HTTP %d for %s — not retrying.", exc.status, input_path.name)
                    break

    # All retries exhausted
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


async def run_batch(input_dir: Path, output_dir: Path, max_concurrency: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.txt"))

    if not files:
        log.warning("No .txt files found in %s", input_dir)
        return

    log.info("Starting batch: %d files, concurrency=%d", len(files), max_concurrency)
    sem = asyncio.Semaphore(max_concurrency)

    connector = aiohttp.TCPConnector(limit=max_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_file(session, sem, f, output_dir)
            for f in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    succeeded = sum(results)
    failed = len(results) - succeeded
    log.info(
        "Batch complete — total: %d, succeeded: %d, failed: %d",
        len(results), succeeded, failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LARQL async batch processor")
    parser.add_argument("--input", default="data/input", help="Directory of .txt prompt files")
    parser.add_argument("--output", default="output", help="Directory for JSON output")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(
        run_batch(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            max_concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
