"""lql_repl.py
Interactive LQL REPL and script runner for the LARQL query engine.

Usage:
    python lql_repl.py                        # interactive mode
    python lql_repl.py --script demo_queries.lql  # run a .lql script file
    python lql_repl.py --vindex ./data/my.vindex  # override vindex path

The LARQL CLI is launched as a persistent subprocess. Queries are sent
line-by-line via stdin and responses read from stdout. Multi-line queries
(ending with ;) are buffered and sent as a single block.

LQL Quick Reference:
    WALK "<prompt>" TOP <k>;                          -- feature activation walk
    WALK "<prompt>" TOP <k> LAYERS <start> TO <end>;  -- layer-scoped walk
    PROBE "<a>" vs "<b>" vs "<c>" AT LAYER <n>;       -- concept comparison
    INFER "<prompt>" TOP <k>;                         -- next-token prediction
    INSERT INTO EDGES (entity, relation, target)      -- knowledge edit
        VALUES ("<e>", "<r>", "<t>");
    SHOW LAYERS;                                      -- model metadata
    SHOW FEATURES AT LAYER <n> TOP <k>;               -- feature inventory
"""
import argparse
import subprocess
import sys
import threading
from pathlib import Path

LARQL_BINARY = "./larql/target/release/larql"
DEFAULT_VINDEX = "./data/gemma3-4b.vindex"
PROMPT = "lql> "


def launch_engine(vindex_path: str) -> subprocess.Popen:
    """Start the larql query subprocess."""
    binary = Path(LARQL_BINARY)
    if not binary.exists():
        print(f"ERROR: LARQL binary not found at {LARQL_BINARY}")
        print("Run: bash setup_env.sh")
        sys.exit(1)

    vindex = Path(vindex_path)
    if not vindex.exists():
        print(f"ERROR: vindex not found at {vindex_path}")
        print("Run: bash fetch_vindex.sh")
        sys.exit(1)

    cmd = [str(binary), "query", str(vindex)]
    print(f"[lql_repl] Launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Stream stderr to terminal so startup messages / errors are visible
    def _drain_stderr():
        for line in proc.stderr:
            print(f"[larql] {line}", end="", flush=True)

    threading.Thread(target=_drain_stderr, daemon=True).start()
    return proc


def send_query(proc: subprocess.Popen, query: str) -> str:
    """Send one LQL query (must end with ;) and collect the response."""
    proc.stdin.write(query.strip() + "\n")
    proc.stdin.flush()

    lines = []
    # Read until we hit the next prompt marker or an empty line after output
    # Adjust the sentinel below if the larql CLI uses a different prompt string
    for line in proc.stdout:
        if line.strip() in ("", "lql>", ">", "larql>"):
            break
        lines.append(line)
    return "".join(lines)


def run_script(proc: subprocess.Popen, script_path: str) -> None:
    """Read a .lql file and execute each ; -terminated statement."""
    text = Path(script_path).read_text(encoding="utf-8")
    # Strip comments and split on semicolons
    statements = []
    buf = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buf.append(stripped)
        if stripped.endswith(";"):
            statements.append(" ".join(buf))
            buf = []

    print(f"[lql_repl] Running {len(statements)} queries from {script_path}\n")
    for i, stmt in enumerate(statements, 1):
        print(f"--- Query {i}/{len(statements)} ---")
        print(f"{PROMPT}{stmt}")
        result = send_query(proc, stmt)
        print(result)
        print()


def run_interactive(proc: subprocess.Popen) -> None:
    """REPL loop: buffer input until ; then send."""
    print("LARQL LQL Interactive Shell")
    print("Type LQL queries ending with ; to execute. Ctrl-C or 'exit;' to quit.")
    print("Example: WALK \"The capital of France is\" TOP 10;\n")

    buf = []
    try:
        while True:
            leader = PROMPT if not buf else "...   "
            try:
                line = input(leader)
            except EOFError:
                break

            stripped = line.strip()
            if stripped.lower() in ("exit;", "quit;", "exit", "quit"):
                break
            if stripped.startswith("--") or not stripped:
                continue

            buf.append(stripped)

            if stripped.endswith(";"):
                query = " ".join(buf)
                buf = []
                result = send_query(proc, query)
                print(result)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="LQL REPL for LARQL")
    parser.add_argument("--vindex", default=DEFAULT_VINDEX, help="Path to .vindex directory")
    parser.add_argument("--script", default=None, help="Path to a .lql script file to run non-interactively")
    args = parser.parse_args()

    proc = launch_engine(args.vindex)

    if args.script:
        run_script(proc, args.script)
        proc.terminate()
        proc.wait(timeout=5)
    else:
        run_interactive(proc)


if __name__ == "__main__":
    main()
