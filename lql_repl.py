"""lql_repl.py
Interactive LQL REPL and script runner.

Usage:
    python lql_repl.py                            # interactive LQL REPL
    python lql_repl.py --script demo_queries.lql  # run a .lql script
    python lql_repl.py --vindex ./data/my.vindex   # override vindex path

Interactive mode wraps: larql repl <vindex>
Script mode runs each statement via: larql lql '<stmt>' --graph <vindex>

LQL Quick Reference (confirmed syntax):
    SELECT ?x WHERE { <subject> <relation> ?x }   -- graph fact lookup
    larql lql 'SELECT ...' --graph <vindex>        -- one-shot CLI form
    larql repl <vindex>                            -- interactive REPL
    larql shannon slot-probe <vindex> --prompt ... -- next-token probing
    larql dev walk --index <vindex> --prompt ...   -- FFN walk
"""
import argparse
import subprocess
import sys
from pathlib import Path

LARQL_BINARY = "./larql/target/release/larql"
DEFAULT_VINDEX = "./data/gemma3-4b.vindex"


def check_prerequisites(vindex_path: str) -> tuple[Path, Path]:
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
    return binary, vindex


def run_interactive(binary: Path, vindex: Path) -> None:
    """Launch larql repl <vindex> directly — takes over the terminal."""
    cmd = [str(binary), "repl", str(vindex)]
    print(f"[lql_repl] Launching interactive REPL: {' '.join(cmd)}")
    print("[lql_repl] Type LQL statements at the prompt. Ctrl-C or 'exit' to quit.\n")
    subprocess.run(cmd)  # hands terminal control to the child process


def run_script(binary: Path, vindex: Path, script_path: str) -> None:
    """Execute each ; -terminated statement via: larql lql '<stmt>' --graph <vindex>"""
    text = Path(script_path).read_text(encoding="utf-8")

    # Parse statements: strip comments, split on ;
    statements = []
    buf = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buf.append(stripped)
        if stripped.endswith(";"):
            stmt = " ".join(buf).rstrip(";")
            statements.append(stmt)
            buf = []

    print(f"[lql_repl] Running {len(statements)} statements from {script_path}\n")

    for i, stmt in enumerate(statements, 1):
        print(f"{'='*60}")
        print(f"[{i}/{len(statements)}] {stmt}")
        print(f"{'='*60}")
        cmd = [str(binary), "lql", stmt, "--graph", str(vindex)]
        result = subprocess.run(cmd, capture_output=False)  # output streams live to terminal
        if result.returncode != 0:
            print(f"[lql_repl] WARNING: statement exited {result.returncode}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LQL REPL / script runner for LARQL")
    parser.add_argument("--vindex", default=DEFAULT_VINDEX, help="Path to .vindex directory")
    parser.add_argument("--script", default=None, help=".lql script file to run non-interactively")
    args = parser.parse_args()

    binary, vindex = check_prerequisites(args.vindex)

    if args.script:
        run_script(binary, vindex, args.script)
    else:
        run_interactive(binary, vindex)


if __name__ == "__main__":
    main()
