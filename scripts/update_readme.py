#!/usr/bin/env python3
"""Regenerate the command output blocks in README.md.

The README shows real output of ``inspect --demo``, ``explain --demo`` and
``stats --demo``. Blocks are delimited by::

    <!-- BEGIN demo:<name> -->
    ```text
    ...
    ```
    <!-- END demo:<name> -->

Run ``python scripts/update_readme.py`` (needs Triton, no GPU) to refresh them;
``--check`` exits 1 if the README is out of date. Timings, sizes, timestamps and
hashes change on every run, so ``tests/test_readme.py`` compares a normalised form.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DEMO_PATH = "/tmp/demo"
COMMANDS: dict[str, list[str]] = {
    "inspect": ["inspect", "--demo"],
    "explain": ["explain", "--demo"],
    "stats": ["stats", "--demo"],
}
_BLOCK = re.compile(
    r"(<!-- BEGIN demo:(?P<name>[a-z-]+) -->\n```text\n)(?P<body>.*?)(```\n<!-- END demo:(?P=name) -->)",
    re.DOTALL,
)
_STAGES = r"(?:frontend|ttir|ttgir|llir|ptx|cubin|store)"


def generate(name: str) -> str:
    """Run one demo command and return ``$ command`` followed by its output."""
    with tempfile.TemporaryDirectory(prefix="tck-readme-") as tmp:
        work = Path(tmp) / "demo"
        command = [*COMMANDS[name], "--demo-dir", str(work)]
        result = subprocess.run(
            [sys.executable, "-m", "triton_cachekit", *command],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")
        output = result.stdout.replace(str(work), DEMO_PATH)
    return f"$ triton-cachekit {' '.join(COMMANDS[name])}\n{output.rstrip()}\n"


def blocks(text: str) -> dict[str, str]:
    return {m.group("name"): m.group("body") for m in _BLOCK.finditer(text)}


def normalise(body: str) -> list[str]:
    """Reduce a block to the parts that must not change between runs."""
    lines = []
    for line in body.splitlines():
        line = re.sub(r"\b[A-Z2-7]{10}\b", "KEY", line)
        line = re.sub(r"\b\d+(?:\.\d+)? (?:us|ms|s)\b", "TIME", line)
        line = re.sub(r"\b\d+(?:\.\d+)? (?:B|KiB|MiB)\b", "SIZE", line)
        line = re.sub(r"\b\d{4}-\d\d-\d\d \d\d:\d\d\b", "DATE", line)
        line = re.sub(r"\b\d+%", "PCT", line)
        line = re.sub(r"\b\d+\.\d+\.\d+\b", "VERSION", line)
        line = re.sub(rf"^({_STAGES})\b", "STAGE", line)
        line = re.sub(rf"\b{_STAGES}$", "STAGE", line)
        line = re.sub(r"-{2,}", "--", line)
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)
    # stats rows are ordered by measured time, which is not stable
    return sorted(line for line in lines if line)


def update(text: str, names: list[str] | None = None) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in COMMANDS or (names and name not in names):
            return match.group(0)
        return match.group(1) + generate(name) + match.group(4)

    return _BLOCK.sub(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if out of date")
    args = parser.parse_args()
    current = README.read_text(encoding="utf-8")
    fresh = update(current)
    if args.check:
        old, new = blocks(current), blocks(fresh)
        stale = [name for name in COMMANDS if normalise(old.get(name, "")) != normalise(new[name])]
        if stale:
            print(f"README.md blocks out of date: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("README.md demo blocks are up to date.")
        return 0
    README.write_text(fresh, encoding="utf-8")
    print(f"updated {len(COMMANDS)} blocks in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
