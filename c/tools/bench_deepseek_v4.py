#!/usr/bin/env python3
"""Repeatable CLI throughput and reference-output comparison for DeepSeek-V4.

The runner measures the engine's own generated-token counter.  With
--cold-expert it asks the engine to issue POSIX DONTNEED after each streamed
expert read; this avoids relying on a privileged system-wide page-cache flush
and is also safe on Windows (where the hint is simply unavailable).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import time
from pathlib import Path


RATE = re.compile(r"generated\s+(\d+)\s+token\(s\)\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+tok/s\)")


def run_once(engine: Path, model: Path, prompt: str, tokens: int, ctx: int, cold: bool):
    env = dict(os.environ, SNAP=str(model), PROMPT=prompt, NGEN=str(tokens), CTX=str(ctx))
    if cold:
        env["DSV4_DROP_EXPERT_CACHE"] = "1"
    start = time.perf_counter()
    result = subprocess.run([str(engine), "0"], env=env, capture_output=True, text=True)
    wall = time.perf_counter() - start
    match = RATE.search(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or f"engine exited with {result.returncode}")
    return {
        "stdout": result.stdout,
        "tok_s": float(match.group(3)) if match else 0.0,
        "generated": int(match.group(1)) if match else 0,
        "wall_s": wall,
        "stderr": result.stderr,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=50000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cold-expert", action="store_true")
    ap.add_argument("--reference", type=Path, help="UTF-8 expected greedy output")
    args = ap.parse_args()
    runs = [run_once(args.engine, args.model, args.prompt, args.tokens, args.ctx, args.cold_expert)
            for _ in range(max(1, args.repeats))]
    rates = [x["tok_s"] for x in runs]
    report = {
        "mode": "cold-expert" if args.cold_expert else "normal",
        "repeats": len(runs),
        "tok_s": rates,
        "mean_tok_s": sum(rates) / len(rates),
        "outputs": [x["stdout"] for x in runs],
    }
    if args.reference:
        expected = args.reference.read_text(encoding="utf-8")
        comparisons = []
        for run in runs:
            actual = run["stdout"]
            comparisons.append({
                "exact": actual == expected,
                "sequence_ratio": difflib.SequenceMatcher(None, expected, actual).ratio(),
                "reference_chars": len(expected),
                "actual_chars": len(actual),
            })
        report["reference"] = str(args.reference)
        report["similarity"] = comparisons
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
