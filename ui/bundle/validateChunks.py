#!/usr/bin/env python3
"""
validate_chunks.py

Validate that the default vendors chunk does NOT overlap with any other chunk.

Input: JSON produced by your chunk generator, shaped like:
{
  "vendors": ["pkg-a", "pkg-b", ...],
  "tools":   [...],
  "graph":   [...],
  ...
}

Exit codes:
  0 = OK (no overlaps with vendors; optionally no overlaps anywhere if --all)
  1 = overlaps found / invalid input

Usage:
  python3 validate_chunks.py chunks.json
  python3 validate_chunks.py chunks.json --all
  python3 validate_chunks.py chunks.json --fail-fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_chunks(path: Path) -> Dict[str, Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Chunk file must be a JSON object mapping chunk -> list[str].")

    out: Dict[str, Set[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError("Chunk keys must be strings.")
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f'Chunk "{k}" must be a JSON array of strings.')
        out[k] = set(v)
    return out


def overlaps(a: Set[str], b: Set[str]) -> Set[str]:
    return a.intersection(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks_json", help="Path to chunks JSON")
    ap.add_argument("--vendors-name", default="vendors", help="Key name used for the vendors chunk")
    ap.add_argument("--all", action="store_true",
                    help="Also check overlaps between ALL non-vendor chunks (pairwise)")
    ap.add_argument("--fail-fast", action="store_true", help="Exit on first overlap found")
    ap.add_argument("--show", type=int, default=200, help="Max overlap entries to print per pair")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.chunks_json))

    vendors_key = args.vendors_name
    if vendors_key not in chunks:
        print(f'ERROR: vendors chunk "{vendors_key}" not found in JSON keys: {sorted(chunks.keys())}', file=sys.stderr)
        return 1

    vendors = chunks[vendors_key]
    others = {k: v for k, v in chunks.items() if k != vendors_key}

    bad = False

    # Check vendors vs each other chunk
    for name, pkgs in sorted(others.items()):
        ov = overlaps(vendors, pkgs)
        if ov:
            bad = True
            print(f"[OVERLAP] {vendors_key} ∩ {name}: {len(ov)} package(s)", file=sys.stderr)
            for p in sorted(ov)[: args.show]:
                print(f"  - {p}", file=sys.stderr)
            if len(ov) > args.show:
                print(f"  ... (truncated, showing first {args.show})", file=sys.stderr)
            if args.fail_fast:
                return 1

    # Optionally check overlaps among all non-vendor chunks too
    if args.all:
        names = sorted(others.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ov = overlaps(others[a], others[b])
                if ov:
                    bad = True
                    print(f"[OVERLAP] {a} ∩ {b}: {len(ov)} package(s)", file=sys.stderr)
                    for p in sorted(ov)[: args.show]:
                        print(f"  - {p}", file=sys.stderr)
                    if len(ov) > args.show:
                        print(f"  ... (truncated, showing first {args.show})", file=sys.stderr)
                    if args.fail_fast:
                        return 1

    if bad:
        return 1

    print("OK: no overlaps found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
