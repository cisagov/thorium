#!/usr/bin/env python3
"""
chunk_cycle_debug.py

Troubleshoot circular chunk dependency warnings from Vite/Rollup manual chunks.

Given:
  - chunks.json (chunkName -> [packageName, ...])
  - an entry module list per chunk (optional, but helpful)
  - your project node_modules

This script builds a *package-level* dependency graph from installed manifests
(node_modules/<pkg>/package.json), then collapses it to a *chunk-level* graph:
  chunk A -> chunk B if any package in A depends on any package in B.

It then finds cycles (SCCs) and prints concrete edges that cause them, e.g.:
  tools (react-markdown) -> vendors (vfile)
  vendors (vfile) -> tools (react-markdown)

Usage:
  python3 chunk_cycle_debug.py chunks.json --project .
  python3 chunk_cycle_debug.py chunks.json --project . --show-edges 50
  python3 chunk_cycle_debug.py chunks.json --project . --only tools vendors
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pkg_manifest_path(node_modules: Path, pkg_name: str) -> Path:
    return node_modules.joinpath(*pkg_name.split("/"), "package.json")


def read_manifest(node_modules: Path, pkg_name: str) -> Optional[dict]:
    p = pkg_manifest_path(node_modules, pkg_name)
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:
        return None


def manifest_deps(manifest: dict) -> Set[str]:
    # For chunk dependency reasoning, "dependencies" is usually what matters.
    # If you want to include optional/peer edges, extend this.
    return set((manifest.get("dependencies") or {}).keys())


def invert_chunks(chunks: Dict[str, List[str]]) -> Dict[str, str]:
    """
    package -> chunk
    If a package appears in multiple chunks, we keep the first and warn.
    """
    pkg_to_chunk: Dict[str, str] = {}
    for chunk, pkgs in chunks.items():
        for p in pkgs:
            if p in pkg_to_chunk and pkg_to_chunk[p] != chunk:
                # This is already a red flag for manual chunk logic.
                print(f'WARNING: package "{p}" appears in multiple chunks: '
                      f'{pkg_to_chunk[p]} and {chunk}', file=sys.stderr)
                continue
            pkg_to_chunk[p] = chunk
    return pkg_to_chunk


@dataclass(frozen=True)
class Edge:
    src_chunk: str
    dst_chunk: str
    src_pkg: str
    dst_pkg: str


def build_chunk_edges(
    node_modules: Path,
    chunks: Dict[str, List[str]],
) -> Tuple[Dict[str, Set[str]], List[Edge]]:
    """
    Returns:
      chunk_graph: chunk -> set(otherChunks it depends on)
      witness_edges: concrete package->package edges crossing chunks
    """
    pkg_to_chunk = invert_chunks(chunks)

    chunk_graph: Dict[str, Set[str]] = {c: set() for c in chunks.keys()}
    witness: List[Edge] = []

    for src_chunk, pkgs in chunks.items():
        for src_pkg in pkgs:
            m = read_manifest(node_modules, src_pkg)
            if not m:
                continue
            for dep in manifest_deps(m):
                dst_chunk = pkg_to_chunk.get(dep)
                if not dst_chunk or dst_chunk == src_chunk:
                    continue
                chunk_graph[src_chunk].add(dst_chunk)
                witness.append(Edge(src_chunk, dst_chunk, src_pkg, dep))

    return chunk_graph, witness


def tarjan_scc(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Tarjan strongly connected components. Returns SCCs of size >= 2 (cycles)
    plus self-loop SCCs if present.
    """
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, set()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            # keep cycles: size>1 or self-loop
            if len(comp) > 1 or (len(comp) == 1 and v in graph.get(v, set())):
                sccs.append(comp)

    for v in graph.keys():
        if v not in indices:
            strongconnect(v)

    return sccs


def format_cycle_edges(
    scc: List[str],
    witness_edges: List[Edge],
    limit: int,
) -> List[str]:
    scc_set = set(scc)
    lines = []
    count = 0
    for e in witness_edges:
        if e.src_chunk in scc_set and e.dst_chunk in scc_set:
            lines.append(f"  {e.src_chunk} ({e.src_pkg}) -> {e.dst_chunk} ({e.dst_pkg})")
            count += 1
            if count >= limit:
                lines.append(f"  ... (truncated at {limit} edges)")
                break
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks_json", help="chunks.json (chunk -> [packages])")
    ap.add_argument("--project", default=".", help="Project directory containing node_modules")
    ap.add_argument("--show-edges", type=int, default=50, help="Max witness edges to show per cycle")
    ap.add_argument("--only", nargs="*", default=None, help="Only analyze these chunks (subset)")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    node_modules = project / "node_modules"
    if not node_modules.exists():
        print(f"ERROR: {node_modules} not found", file=sys.stderr)
        return 2

    chunks_raw = read_json(Path(args.chunks_json))
    if not isinstance(chunks_raw, dict):
        print("ERROR: chunks.json must be a JSON object mapping chunk->list", file=sys.stderr)
        return 2

    chunks: Dict[str, List[str]] = {}
    for k, v in chunks_raw.items():
        if not isinstance(k, str) or not isinstance(v, list):
            print(f"ERROR: invalid chunk entry {k}", file=sys.stderr)
            return 2
        chunks[k] = [x for x in v if isinstance(x, str)]

    if args.only is not None:
        keep = set(args.only)
        missing = keep - set(chunks.keys())
        if missing:
            print(f"ERROR: requested chunks not found: {sorted(missing)}", file=sys.stderr)
            return 2
        chunks = {k: v for k, v in chunks.items() if k in keep}

    chunk_graph, witness = build_chunk_edges(node_modules, chunks)

    # Print a quick adjacency summary
    print("Chunk dependency edges (A -> B means A depends on B):")
    for a in sorted(chunk_graph.keys()):
        outs = sorted(chunk_graph[a])
        if outs:
            print(f"  {a} -> {', '.join(outs)}")
        else:
            print(f"  {a} -> (none)")
    print()

    sccs = tarjan_scc(chunk_graph)
    if not sccs:
        print("No chunk cycles detected in manifest-derived graph.")
        return 0

    print("Detected chunk cycles (SCCs):")
    for comp in sccs:
        comp_sorted = sorted(comp)
        print(f"- Cycle group: {', '.join(comp_sorted)}")
        for line in format_cycle_edges(comp, witness, args.show_edges):
            print(line)
        print()

    print("Interpretation / next steps:")
    print("  - Any edge shown above is a package in one chunk that declares a dependency")
    print("    on a package assigned to another chunk.")
    print("  - To break a tools <-> vendors cycle, move the *shared* packages causing")
    print("    the back-edge into vendors (or into a new shared chunk), or ensure tools")
    print("    only contains packages that do not depend on vendors-assigned packages.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
