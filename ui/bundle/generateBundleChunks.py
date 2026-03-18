#!/usr/bin/env python3
"""
generateBundleChunks.py  (hybrid: npm-ls universe + node_modules manifest edges)

What this does:
1) Uses `npm ls --all --json --omit=dev` (unless --include-dev) to get the
   *production* universe of installed packages (prevents dev toolchain noise).
2) Uses package.json top-level deps as seeds for non-default chunk groups.
3) Computes each group's transitive closure by reading installed manifests:
      node_modules/<pkg>/package.json  (dependencies + optional/peer if enabled)
   but ONLY traversing packages that are in the npm-ls universe.
   (This captures hoisted deps accurately while staying in prod universe.)
4) "vendors" wins on overlap among non-default groups:
      - pkg in >=2 closures => vendors
      - pkg in exactly 1 closure => assigned to that group
      - vendors = universe - uniquely_assigned
5) Cycle detection now uses the same manifest-derived dependency edges to build
   a chunk->chunk graph and find SCC cycles. If a cycle involves any non-vendor
   chunk, those chunks are moved into vendors.

Troubleshooting:
- --print-stats: closure sizes, shared count, vendors size
- --print-overlap: top shared packages and which closures contain them
- --check PKG: where PKG ended up + which closures it appears in
- --print-cycles: SCC cycles + witness edges that create cross-chunk deps
- --cycle-witness-limit: witness print limit per SCC
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------- Non-default chunks (only these are specified) --------------

NON_DEFAULT_GROUPS: List[Tuple[str, List[str]]] = [
    ("graph", [
        r"^cytoscape($|-)",
        r"^@headless-tree($|-|/)",
    ]),
    ("react", [
        r"^react$",
        r"^react-dom$",
        r"^react-is$",
        r"^react-router(-dom)?$",
        r"^scheduler$",
        r"^classnames$",
    ]),
    ("core-js", [
        r"^core-js$",
    ]),
    ("vendors-tools", [
        #r"^remark-gfm$", # used by pipeline descriptions, tools
        #r"^react-markdown$", # used by pipeline descriptions, tools
        r"^react-xml-viewer$",
        r"^react-json-tree$",
        r"^react-syntax-highlighter$", # has babel handler dep
        #r"^sanitize-html$", # has overlapping dependency from domhandler
    ]),
    ("date", [
        r"^react-datepicker$",
        r"^date-fns$",
    ]),
]

DEFAULT_GROUP = "vendors"


# ---------------- Helpers: package.json + npm ls ----------------------------

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_groups(groups_spec) -> List[Tuple[str, List[re.Pattern]]]:
    return [(name, [re.compile(p) for p in pats]) for name, pats in groups_spec]


def match_group(dep_name: str, compiled_groups) -> Optional[str]:
    for gname, pats in compiled_groups:
        if any(p.search(dep_name) for p in pats):
            return gname
    return None


def top_level_deps(pkg: dict, include_dev: bool) -> Set[str]:
    deps = set((pkg.get("dependencies") or {}).keys())
    deps |= set((pkg.get("optionalDependencies") or {}).keys())
    if include_dev:
        deps |= set((pkg.get("devDependencies") or {}).keys())
    return deps


def run_npm_ls(project: Path, include_dev: bool) -> dict:
    cmd = ["npm", "ls", "--all", "--json"]
    if not include_dev:
        cmd.append("--omit=dev")
    p = subprocess.run(cmd, cwd=project, text=True, capture_output=True)
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError(f"`{' '.join(cmd)}` returned empty stdout.\nstderr:\n{p.stderr}")
    return json.loads(out)


def npm_ls_universe(npm_tree: dict) -> Set[str]:
    """
    Universe of package names in npm ls tree (excluding project root).
    """
    out: Set[str] = set()

    def walk(node: dict):
        deps = node.get("dependencies") or {}
        for name, child in deps.items():
            out.add(name)
            if isinstance(child, dict):
                walk(child)

    walk(npm_tree)
    return out


# ---------------- Installed manifest traversal (restricted to universe) ------

def pkg_manifest_path(node_modules: Path, pkg_name: str) -> Path:
    return node_modules.joinpath(*pkg_name.split("/"), "package.json")


def read_installed_manifest(node_modules: Path, pkg_name: str) -> Optional[dict]:
    p = pkg_manifest_path(node_modules, pkg_name)
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:
        return None


def manifest_deps(manifest: dict, include_optional: bool, include_peer: bool) -> Set[str]:
    deps = set((manifest.get("dependencies") or {}).keys())
    if include_optional:
        deps |= set((manifest.get("optionalDependencies") or {}).keys())
    if include_peer:
        deps |= set((manifest.get("peerDependencies") or {}).keys())
    return deps


def closure_from_seeds_manifest(
    node_modules: Path,
    seeds: Iterable[str],
    universe: Set[str],
    include_optional: bool,
    include_peer: bool,
) -> Set[str]:
    """
    Traverse installed package manifests to build transitive closure starting
    from `seeds`, but only include/traverse packages in `universe`.
    """
    seen: Set[str] = set()
    q = deque([s for s in seeds if s in universe])

    while q:
        pkg = q.popleft()
        if pkg in seen:
            continue
        seen.add(pkg)

        m = read_installed_manifest(node_modules, pkg)
        if not m:
            continue

        for dep in manifest_deps(m, include_optional, include_peer):
            if dep in universe and dep not in seen:
                q.append(dep)

    return seen


# ---------------- Cycle detection on chunk graph (manifest edges) ------------

@dataclass(frozen=True)
class WitnessEdge:
    src_chunk: str
    dst_chunk: str
    src_pkg: str
    dst_pkg: str


def tarjan_scc(graph: Dict[str, Set[str]]) -> List[List[str]]:
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
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or (len(comp) == 1 and v in graph.get(v, set())):
                sccs.append(comp)

    for v in graph.keys():
        if v not in indices:
            strongconnect(v)

    return sccs


def build_chunk_graph_and_witnesses_manifest(
    node_modules: Path,
    chunks: Dict[str, Set[str]],
    universe: Set[str],
    include_optional: bool,
    include_peer: bool,
) -> Tuple[Dict[str, Set[str]], List[WitnessEdge]]:
    """
    chunk A -> chunk B if any pkg in A depends on any pkg in B (manifest deps).
    """
    pkg_to_chunk: Dict[str, str] = {}
    for c, pkgs in chunks.items():
        for p in pkgs:
            pkg_to_chunk[p] = c

    graph: Dict[str, Set[str]] = {c: set() for c in chunks.keys()}
    witnesses: List[WitnessEdge] = []

    for src_chunk, pkgs in chunks.items():
        for src_pkg in pkgs:
            if src_pkg not in universe:
                continue
            m = read_installed_manifest(node_modules, src_pkg)
            if not m:
                continue
            for dep in manifest_deps(m, include_optional, include_peer):
                if dep not in universe:
                    continue
                dst_chunk = pkg_to_chunk.get(dep)
                if dst_chunk and dst_chunk != src_chunk:
                    graph[src_chunk].add(dst_chunk)
                    witnesses.append(WitnessEdge(src_chunk, dst_chunk, src_pkg, dep))

    return graph, witnesses


def witnesses_for_component_prioritized(
    comp: Set[str],
    witnesses: List[WitnessEdge],
    vendors_name: str,
    limit: int,
) -> List[WitnessEdge]:
    in_comp = [w for w in witnesses if w.src_chunk in comp and w.dst_chunk in comp]
    v_to = [w for w in in_comp if w.src_chunk == vendors_name and w.dst_chunk != vendors_name]
    to_v = [w for w in in_comp if w.src_chunk != vendors_name and w.dst_chunk == vendors_name]

    picked: List[WitnessEdge] = []
    seen = set()

    def add_list(lst: List[WitnessEdge]) -> bool:
        for w in lst:
            key = (w.src_chunk, w.dst_chunk, w.src_pkg, w.dst_pkg)
            if key in seen:
                continue
            seen.add(key)
            picked.append(w)
            if len(picked) >= limit:
                return True
        return False

    if add_list(v_to):
        return picked
    if add_list(to_v):
        return picked
    add_list(in_comp)
    return picked[:limit]


# ---------------- Main ------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".", help="Project directory containing package.json")
    ap.add_argument("--include-dev", action="store_true",
                    help="Include devDependencies and run npm ls without --omit=dev (usually not desired)")
    ap.add_argument("--include-optional", action="store_true", help="Follow optionalDependencies transitively")
    ap.add_argument("--include-peer", action="store_true", help="Follow peerDependencies transitively (often noisy)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ap.add_argument("--print-seeds", action="store_true", help="Print top-level seeds per group to stderr")
    ap.add_argument("--print-stats", action="store_true", help="Print closure sizes, shared counts, vendors size")
    ap.add_argument("--print-overlap", action="store_true", help="Print top shared packages and which groups contain them")
    ap.add_argument("--overlap-top", type=int, default=60, help="How many shared packages to print")
    ap.add_argument("--check", action="append", default=[],
                    help="Check a package: show which chunk it ended up in and which closures contained it")
    ap.add_argument("--print-cycles", action="store_true", help="Print SCC cycles + witness edges to stderr")
    ap.add_argument("--cycle-witness-limit", type=int, default=200, help="Max witness edges printed per SCC")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    pkg = read_json(project / "package.json")
    node_modules = project / "node_modules"

    compiled = compile_groups(NON_DEFAULT_GROUPS)
    top = top_level_deps(pkg, include_dev=args.include_dev)

    npm_tree = run_npm_ls(project, include_dev=args.include_dev)
    universe = npm_ls_universe(npm_tree)

    # Seeds: only top-level deps that match a non-default group pattern AND are in universe.
    seeds_by_group: Dict[str, Set[str]] = {g: set() for g, _ in NON_DEFAULT_GROUPS}
    for dep in sorted(top):
        if dep not in universe:
            continue
        g = match_group(dep, compiled)
        if g:
            seeds_by_group[g].add(dep)

    # Closures: manifest traversal restricted to universe (captures hoisting)
    closure_by_group: Dict[str, Set[str]] = {}
    for gname, _ in NON_DEFAULT_GROUPS:
        closure_by_group[gname] = closure_from_seeds_manifest(
            node_modules=node_modules,
            seeds=seeds_by_group[gname],
            universe=universe,
            include_optional=args.include_optional,
            include_peer=args.include_peer,
        )

    # Shared counts (overlap among non-default groups)
    counts: Counter[str] = Counter()
    for pkgs in closure_by_group.values():
        counts.update(pkgs)

    # Assign uniquely to non-default; shared goes to vendors
    non_default_chunks: Dict[str, Set[str]] = {g: set() for g, _ in NON_DEFAULT_GROUPS}
    assigned_non_default: Set[str] = set()
    for gname, pkgs in closure_by_group.items():
        for p in pkgs:
            if counts[p] == 1:
                non_default_chunks[gname].add(p)
                assigned_non_default.add(p)

    vendors: Set[str] = set(p for p in universe if p not in assigned_non_default)

    # Cycle detection (manifest-based chunk graph)
    chunks_all: Dict[str, Set[str]] = dict(non_default_chunks)
    chunks_all[DEFAULT_GROUP] = set(vendors)

    graph, witnesses = build_chunk_graph_and_witnesses_manifest(
        node_modules=node_modules,
        chunks=chunks_all,
        universe=universe,
        include_optional=args.include_optional,
        include_peer=args.include_peer,
    )
    sccs = tarjan_scc(graph)

    cycle_chunks_to_vendor: Set[str] = set()
    cycle_components: List[Set[str]] = []
    for comp in sccs:
        comp_set = set(comp)
        if len(comp_set) < 2:
            continue
        if any(c != DEFAULT_GROUP for c in comp_set):
            cycle_components.append(comp_set)
            cycle_chunks_to_vendor |= comp_set
    cycle_chunks_to_vendor.discard(DEFAULT_GROUP)

    moved_packages: Dict[str, List[str]] = {}
    if cycle_chunks_to_vendor:
        for c in sorted(cycle_chunks_to_vendor):
            pkgs = sorted(non_default_chunks.get(c, set()))
            if not pkgs:
                continue
            for p in pkgs:
                vendors.add(p)
            moved_packages[c] = pkgs
            non_default_chunks[c].clear()

    # Final mapping
    final: Dict[str, Set[str]] = dict(non_default_chunks)
    final[DEFAULT_GROUP] = vendors

    # Diagnostics
    if args.print_seeds or args.print_stats or args.print_overlap or args.print_cycles or args.check:
        import sys

        if args.print_seeds:
            for gname, _ in NON_DEFAULT_GROUPS:
                sys.stderr.write(f"{gname} seeds (top-level):\n")
                for s in sorted(seeds_by_group[gname]):
                    sys.stderr.write(f"  - {s}\n")
                sys.stderr.write("\n")

        if args.print_stats:
            shared = sorted([p for p, c in counts.items() if c >= 2])
            sys.stderr.write("=== STATS ===\n")
            sys.stderr.write(f"top-level deps considered (package.json): {len(top)}\n")
            sys.stderr.write(f"production universe packages (npm ls): {len(universe)}\n\n")
            sys.stderr.write("closure sizes:\n")
            for gname, _ in NON_DEFAULT_GROUPS:
                sys.stderr.write(
                    f"  - {gname}: seeds={len(seeds_by_group[gname])}, "
                    f"closure={len(closure_by_group[gname])}, unique_assigned={len(non_default_chunks[gname])}\n"
                )
            sys.stderr.write(f"\nshared packages across non-default closures (count>=2): {len(shared)}\n")
            sys.stderr.write(f"vendors size: {len(vendors)}\n")
            sys.stderr.write("\n")

        if args.print_overlap:
            sys.stderr.write("=== OVERLAP (shared -> vendors candidates) ===\n")
            shared_ranked = sorted([(p, counts[p]) for p in counts if counts[p] >= 2],
                                   key=lambda x: (-x[1], x[0]))
            for p, c in shared_ranked[: args.overlap_top]:
                groups_in = [g for g, s in closure_by_group.items() if p in s]
                sys.stderr.write(f"  {c}  {p}  :: {', '.join(groups_in)}\n")
            sys.stderr.write("\n")

        for pkg_name in args.check:
            in_chunks = [c for c, pkgs in final.items() if pkg_name in pkgs]
            in_closures = [g for g, pkgs in closure_by_group.items() if pkg_name in pkgs]
            sys.stderr.write(f'CHECK "{pkg_name}":\n')
            sys.stderr.write(f"  in final chunks: {', '.join(in_chunks) if in_chunks else '(missing)'}\n")
            sys.stderr.write(f"  in group closures: {', '.join(in_closures) if in_closures else '(none)'}\n")
            if pkg_name in counts:
                sys.stderr.write(f"  overlap count across closures: {counts[pkg_name]}\n")
            sys.stderr.write("\n")

        if args.print_cycles:
            sys.stderr.write("=== SCC CYCLES (manifest-derived, restricted to npm-ls universe) ===\n")
            if not cycle_components:
                sys.stderr.write("  (none)\n\n")
            else:
                for comp_set in cycle_components:
                    sys.stderr.write(f"Cycle component: {', '.join(sorted(comp_set))}\n")
                    ws = witnesses_for_component_prioritized(
                        comp_set, witnesses, DEFAULT_GROUP, limit=args.cycle_witness_limit
                    )
                    if ws:
                        sys.stderr.write("  Witness edges (srcChunk(srcPkg) -> dstChunk(dstPkg)):\n")
                        for w in ws:
                            sys.stderr.write(f"    - {w.src_chunk}({w.src_pkg}) -> {w.dst_chunk}({w.dst_pkg})\n")
                        if len(ws) >= args.cycle_witness_limit:
                            sys.stderr.write("    ... (truncated)\n")
                    else:
                        sys.stderr.write("  (No witness edges captured)\n")
                    sys.stderr.write("\n")

                if moved_packages:
                    sys.stderr.write(f"Moved chunks to {DEFAULT_GROUP} due to cycles:\n")
                    for c in sorted(moved_packages.keys()):
                        sys.stderr.write(f"  - {c} -> {DEFAULT_GROUP}: moved {len(moved_packages[c])} package(s)\n")
                    sys.stderr.write("\n")

    out = {k: sorted(v) for k, v in final.items()}
    print(json.dumps(out, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
