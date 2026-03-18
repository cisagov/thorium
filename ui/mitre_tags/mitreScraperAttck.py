#!/usr/bin/env python3

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mitreattack.stix20 import MitreAttackData


MITRE_ATTACK_KILLCHAIN = "mitre-attack"
REL_TYPE_SUBTECHNIQUE_OF = "subtechnique-of"

ENTERPRISE_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)


@dataclass(frozen=True)
class Config:
    stix_url: str = ENTERPRISE_STIX_URL
    cache_file: Path = Path("enterprise-attack.json")
    out_file: Path = Path("attackTagsList.tags")


def download_if_needed(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    with urllib.request.urlopen(url) as resp:
        dst.write_bytes(resp.read())

    # Validate JSON early so failures are obvious
    json.loads(dst.read_text(encoding="utf-8"))
    return dst


def load_bundle(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def external_attack_id(stix_obj: dict) -> str:
    """Return ATT&CK external id (e.g., T1059 or T1059.001) or ''."""
    for ref in stix_obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"]
    return ""


def iter_attack_tactic_slugs(stix_obj: dict) -> Iterable[str]:
    """Yield tactic slugs like 'command-and-control' from kill_chain_phases."""
    for phase in stix_obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == MITRE_ATTACK_KILLCHAIN and phase.get("phase_name"):
            yield phase["phase_name"]


def build_tactic_display_by_slug(attack: MitreAttackData) -> dict[str, str]:
    """Map tactic slug -> official display name (fixes capitalization)."""
    m: dict[str, str] = {}
    for tac in attack.get_tactics(remove_revoked_deprecated=True):
        slug = tac.get("x_mitre_shortname")
        name = tac.get("name")
        if slug and name:
            m[slug] = name
    return m


def build_parent_name_by_child_id(bundle: dict, tech_by_stix_id: dict[str, dict]) -> dict[str, str]:
    """Map sub-technique stix id -> parent technique name via relationship objects."""
    m: dict[str, str] = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != REL_TYPE_SUBTECHNIQUE_OF:
            continue

        child_id = obj.get("source_ref")
        parent_id = obj.get("target_ref")
        if not child_id or not parent_id:
            continue

        parent = tech_by_stix_id.get(parent_id)
        parent_name = parent.get("name") if parent else None
        if parent_name:
            m[child_id] = parent_name

    return m


def tactic_fallback_title(slug: str) -> str:
    return slug.replace("-", " ").title()


def generate_tags(
    techniques: list[dict],
    tactic_display_by_slug: dict[str, str],
    parent_name_by_child_id: dict[str, str],
) -> set[str]:
    tags: set[str] = set()

    for tech in techniques:
        stix_id = tech.get("id")
        name = tech.get("name", "")
        tid = external_attack_id(tech)
        if not stix_id or not name or not tid:
            continue

        is_sub = bool(tech.get("x_mitre_is_subtechnique", False))
        parent_name = parent_name_by_child_id.get(stix_id, "")

        for slug in iter_attack_tactic_slugs(tech):
            tactic = tactic_display_by_slug.get(slug) or tactic_fallback_title(slug)

            if is_sub and parent_name:
                tags.add(f"{tactic}::{parent_name}::{name} {tid}")
            else:
                tags.add(f"{tactic}::{name} {tid}")

    return tags


def write_sorted_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in sorted(lines):
            f.write(line + "\n")


def main(cfg: Config = Config()) -> None:
    stix_path = download_if_needed(cfg.stix_url, cfg.cache_file)

    attack = MitreAttackData(str(stix_path))
    techniques = attack.get_techniques(remove_revoked_deprecated=True)
    tech_by_stix_id = {t["id"]: t for t in techniques if t.get("id")}

    bundle = load_bundle(stix_path)
    tactic_display_by_slug = build_tactic_display_by_slug(attack)
    parent_name_by_child_id = build_parent_name_by_child_id(bundle, tech_by_stix_id)

    tags = generate_tags(techniques, tactic_display_by_slug, parent_name_by_child_id)
    write_sorted_lines(cfg.out_file, tags)


if __name__ == "__main__":
    main()