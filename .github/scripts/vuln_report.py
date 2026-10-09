#!/usr/bin/env python3
"""Prioritize Trivy vulnerabilities with EPSS and CISA KEV data.

Reads one or more `trivy image --format json` reports, or with `--by-lockfile`
`trivy fs --scanners vuln --format json` reports of the repo's dependency
lockfiles, and ranks every vulnerability by how urgently it should be fixed,
combining its severity with:

* EPSS, FIRST's daily estimate of the probability a CVE is exploited in the
  next 30 days, and the percentile placing it against every scored CVE
* the CISA Known Exploited Vulnerabilities catalog, CVEs with confirmed
  exploitation in the wild

Both feeds are downloaded in full (the EPSS CSV is about 3 MB gzipped and the
KEV catalog about 1.5 MB), so no CVE IDs, image names or other data about the
scanned images leave the runner. `--epss-file` and `--kev-file` read local
copies instead, for mirrors and offline runs, and `--offline` skips both. A
feed that cannot be fetched after retries is noted in the report, and findings
fall back to severity-only ranking; it never fails the run.

Each finding gets one of these tiers, using the same thresholds as OWASP
DockSec, plus KEV:

    in the KEV catalog                          -> Fix Now
    CRITICAL/HIGH and top 10% EPSS              -> Fix Now
    CRITICAL/HIGH                               -> Fix Soon
    MEDIUM/LOW/UNKNOWN and top 10% EPSS         -> Monitor
    MEDIUM/LOW/UNKNOWN                          -> Low Priority

A finding with no EPSS score that is not in KEV is left untiered, since a
missing score is not evidence of low risk.

Three files are written:

* `--sarif` - every finding, for .github/scripts/lint_report.py to report
  (`--tool trivy-image` or `--tool trivy-deps`) and for code scanning; results
  are attributed to line 1 of `--dockerfile`, or with `--by-lockfile` to the
  lockfile each came from, use the CVE as the rule id so
  .github/lint-ignore.toml can ignore one, and carry the severity as a rule tag
  like Trivy's own SARIF
* `--json` - every finding with its EPSS score, KEV status and tier
* `--markdown` - counts by tier and severity per image (or lockfile) and a
  table of the findings worth acting on (Fix Now, Fix Soon and Monitor), plus,
  for image scans run with `--list-all-pkgs`, each image's software bill of
  materials in brief (its OS and package counts by type), for the job summary
  and the report artifact

This script uses only the Python standard library, plus the retrying HTTP
helper beside it. Exit status is 0 whenever the reports were written, whatever
the findings, so vulnerabilities never fail a pipeline, and 2 when an input
cannot be read.
"""

import argparse
import csv
import gzip
import io
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

from http_retry import HttpError, get_bytes

# FIRST's daily EPSS scores for every CVE, linked from first.org/epss/data_stats
EPSS_URL = "https://epss.empiricalsecurity.com/epss_scores-current.csv.gz"
# the CISA Known Exploited Vulnerabilities catalog
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
# a CVE at or above this EPSS percentile is in the top 10% by exploitation likelihood
HIGH_EXPLOITATION_PERCENTILE = 0.90
# the tiers, most urgent first, with the SARIF level each finding gets
TIERS = {
    "fix-now": ("Fix Now", "error"),
    "fix-soon": ("Fix Soon", "warning"),
    "monitor": ("Monitor", "warning"),
    "low-priority": ("Low Priority", "note"),
    "untiered": ("Untiered", "note"),
}
SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")
# code scanning's security-severity score for each severity
SECURITY_SEVERITY = {"CRITICAL": "9.5", "HIGH": "8.0", "MEDIUM": "5.5", "LOW": "2.0", "UNKNOWN": "0.0"}
# the most findings listed in the markdown table
MARKDOWN_ROWS = 200


class InputError(Exception):
    """A Trivy report or feed file that cannot be read."""


def load_epss(data):
    """Parse the EPSS CSV into a map of CVE to its score and percentile.

    # Arguments

    * `data` - The CSV, gzipped or not, starting with a `#model_version` comment line

    Returns a `(scores, score_date)` tuple.
    """
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    text = io.StringIO(data.decode("utf-8"))
    # the first line is a comment such as `#model_version:v2025.03.14,score_date:2026-10-08T...`
    score_date = ""
    first = text.readline()
    if first.startswith("#"):
        for part in first[1:].strip().split(","):
            key, _, value = part.partition(":")
            if key == "score_date":
                score_date = value
    else:
        text.seek(0)
    scores = {}
    for row in csv.DictReader(text):
        try:
            scores[row["cve"].strip().upper()] = (float(row["epss"]), float(row["percentile"]))
        except (KeyError, ValueError, AttributeError):
            continue
    if not scores:
        raise ValueError("no EPSS scores found")
    return scores, score_date


def load_kev(data):
    """Parse the KEV catalog into a map of CVE to the date CISA added it.

    # Arguments

    * `data` - The catalog's JSON

    Returns a `(catalog, version)` tuple.
    """
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError("the KEV catalog is not a JSON object")
    catalog = {
        str(entry["cveID"]).strip().upper(): str(entry.get("dateAdded", ""))
        for entry in document.get("vulnerabilities", [])
        if isinstance(entry, dict) and entry.get("cveID")
    }
    if not catalog:
        raise ValueError("no KEV entries found")
    return catalog, str(document.get("catalogVersion", ""))


def fetch_feed(name, url, path, offline, parse):
    """Load one feed from a file or URL, degrading to no data on failure.

    # Arguments

    * `name` - The feed's name, for messages
    * `url` - The URL to download it from
    * `path` - A local copy to read instead, if given
    * `offline` - Whether to skip downloads
    * `parse` - The function turning the raw bytes into `(data, version)`

    Returns a `(data, version, status)` tuple, with empty data and the reason in
    `status` when the feed is unavailable.
    """
    try:
        if path:
            raw = Path(path).read_bytes()
            source = path
        elif offline:
            return {}, "", f"{name} skipped (--offline)"
        else:
            raw = get_bytes(url, timeout=60)
            source = url
        data, version = parse(raw)
    except (OSError, EOFError, zlib.error, HttpError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        print(f"WARNING: {name} unavailable, ranking without it: {error}", file=sys.stderr)
        return {}, "", f"{name} unavailable: {error}"
    return data, version, f"{name} {version} from {source}".replace("  ", " ")


def tier(severity, percentile, in_kev):
    """Choose a finding's tier.

    # Arguments

    * `severity` - Its Trivy severity
    * `percentile` - Its EPSS percentile, or None when unscored
    * `in_kev` - Whether it is in the KEV catalog
    """
    if in_kev:
        return "fix-now"
    if percentile is None:
        return "untiered"
    high_impact = severity in ("CRITICAL", "HIGH")
    high_exploitation = percentile >= HIGH_EXPLOITATION_PERCENTILE
    if high_impact:
        return "fix-now" if high_exploitation else "fix-soon"
    return "monitor" if high_exploitation else "low-priority"


def load_findings(path, epss, kev, name=None, by_lockfile=False):
    """Read one Trivy JSON report into a list of enriched findings.

    # Arguments

    * `path` - The report
    * `epss` - CVE to `(score, percentile)`
    * `kev` - CVE to the date CISA added it
    * `name` - The image name to report, instead of the report's ArtifactName
    * `by_lockfile` - Report each finding under the lockfile it came from (the
      result's repo-relative Target) instead of the scanned artifact
    """
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise InputError(f"{path}: {error}") from error
    if not isinstance(report, dict):
        raise InputError(f"{path}: not a Trivy JSON report")
    image = name or report.get("ArtifactName") or path
    findings = []
    seen = set()
    for result in report.get("Results") or []:
        for vuln in result.get("Vulnerabilities") or []:
            cve = str(vuln.get("VulnerabilityID", "")).strip()
            package = vuln.get("PkgName", "")
            installed = vuln.get("InstalledVersion", "")
            key = (cve, package, installed, result.get("Target", ""))
            # Trivy can list a package once per target; report each once
            if not cve or key in seen:
                continue
            seen.add(key)
            severity = str(vuln.get("Severity", "UNKNOWN")).upper()
            severity = severity if severity in SEVERITIES else "UNKNOWN"
            score, percentile = epss.get(cve.upper(), (None, None))
            in_kev = cve.upper() in kev
            findings.append({
                "image": (result.get("Target", "") or image) if by_lockfile else image,
                "target": result.get("Target", ""),
                "class": result.get("Class", ""),
                "cve": cve,
                "package": package,
                "installed": installed,
                "fixed": vuln.get("FixedVersion", ""),
                "severity": severity,
                "title": vuln.get("Title", ""),
                "url": vuln.get("PrimaryURL", ""),
                "epss": score,
                "percentile": percentile,
                "kev": in_kev,
                "kev_added": kev.get(cve.upper(), ""),
                "tier": tier(severity, percentile, in_kev),
            })
    return image, findings


def load_inventory(path, name=None):
    """Summarize the packages in one Trivy image report, for the SBOM section.

    # Arguments

    * `path` - The report, already read successfully by `load_findings`
    * `name` - The image name to report, instead of the report's ArtifactName

    Returns `(image, os, counts)`, where `counts` maps each package type (such as
    `ubuntu` or `python-pkg`) to its number of packages, or None when the report
    lists no packages (Trivy was run without `--list-all-pkgs`).
    """
    report = json.loads(Path(path).read_text())
    counts = Counter()
    for result in report.get("Results") or []:
        packages = result.get("Packages") or []
        if packages:
            counts[result.get("Type") or result.get("Class") or "unknown"] += len(packages)
    if not counts:
        return None
    os_info = (report.get("Metadata") or {}).get("OS") or {}
    os_name = " ".join(str(part) for part in (os_info.get("Family"), os_info.get("Name")) if part)
    return name or report.get("ArtifactName") or path, os_name, counts


def sort_key(finding):
    """Order findings by tier, then severity, then EPSS, most urgent first."""
    return (
        list(TIERS).index(finding["tier"]),
        SEVERITIES.index(finding["severity"]),
        -(finding["epss"] or 0.0),
        finding["image"],
        finding["cve"],
        finding["package"],
    )


def describe(finding):
    """Render a finding's one-line description."""
    fixed = f", fixed in {finding['fixed']}" if finding["fixed"] else ", no fix available"
    context = [TIERS[finding["tier"]][0]]
    if finding["epss"] is not None:
        context.append(f"EPSS {finding['epss']:.3f} (p{finding['percentile'] * 100:.0f})")
    if finding["kev"]:
        context.append(f"CISA KEV since {finding['kev_added']}" if finding["kev_added"] else "CISA KEV")
    title = f": {finding['title']}" if finding["title"] else ""
    return (
        f"{finding['cve']} ({finding['severity']}) in {finding['package']} {finding['installed']}"
        f"{fixed} [{'; '.join(context)}] in {finding['image']}{title}"
    )


def repository(image):
    """Strip the tag or digest from an image reference.

    # Arguments

    * `image` - An image reference such as `ghcr.io/org/thorium:main`
    """
    image = image.split("@", 1)[0]
    name, _, tag = image.rpartition(":")
    # a colon before the last slash belongs to a registry port, not a tag
    return name if name and "/" not in tag else image


def to_sarif(findings, dockerfile=None):
    """Build a SARIF log with one result per finding.

    # Arguments

    * `findings` - The enriched findings
    * `dockerfile` - The repo-relative Dockerfile the results are attributed to, or
      None to attribute each to its lockfile
    """
    rules = {}
    results = []
    for finding in findings:
        if finding["cve"] not in rules:
            rules[finding["cve"]] = {
                "id": finding["cve"],
                "shortDescription": {"text": finding["title"] or finding["cve"]},
                "properties": {
                    "tags": ["vulnerability", "security", finding["severity"]],
                    "security-severity": SECURITY_SEVERITY[finding["severity"]],
                },
            }
            if finding["url"].startswith(("https://", "http://")):
                rules[finding["cve"]]["helpUri"] = finding["url"]
        results.append({
            "ruleId": finding["cve"],
            "level": TIERS[finding["tier"]][1],
            "message": {"text": describe(finding)},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": dockerfile or finding["image"]},
                    "region": {"startLine": 1},
                },
            }],
            # one alert per CVE, package and image repository, however the Dockerfile changes;
            # the tag is left out so branch, main and release scans of an image share alerts
            "partialFingerprints": {
                "vulnerability": f"{repository(finding['image'])}|{finding['cve']}|{finding['package']}|{finding['installed']}",
            },
            "properties": {
                "tier": TIERS[finding["tier"]][0],
                "epss": finding["epss"],
                "epssPercentile": finding["percentile"],
                "kev": finding["kev"],
            },
        })
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "trivy-image", "rules": list(rules.values())}},
            "results": results,
        }],
    }


def escape(text):
    """Escape text for a markdown table cell."""
    text = str(text).replace("\r", " ").replace("\n", " ")
    for char, entity in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ("|", "&#124;"), ("`", "&#96;"), ("[", "&#91;"), ("]", "&#93;")):
        text = text.replace(char, entity)
    return text


def to_markdown(images, findings, statuses, subject="Image", inventories=()):
    """Build the markdown priority report.

    # Arguments

    * `images` - The scanned image names (or lockfiles), in input order
    * `findings` - The enriched findings, sorted
    * `statuses` - One line per feed describing where its data came from
    * `subject` - What each of `images` is, for headings
    * `inventories` - `(image, os, counts)` from `load_inventory` for each image
      whose packages were listed
    """
    lines = [f"## {'Image' if subject == 'Image' else 'Dependency'} vulnerabilities by priority", ""]
    lines += [f"- {escape(status)}" for status in statuses]
    lines += [
        "",
        "Findings never fail the pipeline. Fix Now: in CISA KEV, or CRITICAL/HIGH in the top 10% of EPSS; "
        "Fix Soon: other CRITICAL/HIGH; Monitor: lower severity in the top 10% of EPSS; "
        "Untiered: no EPSS score.",
        "",
        f"| {subject} | " + " | ".join(label for label, _ in TIERS.values()) + " | " + " | ".join(SEVERITIES) + " | Fixable |",
        "|---|" + "---:|" * (len(TIERS) + len(SEVERITIES) + 1),
    ]
    for image in images:
        mine = [finding for finding in findings if finding["image"] == image]
        tiers = [sum(1 for finding in mine if finding["tier"] == key) for key in TIERS]
        severities = [sum(1 for finding in mine if finding["severity"] == severity) for severity in SEVERITIES]
        fixable = sum(1 for finding in mine if finding["fixed"])
        lines.append(f"| {escape(image)} | " + " | ".join(str(count) for count in tiers + severities + [fixable]) + " |")
    actionable = [finding for finding in findings if finding["tier"] in ("fix-now", "fix-soon", "monitor")]
    lines += ["", f"### Findings to act on ({len(actionable)})", ""]
    if not actionable:
        lines.append("None.")
    else:
        lines += [
            f"| Tier | CVE | Severity | EPSS (percentile) | KEV | Package | Installed | Fixed | {subject} |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for finding in actionable[:MARKDOWN_ROWS]:
            epss = f"{finding['epss']:.3f} (p{finding['percentile'] * 100:.0f})" if finding["epss"] is not None else ""
            lines.append(
                f"| {TIERS[finding['tier']][0]} | {escape(finding['cve'])} | {finding['severity']} | {epss} | "
                f"{'yes' if finding['kev'] else ''} | {escape(finding['package'])} | {escape(finding['installed'])} | "
                f"{escape(finding['fixed'] or 'none')} | {escape(finding['image'])} |"
            )
        if len(actionable) > MARKDOWN_ROWS:
            lines.append(f"\n{len(actionable) - MARKDOWN_ROWS} more are in the JSON report.")
    if inventories:
        lines += [
            "",
            "### Software bill of materials",
            "",
            "Each image's full CycloneDX SBOM (`*.cdx.json`) is in this run's scan report artifact.",
            "",
            "| Image | OS | Packages | By type |",
            "|---|---|---:|---|",
        ]
        for image, os_name, counts in inventories:
            by_type = ", ".join(f"{escape(kind)} {count}" for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            lines.append(f"| {escape(image)} | {escape(os_name or 'unknown')} | {sum(counts.values())} | {by_type} |")
    # a trailing blank line ends the last table before anything appended after it
    return "\n".join(lines) + "\n\n"


def main():
    """Parse arguments, enrich the reports, and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", action="append", required=True, help="a `trivy image --format json` report; repeatable")
    parser.add_argument("--image-name", help="the image name to report, for a scan of a saved image tar; needs a single --input")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--dockerfile", help="repo-relative Dockerfile the SARIF results of an image scan are attributed to")
    location.add_argument("--by-lockfile", action="store_true", help="for a dependency scan, report and attribute each finding to its lockfile")
    parser.add_argument("--sarif", required=True, help="where to write the SARIF log")
    parser.add_argument("--json", help="where to write every enriched finding")
    parser.add_argument("--markdown", help="where to write the priority report")
    parser.add_argument("--epss-url", default=EPSS_URL, help="the EPSS CSV to download")
    parser.add_argument("--kev-url", default=KEV_URL, help="the KEV catalog to download")
    parser.add_argument("--epss-file", help="a local EPSS CSV (optionally gzipped) to use instead of downloading")
    parser.add_argument("--kev-file", help="a local KEV catalog to use instead of downloading")
    parser.add_argument("--offline", action="store_true", help="skip downloading the feeds and rank by severity only")
    args = parser.parse_args()
    if args.image_name and len(args.input) != 1:
        parser.error("--image-name needs exactly one --input")

    epss, _, epss_status = fetch_feed("EPSS", args.epss_url, args.epss_file, args.offline, load_epss)
    kev, _, kev_status = fetch_feed("CISA KEV", args.kev_url, args.kev_file, args.offline, load_kev)
    images = []
    findings = []
    inventories = []
    try:
        for path in args.input:
            image, found = load_findings(path, epss, kev, args.image_name, args.by_lockfile)
            if args.by_lockfile:
                images += [name for name in dict.fromkeys(finding["image"] for finding in found) if name not in images]
            else:
                images.append(image)
                inventory = load_inventory(path, args.image_name)
                if inventory:
                    inventories.append(inventory)
            findings += found
    except InputError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    findings.sort(key=sort_key)

    Path(args.sarif).write_text(json.dumps(to_sarif(findings, args.dockerfile), indent=2) + "\n")
    subject = "Lockfile" if args.by_lockfile else "Image"
    if args.json:
        Path(args.json).write_text(json.dumps({"feeds": [epss_status, kev_status], "findings": findings}, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(images, findings, [epss_status, kev_status], subject, inventories))
    counts = {label: sum(1 for finding in findings if finding["tier"] == key) for key, (label, _) in TIERS.items()}
    print(f"{len(findings)} vulnerabilities in {len(images)} {subject.lower()}(s): " + ", ".join(f"{count} {label}" for label, count in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
