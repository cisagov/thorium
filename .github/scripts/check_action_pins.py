#!/usr/bin/env python3
"""Report GitHub Actions `uses:` references that are unpinned or behind the latest release.

Every `uses: owner/repo[/path]@ref` in the repo's workflows and local composite
actions is checked:

* refs that are not a full 40-character commit SHA are reported as unpinned
* SHA pins must carry a `# vX.Y.Z` comment naming the release they correspond to
* that version is compared with the repo's latest GitHub release (or highest
  semver tag when the repo publishes no releases), and older pins are reported

This script only reads workflow files and calls the GitHub REST API on
api.github.com; it uses only the Python standard library so it adds no
dependencies of its own. zizmor's `ref-version-mismatch` and `impostor-commit`
audits separately verify that each SHA really matches its comment and repo.

Two files are written:

* `--json` - every reference with its pinned and latest version and status
* `--sarif` - only the problems, for .github/scripts/lint_report.py to report
  and gate on like any other linter: unpinned references are errors, outdated
  pins and missing version comments are warnings, and references whose latest
  release is unknown are notes

Outdated pins are warnings rather than errors since Dependabot opens the
corresponding update pull requests.

GitHub API calls are retried on network errors, rate limits and server
errors (see http_retry.py). A repository whose latest release still cannot be
looked up is reported as `unknown-latest` rather than failing the run.

Exit status is 0 when both files were written.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import http_retry

# the GitHub REST API root; the only host this script contacts
API_ROOT = "https://api.github.com"
# matches a `uses:` line and captures the action, its ref and an optional trailing comment
USES_RE = re.compile(
    r"""^\s*-?\s*uses:\s*["']?(?P<action>[^@\s"']+)@(?P<ref>[^\s"'#]+)["']?\s*(?:\#\s*(?P<comment>.*))?$"""
)
# matches `uses:` in a YAML flow mapping, such as `- {uses: owner/repo@ref, with: ...}`
FLOW_USES_RE = re.compile(r"""[{,]\s*uses:\s*["']?(?P<action>[^@\s"',}]+)@(?P<ref>[^\s"',}#]+)""")
# any `uses:` key, so references neither pattern understands are reported instead of skipped
ANY_USES_RE = re.compile(r"""(?:^|[\s{,-])uses\s*:""")
# the most tag pages to read when a repository publishes no releases
MAX_TAG_PAGES = 5
# a valid owner/repo pair, used to keep API paths well-formed
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# a full commit SHA
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# a semantic version, with an optional leading `v` and an optional patch number
SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?$")
# every problem this script reports, as rule id -> (SARIF level, description)
RULES = {
    "unpinned": ("error", "Action is not pinned to a full commit SHA"),
    "outdated-major": ("warning", "Pinned action is behind a newer major release"),
    "outdated": ("warning", "Pinned action is behind its latest release"),
    "missing-version-comment": ("warning", "SHA pin has no `# vX.Y.Z` comment naming its release"),
    "unknown-latest": ("note", "The action's latest release could not be determined"),
    "unrecognized": ("note", "The `uses:` reference is not a recognizable owner/repo action"),
}


def parse_version(text):
    """Parse a version string into a comparable tuple.

    # Arguments

    * `text` - The version string, such as `v4.38.2`

    Returns a `(major, minor, patch)` tuple, or None when `text` is not a version.
    """
    # pull out the numeric components, treating a missing patch as zero
    match = SEMVER_RE.match(text.strip()) if text else None
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def api_get(path, token):
    """Fetch and decode a JSON document from the GitHub REST API.

    # Arguments

    * `path` - The API path, starting with `/`
    * `token` - An optional GitHub token used to raise the rate limit

    Returns None for missing resources so callers can fall back, and raises
    `http_retry.HttpError` for any other failure after retries.
    """
    # build the request with the recommended headers and optional auth
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return http_retry.get_json(API_ROOT + path, headers=headers)
    except http_retry.HttpError as error:
        if error.status == 404:
            return None
        raise


def latest_version(repo, token):
    """Find the newest released version of an action repository.

    # Arguments

    * `repo` - The `owner/repo` of the action
    * `token` - An optional GitHub token used to raise the rate limit

    Returns a `(tag, version_tuple)` pair, or None when no version could be found.
    """
    # prefer the latest non-prerelease GitHub release
    release = api_get(f"/repos/{repo}/releases/latest", token)
    if isinstance(release, dict) and parse_version(release.get("tag_name", "")):
        return release["tag_name"], parse_version(release["tag_name"])
    # otherwise fall back to the highest semver tag, which may not be on the first page
    tags = []
    for page in range(1, MAX_TAG_PAGES + 1):
        batch = api_get(f"/repos/{repo}/tags?per_page=100&page={page}", token) or []
        tags.extend(batch)
        if len(batch) < 100:
            break
    versions = [(tag.get("name", ""), parse_version(tag.get("name", ""))) for tag in tags if isinstance(tag, dict)]
    versions = [entry for entry in versions if entry[1]]
    return max(versions, key=lambda entry: entry[1]) if versions else None


def collect_uses(root):
    """Collect every remote `uses:` reference in workflows and local actions.

    # Arguments

    * `root` - The repository root
    """
    # gather workflow files and any local composite action definitions
    github_dir = root / ".github"
    files = sorted(github_dir.glob("workflows/*.y*ml")) + sorted(github_dir.glob("actions/**/action.y*ml"))
    found = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = USES_RE.match(line) or FLOW_USES_RE.search(line)
            if not match:
                # a `uses:` this script cannot parse is reported rather than silently
                # skipped, except local actions, which have no `@ref`
                code = line.split("#", 1)[0]
                value = re.split(r"uses\s*:", code, maxsplit=1)[-1].strip().strip("\"'{} ")
                if ANY_USES_RE.search(code) and not value.startswith(("./", "docker://")):
                    found.append(
                        {
                            "file": str(path.relative_to(root)),
                            "line": number,
                            "action": line.strip(),
                            "repo": "",
                            "ref": "",
                            "pinned_tag": "",
                        }
                    )
                continue
            # skip local actions and docker image references
            if match["action"].startswith(("./", "docker://")):
                continue
            # reduce `owner/repo/sub/path` to the repository that hosts it
            repo = "/".join(match["action"].split("/")[:2])
            comment = (match.groupdict().get("comment") or "").split()
            found.append(
                {
                    "file": str(path.relative_to(root)),
                    "line": number,
                    "action": match["action"],
                    "repo": repo,
                    "ref": match["ref"],
                    "pinned_tag": comment[0] if comment else "",
                }
            )
    return found


def check(use, token, latest_cache):
    """Classify one action reference.

    # Arguments

    * `use` - The reference, as collected by `collect_uses`
    * `token` - An optional GitHub token used to raise the rate limit
    * `latest_cache` - Latest releases already looked up, by repository

    Returns `(status, latest_tag, message)`, where status is `up-to-date` or a rule id.
    """
    if not use["repo"]:
        return "unrecognized", "", f"`{use['action']}` has a `uses:` reference this check cannot parse"
    if not REPO_RE.match(use["repo"]):
        return "unrecognized", "", f"{use['action']} is not an owner/repo action reference"
    if not SHA_RE.match(use["ref"]):
        return "unpinned", "", f"{use['action']}@{use['ref']} is not pinned to a full commit SHA"
    # look up each repository's latest release only once
    if use["repo"] not in latest_cache:
        try:
            latest_cache[use["repo"]] = latest_version(use["repo"], token)
        except http_retry.HttpError as error:
            # an unreachable API leaves this repository's latest release unknown
            print(f"could not look up the latest release of {use['repo']}: {error}", file=sys.stderr)
            latest_cache[use["repo"]] = None
    newest = latest_cache[use["repo"]]
    latest = newest[0] if newest else ""
    pinned = parse_version(use["pinned_tag"])
    if pinned is None:
        return "missing-version-comment", latest, f"{use['action']} is pinned to a SHA with no `# vX.Y.Z` comment naming its release"
    if newest is None:
        return "unknown-latest", latest, f"could not determine the latest release of {use['repo']}"
    if pinned < newest[1]:
        status = "outdated" if pinned[0] == newest[1][0] else "outdated-major"
        return status, latest, f"{use['repo']} is pinned at {use['pinned_tag']}; latest release is {latest}"
    return "up-to-date", latest, ""


def to_sarif(rows):
    """Build a SARIF log of the references with problems.

    # Arguments

    * `rows` - The checked references
    """
    rules = [
        {"id": rule, "shortDescription": {"text": description}, "defaultConfiguration": {"level": level}}
        for rule, (level, description) in RULES.items()
    ]
    results = [
        {
            "ruleId": row["status"],
            "level": RULES[row["status"]][0],
            "message": {"text": row["message"]},
            "locations": [
                {"physicalLocation": {"artifactLocation": {"uri": row["file"]}, "region": {"startLine": row["line"]}}}
            ],
        }
        for row in rows
        if row["status"] != "up-to-date"
    ]
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": "action-pins", "rules": rules}}, "results": results}],
    }


def main():
    """Check every action reference and write the JSON table and SARIF findings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, help="where to write every reference and its status")
    parser.add_argument("--sarif", required=True, help="where to write the problems as SARIF")
    args = parser.parse_args()
    # read the optional token and locate the repo root
    # GitHub Actions provides GITHUB_TOKEN and GITHUB_WORKSPACE; GitLab CI
    # provides CI_PROJECT_DIR and a GH_TOKEN project variable when configured
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    root = Path(os.environ.get("GITHUB_WORKSPACE") or os.environ.get("CI_PROJECT_DIR") or ".").resolve()
    latest_cache = {}
    rows = []
    for use in collect_uses(root):
        status, latest, message = check(use, token, latest_cache)
        rows.append(
            {
                "file": use["file"],
                "line": use["line"],
                "action": use["action"],
                "ref": use["ref"],
                "pinned_version": use["pinned_tag"],
                "latest_version": latest,
                "status": status,
                "message": message,
            }
        )
    Path(args.json).write_text(json.dumps({"actions": rows}, indent=2) + "\n", encoding="utf-8")
    Path(args.sarif).write_text(json.dumps(to_sarif(rows), indent=2) + "\n", encoding="utf-8")
    problems = sum(row["status"] != "up-to-date" for row in rows)
    print(f"checked {len(rows)} action references; {problems} have problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
