#!/usr/bin/env python3
"""Report pinned tool versions in Dockerfiles that are behind their upstream release.

Pins are declared as Dockerfile ARG or ENV values with a Renovate-style comment on the
line directly above, so the source of each version lives next to the pin:

    # renovate: datasource=github-releases depName=mandiant/flare-floss extractVersion=^v(?<version>.+)$
    ARG FLOSS_VERSION=3.1.1

Supported keys (a subset of Renovate's, so Renovate can read the same comments):

* `datasource` - `github-releases`, `github-tags`, `pypi`, or `git-refs` (commit pins)
* `depName` - `owner/repo` for GitHub, the package name for PyPI, or the repository URL
  for `git-refs`
* `extractVersion` - regex with a `version` group that turns a tag into a version
* `allowedVersions` - limit candidates: `12.0.x`, `<2026.8.28`, `<=1.2`, `>=1`, or `/regex/`
* `currentValue` - for `git-refs`, the tag or branch the pinned commit should match or
  track; `${NAME}` is replaced with the value of another ARG/ENV in the same Dockerfile

Checks:

* a version behind the newest allowed upstream release is reported as `outdated`
* a commit that does not match its `currentValue` tag is reported as `commit-mismatch`
* a commit that tracks a branch and is behind it is reported as `behind-branch`
* an unannotated `*_VERSION` or `*_COMMIT` value is reported as `unannotated-pin`, so
  new pins do not go unchecked
* a `git-refs` pin that is not a full 40-character commit SHA is reported as
  `invalid-commit`
* a version that could not be looked up is reported as `lookup-failed`

Two files are written: `--json` lists every pin with its current and latest versions,
and `--sarif` holds only the problems, for .github/scripts/lint_report.py to report and
gate on like any other linter (commit mismatches are errors, outdated versions
warnings, the rest notes).

This script uses only the Python standard library and calls api.github.com (with
GITHUB_TOKEN or GH_TOKEN when set, to raise the rate limit) and pypi.org.

API calls are retried on network errors, rate limits and server errors (see
http_retry.py). Exit status is 0 when both files were written; API calls that still
fail, unexpected API responses, and invalid annotation regexes are reported as
`lookup-failed` findings rather than aborting the run.
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

import http_retry

# the hosts this script contacts
GITHUB_API = "https://api.github.com"
PYPI_API = "https://pypi.org/pypi"
# the Dockerfiles checked, relative to the repo root
DOCKERFILE_GLOBS = ("Dockerfile", "thorctl/Dockerfile", "base/**/Dockerfile", "tools/images/**/Dockerfile")
# a Renovate-style annotation comment
ANNOTATION_RE = re.compile(r"^\s*#\s*renovate:\s*(?P<fields>.+?)\s*$")
# one key=value field in an annotation; values may be quoted
FIELD_RE = re.compile(r"""(\w+)=("[^"]*"|'[^']*'|\S+)""")
# an ARG or ENV instruction with a value
PIN_RE = re.compile(r"""^\s*(?P<kind>ARG|ENV)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>"[^"]*"|'[^']*'|\S*)""")
# pin names that should carry an annotation
PIN_NAME_RE = re.compile(r"_(VERSION|COMMIT)$")
# a full commit SHA
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Renovate's named group syntax, `(?<name>`, which Python spells `(?P<name>`
NAMED_GROUP_RE = re.compile(r"\(\?<([A-Za-z_]\w*)>")
# version markers that make a version a pre-release
UNSTABLE_RE = re.compile(r"(alpha|beta|preview|pre|rc|dev|a\d|b\d)", re.IGNORECASE)
# every problem this script reports, as rule id -> (SARIF level, description)
RULES = {
    "commit-mismatch": ("error", "Pinned commit does not match the tag it declares"),
    "invalid-commit": ("error", "Commit pin is not a full 40-character commit SHA"),
    "outdated": ("warning", "Pinned version is behind the newest upstream release"),
    "behind-branch": ("note", "Pinned commit is behind the branch it tracks"),
    "lookup-failed": ("note", "The pinned version could not be checked upstream"),
    "unannotated-pin": ("note", "Version or commit pin has no '# renovate:' annotation"),
}


class LookupError_(Exception):
    """An upstream lookup failed in a way worth reporting."""


class Upstream:
    """Cached access to the GitHub and PyPI APIs."""

    def __init__(self, token):
        """Create the client.

        # Arguments

        * `token` - An optional GitHub token used to raise the rate limit
        """
        self.token = token
        self.cache = {}

    def get(self, url):
        """Fetch and decode a JSON document, caching by URL.

        # Arguments

        * `url` - The full URL to fetch

        Returns the decoded JSON, or None for a 404.
        """
        if url in self.cache:
            return self.cache[url]
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "thorium-tool-version-check"}
        if self.token and url.startswith(GITHUB_API):
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            data = http_retry.get_json(url, headers=headers)
        except http_retry.HttpError as error:
            if error.status != 404:
                raise LookupError_(str(error)) from error
            data = None
        self.cache[url] = data
        return data

    def github_pages(self, path, pages=3):
        """Fetch up to `pages` pages of a GitHub list endpoint.

        # Arguments

        * `path` - The API path, starting with `/`, without paging parameters
        * `pages` - The most pages of 100 items to fetch
        """
        items = []
        for page in range(1, pages + 1):
            separator = "&" if "?" in path else "?"
            batch = self.get(f"{GITHUB_API}{path}{separator}per_page=100&page={page}") or []
            if not isinstance(batch, list):
                raise LookupError_(f"{path} did not return a list")
            items.extend(batch)
            if len(batch) < 100:
                break
        return items


def parse_fields(text):
    """Parse `key=value` pairs from an annotation, allowing quoted values.

    Backslashes are kept as written, since extractVersion values are regexes.

    # Arguments

    * `text` - The text after `renovate:`
    """
    return {key: value.strip("\"'") for key, value in FIELD_RE.findall(text)}


def version_key(version):
    """Build a sortable key for a loosely formatted version string.

    # Arguments

    * `version` - The version, such as `1.2.3`, `0.0.0rc18` or `2026.7.1`
    """
    numbers = [int(part) for part in re.findall(r"\d+", version.split("+")[0])]
    stable = 0 if UNSTABLE_RE.search(version) else 1
    return (numbers[:3] + [0] * (3 - len(numbers[:3])), stable, numbers[3:], version)


def compile_regex(pattern):
    """Compile an annotation regex, accepting Renovate's `(?<name>...)` groups.

    # Arguments

    * `pattern` - The regex from an annotation

    Raises `LookupError_` when the regex is invalid, so the pin is reported rather
    than aborting the run.
    """
    try:
        return re.compile(NAMED_GROUP_RE.sub(r"(?P<\1>", pattern))
    except re.error as error:
        raise LookupError_(f"invalid regex {pattern!r}: {error}") from error


def allowed(version, rule):
    """Check a candidate version against an `allowedVersions` rule.

    # Arguments

    * `version` - The candidate version
    * `rule` - The rule, or None to allow everything
    """
    if not rule:
        return True
    if rule.startswith("/") and rule.endswith("/"):
        return compile_regex(rule[1:-1]).search(version) is not None
    if rule.endswith(".x"):
        return version == rule[:-2] or version.startswith(rule[:-1])
    match = re.match(r"^(<=|<|>=|>)\s*(.+)$", rule)
    if match:
        operator, bound = match.groups()
        left, right = version_key(version)[:3], version_key(bound)[:3]
        return {"<": left < right, "<=": left <= right, ">=": left >= right, ">": left > right}[operator]
    return version == rule


def extract(tag, pattern):
    """Turn a tag name into a version using an `extractVersion` regex.

    # Arguments

    * `tag` - The tag name
    * `pattern` - The regex with a `version` group (Renovate's `(?<version>...)` syntax
      is accepted), or None to use the tag as is

    Returns the version, or None when the tag does not match.
    """
    if not pattern:
        return tag
    match = compile_regex(pattern).match(tag)
    if not match:
        return None
    return match.groupdict().get("version") or match.group(0)


def candidates(pin, upstream):
    """List the upstream versions for a pin, highest version first.

    # Arguments

    * `pin` - The pin, with its annotation fields
    * `upstream` - The API client

    Returns a list of (version, is_prerelease) pairs.
    """
    fields = pin["fields"]
    source, name = fields.get("datasource"), fields.get("depName", "")
    pattern = fields.get("extractVersion")
    if source == "github-releases":
        releases = [r for r in upstream.github_pages(f"/repos/{name}/releases") if not r.get("draft")]
        found = [(extract(r["tag_name"], pattern), bool(r.get("prerelease")), r.get("published_at") or "") for r in releases]
        # highest version first, so a backport released after a newer version is not
        # taken as the latest; publish date only orders names that compare equal
        found.sort(key=lambda item: (version_key(item[0] or ""), item[2]), reverse=True)
        found = [(version, pre) for version, pre, _ in found]
    elif source == "github-tags":
        tags = upstream.github_pages(f"/repos/{name}/tags")
        found = [(extract(t["name"], pattern), False) for t in tags]
        found = [(v, bool(v and UNSTABLE_RE.search(v))) for v, _ in found]
        found.sort(key=lambda item: version_key(item[0] or ""), reverse=True)
    elif source == "pypi":
        data = upstream.get(f"{PYPI_API}/{urllib.parse.quote(name)}/json")
        if data is None:
            raise LookupError_(f"PyPI has no package {name}")
        uploads = []
        for version, files in data.get("releases", {}).items():
            if files and not all(f.get("yanked") for f in files):
                uploads.append((max(f.get("upload_time_iso_8601", "") for f in files), version))
        # highest version first; upload time only orders versions that compare equal
        uploads.sort(key=lambda item: (version_key(item[1]), item[0]), reverse=True)
        found = [(version, bool(UNSTABLE_RE.search(version))) for _, version in uploads]
    else:
        raise LookupError_(f"unsupported datasource {source!r}")
    return [(v, pre) for v, pre in found if v and allowed(v, fields.get("allowedVersions"))]


def check_version(pin, upstream):
    """Compare a version pin with its newest allowed upstream release.

    # Arguments

    * `pin` - The pin, with its annotation fields
    * `upstream` - The API client

    Returns `(status, latest, message)`.
    """
    found = candidates(pin, upstream)
    current = pin["value"]
    pinned = next((pre for version, pre in found if version == current), None)
    # pre-releases count only when the pin itself is a pre-release
    unstable = bool(pinned) or bool(UNSTABLE_RE.search(current))
    eligible = [version for version, pre in found if unstable or not pre]
    if not eligible:
        raise LookupError_("no upstream versions matched the annotation")
    # candidates are highest first, so anything other than the first one is behind
    latest = eligible[0]
    if current == latest:
        return "up-to-date", latest, ""
    if current not in eligible and version_key(current) >= version_key(latest):
        return "up-to-date", latest, ""
    return "outdated", latest, f"{pin['name']} is {current}; the newest upstream release of {pin['fields'].get('depName')} is {latest}"


def resolve_ref(repo, ref, upstream):
    """Resolve a tag or branch to a commit SHA.

    # Arguments

    * `repo` - The `owner/repo`
    * `ref` - The tag or branch name
    * `upstream` - The API client

    Returns `(kind, sha)`, where kind is `tag` or `branch`.
    """
    data = upstream.get(f"{GITHUB_API}/repos/{repo}/git/ref/tags/{urllib.parse.quote(ref)}")
    if data:
        obj = data["object"]
        # annotated tags point at a tag object that points at the commit
        while obj["type"] == "tag":
            tag = upstream.get(f"{GITHUB_API}/repos/{repo}/git/tags/{obj['sha']}")
            if not tag:
                raise LookupError_(f"{repo} tag object {obj['sha']} for {ref} was not found")
            obj = tag["object"]
        return "tag", obj["sha"]
    data = upstream.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{urllib.parse.quote(ref)}")
    if data:
        return "branch", data["object"]["sha"]
    raise LookupError_(f"{repo} has no tag or branch named {ref}")


def check_commit(pin, upstream, values):
    """Check a commit pin against the tag it declares or the branch it tracks.

    # Arguments

    * `pin` - The pin, with its annotation fields
    * `upstream` - The API client
    * `values` - Every ARG/ENV value in the Dockerfile, for `${NAME}` references

    Returns `(status, latest, message)`.
    """
    fields = pin["fields"]
    if not SHA_RE.match(pin["value"]):
        return "invalid-commit", "", f"{pin['name']} is {pin['value']!r}, which is not a full 40-character commit SHA"
    repo = re.sub(r"^https://github\.com/|\.git$", "", fields.get("depName", "")).strip("/")
    ref = re.sub(r"\$\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), fields.get("currentValue", ""))
    if not ref or "${" in ref:
        raise LookupError_(f"currentValue {fields.get('currentValue')!r} could not be resolved")
    kind, sha = resolve_ref(repo, ref, upstream)
    if kind == "tag":
        if sha != pin["value"]:
            return "commit-mismatch", sha, f"{pin['name']} is {pin['value'][:12]} but {repo} tag {ref} is {sha[:12]}"
        return "up-to-date", ref, ""
    if sha == pin["value"]:
        return "up-to-date", ref, ""
    compare = upstream.get(f"{GITHUB_API}/repos/{repo}/compare/{pin['value']}...{sha}")
    behind = compare.get("ahead_by", 0) if compare else 0
    if behind == 0:
        return "up-to-date", ref, ""
    return "behind-branch", sha, f"{pin['name']} is {behind} commits behind {repo} {ref} ({sha[:12]})"


def collect(root):
    """Collect every ARG/ENV pin and its annotation from the checked Dockerfiles.

    # Arguments

    * `root` - The repository root

    Returns a list of pins and a map of Dockerfile -> {name: value}.
    """
    pins, values = [], {}
    files = sorted({path for pattern in DOCKERFILE_GLOBS for path in root.glob(pattern) if path.is_file()})
    for path in files:
        rel = str(path.relative_to(root))
        lines = path.read_text(encoding="utf-8").splitlines()
        values[rel] = {}
        for number, line in enumerate(lines, start=1):
            match = PIN_RE.match(line)
            if not match or not match["value"].strip("\"'"):
                continue
            value = match["value"].strip("\"'")
            values[rel][match["name"]] = value
            previous = lines[number - 2] if number > 1 else ""
            annotation = ANNOTATION_RE.match(previous)
            if annotation or PIN_NAME_RE.search(match["name"]):
                pins.append(
                    {
                        "file": rel,
                        "line": number,
                        "name": match["name"],
                        "value": value,
                        "fields": parse_fields(annotation["fields"]) if annotation else None,
                    }
                )
    return pins, values


def to_sarif(rows):
    """Build a SARIF log of the pins with problems.

    # Arguments

    * `rows` - The checked pins
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
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": row["file"]}, "region": {"startLine": row["line"]}}}],
        }
        for row in rows
        if row["status"] in RULES
    ]
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": "tool-versions", "rules": rules}}, "results": results}],
    }


def main():
    """Check every annotated pin and write the JSON table and SARIF findings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", required=True, help="where to write every pin and its status")
    parser.add_argument("--sarif", required=True, help="where to write the problems as SARIF")
    args = parser.parse_args()
    # GitHub Actions provides GITHUB_TOKEN and GITHUB_WORKSPACE; GitLab CI provides
    # CI_PROJECT_DIR and a GH_TOKEN project variable when configured
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    root = Path(os.environ.get("GITHUB_WORKSPACE") or os.environ.get("CI_PROJECT_DIR") or ".").resolve()
    upstream = Upstream(token)
    pins, values = collect(root)
    rows = []
    for pin in pins:
        fields = pin["fields"]
        row = {key: pin[key] for key in ("file", "line", "name", "value")}
        row.update(datasource=(fields or {}).get("datasource", ""), dep_name=(fields or {}).get("depName", ""), latest="")
        if fields is None:
            row.update(status="unannotated-pin", message=f"{pin['name']} has no '# renovate:' annotation, so its version is not checked")
        else:
            try:
                check = check_commit(pin, upstream, values[pin["file"]]) if fields.get("datasource") == "git-refs" else check_version(pin, upstream)
                row["status"], row["latest"], row["message"] = check
            except LookupError_ as error:
                row.update(status="lookup-failed", message=f"{pin['name']} could not be checked: {error}")
            # an upstream response without the expected fields
            except (KeyError, TypeError, AttributeError) as error:
                row.update(status="lookup-failed", message=f"{pin['name']} could not be checked: unexpected API response ({error!r})")
        rows.append(row)
    Path(args.json).write_text(json.dumps({"pins": rows}, indent=2) + "\n", encoding="utf-8")
    Path(args.sarif).write_text(json.dumps(to_sarif(rows), indent=2) + "\n", encoding="utf-8")
    problems = sum(row["status"] != "up-to-date" for row in rows)
    print(f"checked {len(rows)} pins in {len(values)} Dockerfiles; {problems} have findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
