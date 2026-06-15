// project imports
import { ImageVersion } from '@models/images';

/**
 * Format an {@link ImageVersion} for display.
 *
 * Renders a `SemVer` as `vMAJOR.MINOR.PATCH` with optional `-pre` and `+build` suffixes,
 * or a `Custom` version as its raw string.
 *
 * @param version - The image/tool version, or `undefined`.
 * @returns A display string (e.g. `v1.2.3`, `v1.2.3-alpha.1+build.5`, or the custom string),
 *   or an empty string when no version is present.
 */
export function formatImageVersion(version?: ImageVersion): string {
  if (!version) return '';
  if (version.Custom) return version.Custom;
  const semver = version.SemVer;
  if (!semver) return '';
  let out = `v${semver.major}.${semver.minor}.${semver.patch}`;
  if (semver.pre) out += `-${semver.pre}`;
  if (semver.build) out += `+${semver.build}`;
  return out;
}

/**
 * Label a tool-result version by its upload time, appending the tool version when known.
 *
 * Used by the result tile's version selector and the diff view's version dropdowns/titles.
 *
 * @param uploaded - The result's upload timestamp (ISO string).
 * @param version - The tool version, or `undefined`.
 * @returns `"<localized date> - <version>"`, or just the date when no version is present.
 */
export function versionLabel(uploaded: string, version?: ImageVersion): string {
  const date = new Date(uploaded);
  const when = isNaN(date.getTime()) ? uploaded : date.toLocaleString();
  const formatted = formatImageVersion(version);
  return formatted ? `${when} - ${formatted}` : when;
}
