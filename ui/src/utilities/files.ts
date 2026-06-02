// project imports
import { SubmissionChunk } from '@models/files';

/**
 * Heuristically test whether a string looks like a SHA256 hash.
 *
 * Checks for a 64-character string of hexadecimal digits; does not verify the value is an
 * actual hash of anything.
 *
 * @param someString - The string to test.
 * @returns `true` if the string is 64 hex characters, otherwise `false`.
 */
export function isValidSha256(someString: string): boolean {
  if (someString.length !== 64) return false;
  return /^[A-Fa-f0-9]+$/.test(someString);
}

/**
 * Build a comma-separated display string of the unique, human-friendly file names from a file's submissions.
 *
 * Strips the agent-prepended `/tmp/thorium/children/<id>/` and leading `./` paths, de-duplicates,
 * and sorts. Names that are themselves SHA256 hashes are filtered out *if* any non-hash names
 * exist; if every name is a hash, the full (hash) list is returned so the caller still has something
 * to display.
 *
 * @param submissions - The submission chunks to derive names from.
 * @returns A `',   '`-joined string of the chosen file names.
 */
export const getUniqueFileNames = (submissions: SubmissionChunk[]) => {
  // remove any Thorium prepended paths added by agent
  const fullNames = [
    ...new Set(
      submissions.map((submission: SubmissionChunk) => {
        let strippedPath = (submission.name ?? '').replace(/^\/tmp\/thorium\/children\/[^/]+\//, '');
        strippedPath = strippedPath.replace(/^.\//, '');
        return strippedPath;
      }),
    ),
  ].sort();
  // get only non-sha256 hash names
  const filteredNames = fullNames.filter((name) => !isValidSha256(name));
  // return non-sha256 names if any are found, otherwise return full names
  return filteredNames.length > 0 ? filteredNames.join(',   ') : fullNames.join(',   ');
};
