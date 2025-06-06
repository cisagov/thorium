import { SubmissionChunk } from 'models';

// return if thing is likely sha256 hash based on length and base64 characters
export function isValidSha256(someString: string): boolean {
  if (someString.length !== 64) return false;
  return /^[A-Fa-f0-9]+$/.test(someString);
}

// Get unique file names from submissions
export const getUniqueFileNames = (submissions: SubmissionChunk[]) => {
  // remove any Thorium prepended paths added by agent
  const fullNames = [
    ...new Set(
      submissions.map((submission: any) => {
        let strippedPath = submission.name.replace(/^\/tmp\/thorium\/children\/[^\/]+\//, '');
        strippedPath = strippedPath.replace(/^.\//, '');
        return strippedPath;
      }),
    ),
  ].sort() as string[];
  // get only non-sha256 hash names
  const filteredNames = fullNames.filter((name) => !isValidSha256(name));
  // return non-sha256 names if any are found, otherwise return full names
  return filteredNames.length > 0 ? filteredNames.join(',   ') : fullNames.join(',   ');
};
