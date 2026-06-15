// project imports
import { downloadBlob } from './download';
import { fetchResultFileCached } from './resultFiles';
import { formatImageVersion } from './version';
import { Output } from '@models/results';

/** Name of the manifest describing the run that {@link buildResultsZip} adds to each bundle. */
const RESULT_METADATA_NAME = 'result-metadata.json';

/**
 * Normalize a tool-supplied result-file name into a safe zip entry path.
 *
 * Result files are tool output over semi-trusted samples, so a name may contain absolute-looking
 * prefixes or `..` traversal segments that a naive extractor would follow outside the target
 * directory (zip-slip). Leading slashes and any `..` component are dropped while legitimate nested
 * paths are preserved.
 *
 * @param name - The raw result-file name reported by the tool.
 * @returns A sanitized entry path, or an empty string if nothing safe remains.
 */
function safeZipEntryName(name: string): string {
  // split on both separators, drop empty (leading '/', doubled slashes) and traversal ('..') parts
  return name
    .split(/[/\\]+/)
    .filter((part) => part !== '' && part !== '..' && part !== '.')
    .join('/');
}

/** Manifest describing the run a results bundle was built from. */
export interface ResultsMetadata {
  /** The tool that produced the result. */
  tool: string;
  /** The formatted tool version, when known. */
  tool_version?: string;
  /** When the result was uploaded (ISO string). */
  uploaded: string;
  /** The result files that were successfully downloaded into the bundle. */
  artifacts: string[];
}

/**
 * Build a unique, filesystem-safe name for a results bundle.
 *
 * A bare `${tool}.zip` collides whenever the same tool is downloaded for a different file, or
 * for a different run/version of the same file. Combining the tool name, the file's SHA256, and
 * the result's upload timestamp makes every bundle distinct. Each part is sanitized so the name
 * is valid across operating systems (path separators, the colons in the ISO timestamp, etc.).
 *
 * @param sha256 - The SHA256 of the file the result belongs to.
 * @param tool - The tool that produced the result.
 * @param output - The result being archived (its `uploaded` timestamp identifies the run).
 * @returns A filename of the form `<tool>_<sha256>_<uploaded>.zip`.
 */
export function resultsBundleName(sha256: string, tool: string, output: Output): string {
  // collapse any run of filename-unsafe characters to a single dash, then trim leading/trailing dashes
  const safe = (s: string) => s.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
  const parts = [safe(tool), safe(sha256)];
  const uploaded = safe(output.uploaded ?? '');
  if (uploaded) parts.push(uploaded);
  return `${parts.join('_')}.zip`;
}

/**
 * Assemble a zip archive (in memory) containing a tool result and all of its result files.
 *
 * Mirrors how thorctl packages results: the structured result is written as `result.json`
 * and each result file is fetched via `fetchFile` and added under its own name. A
 * `result-metadata.json` manifest (tool, version, upload time, downloaded artifacts) is also
 * written so a downloaded bundle is self-describing — unless the result already ships a file
 * by that name, which we leave untouched. This is the interim client-side implementation until
 * the API exposes a single download-all route.
 *
 * Split from {@link downloadResultsAsZip} (which supplies the real fetcher) so the assembly
 * logic can be unit-tested with a mocked `fetchFile`.
 *
 * @param tool - The tool that produced the result (recorded in the manifest).
 * @param output - The tool result whose JSON + files should be archived.
 * @param fetchFile - Fetches the bytes of a named result file, or `null` on failure.
 * @returns A `Blob` containing the generated zip archive.
 */
export async function buildResultsZip(
  tool: string,
  output: Output,
  fetchFile: (name: string) => Promise<ArrayBuffer | null>,
): Promise<Blob> {
  // dynamic import keeps jszip out of the main bundle until a download is requested
  const JSZip = (await import('jszip')).default;
  const zip = new JSZip();
  // write the structured result as pretty-printed json
  zip.file('result.json', JSON.stringify(output.result, null, 2));
  // fetch and add each result file (skip any that fail to download), tracking what actually landed
  const artifacts: string[] = [];
  for (const name of output.files ?? []) {
    const bytes = await fetchFile(name);
    if (bytes) {
      // sanitize the tool-supplied name so a hostile path can't escape the archive on extraction
      const entryName = safeZipEntryName(name) || name;
      zip.file(entryName, bytes);
      artifacts.push(entryName);
    }
  }
  // add a manifest describing the run, unless one was actually downloaded into the bundle; gating on
  // the downloaded artifacts (not the declared file list) ensures a bundle whose own manifest failed
  // to download still ends up self-describing
  if (!artifacts.includes(RESULT_METADATA_NAME)) {
    const metadata: ResultsMetadata = {
      tool,
      tool_version: formatImageVersion(output.tool_version) || undefined,
      uploaded: output.uploaded,
      artifacts,
    };
    zip.file(RESULT_METADATA_NAME, JSON.stringify(metadata, null, 2));
  }
  return zip.generateAsync({ type: 'blob' });
}

/**
 * Download a tool result and all of its result files as a single zip archive. The archive is
 * named `<tool>_<sha256>_<uploaded>.zip` (see {@link resultsBundleName}) so bundles for
 * different files, tools, or runs don't collide when saved to the same folder.
 *
 * @param sha256 - The SHA256 of the file the result belongs to.
 * @param tool - The tool that produced the result (also used in the archive name and manifest).
 * @param output - The result to archive (its `result`, `files`, and `id`).
 * @param errorHandler - Called with a formatted message if a result-file download fails.
 * @returns `true` if the archive was built and the download triggered, `false` on failure.
 */
export async function downloadResultsAsZip(
  sha256: string,
  tool: string,
  output: Output,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  try {
    // reuse any bytes already cached from previewing/downloading files of this exact result
    const blob = await buildResultsZip(tool, output, (name) => fetchResultFileCached(sha256, tool, output.id, name, errorHandler));
    downloadBlob(blob, resultsBundleName(sha256, tool, output));
    return true;
  } catch (err) {
    errorHandler(`Failed to build results archive: ${err instanceof Error ? err.message : String(err)}`);
    return false;
  }
}
