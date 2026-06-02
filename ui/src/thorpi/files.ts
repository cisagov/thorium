import client, { parseRequestError } from './client';

// project imports
import { RequestTags, TagCounts } from '@models/tags';
import { Sample, SampleSubmissionResponse } from '@models/files';
import { Filters } from '@models/search';

// Debugging errors randomly inserted.
// Valid values 0-100 (percentage chance of error)
const RANDOM_DEBUG_ERRORS = 0;

interface ApiCursor<T> {
  cursor?: string;
  data: T[];
}

/**
 * List files matching the given filters (`GET /files` or `/files/details/`).
 *
 * Results are paginated via an opaque cursor: pass the returned `cursor` back in on the next
 * call to fetch the following page (`null` cursor means no more pages).
 *
 * @param data - Search/filter parameters (groups, tags, time range, limit, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param details - When `true`, request full {@link Sample} objects (`/details/`) instead of summaries.
 * @param cursor - Pagination cursor from a previous call, or `null`/omitted for the first page.
 * @returns The page of files and the next-page cursor (`cursor` is `null` when exhausted).
 *          On failure, returns an empty list and a `null` cursor.
 */
export async function listFiles(
  data: Filters,
  errorHandler: (error: string) => void,
  details?: boolean | null,
  cursor?: string | null,
): Promise<{ files: Sample[]; cursor: string | null }> {
  let url = '/files';
  if (details) {
    url += '/details/';
  }
  if (cursor) {
    data.cursor = cursor;
  }
  return client
    .get<ApiCursor<Sample>>(url, { params: data })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        const cursor = res.data.cursor ?? null;
        return { files: res.data.data, cursor: cursor };
      }
      return { files: [], cursor: null };
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Files');
      return { files: [], cursor: null };
    });
}

/**
 * Upload a file to Thorium (`POST /files/`).
 *
 * Upload progress is reported through `progressHandler` and the upload can be cancelled via the
 * supplied `AbortController`. A 409 conflict (file already exists) is treated as success: the
 * existing file's SHA256 is returned in the response so callers can navigate to it.
 *
 * @param form - Multipart form data containing the file bytes plus submission metadata (groups, tags, etc.).
 * @param errorHandler - Called with a formatted message if the upload fails or is cancelled.
 * @param progressHandler - Called with upload progress in the range 0–1 as bytes are sent.
 * @param controller - Abort controller used to cancel the in-flight upload.
 * @returns The submission response (including the file's SHA256), or `false` if the upload failed or was cancelled.
 */
export async function uploadFile(
  form: FormData,
  errorHandler: (error: string) => void,
  progressHandler: (progress: number) => void,
  controller: AbortController,
): Promise<SampleSubmissionResponse | false> {
  const url = '/files/';
  const config = {
    onUploadProgress: (progressEvent: { progress?: number }) => progressHandler(progressEvent.progress ?? 0),
    signal: controller.signal,
  };
  if (RANDOM_DEBUG_ERRORS) {
    if (Math.floor(Math.random() * 100) < RANDOM_DEBUG_ERRORS) {
      errorHandler(`Failed to upload file: Permission Denied`);
      return false;
    }
  }
  return client
    .post<SampleSubmissionResponse>(url, form, config)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if ('sha256' in res.data) {
          return res.data;
        } else {
          errorHandler('Error: file upload response did not contain a hash (proxy error?).');
        }
      }
      if (controller.signal.aborted) {
        errorHandler('Upload cancelled by user');
      }
      return false;
    })
    .catch((error: unknown) => {
      if (
        typeof error === 'object' &&
        error !== null &&
        'response' in error &&
        typeof (error as Record<string, unknown>).response === 'object'
      ) {
        const axiosError = error as { response: { status: number; data: { error?: string } } };
        if (axiosError.response?.status == 409 && axiosError.response?.data?.error) {
          return { sha256: axiosError.response.data.error } as SampleSubmissionResponse;
        }
      }
      parseRequestError(error, errorHandler, 'Upload File');
      return false;
    });
}

/**
 * Download a file's bytes as an archive (`GET /files/sample/{sha256}/download`).
 *
 * Defaults to Thorium's CaRT format. When `archiveFormat` is `'Encrypted ZIP'` the file is
 * instead returned as a password-protected zip (the `/zip` endpoint) using `archivePassword`.
 *
 * @param sha256 - The SHA256 of the file to download.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param archiveFormat - Archive format to request: `'CaRT'` (default) or `'Encrypted ZIP'`.
 * @param archivePassword - Password applied to the zip when `archiveFormat` is `'Encrypted ZIP'` (defaults to `'infected'`).
 * @returns The archived file bytes as an `ArrayBuffer`, or `null` if the request failed.
 */
export async function getFile(
  sha256: string,
  errorHandler: (error: string) => void,
  archiveFormat = 'CaRT',
  archivePassword = 'infected',
): Promise<ArrayBuffer | null> {
  let url = '/files/sample/' + sha256 + '/download';
  const options: { responseType: 'arraybuffer'; params?: { password: string } } = { responseType: 'arraybuffer' };
  if (archiveFormat == 'Encrypted ZIP') {
    url = url + '/zip';
    if (archivePassword) {
      options['params'] = { password: archivePassword };
    }
  }
  return client
    .get<ArrayBuffer>(url, options)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Download File');
      return null;
    });
}

/**
 * Fetch a file's full metadata (`GET /files/sample/{sha256}`).
 *
 * @param sha256 - The SHA256 of the file to fetch.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The {@link Sample} details, or `null` if not found or the request failed.
 */
export async function getFileDetails(sha256: string, errorHandler: (error: string) => void): Promise<Sample | null> {
  const url = '/files/sample/' + sha256;
  return client
    .get<Sample>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get File Details');
      return null;
    });
}

/**
 * Add tags to a file (`POST /files/tags/{sha256}`).
 *
 * @param sha256 - The SHA256 of the file to tag.
 * @param tags - Map of tag keys to value lists to add.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the tags were added (HTTP 204), otherwise `false`.
 */
export async function uploadTags(sha256: string, tags: RequestTags, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/files/tags/' + sha256;
  return client
    .post(url, tags)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create File Tags');
      return false;
    });
}

/**
 * Remove tags from a file (`DELETE /files/tags/{sha256}`).
 *
 * @param sha256 - The SHA256 of the file to untag.
 * @param tags - Map of tag keys to value lists to remove.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the tags were removed (HTTP 204), otherwise `false`.
 */
export async function deleteTags(sha256: string, tags: RequestTags, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/files/tags/' + sha256;
  return client
    .delete(url, { data: tags })
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete File Tags');
      return false;
    });
}

/**
 * Delete a single submission of a file (`DELETE /files/sample/{sha256}/{id}`).
 *
 * A file can be submitted multiple times (by different users/groups); this removes one
 * submission scoped to the given groups rather than the underlying sample.
 *
 * @param sha256 - The SHA256 of the file the submission belongs to.
 * @param id - The id of the submission to delete.
 * @param groups - The groups to remove the submission from.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the submission was deleted (HTTP 204), otherwise `false`.
 */
export async function deleteSubmission(
  sha256: string,
  id: string,
  groups: string[],
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const params = { groups: groups };
  const url = '/files/sample/' + sha256 + '/' + id;
  return client
    .delete(url, { params: params })
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete File Submission');
      return false;
    });
}

/**
 * Update a file's submission metadata (`PATCH /files/sample/{sha256}`).
 *
 * @param sha256 - The SHA256 of the file to update.
 * @param data - The submission fields to change (e.g. groups, description).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the update succeeded (HTTP 204), otherwise `false`.
 */
export async function updateFileSubmission(
  sha256: string,
  data: Record<string, unknown>,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const url = '/files/sample/' + sha256;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update File Submission');
      return false;
    });
}

/**
 * Fetch aggregate tag counts across files matching the filters (`GET /files/count/`).
 *
 * Used to populate tag autocomplete options with their occurrence counts.
 *
 * @param data - Search/filter parameters constraining which files are counted (groups, limit, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param cursor - Pagination cursor from a previous call, or `null`/omitted for the first page.
 * @returns The {@link TagCounts} aggregation, or `null` if the request failed.
 */
export async function countFileTags(
  data: Filters,
  errorHandler: (error: string) => void,
  cursor?: string | null,
): Promise<TagCounts | null> {
  const url = '/files/count/';
  if (cursor) {
    data.cursor = cursor;
  }
  return client
    .get<TagCounts>(url, { params: data })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Files');
      return null;
    });
}
