// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';
import { CreateTags, Filters, FilterTags } from '@models';

// Debugging errors randomly inserted.
// Valid values 0-100 (percentage chance of error)
const RANDOM_DEBUG_ERRORS = 0;

/**
 * Get a list of files
 * @async
 * @function
 * @param {object} [data] - optional request parameters which includes:
 *   - groups:  to which the files are viewable
 *   - start: start date for search range
 *   - submission: uuid for previously searched submission
 *   - end: end date for search range
 *   - limit:  the max number of submissions to return
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {boolean} details - whether to return details for listed submissions
 * @param {string} cursor - the cursor value to continue listing from
 * @returns {Promise<any[]>} - Promise object representing a list of file details.
 */
export async function listFiles(
  data: Filters,
  errorHandler: (error: string) => void,
  details?: boolean | null,
  cursor?: string | null,
): Promise<{ files: any[]; cursor: string | null }> {
  // build url parameters including optional args if specified
  let url = '/files';
  if (details) {
    url += '/details/';
  }
  // pass in limit and cursor value
  if (cursor) {
    data.cursor = cursor;
  }
  return client
    .get(url, { params: data })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        const cursor = res.data.cursor ? (res.data.cursor as string) : null;
        return { files: res.data.data as any[], cursor: cursor };
      }
      return { files: [] as any[], cursor: null };
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'List Files');
      return { files: [] as any[], cursor: null };
    });
}

/**
 * Upload a file.
 * @async
 * @function
 * @param {FormData} form - form object containing file and submission info
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {any} progressHandler - progress report handler function
 * @param {any} controller - abort controller
 * @returns {object} - promise object representing file post response
 */
export async function uploadFile(
  form: FormData,
  errorHandler: (error: string) => void,
  progressHandler: any,
  controller: any,
): Promise<any | boolean> {
  const url = '/files/';
  const config = {
    onUploadProgress: (progressEvent: any) => progressHandler(progressEvent.progress),
    signal: controller.signal,
  };
  if (RANDOM_DEBUG_ERRORS) {
    if (Math.floor(Math.random() * 100) < RANDOM_DEBUG_ERRORS) {
      errorHandler(`Failed to upload file: Permission Denied`);
      return false;
    }
  }
  return client
    .post(url, form, config)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if ('sha256' in res.data) {
          return res.data;
        } else {
          // got a valid response but it didn't contain a sha256, probably a proxy error
          errorHandler('Error: file upload response did not contain a hash (proxy error?).');
        }
      }
      if (controller.signal.aborted) {
        errorHandler('Upload cancelled by user');
        // no message or error returned
      }
      return false;
    })
    .catch((error) => {
      // special handler for file already exists
      if (error?.response?.status == 409 && error?.response?.data?.error) {
        return { sha256: error.response.data.error };
      }
      parseRequestError(error, errorHandler, 'List Files');
      return { files: [] as any[], cursor: null };
    });
}

/**
 * Download a file.
 * @async
 * @function
 * @param {string} sha256 - sha256 hash of sample to update
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {string} [archiveFormat=CaRT] - the archive format used for encapsulating the downloaded file
 * @param {string} [archivePassword=infected] - password to unencrypt the downloaded archive
 * @returns {Promise<ArrayBuffer | null>} - promise of the downloaded file
 */
export async function getFile(
  sha256: string,
  errorHandler: (error: string) => void,
  archiveFormat = 'CaRT',
  archivePassword = 'infected',
): Promise<ArrayBuffer | null> {
  let url = '/files/sample/' + sha256 + '/download';
  const options: any = { responseType: 'arraybuffer' };
  // downloading a zip is a url subpath and can take a password
  if (archiveFormat == 'Encrypted ZIP') {
    url = url + '/zip';
    if (archivePassword) {
      options['params'] = { password: archivePassword };
    }
  }
  return client
    .get(url, options)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        // pass back full response because data and content type headers are needed
        return res.data as ArrayBuffer;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Download File');
      return null;
    });
}

/**
 * Get file details
 * @async
 * @function
 * @param {string} sha256 - sha256 hash of sample info to get
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - promise of the submission info for a file
 */
export async function getFileDetails(sha256: string, errorHandler: (error: string) => void): Promise<any | null> {
  const url = '/files/sample/' + sha256;
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get File Details');
      return null;
    });
}

/**
 * Update file submission details.
 * @async
 * @function
 * @param {string} sha256 - sha256 hash of sample to update
 * @param {CreateTags} tags - tags to add (and optionally groups to add them to)
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - promise object representing tags post response.
 */
export async function uploadTags(sha256: string, tags: CreateTags, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/files/tags/' + sha256;
  return client
    .post(url, tags)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create File Tags');
      return false;
    });
}

/**
 * Delete file tags
 * @async
 * @function
 * @param {string} sha256 - sha256 hash of sample to update
 * @param {CreateTags} tags - tags to delete (and optionally groups to delete them from)
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - promise of delete file tags success boolean
 */
export async function deleteTags(sha256: string, tags: CreateTags, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/files/tags/' + sha256;
  return client
    .delete(url, { data: tags })
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete File Tags');
      return false;
    });
}

/**
 * Delete a file submission
 * @async
 * @function
 * @param {string} sha256 - sha256 hash of sample to update
 * @param {string} id - submission id to delete
 * @param {string[]} groups - groups to delete from, otherwise ALL of them
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - promise of delete submission success boolean
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
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete File Submission');
      return false;
    });
}

/**
 * Update file submission details
 * @async
 * @function
 * @param {string} sha256 - sha256 hash of sample to update
 * @param {object} data - data to update submission with
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - promise of update submission success boolean
 */
export async function updateFileSubmission(sha256: string, data: any, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/files/sample/' + sha256;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update File Submission');
      return false;
    });
}
