// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';

/**
 * Submit a comment about a file.
 * @async
 * @function
 * @param {string} sha256 - the hash of the sample the comment is about
 * @param {FormData} postForm - json formatted params to post including comment
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - promise object representing comment post response
 */
export async function postFileComments(sha256: string, postForm: FormData, errorHandler: (error: string) => void): Promise<any | null> {
  const url = `/files/comment/${sha256}`;
  return client
    .post(url, postForm)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Post Comments');
      return null;
    });
}

/**
 * Download a comment attachment.
 * @async
 * @function
 * @param {string} sha256 - the sha256 hash of the sample the comment file is about
 * @param {string} commentId - the UUID of the comment to download from
 * @param {string} attachmentID - the UUID of the comment attachment to download
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - promise object representing comment post response
 */
export async function downloadAttachment(
  sha256: string,
  commentId: string,
  attachmentID: string,
  errorHandler: (error: string) => void,
): Promise<any | null> {
  const url = `/files/comment/download/${sha256}/${commentId}/${attachmentID}`;
  return client
    .get(url, { responseType: 'arraybuffer' })
    .then((res) => {
      if (res?.status && res.status == 200) {
        // pass back full response because data and content type headers are needed
        return res;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Download Comment Attachment');
      return null;
    });
}
