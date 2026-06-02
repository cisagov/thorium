import client, { parseRequestError } from './client';
import type { AxiosResponse } from './client';

// project imports
import { CommentResponse } from '@models/files';

/**
 * Post a comment (with optional file attachments) on a file (`POST /files/comment/{sha256}`).
 *
 * @param sha256 - The SHA256 of the file to comment on.
 * @param postForm - Multipart form data containing the comment body, groups, and any attachments.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The created comment response, or `null` if the request failed.
 */
export async function postFileComments(
  sha256: string,
  postForm: FormData,
  errorHandler: (error: string) => void,
): Promise<CommentResponse | null> {
  const url = `/files/comment/${sha256}`;
  return client
    .post<CommentResponse>(url, postForm)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Post Comments');
      return null;
    });
}

/**
 * Download a file attached to a comment (`GET /files/comment/download/{sha256}/{commentId}/{attachmentID}`).
 *
 * The raw axios response is returned (rather than just the body) so callers can read
 * headers such as `Content-Disposition` to recover the original filename.
 *
 * @param sha256 - The SHA256 of the file the comment belongs to.
 * @param commentId - The id of the comment holding the attachment.
 * @param attachmentID - The id of the specific attachment to download.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The axios response with the attachment bytes as an `ArrayBuffer`, or `null` if the request failed.
 */
export async function downloadAttachment(
  sha256: string,
  commentId: string,
  attachmentID: string,
  errorHandler: (error: string) => void,
): Promise<AxiosResponse<ArrayBuffer> | null> {
  const url = `/files/comment/download/${sha256}/${commentId}/${attachmentID}`;
  return client
    .get<ArrayBuffer>(url, { responseType: 'arraybuffer' })
    .then((res) => {
      if (res?.status && res.status == 200) {
        return res;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Download Comment Attachment');
      return null;
    });
}
