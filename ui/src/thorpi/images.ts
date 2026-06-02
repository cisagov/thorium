import client, { parseRequestError } from './client';

// project imports
import { Image, ImageDetailsList, ImageList, ImageRequest } from '@models/images';

/**
 * Create a new image (analysis tool definition) (`POST /images/`).
 *
 * @param data - The image definition to create (group, name, scaler, resources, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the image was created (HTTP 204), otherwise `false`.
 */
export async function createImage(data: ImageRequest, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/images/';
  return client
    .post(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create Image');
      return false;
    });
}

/**
 * Delete an image from a group (`DELETE /images/{group}/{image}`).
 *
 * @param group - The group the image belongs to.
 * @param image - The name of the image to delete.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the image was deleted (HTTP 204), otherwise `false`.
 */
export async function deleteImage(group: string, image: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/images/' + group + '/' + image;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete Image');
      return false;
    });
}

/**
 * Fetch a single image by name, scoped to a group (`GET /images/data/{group}/{image}`).
 *
 * Note: this function does not take an `errorHandler`; failures are logged to the console
 * and surfaced to callers as a `null` return.
 *
 * @param group - The group the image belongs to.
 * @param image - The name of the image to fetch.
 * @returns The {@link Image}, or `null` if not found or the request failed.
 */
export async function getImage(group: string, image: string): Promise<Image | null> {
  const url = '/images/data/' + group + '/' + image;
  return client
    .get<Image>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, console.log, 'Get Image');
      return null;
    });
}

/**
 * List the images in a group, as names or full details (`GET /images/{group}/` or `/details/`).
 *
 * @param group - The group whose images to list.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param details - When `true`, return full {@link Image} objects; when `false`, return an {@link ImageList} of names.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @param limit - Maximum number of images to return per page (defaults to 100).
 * @returns An array of {@link Image} details (when `details`) or an {@link ImageList}, or `null` if the request failed.
 */
export async function listImages(
  group: string,
  errorHandler: (error: string) => void,
  details = false,
  cursor: string | null = null,
  limit = 100,
): Promise<Image[] | ImageList | null> {
  let url = '/images/' + group + '/';
  if (details) {
    url += 'details/';
  }
  const params: { limit: number; cursor?: string } = { limit };
  if (cursor) {
    params['cursor'] = cursor;
  }
  return client
    .get<ImageList | ImageDetailsList>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if (details && 'details' in res.data) {
          return res.data.details;
        }
        return res.data as ImageList;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Images');
      return null;
    });
}

/**
 * Update an existing image (`PATCH /images/{group}/{image}`).
 *
 * @param group - The group the image belongs to.
 * @param image - The name of the image to update.
 * @param data - The subset of image fields to change.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the update succeeded (HTTP 204), otherwise `false`.
 */
export async function updateImage(
  group: string,
  image: string,
  data: Partial<ImageRequest>,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const url = '/images/' + group + '/' + image;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update Image');
      return false;
    });
}
