// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';

/**
 * Create a new Thorium image.
 * @async
 * @function
 * @param {string} data - Image details to submit when creating new image
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {object} - Request response
 */
export async function createImage(data: any, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/images/';
  return client
    .post(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create Image');
      return false;
    });
}

/**
 * Delete an image by name.
 * @async
 * @function
 * @param {string} group - Group name target image is owned by
 * @param {string} image - Name of image being deleted
 * @param {(error: string) => void)} errorHandler - error handler function
 * @returns {object} - Request response
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
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete Image');
      return false;
    });
}

/**
 * Get details for an image.
 * @async
 * @function
 * @param {string} group - Group name target image is owned by
 * @param {string} image - Name of image to retrieve details about
 * @returns {object} - Image details
 */
export async function getImage(group: string, image: string): Promise<any | null> {
  const url = '/images/data/' + group + '/' + image;
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200) {
        return res;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, console.log, 'Get Image');
      return null;
    });
}

/**
 * Get a list of images with optional details for a particular group.
 * @async
 * @function
 * @param {string} group - Group name to list owned images for
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {boolean} details - Whether to return details for listed images
 * @param {string} cursor - the cursor value to continue listing from
 * @param {number} limit - number of images to return
 * @returns {object} - images list with optional details
 */
export async function listImages(
  group: string,
  errorHandler: (error: string) => void,
  details = false,
  cursor = null,
  limit = 100,
): Promise<any | null> {
  let url = '/images/' + group + '/';
  if (details) {
    url += 'details/';
  }
  // pass in limit and cursor value
  const params: any = {};
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;
  return client
    .get(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if (details && res.data.details) {
          return res.data.details;
        }
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'List Images');
      return null;
    });
}

/**
 * Update details for an image.
 * @async
 * @function
 * @param {string} group - Group name target image is owned by
 * @param {string} image - Name of image to patch
 * @param {object} data - Json body to patch image with
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {object} - image details
 */
export async function updateImage(group: string, image: string, data: any, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/images/' + group + '/' + image;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update Image');
      return false;
    });
}
