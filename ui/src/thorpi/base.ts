// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';

/**
 * Get Thorium API and UI version
 * @async
 * @function
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<string | boolean>} - Request response
 */
export async function getVersion(errorHandler: (error: string) => void): Promise<string | boolean> {
  const url = '/version';
  return client
    .get(url)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get API Version');
      return false;
    });
}

/**
 * Get Thorium banner
 * @async
 * @function
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<string | boolean>} - Request response
 */
export async function getBanner(errorHandler: (error: string) => void): Promise<string | boolean> {
  const url = '/banner';
  return client
    .get(url)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get Banner');
      return false;
    });
}
