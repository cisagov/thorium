// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';

/**
 * Get Thorium system stats
 * @async
 * @function
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - Request response
 */
export async function getSystemStats(errorHandler: (error: string) => void): Promise<any | null> {
  const url = '/system/stats';
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get System Stats');
      return null;
    });
}

/**
 * Get Thorium system settings
 * @async
 * @function
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - Request response
 */
export async function getSystemSettings(errorHandler: (error: string) => void): Promise<any | null> {
  const url = '/system/settings';
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200) {
        return res.data;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get System Settings');
      return false;
    });
}
