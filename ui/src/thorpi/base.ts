import client, { parseRequestError } from './client';

/**
 * Fetch the running API server version (`GET /version`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The version string, or `false` if the request failed.
 */
export async function getVersion(errorHandler: (error: string) => void): Promise<string | boolean> {
  const url = '/version';
  return client
    .get<string>(url)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get API Version');
      return false;
    });
}

/**
 * Fetch the instance login/notification banner text (`GET /banner`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The banner string, or `null` if not set or the request failed.
 */
export async function getBanner(errorHandler: (error: string) => void): Promise<string | null> {
  const url = '/banner';
  return client
    .get<string>(url)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Banner');
      return null;
    });
}
