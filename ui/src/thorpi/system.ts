import client, { parseRequestError } from './client';

// project imports
import { Stats, SystemSettings } from '@models/system';

/**
 * Fetch system-wide runtime statistics (`GET /system/stats`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The system {@link Stats}, or `null` if the request failed.
 */
export async function getSystemStats(errorHandler: (error: string) => void): Promise<Stats | null> {
  const url = '/system/stats';
  return client
    .get<Stats>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get System Stats');
      return null;
    });
}

/**
 * Fetch the system configuration settings (`GET /system/settings`).
 *
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The {@link SystemSettings}, or `null` if the request failed.
 */
export async function getSystemSettings(errorHandler: (error: string) => void): Promise<SystemSettings | null> {
  const url = '/system/settings';
  return client
    .get<SystemSettings>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get System Settings');
      return null;
    });
}
