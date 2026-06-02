import client, { parseRequestError } from './client';

// project imports
import { AssociationCreate } from '@models/associations';

/**
 * Create an association between two entities (`POST /associations/`).
 *
 * @param data - The association to create (source/target entities and metadata).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns An object with the new association's `id`, or `null` if the request failed.
 */
export const createAssociation = async (data: AssociationCreate, errorHandler: (error: string) => void): Promise<{ id: string } | null> => {
  const url = '/associations/';
  return client
    .post<{ id: string }>(url, data)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create Associations');
      return null;
    });
};
