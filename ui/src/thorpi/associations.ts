import client, { parseRequestError } from './client';

/**
 * Create a new entity instance
 * @async
 * @function
 * @param {FormData} [data] - new entity spec
 * @param {object} errorHandler - error handler function
 * @returns {object} - Promise object representing response data
 */
export const createAssociation = async (data: any, errorHandler: (error: string) => void): Promise<{ id: string } | null> => {
  // build url parameters including optional args if specified
  let url = '/associations/';
  return client
    .post(url, data)
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create Associations');
      return null;
    });
};
