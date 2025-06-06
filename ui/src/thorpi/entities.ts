// import the base client function that loads from the config
// and injects the token via axios intercepts
import { EntityTypes, Filters, UpdateEntityForm } from 'models';
import client, { parseRequestError } from './client';

/**
 * Create a new entity instance
 * @async
 * @function
 * @param {FormData} [data] - new entity spec
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<{id: string} | null>} - Promise object representing response data
 */
export const createEntity = async (data: FormData, errorHandler: (error: string) => void): Promise<{ id: string } | null> => {
  // build url parameters including optional args if specified
  let url = '/entities/';
  return client
    .post(url, data)
    .then((res) => {
      if (res && res.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create Entity');
      return null;
    });
};

/**
 * Get an existing entity instance
 * @async
 * @function
 * @param {string} [id] - uuid for requested entity
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<EntityTypes | null>} - Promise object representing response data
 */
export const getEntity = async (id: string, errorHandler: (error: string) => void): Promise<EntityTypes | null> => {
  // build url parameters including optional args if specified
  let url = `/entities/${id}`;
  return client
    .get(url)
    .then((res) => {
      if (res && res.status && res.status == 200 && res.data) {
        return res.data;
      }
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update Entity');
      return null;
    });
};

/**
 * Update an existing entity
 * @async
 * @function
 * @param {FormData} data - update params
 * @param {string} [id] - uuid for requested entity
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - Promise object representing response data
 */
export const updateEntity = async (id: string, data: FormData, errorHandler: (error: string) => void): Promise<boolean> => {
  // build url parameters including optional args if specified
  let url = `/entities/${id}`;
  return client
    .patch(url, data)
    .then((res) => {
      if (res && res.status && res.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update Entity');
      return false;
    });
};

/**
 * Delete an existing entity
 * @async
 * @function
 * @param {string} [id] - uuid for target entity
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<boolean>} - Promise object representing response data
 */
export const deleteEntity = async (id: string, errorHandler: (error: string) => void): Promise<boolean> => {
  // build url parameters including optional args if specified
  let url = `/entities/${id}`;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status && res.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete Entity');
      return false;
    });
};

/**
 * Get a list of entities
 * @async
 * @function
 * @param {Filters} [data] - optional filter parameters which includes:
 *   - groups: to which the entities are viewable
 *   - start: start date for search range
 *   - end: end date for search range
 *   - limit:  the max number of submissions to return
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {boolean} details - whether to return details for listed submissions
 * @param {string} cursor - the cursor value to continue listing from
 * @returns {Promise<{entityList: EntityTypes[]: entityCursor} | null>} - Promise object representing a list of file details.
 */
export const listEntities = async (
  data: Filters,
  errorHandler: (error: string) => void,
  details: boolean,
  cursor: string | null,
): Promise<{ entityList: EntityTypes[]; entityCursor: string | null }> => {
  // build url parameters including optional args if specified
  let url = '/entities';
  if (details) {
    url += '/details/';
  }
  // pass in cursor value
  if (cursor) {
    data.cursor = cursor;
  }
  return client
    .get(url, { params: data })
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        const cursor = res.data.cursor ? (res.data.cursor as string) : null;
        return { entityList: res.data.data as any[], entityCursor: cursor };
      }
      return { entityList: [] as any[], entityCursor: null };
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update Entity');
      return { entityList: [] as any[], entityCursor: null };
    });
};
