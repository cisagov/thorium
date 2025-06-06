// import the base client function that loads from the config
// and injects the token via axios intercepts
import { Graph } from 'models';
import client, { parseRequestError } from './client';
// @ts-ignore
import JSONBigInt from 'json-bigint';

// this will force big ints to be strings
const JSONBigString = JSONBigInt({ storeAsString: true });

/**
 * Build a new tree
 * @async
 * @function
 * @param {object} data - things to search for
 * @param {boolean} filterChildless - filter childless nodes
 * @param {number} limit - max depth of graph of tree
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<Graph | null> } - Request response
 */
export const getInitialTree = async (
  data: any,
  filterChildless: boolean,
  limit: number,
  errorHandler: (error: string) => void,
): Promise<Graph | null> => {
  const url = '/trees/';
  const params: any = {};
  if (filterChildless) {
    params['filter_childless'] = filterChildless;
  }
  params['limit'] = limit;
  return client
    .post(url, { ...data }, { transformResponse: [(data) => data], params: params })
    .then((res) => {
      if (res && res.status && res.status == 200) {
        return JSONBigString.parse(res.data);
      }
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Build Tree');
      return null;
    });
};

/**
 * Grow an existing tree from cursor
 * @async
 * @function
 * @param {string} id -
 * @param {string[]} nodes - nodes to grow graph for
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<Graph | null>} - Request response
 */
export const growTree = async (id: string, nodes: string[], errorHandler: (error: string) => void): Promise<Graph | null> => {
  const url = `/trees/${id}`;
  const params: any = {};
  params['limit'] = 1;
  return client
    .patch(url, { growable: nodes }, { transformResponse: [(data) => data], params: params })
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return JSONBigString.parse(res.data);
      }
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Grow Tree');
      return null;
    });
};
