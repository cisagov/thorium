// import the base client function that loads from the config
// and injects the token via axios intercepts
import client, { parseRequestError } from './client';

/**
 * Get results for a hash.
 * @async
 * @function
 * @param {string} sha256 - hash of sample to get results for
 * @param {(error: string) => void} errorHandler - error handler function
 * @param {object} data - search parameters to retrieve results can include
 *     groups: list of groups results can be member of
 *     tools: list of tools to get results for
 * @returns {Promise<any | null>} - results dict
 */
export async function getResults(sha256: string, errorHandler: (error: string) => void, data = {}): Promise<any | null> {
  const url = '/files/results/' + sha256;
  return client
    .get(url, data)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete User');
      return null;
    });
}

/**
 * Get results for a hash.
 * @async
 * @function
 * @param {string} sha256 - hash of sample to get results for
 * @param {string} tool - name of image that created the result
 * @param {string} id - id of the result
 * @param {string} name - name of result file to retrieve
 * @param {(error: string) => void} errorHandler - error handler function
 * @returns {Promise<any | null>} - results dict
 */
export async function getResultsFile(
  sha256: string,
  tool: string,
  id: string,
  name: string,
  errorHandler: (error: string) => void,
): Promise<any | null> {
  const url = `/files/result-files/${sha256}/${tool}/${id}`;
  // add the name of the result file to our params
  const data = {
    result_file: name,
  };
  return client
    .get(url, { params: data, responseType: 'arraybuffer' })
    .then((res) => {
      if (res?.status == 200) {
        // return full res, we may need the headers to build out a file object
        return res;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get Results File');
      return null;
    });
}
