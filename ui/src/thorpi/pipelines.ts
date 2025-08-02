// import the base client function that loads from the config
// and injects the token via axios intercepts
import { Pipeline, PipelineCreate } from 'models/pipelines';
import client, { parseRequestError } from './client';

/**
 * Create a new Thorium pipeline.
 * @async
 * @function
 * @param {any} pipeline - Pipeline details to submit when creating new pipeline
 * @param {(error: string) => void} errorHandler - Error handler function
 * @returns {Promise<boolean>} - Request response
 */
export async function createPipeline(pipeline: PipelineCreate, errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .post('/pipelines/', pipeline)
    .then((res) => {
      // check for errors in the request response
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Create Pipeline');
      return false;
    });
}

/**
 * Delete a pipeline by name.
 * @async
 * @function
 * @param {string} group - Group name target pipeline is owned by
 * @param {string} pipeline - Name of pipeline being deleted
 * @param {(error: string) => void} errorHandler - Error handler function
 * @returns {Promise<boolean>} - Request response
 */
export async function deletePipeline(group: string, pipeline: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/pipelines/' + group + '/' + pipeline;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Delete Pipeline');
      return false;
    });
}

/**
 * Get details for a pipeline.
 * @async
 * @function
 * @param {string} group - Group of pipeline
 * @param {string} pipeline - Name of pipeline
 * @param {(error: string) => void} errorHandler - Error handler function
 * @returns {Promise<any | null>} - pipeline details
 */
export async function getPipeline(group: string, pipeline: string, errorHandler: (error: string) => void): Promise<any | null> {
  const url = '/pipelines/data/' + group + '/' + pipeline;
  return client
    .get(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Get Pipeline');
      return null;
    });
}

/**
 * Get a list of pipelines with optional details for a particular group.
 * @async
 * @function
 * @param {string} group - group name to list owned pipelines for
 * @param {(error: string) => void} errorHandler - Error handler function
 * @param {boolean} details - whether to return details for listed pipelines
 * @param {string} cursor - the cursor value to continue listing from
 * @param {number} limit - number of pipelines to return
 * @returns {Promise<Pipeline[] | string[] | null>} - pipelines list with optional details
 */
export async function listPipelines(
  group: string,
  errorHandler: (error: string) => void,
  details = false,
  cursor = null,
  limit = 100,
): Promise<Pipeline[] | string[] | null> {
  let url = '/pipelines/list/' + group + '/';
  if (details) {
    url += 'details/';
  }
  // pass in limit and cursor value
  const params: any = { limit: limit };
  if (cursor) {
    params['cursor'] = cursor;
  }
  return client
    .get(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if (details && res.data.details) {
          return res.data.details;
        } else if (!details && res.data.names) {
          return res.data.names;
        }
      }
      return null;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'List Pipelines');
      return null;
    });
}

/**
 * Update details for a pipeline.
 * @async
 * @function
 * @param {string} group - Group name target pipeline is owned by
 * @param {string} pipeline - Name of pipeline to patch
 * @param {any} data - Json body to patch pipeline with
 * @param {(error: string) => void} errorHandler - Error handler function
 * @returns {Promise<boolean>} - request response
 */
export async function updatePipeline(group: string, pipeline: string, data: any, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/pipelines/' + group + '/' + pipeline;
  return client
    .patch(url, data)
    .then((res) => {
      // response was successful, reload pipeline resource
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error) => {
      parseRequestError(error, errorHandler, 'Update Pipeline');
      return false;
    });
}
