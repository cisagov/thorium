import { Pipeline, PipelineCreate, PipelineUpdate } from '@models/pipelines';
import client, { parseRequestError } from './client';

interface PipelineListResponse {
  cursor?: number;
  names: string[];
}

interface PipelineDetailsListResponse {
  cursor?: number;
  details: Pipeline[];
}

/**
 * Create a new pipeline (`POST /pipelines/`).
 *
 * @param pipeline - The pipeline definition to create (group, name, stage order, triggers, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the pipeline was created (HTTP 204), otherwise `false`.
 */
export async function createPipeline(pipeline: PipelineCreate, errorHandler: (error: string) => void): Promise<boolean> {
  return client
    .post('/pipelines/', pipeline)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create Pipeline');
      return false;
    });
}

/**
 * Delete a pipeline from a group (`DELETE /pipelines/{group}/{pipeline}`).
 *
 * @param group - The group the pipeline belongs to.
 * @param pipeline - The name of the pipeline to delete.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the pipeline was deleted (HTTP 204), otherwise `false`.
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
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete Pipeline');
      return false;
    });
}

/**
 * Fetch a single pipeline by name, scoped to a group (`GET /pipelines/data/{group}/{pipeline}`).
 *
 * @param group - The group the pipeline belongs to.
 * @param pipeline - The name of the pipeline to fetch.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The {@link Pipeline}, or `null` if not found or the request failed.
 */
export async function getPipeline(group: string, pipeline: string, errorHandler: (error: string) => void): Promise<Pipeline | null> {
  const url = '/pipelines/data/' + group + '/' + pipeline;
  return client
    .get<Pipeline>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Pipeline');
      return null;
    });
}

/**
 * List the pipelines in a group, as names or full details (`GET /pipelines/list/{group}/` or `/details/`).
 *
 * @param group - The group whose pipelines to list.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param details - When `true`, return full {@link Pipeline} objects; when `false`, return pipeline names.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @param limit - Maximum number of pipelines to return per page (defaults to 100).
 * @returns An array of {@link Pipeline} details (when `details`) or names, or `null` if the request failed.
 */
export async function listPipelines(
  group: string,
  errorHandler: (error: string) => void,
  details = false,
  cursor: string | null = null,
  limit = 100,
): Promise<Pipeline[] | string[] | null> {
  let url = '/pipelines/list/' + group + '/';
  if (details) {
    url += 'details/';
  }
  const params: { limit: number; cursor?: string } = { limit: limit };
  if (cursor) {
    params['cursor'] = cursor;
  }
  return client
    .get<PipelineListResponse | PipelineDetailsListResponse>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        if (details && 'details' in res.data) {
          return res.data.details;
        } else if (!details && 'names' in res.data) {
          return res.data.names;
        }
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Pipelines');
      return null;
    });
}

/**
 * Update an existing pipeline (`PATCH /pipelines/{group}/{pipeline}`).
 *
 * @param group - The group the pipeline belongs to.
 * @param pipeline - The name of the pipeline to update.
 * @param data - The pipeline fields to change (stage order, SLA, triggers, description, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the update succeeded (HTTP 204), otherwise `false`.
 */
export async function updatePipeline(
  group: string,
  pipeline: string,
  data: PipelineUpdate,
  errorHandler: (error: string) => void,
): Promise<boolean> {
  const url = '/pipelines/' + group + '/' + pipeline;
  return client
    .patch(url, data)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update Pipeline');
      return false;
    });
}
