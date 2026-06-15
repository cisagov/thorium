import client, { parseRequestError } from './client';
import type { AxiosResponse } from './client';

// project imports
import { EntityRequest } from '@models/entities/requests';
import { OutputMap } from '@models/results';

/**
 * Fetch all tool results for a file (`GET /files/results/{sha256}`).
 *
 * @param sha256 - The SHA256 of the file whose results to fetch.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param data - Optional axios request config (e.g. `params` to filter by tool/group).
 * @returns The {@link OutputMap} of tool results keyed by tool, or `null` if the request failed.
 */
export async function getResults(
  sha256: string,
  errorHandler: (error: string) => void,
  data: Record<string, unknown> = {},
): Promise<OutputMap | null> {
  const url = '/files/results/' + sha256;
  return client
    .get<OutputMap>(url, data)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Results');
      return null;
    });
}

/**
 * Download a single result file produced by a tool run (`GET /files/result-files/{sha256}/{tool}/{id}`).
 *
 * The raw axios response is returned (rather than just the body) so callers can read response
 * headers when saving the file.
 *
 * @param sha256 - The SHA256 of the file the result belongs to.
 * @param tool - The tool that produced the result.
 * @param id - The id of the specific result/run.
 * @param name - The name of the result file to download (sent as the `result_file` query param).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The axios response with the result file bytes as an `ArrayBuffer`, or `null` if the request failed.
 */
export async function getResultsFile(
  sha256: string,
  tool: string,
  id: string,
  name: string,
  errorHandler: (error: string) => void,
): Promise<AxiosResponse<ArrayBuffer> | null> {
  const url = `/files/result-files/${sha256}/${tool}/${id}`;
  const data = {
    result_file: name,
  };
  return client
    .get<ArrayBuffer>(url, { params: data, responseType: 'arraybuffer' })
    .then((res) => {
      if (res?.status == 200) {
        return res;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Results File');
      return null;
    });
}

/**
 * Download the entities of a single kind produced by a tool run
 * (`GET /files/result-files/{sha256}/{tool}/{result_id}/{entity_kind}`).
 *
 * The body is a JSON array of {@link EntityRequest}. Unlike {@link getResultsFile}, the target is
 * identified by an `entity_kind` path segment rather than a `result_file` query param.
 *
 * @param sha256 - The SHA256 of the file the result belongs to.
 * @param tool - The tool that produced the result.
 * @param resultId - The id of the specific result/run.
 * @param entityKind - The entity kind to fetch (e.g. `Device`, `Other`).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The parsed array of entity requests, or `null` if the request failed.
 */
export async function getResultEntities(
  sha256: string,
  tool: string,
  resultId: string,
  entityKind: string,
  errorHandler: (error: string) => void,
): Promise<EntityRequest[] | null> {
  const url = `/files/result-files/${sha256}/${tool}/${resultId}/${entityKind}`;
  return client
    .get<EntityRequest[] | string>(url)
    .then((res) => {
      if (res?.status == 200 && res.data != null) {
        // tolerate a string body (e.g. when the server doesn't set a JSON content-type)
        return typeof res.data === 'string' ? (JSON.parse(res.data) as EntityRequest[]) : res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Result Entities');
      return null;
    });
}
