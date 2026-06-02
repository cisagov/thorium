import client, { parseRequestError } from './client';

// project imports
import { Reaction, ReactionIdResponse, ReactionRequest, StageLogs, StatusUpdate } from '@models/reactions';

// Debugging errors randomly inserted.
// Valid values 0-100 (percentage chance of error)
const RANDOM_DEBUG_ERRORS = 0;

interface ReactionListResponse {
  cursor?: number;
  names: string[];
}

interface ReactionDetailsListResponse {
  cursor?: number;
  details: Reaction[];
}

/**
 * Create (launch) a reaction — a run of a pipeline against a target (`POST /reactions/`).
 *
 * @param reaction - The reaction request (group, pipeline, target sample(s), args, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param tags - Optional tags to attach to the reaction; merged into the request when not `null`.
 * @returns The created reaction's id response, or `null` if the request failed.
 */
export async function createReaction(
  reaction: ReactionRequest,
  errorHandler: (error: string) => void,
  tags: string[] | null = null,
): Promise<ReactionIdResponse | null> {
  const url = '/reactions/';
  if (tags != null) {
    reaction['tags'] = tags;
  }
  if (RANDOM_DEBUG_ERRORS) {
    if (Math.floor(Math.random() * 100) < RANDOM_DEBUG_ERRORS) {
      errorHandler(`Failed to create reaction: Permission Denied`);
      return null;
    }
  }
  return client
    .post<ReactionIdResponse>(url, reaction)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create Reaction');
      return null;
    });
}

/**
 * Fetch a single reaction by id (`GET /reactions/{group}/{uuid}`).
 *
 * @param group - The group the reaction belongs to.
 * @param uuid - The reaction's unique id.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The {@link Reaction}, or `null` if not found or the request failed.
 */
export async function getReaction(group: string, uuid: string, errorHandler: (error: string) => void): Promise<Reaction | null> {
  const url = '/reactions/' + group + '/' + uuid;
  return client
    .get<Reaction>(url)
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Reaction');
      return null;
    });
}

/**
 * Fetch a reaction's status-update log (`GET /reactions/logs/{group}/{uuid}`).
 *
 * These are the high-level lifecycle status updates for the reaction (not per-stage stdout —
 * see {@link getReactionStageLogs} for that).
 *
 * @param group - The group the reaction belongs to.
 * @param uuid - The reaction's unique id.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @param limit - Maximum number of status updates to return per page (defaults to 100).
 * @returns An array of {@link StatusUpdate}s, or `null` if the request failed.
 */
export async function getReactionLogs(
  group: string,
  uuid: string,
  errorHandler: (error: string) => void,
  cursor: string | null = null,
  limit = 100,
): Promise<StatusUpdate[] | null> {
  const url = '/reactions/logs/' + group + '/' + uuid;
  const params: { cursor?: string; limit?: number } = {};
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;
  return client
    .get<StatusUpdate[]>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Reaction Logs');
      return null;
    });
}

/**
 * Fetch the captured stdout/stderr log lines for a single stage of a reaction
 * (`GET /reactions/logs/{group}/{uuid}/{stage}`).
 *
 * @param group - The group the reaction belongs to.
 * @param uuid - The reaction's unique id.
 * @param stage - The name of the pipeline stage whose logs to fetch.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @param limit - Maximum number of log lines to return per page (defaults to 100).
 * @returns The stage's log lines, or `null` if the request failed or no logs exist.
 */
export const getReactionStageLogs = async (
  group: string,
  uuid: string,
  stage: string,
  errorHandler: (error: string) => void,
  cursor: string | null = null,
  limit = 100,
): Promise<string[] | null> => {
  const url = '/reactions/logs/' + group + '/' + uuid + '/' + stage;
  const params: { cursor?: string; limit?: number } = {};
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;
  return client
    .get<StageLogs>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data?.logs) {
        return res.data.logs;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Reaction Stage Logs');
      return null;
    });
};

/**
 * List reactions in a group, filtered by pipeline or by tag (`GET /reactions/...`).
 *
 * When `tag` is empty the list is scoped by `pipeline` (`/reactions/list/{group}/{pipeline}/`);
 * when `tag` is provided it takes precedence and the list is scoped by tag
 * (`/reactions/tag/{group}/{tag}/`).
 *
 * @param group - The group whose reactions to list.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param pipeline - Pipeline name to filter by (used only when `tag` is empty).
 * @param tag - Tag to filter by; when non-empty, overrides the pipeline filter.
 * @param details - When `true`, return full {@link Reaction} details; when `false`, return ids/names.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @param limit - Maximum number of reactions to return per page (defaults to 1000).
 * @returns The list response (names or details, with a cursor), or `null` if the request failed.
 */
export async function listReactions(
  group: string,
  errorHandler: (error: string) => void,
  pipeline = '',
  tag = '',
  details = false,
  cursor: string | null = null,
  limit = 1000,
): Promise<ReactionDetailsListResponse | ReactionListResponse | null> {
  let url = '/reactions/';
  if (tag == '') {
    url += 'list/' + group + '/' + pipeline + '/';
  } else {
    url += 'tag/' + group + '/' + tag + '/';
  }
  if (details) {
    url += 'details/';
  }
  const params: { cursor?: string; limit?: number } = {};
  if (cursor) {
    params['cursor'] = cursor;
  }
  params['limit'] = limit;
  return client
    .get<ReactionDetailsListResponse | ReactionListResponse>(url, { params: params })
    .then((res) => {
      if (res?.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Reactions');
      return null;
    });
}

/**
 * Delete a reaction by id (`DELETE /reactions/{group}/{uuid}`).
 *
 * @param group - The group the reaction belongs to.
 * @param uuid - The reaction's unique id.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the reaction was deleted (HTTP 204), otherwise `false`.
 */
export async function deleteReaction(group: string, uuid: string, errorHandler: (error: string) => void): Promise<boolean> {
  const url = '/reactions/' + group + '/' + uuid;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete Reaction');
      return false;
    });
}
