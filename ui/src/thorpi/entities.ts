import { Filters } from '@models/search';
import { EntityTypes } from '@models/entities/entities';
import client, { parseRequestError } from './client';

interface ApiCursor<T> {
  cursor?: string;
  data: T[];
}

/**
 * Create a new entity (`POST /entities/`).
 *
 * @param data - Multipart form data describing the entity (type, fields, groups, optional image).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns An object with the new entity's `id`, or `null` if the request failed.
 */
export const createEntity = async (data: FormData, errorHandler: (error: string) => void): Promise<{ id: string } | null> => {
  const url = '/entities/';
  return client
    .post<{ id: string }>(url, data)
    .then((res) => {
      if (res && res.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Create Entity');
      return null;
    });
};

/**
 * Fetch a single entity by id (`GET /entities/{id}`).
 *
 * @param id - The id of the entity to fetch.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The entity, or `null` if not found or the request failed.
 */
export const getEntity = async (id: string, errorHandler: (error: string) => void): Promise<EntityTypes | null> => {
  const url = `/entities/${id}`;
  return client
    .get<EntityTypes>(url)
    .then((res) => {
      if (res && res.status && res.status == 200 && res.data) {
        return res.data;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Get Entity');
      return null;
    });
};

/**
 * Update an existing entity (`PATCH /entities/{id}`).
 *
 * @param id - The id of the entity to update.
 * @param data - Multipart form data with the fields to change (and optionally a new image).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the update succeeded (HTTP 204), otherwise `false`.
 */
export const updateEntity = async (id: string, data: FormData, errorHandler: (error: string) => void): Promise<boolean> => {
  const url = `/entities/${id}`;
  return client
    .patch(url, data)
    .then((res) => {
      if (res && res.status && res.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Update Entity');
      return false;
    });
};

/**
 * Delete an entity by id (`DELETE /entities/{id}`).
 *
 * @param id - The id of the entity to delete.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns `true` if the deletion succeeded (HTTP 204), otherwise `false`.
 */
export const deleteEntity = async (id: string, errorHandler: (error: string) => void): Promise<boolean> => {
  const url = `/entities/${id}`;
  return client
    .delete(url)
    .then((res) => {
      if (res?.status && res.status == 204) {
        return true;
      }
      return false;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Delete Entity');
      return false;
    });
};

export interface EntityImage {
  url: string;
  isSvg: boolean;
}

/**
 * Fetch an entity's image as a browser-displayable object URL (`GET /entities/{id}/image`).
 *
 * The blob is wrapped in an object URL via `URL.createObjectURL`; callers are responsible
 * for revoking it with `URL.revokeObjectURL` when done to avoid leaking memory. This
 * function intentionally swallows errors (returns `null`) since a missing image is expected.
 *
 * @param entityId - The id of the entity whose image to fetch.
 * @returns An {@link EntityImage} with the object URL and an `isSvg` flag, or `null` if there is no image.
 */
export const fetchEntityImage = async (entityId: string): Promise<EntityImage | null> => {
  return client
    .get<Blob>(`/entities/${entityId}/image`, { responseType: 'blob' })
    .then((res) => {
      if (res?.status === 200 && res.data) {
        const blob = res.data;
        return {
          url: URL.createObjectURL(blob),
          isSvg: blob.type === 'image/svg+xml',
        };
      }
      return null;
    })
    .catch(() => null);
};

/**
 * List entities matching the given filters (`GET /entities` or `/entities/details/`).
 *
 * Results are paginated via an opaque cursor: pass the returned `entityCursor` back in on the
 * next call to fetch the following page (`null` cursor means no more pages).
 *
 * @param data - Search/filter parameters (groups, tags, time range, limit, etc.).
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param details - When `true`, request full entity objects (`/details/`) instead of summaries.
 * @param cursor - Pagination cursor from a previous call, or `null` for the first page.
 * @returns The page of entities and the next-page cursor (`entityCursor` is `null` when exhausted).
 *          On failure, returns an empty list and a `null` cursor.
 */
export const listEntities = async (
  data: Filters,
  errorHandler: (error: string) => void,
  details: boolean,
  cursor: string | null,
): Promise<{ entityList: EntityTypes[]; entityCursor: string | null }> => {
  let url = '/entities';
  if (details) {
    url += '/details/';
  }
  if (cursor) {
    data.cursor = cursor;
  }
  return client
    .get<ApiCursor<EntityTypes>>(url, { params: data })
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        const cursor = res.data.cursor ?? null;
        return { entityList: res.data.data, entityCursor: cursor };
      }
      return { entityList: [], entityCursor: null };
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'List Entity');
      return { entityList: [], entityCursor: null };
    });
};
