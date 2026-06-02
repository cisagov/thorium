import { Graph, Seed } from '@models/trees';
import client, { parseRequestError } from './client';
import JSONBigInt from 'json-bigint';

// this will force big ints to be strings
const JSONBigString = JSONBigInt({ storeAsString: true });

/**
 * Build the initial association graph from a set of seed nodes (`POST /trees/`).
 *
 * The response is parsed with a BigInt-aware JSON parser (storing big integers as strings) so
 * large numeric ids survive the round-trip without precision loss.
 *
 * @param data - The {@link Seed} describing the starting node(s) for the graph.
 * @param filterChildless - When `true`, omit nodes that have no children from the result.
 * @param limit - Maximum number of nodes to expand per growable node.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The built {@link Graph}, or `null` if the request failed.
 */
export const getInitialTree = async (
  data: Seed,
  filterChildless: boolean,
  limit: number,
  errorHandler: (error: string) => void,
): Promise<Graph | null> => {
  const url = '/trees/';
  const params: { filter_childless?: boolean; limit: number } = { limit };
  if (filterChildless) {
    params['filter_childless'] = filterChildless;
  }
  return client
    .post<string>(url, { ...data }, { transformResponse: [(data: string) => data], params: params })
    .then((res) => {
      if (res && res.status && res.status == 200) {
        return JSONBigString.parse(res.data) as Graph;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Build Tree');
      return null;
    });
};

/**
 * Expand an existing association graph by growing from the given nodes (`PATCH /trees/{id}`).
 *
 * Like {@link getInitialTree}, the response is parsed with a BigInt-aware JSON parser to
 * preserve large numeric ids.
 *
 * @param id - The id of the existing graph/tree to grow.
 * @param nodes - The ids of the growable nodes to expand.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param limit - Maximum number of children to expand per node (defaults to 1).
 * @returns The updated {@link Graph}, or `null` if the request failed.
 */
export const growTree = async (id: string, nodes: string[], errorHandler: (error: string) => void, limit = 1): Promise<Graph | null> => {
  const url = `/trees/${id}`;
  const params: { limit: number } = { limit };
  return client
    .patch<string>(url, { growable: nodes }, { transformResponse: [(data: string) => data], params: params })
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        return JSONBigString.parse(res.data) as Graph;
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Grow Tree');
      return null;
    });
};
