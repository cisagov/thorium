// project imports
import { getImage, listImages } from '@thorpi/images';
import { listGroups } from '@thorpi/groups';
import { Image, ImageList } from '@models/images';
import { Group } from '@models/groups';

/**
 * Fetch a single image and push it into component state, toggling a loading flag around the request.
 *
 * A no-op (other than the loading toggle) if `group` or `name` is missing, or if the image isn't found.
 *
 * @param image - Partial image identity; both `group` and `name` must be present to fetch.
 * @param setImage - State setter called with the fetched {@link Image} on success.
 * @param setLoading - State setter toggled `true` before the request and `false` after.
 */
export async function fetchSingleImage(
  image: { group?: string; name?: string },
  setImage: (image: Image) => void,
  setLoading: (loading: boolean) => void,
) {
  setLoading(true);
  if (image?.group && image?.name) {
    const reqImage = await getImage(image.group, image.name);
    if (reqImage) {
      setImage(reqImage);
    }
  }
  setLoading(false);
}

/**
 * Fetch and aggregate the images across multiple groups, then push them into component state.
 *
 * Iterates each group, concatenating either full {@link Image} objects (when `details`) or image
 * names. Detailed results are sorted by `group + name`. The `cancelUpdate` flag lets a caller
 * (e.g. an unmounting component or a superseded request) skip the final state update to avoid
 * setting state after the relevant view is gone.
 *
 * @param groups - The groups whose images to fetch.
 * @param setImages - State setter called with the aggregated images (typed by `details`).
 * @param cancelUpdate - When `true`, skip the final `setImages` call (results are discarded).
 * @param setError - Error callback forwarded to the underlying list request.
 * @param setLoading - State setter toggled `true` before the requests and `false` after.
 * @param details - When `true`, fetch full {@link Image} objects; when `false`, fetch only names.
 */
export async function fetchImages(
  groups: string[],
  setImages: ((images: Image[]) => void) | ((images: string[]) => void),
  cancelUpdate: boolean,
  setError: (error: string) => void,
  setLoading: (loading: boolean) => void,
  details = false,
) {
  if (typeof setLoading == 'function') setLoading(true);

  const images: (Image | string)[] = [];
  if (groups && Array.isArray(groups) && groups.length) {
    for (const group of groups) {
      const reqImages = await listImages(group, setError, details, null, 1000);
      if (reqImages) {
        if (details) {
          images.push(...(reqImages as Image[]));
        } else {
          const nameList = reqImages as ImageList;
          images.push(...nameList.names);
        }
      } else {
        (setImages as (images: never[]) => void)([]);
      }
    }
    if (!cancelUpdate) {
      if (details) {
        const detailImages = images as Image[];
        (setImages as (images: Image[]) => void)(detailImages.sort((a, b) => (a.group + a.name).localeCompare(b.group + b.name)));
      } else {
        (setImages as (images: string[]) => void)(images as string[]);
      }
    }
  }
  if (typeof setLoading == 'function') setLoading(false);
}

/**
 * Fetch the current user's groups and push them into component state, toggling a loading flag.
 *
 * When `details` is `true` the groups are returned as a `{ [name]: Group }` map keyed by group
 * name; otherwise as a sorted array of group names. On failure the empty shape is chosen by
 * `returnType` (`'Object'` → `{}`, anything else → `[]`).
 *
 * @param setGroups - State setter called with the groups as a map, details array, or name array.
 * @param setLoading - State setter toggled `true` before the request and `false` after.
 * @param details - When `true`, fetch full {@link Group} objects (returned as a name-keyed map).
 * @param returnType - Selects the empty value used on failure: `'Object'` for `{}`, otherwise `[]`.
 */
export async function fetchGroups(
  setGroups: (groups: { [name: string]: Group } | Group[] | string[]) => void,
  setLoading: (isLoading: boolean) => void,
  details = false,
  returnType = 'Object',
) {
  if (typeof setLoading == 'function') setLoading(true);
  const reqGroups = await listGroups(console.log, details);
  if (reqGroups !== null) {
    if (details) {
      const groupDetailsList = reqGroups as Group[];
      const allGroups: { [name: string]: Group } = {};
      for (const group of groupDetailsList) {
        allGroups[group.name] = group;
      }
      setGroups(allGroups);
    } else {
      const groupNameList = reqGroups as string[];
      setGroups([...groupNameList.sort()]);
    }
  } else {
    if (returnType == 'Object') {
      setGroups({});
    } else {
      setGroups([]);
    }
  }

  if (typeof setLoading == 'function') setLoading(false);
}
