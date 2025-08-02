import { getImage, listGroups, listImages } from '@thorpi';
import { Group } from 'models';

/**
 * Get a single image spec from the Thorium API
 * @param {any} image image object containing a name and group field
 * @param {(image: any) => void} setImage function to set the updated image
 * @param {(loading: boolean) => void} setLoading function to set whether currently making API request
 * @returns {Promise} a promise for the request action
 */
export async function fetchSingleImage(image: any, setImage: (image: any) => void, setLoading: (loading: boolean) => void) {
  setLoading(true);
  if (image?.group && image?.name) {
    const reqImage = await getImage(image.group, image.name);
    if (reqImage) {
      setImage(reqImage.data);
    }
  }
  setLoading(false);
}

/**
 * Get a list of images from a users accessible groups
 * @param {string[]} groups A list of string groups a user can see
 * @param {(images: any[]) => void} setImages useState setter for image details
 * @param {boolean} cancelUpdate boolean indicating do not update if component is mounted
 * @param {(error: string) => void} setError an auth hook for validating cookies when 401s are returned
 * @param {(loading: boolean) => void} setLoading is the fetch function loading new data
 * @param {boolean} details whether to fetch image details or just names
 * @returns {object} async promise for request
 */
export async function fetchImages(
  groups: string[],
  setImages: (images: any[]) => void,
  cancelUpdate: boolean,
  setError: (error: string) => void,
  setLoading: (loading: boolean) => void,
  details = false,
) {
  if (typeof setLoading == 'function') setLoading(true);

  const images = [];
  if (groups && Array.isArray(groups) && groups.length) {
    // Get image details for each group
    for (const group of groups) {
      const reqImages = await listImages(group, setError, details, null, 1000);
      if (reqImages) {
        // request successful, pass back image details in a list or just a list of names
        if (details) images.push(...reqImages);
        else images.push(...reqImages.names);
      } else {
        // when request failed, reset images to empty
        setImages([]);
      }
    }
    // don't update if component isn't mounted
    // this occurs on redirect to login page
    if (!cancelUpdate) {
      if (details) {
        setImages(images.sort((a, b) => (a.group + a.name).localeCompare(b.group + b.name)));
      } else {
        setImages(images);
      }
    }
  }
  if (typeof setLoading == 'function') setLoading(false);
}

// get a list of groups to get group roles
/**
 * Get details for a users groups
 * @param {(groups: {[name: string]: Group} | Group[] | string[]) => void} setGroups useState setter for group details
 * @param {() => void} checkCookie an auth hook for validating cookies when 401s are returned
 * @param {(isLoading: boolean) => void} setLoading whether there is currently an outstanding API request
 * @param {boolean} details whether to return group details in object or list of group names
 * @returns {object} async promise for request
 */
export async function fetchGroups(
  setGroups: (groups: { [name: string]: Group } | Group[] | string[]) => void,
  setLoading: (isLoading: boolean) => void,
  details = false,
  returnType = 'Object',
) {
  // set loading when interacting w/ API
  if (typeof setLoading == 'function') setLoading(true);
  // get groups list from API
  const reqGroups = await listGroups(console.log, details);
  if (reqGroups !== null) {
    // get group details
    if (details) {
      const groupDetailsList = reqGroups as Group[];
      const allGroups: { [name: string]: Group } = {};
      for (const group of groupDetailsList) {
        allGroups[group.name] = group;
      }
      setGroups(allGroups);
      // set group names list
    } else {
      const groupNameList = reqGroups as string[];
      setGroups([...groupNameList.sort()]);
    }
  } else {
    // request failed for some reason, reset groups to falsey
    if (returnType == 'Object') {
      setGroups({});
    } else {
      setGroups([]);
    }
  }

  // done interacting w/ API, set loading to false
  if (typeof setLoading == 'function') setLoading(false);
}
