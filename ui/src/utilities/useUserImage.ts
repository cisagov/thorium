import { useEffect, useState } from 'react';

// project imports
import { fetchUserImage } from '@thorpi/users';

/**
 * Lazily load a user's profile icon as an object URL.
 *
 * Fetches the icon from `GET /users/user/{username}/image`. When `hasImage` is explicitly `false`
 * the request is skipped, so the common "no icon" case costs nothing for callers that know the flag
 * (whoami-backed views). When `hasImage` is `undefined` — e.g. a submitter where only the username
 * is known — the fetch is attempted and a missing icon falls back silently (`fetchUserImage` treats
 * a 404 as an expected "no icon"). The created object URL is revoked automatically when the
 * user/flag changes or the component unmounts. Call the returned `reload` after the icon changes
 * (upload/remove) to force a refetch even when `hasImage` is unchanged.
 *
 * @param username - The user whose icon to load (skipped when undefined).
 * @param hasImage - Whether the user has an icon set (from `UserInfo.has_image`); `undefined` means
 *   unknown, in which case the fetch is attempted and falls back to no icon on a 404.
 * @param externalVersion - Optional caller-owned counter; changing it forces a refetch (e.g. after a
 *   parent uploads/removes the icon) without the caller needing the returned `reload`.
 * @returns The current object URL and its MIME type (both `null` when there is no icon) and a `reload`
 *   function to force a refetch.
 */
export function useUserImage(
  username: string | undefined,
  hasImage: boolean | undefined,
  externalVersion = 0,
): { imageUrl: string | null; imageType: string | null; reload: () => void } {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  // MIME type of the fetched icon, so callers can render <img> vs <video>
  const [imageType, setImageType] = useState<string | null>(null);
  // bumping this forces a refetch (e.g. after the icon is replaced at the same path)
  const [version, setVersion] = useState(0);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;

    // fetch unless the caller explicitly told us there is no icon (hasImage === false)
    if (username && hasImage !== false) {
      void fetchUserImage(username).then((image) => {
        // the effect was cleaned up before the fetch resolved — revoke and bail
        if (!active) {
          if (image) URL.revokeObjectURL(image.url);
          return;
        }
        objectUrl = image?.url ?? null;
        setImageUrl(image?.url ?? null);
        setImageType(image?.type ?? null);
      });
    } else {
      setImageUrl(null);
      setImageType(null);
    }

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [username, hasImage, version, externalVersion]);

  return { imageUrl, imageType, reload: () => setVersion((v) => v + 1) };
}
