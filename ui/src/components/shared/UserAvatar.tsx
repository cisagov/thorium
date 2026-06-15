import styled from 'styled-components';
import { FaCircleUser } from 'react-icons/fa6';

// project imports
import { useUserImage } from '@utilities/useUserImage';

// spec: ./UserAvatar.spec.md

/** Round avatar image sized to the caller-provided pixel dimension. */
const AvatarImage = styled.img<{ $size: number }>`
  width: ${(p) => p.$size}px;
  height: ${(p) => p.$size}px;
  border-radius: 50%;
  object-fit: cover;
  border: 1px solid var(--thorium-panel-border);
  flex: 0 0 auto;
`;

/** Round avatar video (MP4/WebM) sharing the image's sizing and cropping so both render identically. */
const AvatarVideo = styled.video<{ $size: number }>`
  width: ${(p) => p.$size}px;
  height: ${(p) => p.$size}px;
  border-radius: 50%;
  object-fit: cover;
  border: 1px solid var(--thorium-panel-border);
  flex: 0 0 auto;
`;

/**
 * Props for {@link UserAvatar}.
 */
export interface UserAvatarProps {
  /** The user whose profile icon to display. */
  username: string | undefined;
  /** The rendered width/height of the avatar in pixels. */
  size: number;
  /**
   * Whether the user is known to have an icon (from `UserInfo.has_image`). Pass `false` to skip the
   * fetch entirely; leave `undefined` when only the username is known (a fetch is attempted and the
   * placeholder is shown on a 404).
   */
  hasImage?: boolean;
  /** Caller-owned counter; changing it forces a refetch (e.g. after an upload/remove). */
  version?: number;
  /** Alt text for the image; defaults to a generic profile-icon label. */
  alt?: string;
  /** Optional class applied to the rendered image or placeholder. */
  className?: string;
}

/**
 * Display a user's profile icon, or a neutral placeholder when the user has no icon set.
 *
 * Wraps {@link useUserImage} so avatar rendering lives in one place (profile page, nav banner, user
 * list, and submitter lines). Video icons (MP4/WebM) render in a muted, looping, autoplaying
 * `<video>`; images and animated GIFs render in an `<img>`; when no icon is set the `FaCircleUser`
 * placeholder is shown at the same size.
 *
 * @param props - See {@link UserAvatarProps}.
 * @returns The avatar image, video, or placeholder icon.
 */
const UserAvatar: React.FC<UserAvatarProps> = ({ username, size, hasImage, version, alt = 'Profile icon', className }) => {
  const { imageUrl, imageType } = useUserImage(username, hasImage, version);
  // no icon set (or fetch failed) — show the neutral placeholder at the same footprint
  if (!imageUrl) {
    return <FaCircleUser size={size} className={className} />;
  }
  // video icons can't render in an <img>; play them muted/looping so they behave like an animated avatar
  if (imageType?.startsWith('video/')) {
    return <AvatarVideo src={imageUrl} $size={size} className={className} autoPlay loop muted playsInline aria-label={alt} />;
  }
  // images and animated GIFs (GIFs animate natively in an <img>)
  return <AvatarImage src={imageUrl} alt={alt} $size={size} className={className} />;
};

export default UserAvatar;
