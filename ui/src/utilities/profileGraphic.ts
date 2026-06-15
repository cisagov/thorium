/**
 * Shared rules for what may be used as a user's profile graphic (avatar).
 *
 * The backend accepts static images plus animated GIFs and short MP4/WebM clips. Static raster images
 * are downscaled client-side to a small PNG before upload; animated GIFs and video are uploaded as-is
 * (a canvas resize would flatten a GIF to a single frame and cannot encode video), so a byte cap bounds
 * what a user can stream into the graphics bucket. Keeping the type/extension rules here keeps the file
 * picker, the upload call, and the avatar renderer in agreement.
 */

/**
 * Maximum size (in bytes) for a profile graphic that is uploaded without client-side resizing.
 *
 * Applies to animated GIFs and video, which can't be cheaply downscaled in the browser. Mirrors the
 * `ICON_MAX_BYTES` limit enforced by the API's upload route.
 */
export const PROFILE_GRAPHIC_MAX_BYTES = 10 * 1024 * 1024;

/**
 * Accepted profile-graphic MIME types mapped to the file extension the backend stores them under.
 *
 * The keys are the single source of truth for both the file input's `accept` attribute and client-side
 * validation; the values match the extensions the API derives from each content type.
 */
export const PROFILE_GRAPHIC_EXTENSIONS: Readonly<Record<string, string>> = {
  'image/png': 'png',
  'image/jpeg': 'jpeg',
  'image/bmp': 'bmp',
  'image/gif': 'gif',
  'video/mp4': 'mp4',
  'video/webm': 'webm',
};

/**
 * MIME types that must be uploaded as-is rather than resized.
 *
 * A canvas resize would collapse an animated GIF to a single frame and cannot encode video at all, so
 * these bypass {@link fileToResizedBlob} and are size-checked against {@link PROFILE_GRAPHIC_MAX_BYTES}
 * instead.
 */
const UNRESIZABLE_GRAPHIC_TYPES: ReadonlySet<string> = new Set(['image/gif', 'video/mp4', 'video/webm']);

/** The `accept` attribute value for a profile-graphic file input, derived from the accepted types. */
export const PROFILE_GRAPHIC_ACCEPT = Object.keys(PROFILE_GRAPHIC_EXTENSIONS).join(',');

/**
 * Whether a MIME type is accepted as a profile graphic.
 *
 * @param type - The file's MIME type (e.g. from `File.type`).
 * @returns `true` if the type is in the accepted set.
 */
export function isAllowedProfileGraphic(type: string): boolean {
  return type in PROFILE_GRAPHIC_EXTENSIONS;
}

/**
 * Whether a graphic must be uploaded as-is (animated GIF or video) rather than resized to a PNG.
 *
 * @param type - The file's MIME type.
 * @returns `true` if the type cannot be safely resized on a canvas.
 */
export function isUnresizableGraphic(type: string): boolean {
  return UNRESIZABLE_GRAPHIC_TYPES.has(type);
}

/**
 * The outcome of validating a selected file for use as a profile graphic.
 *
 * On success, `resize` says whether the file should be downscaled on a canvas before upload (static
 * images) or uploaded as-is (animated GIF / video). On failure, `error` is a user-facing message.
 */
export type ProfileGraphicCheck = { ok: true; resize: boolean } | { ok: false; error: string };

/**
 * Validate a selected file against the profile-graphic type and size rules.
 *
 * Rejects unsupported MIME types, and rejects unresizable graphics (animated GIF / video) larger than
 * {@link PROFILE_GRAPHIC_MAX_BYTES} — static images have no client size limit here since they're
 * downscaled before upload. Kept pure (no DOM) so it can be unit-tested directly.
 *
 * @param file - The selected file's MIME `type` and `size` in bytes.
 * @returns Whether the file is acceptable and, if so, whether it needs resizing.
 */
export function validateProfileGraphic(file: { type: string; size: number }): ProfileGraphicCheck {
  // reject anything outside the accepted image/video set
  if (!isAllowedProfileGraphic(file.type)) {
    return { ok: false, error: 'Please choose a PNG, JPEG, BMP, GIF, MP4, or WebM file.' };
  }
  // animated GIFs and video are uploaded as-is, so bound their size here (static images are resized)
  const resize = !isUnresizableGraphic(file.type);
  if (!resize && file.size > PROFILE_GRAPHIC_MAX_BYTES) {
    const maxMb = Math.round(PROFILE_GRAPHIC_MAX_BYTES / (1024 * 1024));
    return { ok: false, error: `Animated and video profile pictures must be ${maxMb} MB or smaller.` };
  }
  return { ok: true, resize };
}

/**
 * Build the multipart filename for an uploaded profile graphic based on its MIME type.
 *
 * The API derives the stored extension from the part's Content-Type, not this name, but a matching
 * extension keeps the request self-consistent. Unknown types fall back to `.png`.
 *
 * @param type - The blob's MIME type.
 * @returns A filename like `icon.mp4`.
 */
export function profileGraphicFilename(type: string): string {
  const ext = PROFILE_GRAPHIC_EXTENSIONS[type] ?? 'png';
  return `icon.${ext}`;
}
