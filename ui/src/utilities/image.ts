/**
 * Compute dimensions that fit an image within a square bound without upscaling.
 *
 * Scales `width`/`height` down proportionally so the longest edge is at most `maxPx`.
 * Images already within the bound are returned unchanged (we never enlarge).
 *
 * Exported separately from {@link fileToResizedBlob} so the pure scaling math can be
 * unit-tested without a DOM/canvas.
 *
 * @param width - The source image width in pixels.
 * @param height - The source image height in pixels.
 * @param maxPx - The maximum allowed length of the longest edge.
 * @returns The scaled `{ width, height }`, rounded to whole pixels (minimum 1).
 */
export function computeScaledDimensions(width: number, height: number, maxPx: number): { width: number; height: number } {
  const longest = Math.max(width, height);
  // never upscale — only shrink images larger than the bound
  if (longest <= maxPx || longest === 0) {
    return { width, height };
  }
  const scale = maxPx / longest;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/**
 * Read an image {@link File} and produce a downscaled image {@link Blob}.
 *
 * The image is drawn onto an offscreen canvas scaled so its longest edge is at most
 * `maxPx` (see {@link computeScaledDimensions}), keeping the uploaded file small. Used to
 * build a user's profile icon before sending it to the API as multipart form data.
 *
 * @param file - The image file the user selected.
 * @param maxPx - The maximum length of the longest edge (defaults to 256).
 * @param mimeType - The output encoding (defaults to `image/png`).
 * @param quality - Encoder quality for lossy types like `image/jpeg` (0–1, defaults to 0.9).
 * @returns A promise resolving to the resized image as a {@link Blob}.
 * @throws If the file cannot be read or decoded as an image, the canvas context is unavailable,
 *   or the canvas fails to encode the image.
 */
export function fileToResizedBlob(file: File, maxPx = 256, mimeType = 'image/png', quality = 0.9): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read image file'));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('Failed to decode image file'));
      img.onload = () => {
        const { width, height } = computeScaledDimensions(img.naturalWidth, img.naturalHeight, maxPx);
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Canvas 2D context is unavailable'));
          return;
        }
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Failed to encode image'))), mimeType, quality);
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  });
}
