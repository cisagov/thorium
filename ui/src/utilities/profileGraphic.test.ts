import { describe, it, expect } from 'vitest';

// project imports
import {
  isAllowedProfileGraphic,
  isUnresizableGraphic,
  PROFILE_GRAPHIC_ACCEPT,
  PROFILE_GRAPHIC_MAX_BYTES,
  profileGraphicFilename,
  validateProfileGraphic,
} from './profileGraphic';

const MB = 1024 * 1024;

describe('isAllowedProfileGraphic', () => {
  it('accepts the supported image and video types', () => {
    for (const type of ['image/png', 'image/jpeg', 'image/bmp', 'image/gif', 'video/mp4', 'video/webm']) {
      expect(isAllowedProfileGraphic(type)).toBe(true);
    }
  });

  it('rejects unsupported types', () => {
    for (const type of ['image/svg+xml', 'video/quicktime', 'image/tiff', 'application/pdf', '']) {
      expect(isAllowedProfileGraphic(type)).toBe(false);
    }
  });
});

describe('isUnresizableGraphic', () => {
  it('flags animated and video types as unresizable', () => {
    expect(isUnresizableGraphic('image/gif')).toBe(true);
    expect(isUnresizableGraphic('video/mp4')).toBe(true);
    expect(isUnresizableGraphic('video/webm')).toBe(true);
  });

  it('treats static raster images as resizable', () => {
    expect(isUnresizableGraphic('image/png')).toBe(false);
    expect(isUnresizableGraphic('image/jpeg')).toBe(false);
    expect(isUnresizableGraphic('image/bmp')).toBe(false);
  });
});

describe('validateProfileGraphic', () => {
  it('accepts a static image and marks it for resizing regardless of size', () => {
    expect(validateProfileGraphic({ type: 'image/png', size: 50 * MB })).toEqual({ ok: true, resize: true });
  });

  it('accepts an animated GIF or video under the cap without resizing', () => {
    expect(validateProfileGraphic({ type: 'image/gif', size: 9 * MB })).toEqual({ ok: true, resize: false });
    expect(validateProfileGraphic({ type: 'video/mp4', size: 1 })).toEqual({ ok: true, resize: false });
  });

  it('accepts an unresizable graphic exactly at the cap', () => {
    expect(validateProfileGraphic({ type: 'video/webm', size: PROFILE_GRAPHIC_MAX_BYTES })).toEqual({ ok: true, resize: false });
  });

  it('rejects an unresizable graphic over the cap with a size message', () => {
    const result = validateProfileGraphic({ type: 'video/mp4', size: PROFILE_GRAPHIC_MAX_BYTES + 1 });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('10 MB');
  });

  it('rejects an unsupported type before checking size', () => {
    const result = validateProfileGraphic({ type: 'video/quicktime', size: 1 });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('PNG');
  });
});

describe('profileGraphicFilename', () => {
  it('maps each accepted type to a matching extension', () => {
    expect(profileGraphicFilename('image/png')).toBe('icon.png');
    expect(profileGraphicFilename('image/jpeg')).toBe('icon.jpeg');
    expect(profileGraphicFilename('video/mp4')).toBe('icon.mp4');
    expect(profileGraphicFilename('video/webm')).toBe('icon.webm');
  });

  it('falls back to png for unknown types', () => {
    expect(profileGraphicFilename('application/octet-stream')).toBe('icon.png');
    expect(profileGraphicFilename('')).toBe('icon.png');
  });
});

describe('PROFILE_GRAPHIC_ACCEPT', () => {
  it('lists every accepted MIME type', () => {
    const types = PROFILE_GRAPHIC_ACCEPT.split(',');
    expect(types).toEqual(expect.arrayContaining(['image/png', 'image/jpeg', 'image/bmp', 'image/gif', 'video/mp4', 'video/webm']));
  });
});
