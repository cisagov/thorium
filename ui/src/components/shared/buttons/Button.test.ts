import { describe, it, expect } from 'vitest';

import { ButtonVariant, ButtonSize } from './types';
import { SIZE_TOKENS, VARIANT_TOKENS, ICON_SIZE } from './tokens';

describe('ButtonVariant enum', () => {
  it('contains all expected variants', () => {
    expect(ButtonVariant.Primary).toBe('primary');
    expect(ButtonVariant.Secondary).toBe('secondary');
    expect(ButtonVariant.Ok).toBe('ok');
    expect(ButtonVariant.Danger).toBe('danger');
    expect(ButtonVariant.Warning).toBe('warning');
    expect(ButtonVariant.Info).toBe('info');
    expect(ButtonVariant.Ghost).toBe('ghost');
    expect(ButtonVariant.Icon).toBe('icon');
  });

  it('has exactly 8 variants', () => {
    expect(Object.values(ButtonVariant)).toHaveLength(8);
  });
});

describe('ButtonSize enum', () => {
  it('contains all expected sizes', () => {
    expect(ButtonSize.XSmall).toBe('xs');
    expect(ButtonSize.Small).toBe('sm');
    expect(ButtonSize.Medium).toBe('md');
    expect(ButtonSize.Large).toBe('lg');
  });

  it('has exactly 4 sizes', () => {
    expect(Object.values(ButtonSize)).toHaveLength(4);
  });
});

describe('SIZE_TOKENS', () => {
  it('has an entry for every ButtonSize', () => {
    for (const size of Object.values(ButtonSize)) {
      expect(SIZE_TOKENS[size]).toBeDefined();
      expect(SIZE_TOKENS[size].padding).toBeTruthy();
      expect(SIZE_TOKENS[size].fontSize).toBeTruthy();
    }
  });

  it('padding increases with size', () => {
    const sizes = [ButtonSize.XSmall, ButtonSize.Small, ButtonSize.Medium, ButtonSize.Large];
    for (let i = 1; i < sizes.length; i++) {
      const prevVertical = parseFloat(SIZE_TOKENS[sizes[i - 1]].padding.split(' ')[0]);
      const currVertical = parseFloat(SIZE_TOKENS[sizes[i]].padding.split(' ')[0]);
      expect(currVertical).toBeGreaterThanOrEqual(prevVertical);
    }
  });

  it('fontSize increases with size', () => {
    const sizes = [ButtonSize.XSmall, ButtonSize.Small, ButtonSize.Medium, ButtonSize.Large];
    for (let i = 1; i < sizes.length; i++) {
      const prevSize = parseFloat(SIZE_TOKENS[sizes[i - 1]].fontSize);
      const currSize = parseFloat(SIZE_TOKENS[sizes[i]].fontSize);
      expect(currSize).toBeGreaterThan(prevSize);
    }
  });
});

describe('VARIANT_TOKENS', () => {
  it('has an entry for every ButtonVariant', () => {
    for (const variant of Object.values(ButtonVariant)) {
      const token = VARIANT_TOKENS[variant];
      expect(token).toBeDefined();
      expect(token.bg).toBeTruthy();
      expect(token.border).toBeTruthy();
      expect(token.text).toBeTruthy();
      expect(token.hoverBg).toBeTruthy();
      expect(token.hoverBorder).toBeTruthy();
      expect(token.focusRing).toBeTruthy();
    }
  });

  it('filled variants use --thorium-button-text for text color', () => {
    const filled = [
      ButtonVariant.Primary,
      ButtonVariant.Secondary,
      ButtonVariant.Ok,
      ButtonVariant.Danger,
      ButtonVariant.Warning,
      ButtonVariant.Info,
    ];
    for (const v of filled) {
      expect(VARIANT_TOKENS[v].text).toBe('var(--thorium-button-text)');
    }
  });

  it('ghost and icon variants use --thorium-text for text color', () => {
    expect(VARIANT_TOKENS[ButtonVariant.Ghost].text).toBe('var(--thorium-text)');
    expect(VARIANT_TOKENS[ButtonVariant.Icon].text).toBe('var(--thorium-text)');
  });

  it('ghost and icon variants have transparent backgrounds', () => {
    expect(VARIANT_TOKENS[ButtonVariant.Ghost].bg).toBe('transparent');
    expect(VARIANT_TOKENS[ButtonVariant.Icon].bg).toBe('transparent');
  });

  it('filled variants reference thorium CSS variables for backgrounds', () => {
    const filled = [
      ButtonVariant.Primary,
      ButtonVariant.Secondary,
      ButtonVariant.Ok,
      ButtonVariant.Danger,
      ButtonVariant.Warning,
      ButtonVariant.Info,
    ];
    for (const v of filled) {
      expect(VARIANT_TOKENS[v].bg).toMatch(/^var\(--thorium-/);
    }
  });

  it('each variant maps to a distinct background', () => {
    const bgs = Object.values(ButtonVariant).map((v) => VARIANT_TOKENS[v].bg);
    const filled = bgs.filter((bg) => bg !== 'transparent');
    const unique = new Set(filled);
    expect(unique.size).toBe(filled.length);
  });
});

describe('ICON_SIZE', () => {
  it('has an entry for every ButtonSize', () => {
    for (const size of Object.values(ButtonSize)) {
      expect(ICON_SIZE[size]).toBeDefined();
      expect(ICON_SIZE[size]).toMatch(/^\d+px$/);
    }
  });

  it('icon dimensions increase with size', () => {
    const sizes = [ButtonSize.XSmall, ButtonSize.Small, ButtonSize.Medium, ButtonSize.Large];
    for (let i = 1; i < sizes.length; i++) {
      const prevDim = parseInt(ICON_SIZE[sizes[i - 1]]);
      const currDim = parseInt(ICON_SIZE[sizes[i]]);
      expect(currDim).toBeGreaterThan(prevDim);
    }
  });
});
