import { describe, it, expect } from 'vitest';

// project imports
import { PanelVariant } from './types';
import { VARIANT_TOKENS } from './tokens';

describe('PanelVariant enum', () => {
  it('contains all expected variants', () => {
    expect(PanelVariant.Standard).toBe('standard');
    expect(PanelVariant.Flush).toBe('flush');
    expect(PanelVariant.Result).toBe('result');
    expect(PanelVariant.Outlined).toBe('outlined');
  });

  it('has exactly 4 variants', () => {
    expect(Object.values(PanelVariant)).toHaveLength(4);
  });
});

describe('VARIANT_TOKENS', () => {
  it('has an entry for every PanelVariant', () => {
    for (const variant of Object.values(PanelVariant)) {
      const token = VARIANT_TOKENS[variant];
      expect(token).toBeDefined();
      expect(token.bg).toBeTruthy();
      expect(token.text).toBeTruthy();
      expect(token.border).toBeTruthy();
      expect(token.borderRadius).toBeTruthy();
      expect(token.overflow).toBeTruthy();
    }
  });

  it('Standard variant uses panel bg and panel border', () => {
    const token = VARIANT_TOKENS[PanelVariant.Standard];
    expect(token.bg).toBe('var(--thorium-panel-bg)');
    expect(token.border).toContain('var(--thorium-panel-border)');
    expect(token.overflow).toBe('visible');
  });

  it('Flush variant has no border', () => {
    const token = VARIANT_TOKENS[PanelVariant.Flush];
    expect(token.bg).toBe('var(--thorium-panel-bg)');
    expect(token.border).toBe('none');
    expect(token.overflow).toBe('visible');
  });

  it('Result variant scrolls overflow and has no border', () => {
    const token = VARIANT_TOKENS[PanelVariant.Result];
    expect(token.bg).toBe('var(--thorium-panel-bg)');
    expect(token.border).toBe('none');
    expect(token.overflow).toBe('auto');
  });

  it('Outlined variant has transparent bg with visible border', () => {
    const token = VARIANT_TOKENS[PanelVariant.Outlined];
    expect(token.bg).toBe('transparent');
    expect(token.border).toContain('var(--thorium-panel-border)');
    expect(token.overflow).toBe('visible');
  });

  it('all variants use --thorium-text for text color', () => {
    for (const variant of Object.values(PanelVariant)) {
      expect(VARIANT_TOKENS[variant].text).toBe('var(--thorium-text)');
    }
  });

  it('all variants have consistent border-radius', () => {
    for (const variant of Object.values(PanelVariant)) {
      expect(VARIANT_TOKENS[variant].borderRadius).toBe('0.25rem');
    }
  });

  it('bordered variants reference --thorium-panel-border', () => {
    const bordered = [PanelVariant.Standard, PanelVariant.Outlined];
    for (const v of bordered) {
      expect(VARIANT_TOKENS[v].border).toMatch(/var\(--thorium-panel-border\)/);
    }
  });

  it('borderless variants use none', () => {
    const borderless = [PanelVariant.Flush, PanelVariant.Result];
    for (const v of borderless) {
      expect(VARIANT_TOKENS[v].border).toBe('none');
    }
  });
});
