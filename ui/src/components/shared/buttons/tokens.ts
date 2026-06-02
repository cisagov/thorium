import { ButtonVariant, ButtonSize } from './types';

export interface SizeToken {
  padding: string;
  fontSize: string;
}

export interface VariantToken {
  bg: string;
  border: string;
  text: string;
  hoverBg: string;
  hoverBorder: string;
  focusRing: string;
}

export const SIZE_TOKENS: Record<ButtonSize, SizeToken> = {
  [ButtonSize.XSmall]: { padding: '2px 6px', fontSize: '0.7rem' },
  [ButtonSize.Small]: { padding: '4px 10px', fontSize: '0.78rem' },
  [ButtonSize.Medium]: { padding: '6px 14px', fontSize: '0.85rem' },
  [ButtonSize.Large]: { padding: '8px 20px', fontSize: '0.95rem' },
};

const FILLED_BASE: Pick<VariantToken, 'text' | 'focusRing'> = {
  text: 'var(--thorium-button-text)',
  focusRing: 'var(--thorium-highlight-panel-border)',
};

export const VARIANT_TOKENS: Record<ButtonVariant, VariantToken> = {
  [ButtonVariant.Primary]: {
    ...FILLED_BASE,
    bg: 'var(--thorium-info-bg)',
    border: 'var(--thorium-info-bg)',
    hoverBg: 'var(--thorium-info-bg)',
    hoverBorder: 'var(--thorium-info-bg)',
  },
  [ButtonVariant.Secondary]: {
    ...FILLED_BASE,
    bg: 'var(--thorium-empty-bg)',
    border: 'var(--thorium-empty-bg)',
    hoverBg: 'var(--thorium-empty-bg)',
    hoverBorder: 'var(--thorium-empty-bg)',
  },
  [ButtonVariant.Ok]: {
    ...FILLED_BASE,
    bg: 'var(--thorium-ok-bg)',
    border: 'var(--thorium-ok-bg)',
    hoverBg: 'var(--thorium-ok-bg)',
    hoverBorder: 'var(--thorium-ok-bg)',
  },
  [ButtonVariant.Danger]: {
    ...FILLED_BASE,
    bg: 'var(--thorium-danger-bg)',
    border: 'var(--thorium-danger-bg)',
    hoverBg: 'var(--thorium-danger-bg)',
    hoverBorder: 'var(--thorium-danger-bg)',
  },
  [ButtonVariant.Warning]: {
    ...FILLED_BASE,
    bg: 'var(--thorium-warning-bg)',
    border: 'var(--thorium-warning-bg)',
    hoverBg: 'var(--thorium-warning-bg)',
    hoverBorder: 'var(--thorium-warning-bg)',
  },
  [ButtonVariant.Info]: {
    ...FILLED_BASE,
    bg: 'var(--thorium-info-secondary-bg)',
    border: 'var(--thorium-info-secondary-bg)',
    hoverBg: 'var(--thorium-info-secondary-bg)',
    hoverBorder: 'var(--thorium-info-secondary-bg)',
  },
  [ButtonVariant.Ghost]: {
    bg: 'transparent',
    border: 'transparent',
    text: 'var(--thorium-text)',
    hoverBg: 'var(--thorium-highlight-panel-bg)',
    hoverBorder: 'transparent',
    focusRing: 'var(--thorium-highlight-panel-border)',
  },
  [ButtonVariant.Icon]: {
    bg: 'transparent',
    border: 'transparent',
    text: 'var(--thorium-text)',
    hoverBg: 'transparent',
    hoverBorder: 'transparent',
    focusRing: 'var(--thorium-highlight-panel-border)',
  },
};

export const BUTTON_BAR_GAP = '2px';
export const BUTTON_BAR_MARGIN = '12px';

export const ICON_SIZE: Record<ButtonSize, string> = {
  [ButtonSize.XSmall]: '24px',
  [ButtonSize.Small]: '30px',
  [ButtonSize.Medium]: '36px',
  [ButtonSize.Large]: '44px',
};
