import { PanelVariant } from './types';

export interface VariantToken {
  bg: string;
  text: string;
  border: string;
  borderRadius: string;
  overflow: string;
}

export const VARIANT_TOKENS: Record<PanelVariant, VariantToken> = {
  [PanelVariant.Standard]: {
    bg: 'var(--thorium-panel-bg)',
    text: 'var(--thorium-text)',
    border: '1px solid var(--thorium-panel-border)',
    borderRadius: '0.25rem',
    overflow: 'visible',
  },
  [PanelVariant.Flush]: {
    bg: 'var(--thorium-panel-bg)',
    text: 'var(--thorium-text)',
    border: 'none',
    borderRadius: '0.25rem',
    overflow: 'visible',
  },
  [PanelVariant.Result]: {
    bg: 'var(--thorium-panel-bg)',
    text: 'var(--thorium-text)',
    border: 'none',
    borderRadius: '0.25rem',
    overflow: 'auto',
  },
  [PanelVariant.Outlined]: {
    bg: 'transparent',
    text: 'var(--thorium-text)',
    border: '1px solid var(--thorium-panel-border)',
    borderRadius: '0.25rem',
    overflow: 'visible',
  },
};
