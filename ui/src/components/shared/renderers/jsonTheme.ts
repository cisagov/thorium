import { useEffect, useState } from 'react';

// spec: ./SPEC.md

/**
 * base16 "Ocean" theme for `react-json-tree`, theme-aware via `--thorium-*` for the base background.
 *
 * The token colors are dark-scheme pastels; on the light-background themes (Light/Crab) they must be
 * inverted to remain legible — pair this with {@link useJsonTreeInvert} so the tree reads at adequate
 * contrast in all four themes.
 *
 * Shared by the tool-result JSON display and the modular {@link JsonRenderer}.
 */
export const OceanJsonTheme = {
  scheme: 'Ocean',
  author: 'Chris Kempson (http://chriskempson.com)',
  // theme-aware panel background pulled from styles/colors.scss
  base00: 'var(--thorium-panel-bg)',
  base01: '#343d46',
  base02: '#4f5b66',
  base03: '#65737e',
  base04: '#a7adba',
  base05: '#c0c5ce',
  base06: '#dfe1e8',
  base07: '#eff1f5',
  base08: '#bf616a',
  base09: '#d08770',
  base0A: '#ebcb8b',
  base0B: '#a3be8c',
  base0C: '#96b5b4',
  base0D: '#8fa1b3',
  base0E: '#b48ead',
  base0F: '#ab7967',
};

/** The theme names that use a light (near-white / cream) panel background. */
const LIGHT_BACKGROUND_THEMES = new Set(['Light', 'Crab']);

/**
 * Reactively report whether `react-json-tree` should invert {@link OceanJsonTheme}.
 *
 * The active theme is stored as a `theme` attribute on the `#root` element (set by the auth context).
 * Its token palette is tuned for dark backgrounds, so on the light-background themes (Light/Crab) the
 * palette must be inverted to keep the JSON tree legible. This hook reads that attribute and re-renders
 * when it changes so a live theme switch updates the tree colors without a reload.
 *
 * @returns `true` when the active theme has a light background and the palette should be inverted.
 */
export function useJsonTreeInvert(): boolean {
  const [invert, setInvert] = useState<boolean>(() => readInvert());
  useEffect(() => {
    // the theme attribute lives on #root; re-read it whenever that attribute mutates
    const root = document.getElementById('root');
    if (!root) return;
    const update = () => setInvert(readInvert());
    update();
    const observer = new MutationObserver(update);
    observer.observe(root, { attributes: true, attributeFilter: ['theme'] });
    return () => observer.disconnect();
  }, []);
  return invert;
}

/** Read the current `#root` theme attribute and decide whether the JSON palette needs inverting. */
function readInvert(): boolean {
  if (typeof document === 'undefined') return false;
  const theme = document.getElementById('root')?.getAttribute('theme') ?? '';
  return LIGHT_BACKGROUND_THEMES.has(theme);
}
