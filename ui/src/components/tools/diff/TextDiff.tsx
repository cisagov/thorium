// spec: ../ToolResult.spec.md
import React from 'react';
import ReactDiffViewer, { DiffMethod } from 'react-diff-viewer-continued';
import styled from 'styled-components';

const DiffWrapper = styled.div`
  font-size: 0.85rem;
  /* let the emotion-styled diff table scroll within the overlay window */
  overflow: auto;
  background: var(--thorium-panel-bg);
`;

// Map the diff viewer's color variables onto the Thorium theme so it matches the rest of the UI.
// Added/removed keep tinted green/red (with theme text) for readability; neutral surfaces use theme
// panel/gutter vars. The same set is used for both light and dark since our `--thorium-*` vars
// already switch with the active theme.
const THEME_VARS = {
  diffViewerBackground: 'var(--thorium-panel-bg)',
  diffViewerColor: 'var(--thorium-text)',
  diffViewerTitleBackground: 'var(--thorium-nav-panel-bg)',
  diffViewerTitleColor: 'var(--thorium-text)',
  diffViewerTitleBorderColor: 'var(--thorium-panel-border)',
  gutterBackground: 'var(--thorium-secondary-panel-bg)',
  gutterBackgroundDark: 'var(--thorium-secondary-panel-bg)',
  gutterColor: 'var(--thorium-secondary-text)',
  emptyLineBackground: 'var(--thorium-panel-bg)',
  highlightBackground: 'var(--thorium-highlight-panel-bg)',
  highlightGutterBackground: 'var(--thorium-highlight-panel-bg)',
  codeFoldBackground: 'var(--thorium-secondary-panel-bg)',
  codeFoldGutterBackground: 'var(--thorium-secondary-panel-bg)',
  codeFoldContentColor: 'var(--thorium-secondary-text)',
  addedBackground: 'rgba(46, 160, 67, 0.18)',
  addedColor: 'var(--thorium-text)',
  addedGutterBackground: 'rgba(46, 160, 67, 0.30)',
  addedGutterColor: 'var(--thorium-text)',
  wordAddedBackground: 'rgba(46, 160, 67, 0.40)',
  removedBackground: 'rgba(248, 81, 73, 0.18)',
  removedColor: 'var(--thorium-text)',
  removedGutterBackground: 'rgba(248, 81, 73, 0.30)',
  removedGutterColor: 'var(--thorium-text)',
  wordRemovedBackground: 'rgba(248, 81, 73, 0.40)',
};

const DIFF_STYLES = { variables: { dark: THEME_VARS, light: THEME_VARS } };

export interface TextDiffProps {
  oldValue: string;
  newValue: string;
  /** jsdiff comparison granularity; defaults to line-based. */
  method?: DiffMethod;
  oldTitle?: string;
  newTitle?: string;
}

/**
 * Thin adapter around `react-diff-viewer-continued` for diffing two text/JSON payloads, themed via
 * {@link DIFF_STYLES} to match the active Thorium theme.
 */
const TextDiff: React.FC<TextDiffProps> = ({ oldValue, newValue, method = DiffMethod.LINES, oldTitle, newTitle }) => {
  // the theme attribute lives on #root (set by the auth context), not documentElement; Light and Crab
  // both use light backgrounds, so only the remaining themes count as dark
  const theme = typeof document !== 'undefined' ? document.getElementById('root')?.getAttribute('theme') : null;
  const useDarkTheme = theme !== 'Light' && theme !== 'Crab';
  // react-diff-viewer-continued handles the YAML method structurally only at the line level; for
  // sub-line word highlighting it calls jsdiff's `diffYaml`, which does not exist and throws
  // `compareFunc is not a function` on any modified (not purely added/removed) line. Disabling word
  // diff for YAML keeps the structural line diff and avoids the crash.
  const disableWordDiff = method === DiffMethod.YAML;
  return (
    <DiffWrapper>
      <ReactDiffViewer
        oldValue={oldValue}
        newValue={newValue}
        compareMethod={method}
        disableWordDiff={disableWordDiff}
        useDarkTheme={useDarkTheme}
        styles={DIFF_STYLES}
        leftTitle={oldTitle}
        rightTitle={newTitle}
      />
    </DiffWrapper>
  );
};

export default TextDiff;
