import { EditorView } from '@codemirror/view';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { tags } from '@lezer/highlight';

export const thoriumEditorTheme = EditorView.theme({
  '&': {
    backgroundColor: 'var(--thorium-secondary-panel-bg)',
    color: 'var(--thorium-text)',
    border: '1px solid var(--thorium-panel-border)',
    borderRadius: '4px',
    fontSize: '13px',
    fontFamily: 'monospace',
  },
  '&.cm-focused': {
    outline: '2px solid var(--thorium-info-secondary-bg)',
    outlineOffset: '-1px',
  },
  '.cm-content': {
    caretColor: 'var(--thorium-text)',
    padding: '8px 0',
  },
  '.cm-gutters': {
    backgroundColor: 'var(--thorium-panel-bg)',
    color: 'var(--thorium-secondary-text)',
    borderRight: '1px solid var(--thorium-panel-border)',
    minWidth: '40px',
  },
  '.cm-activeLineGutter': {
    backgroundColor: 'var(--thorium-highlight-panel-bg)',
  },
  '.cm-activeLine': {
    backgroundColor: 'var(--thorium-highlight-panel-bg)',
  },
  '.cm-cursor, .cm-dropCursor': {
    borderLeftColor: 'var(--thorium-text)',
  },
  '.cm-selectionBackground': {
    backgroundColor: 'var(--thorium-highlight-panel-bg) !important',
  },
  '&.cm-focused .cm-selectionBackground': {
    backgroundColor: 'var(--thorium-info-secondary-bg) !important',
    opacity: '0.3',
  },
  '.cm-panels': {
    backgroundColor: 'var(--thorium-panel-bg)',
    color: 'var(--thorium-text)',
  },
  '.cm-tooltip': {
    backgroundColor: 'var(--thorium-panel-bg)',
    color: 'var(--thorium-text)',
    border: '1px solid var(--thorium-panel-border)',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  '.cm-tooltip-lint': {
    maxWidth: '500px',
  },
  '.cm-diagnostic': {
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word',
  },
  '.cm-tooltip-autocomplete': {
    '& > ul > li[aria-selected]': {
      backgroundColor: 'var(--thorium-highlight-panel-bg)',
    },
  },
  '.cm-searchMatch': {
    backgroundColor: 'var(--thorium-warning-bg)',
    opacity: '0.3',
  },
  '.cm-searchMatch.cm-searchMatch-selected': {
    backgroundColor: 'var(--thorium-info-secondary-bg)',
    opacity: '0.4',
  },
  '.cm-diagnostic-error': {
    borderLeft: '4px solid var(--thorium-danger-bg)',
    borderBottom: '2px solid var(--thorium-danger-bg)',
  },
  '.cm-diagnostic-warning': {
    borderLeft: '4px solid var(--thorium-warning-bg)',
    borderBottom: '2px solid var(--thorium-warning-bg)',
  },
  '.cm-diagnostic-info': {
    borderLeft: '4px solid var(--thorium-info-secondary-bg)',
    borderBottom: '2px solid var(--thorium-info-secondary-bg)',
  },
  '.cm-lintRange': {
    backgroundImage: 'none',
    textDecoration: 'underline wavy',
    textDecorationThickness: '2px',
    textUnderlineOffset: '3px',
    textDecorationSkipInk: 'none',
  },
  '.cm-lintRange-error': {
    textDecorationColor: 'var(--thorium-danger-bg)',
  },
  '.cm-lintRange-warning': {
    textDecorationColor: 'var(--thorium-warning-bg)',
  },
  '.cm-lintRange-info': {
    textDecorationColor: 'var(--thorium-info-secondary-bg)',
  },
  '.cm-lintRange-error.cm-lint-has-warning, .cm-lintRange-error.cm-lint-has-info': {
    textDecorationColor: 'var(--thorium-lint-overlap)',
  },
  '.cm-lintRange-warning.cm-lint-has-info': {
    textDecorationColor: 'var(--thorium-lint-overlap)',
  },
  '.cm-lint-marker-error': {
    content: '"!"',
  },
  '.cm-lint-marker-warning': {
    content: '"?"',
  },
});

const thoriumHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: 'var(--thorium-link-text)' },
  { tag: tags.string, color: 'var(--thorium-ok-bg)' },
  { tag: tags.number, color: 'var(--thorium-warning-bg)' },
  { tag: tags.bool, color: 'var(--thorium-warning-bg)' },
  { tag: tags.null, color: 'var(--thorium-secondary-text)' },
  { tag: tags.comment, color: 'var(--thorium-secondary-text)', fontStyle: 'italic' },
  { tag: tags.propertyName, color: 'var(--thorium-link-text-alt)' },
  { tag: tags.punctuation, color: 'var(--thorium-secondary-text)' },
  { tag: tags.meta, color: 'var(--thorium-highlight-text)' },
  { tag: tags.atom, color: 'var(--thorium-warning-bg)' },
  { tag: tags.definition(tags.variableName), color: 'var(--thorium-link-text-alt)' },
  { tag: tags.typeName, color: 'var(--thorium-link-text)' },
]);

export const thoriumHighlighting = syntaxHighlighting(thoriumHighlightStyle);
